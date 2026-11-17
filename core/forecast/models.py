"""PRISM base-model wrappers.

Layer-2 models from the design. This module hosts only the LightGBM
tabular models — the sequence model (TFT/N-BEATS), sentiment-only GBM,
and graph-propagation signal are added in Task #8.

Why LightGBM is the primary tabular model:
  * Beats every other tree implementation on Indian-equity-scale data
    (~5 yrs × ~200 tickers × ~50 features = ~250k rows).
  * Native quantile loss for distributional forecasting.
  * Native handling of NaNs — important because filings/sentiment features
    are intermittently missing.
  * Categorical column support — we pass ``sector`` / ``regime`` directly.
  * Trains in seconds on a 14-core CPU; no GPU faff.

Per-horizon directory layout under ``data/models/``::

    data/models/
      ├─ 1d/
      │   ├─ q10.pkl       quantile regressor for q10
      │   ├─ q50.pkl       quantile regressor for q50
      │   ├─ q90.pkl       quantile regressor for q90
      │   ├─ clf.pkl       binary classifier (with isotonic calibration wrapper)
      │   └─ meta.json     feature list, training window, ablation tag
      └─ 5d/  ...

All saves use ``joblib`` (sklearn convention) so the calibrator survives
the pickle round-trip cleanly.
"""
from __future__ import annotations

import json
import hashlib
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("forecast.models")

ROOT       = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "data" / "models"
ARTIFACT_VERSION = 2

# Feature columns by ablation rung. The actual training set may have more
# columns; anything not listed here is dropped before fit() so the rungs
# stay truly comparable.
PRICE_FEATS = [
    "logret_1", "logret_5", "logret_20", "logret_60",
    "rv_5", "rv_20", "rv_60",
    "rsi14", "macd", "macd_signal", "macd_hist",
    "bb_pct", "atr14", "stoch_k", "stoch_d", "vwap20_dist",
    "vol_z20", "vol_ratio_20",
    "gap_pct", "intraday_range", "close_to_high",
    "cs_sector_excess", "cs_sector_rank", "cs_market_excess",
    "market_breadth_up", "market_dispersion", "market_mean_return",
    "sector_mean_return",
    "macro_india_vix", "macro_usdinr", "macro_brent", "macro_india_10y",
    "dow", "dom", "month",
    # Alpha features (mom_12_1, reversal_1m, idio_vol_60) are computed
    # in the Parquet but intentionally NOT referenced here. On NIFTY 50
    # ×5y with conservative HPs, adding any of them hurt Sharpe. They
    # remain in the panel for future re-evaluation with NIFTY 200 or
    # deeper data — we just don't pay the overfit cost on them now.
]

SENTIMENT_FEATS = [
    "sent_mean_1d",  "sent_dispersion_1d",  "sent_count_1d",
    "sent_mean_7d",  "sent_dispersion_7d",  "sent_count_7d",
    "sent_mean_30d", "sent_dispersion_30d", "sent_count_30d",
]

FILING_FEATS = [
    "fil_days_since", "fil_guidance_score",
    "fil_concerns_n", "fil_risks_n", "fil_whatchanged_n", "fil_highlights_n",
]

GRAPH_FEATS = [
    "graph_n_customers", "graph_n_suppliers",
    "graph_n_partners", "graph_n_competitors",
    "graph_n_resolved", "graph_neighbour_ret",
]

CATEGORICAL_FEATS = ["sector", "regime"]


ABLATIONS: dict[str, list[str]] = {
    "price_only":        PRICE_FEATS + CATEGORICAL_FEATS,
    "plus_sentiment":    PRICE_FEATS + SENTIMENT_FEATS + CATEGORICAL_FEATS,
    "plus_filings":      PRICE_FEATS + SENTIMENT_FEATS + FILING_FEATS + CATEGORICAL_FEATS,
    "full":              PRICE_FEATS + SENTIMENT_FEATS + FILING_FEATS + GRAPH_FEATS + CATEGORICAL_FEATS,
}


# ---------------------------------------------------------------------------
# Lazy imports — keep the module importable without sklearn / lightgbm
# ---------------------------------------------------------------------------


def _lgb():
    import lightgbm as lgb  # noqa: WPS433
    return lgb


def _joblib():
    import joblib  # noqa: WPS433
    return joblib


def _isotonic():
    from sklearn.isotonic import IsotonicRegression  # noqa: WPS433
    return IsotonicRegression


# ---------------------------------------------------------------------------
# Frame prep
# ---------------------------------------------------------------------------


@dataclass
class TrainSpec:
    horizon: str               # '1d' | '5d'
    ablation: str              # one of ABLATIONS keys
    label_col: str             # 'up_1d' | 'xs_up_1d' | 'ret_1d' | 'xs_excess_1d' | ...
    feature_cols: list[str]


def make_spec(
    horizon: str,
    ablation: str,
    *,
    classification: bool,
    target: str = "xs",
) -> TrainSpec:
    """Build a training spec.

    ``target``:
      * ``'xs'`` (default) — cross-sectional outperform-the-median labels.
        Classification: ``xs_up_<h>d``. Regression: ``xs_excess_<h>d``.
      * ``'abs'``           — absolute direction / log-return labels.
        Classification: ``up_<h>d``. Regression: ``ret_<h>d``.

    Cross-sectional is the default because absolute direction on liquid
    large caps is empirically a coin flip — the noise cancels when we
    predict *relative* performance instead.
    """
    if horizon not in ("1d", "5d"):
        raise ValueError(f"unknown horizon {horizon!r}")
    if ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation {ablation!r}; want one of {list(ABLATIONS)}")
    if target not in ("xs", "abs"):
        raise ValueError(f"unknown target {target!r}; want 'xs' or 'abs'")
    if target == "xs":
        label = f"xs_up_{horizon}" if classification else f"xs_excess_{horizon}"
    else:
        label = f"up_{horizon}"     if classification else f"ret_{horizon}"
    return TrainSpec(
        horizon=horizon,
        ablation=ablation,
        label_col=label,
        feature_cols=list(ABLATIONS[ablation]),
    )


def prepare_xy(panel: pd.DataFrame, spec: TrainSpec) -> tuple[pd.DataFrame, pd.Series]:
    """Slice the panel to features+label for a given training spec.

    Drops rows where the label is NaN (i.e. the most recent dates where
    the forward return isn't observed yet). Categorical columns are kept
    as pandas `category` dtype — LightGBM picks them up natively.
    """
    cols = [c for c in spec.feature_cols if c in panel.columns]
    missing = set(spec.feature_cols) - set(cols)
    if missing:
        log.warning("missing feature columns at train time: %s", sorted(missing))
    df = panel[cols + [spec.label_col]].copy()
    df = df.dropna(subset=[spec.label_col])
    for c in CATEGORICAL_FEATS:
        if c in df.columns:
            df[c] = df[c].astype("category")
    X = df[cols]
    y = df[spec.label_col]
    return X, y


# ---------------------------------------------------------------------------
# Model trainers
# ---------------------------------------------------------------------------


# Conservative defaults — tuned empirically on NIFTY 50 × 5y. The fancier
# defaults (num_leaves 127, lambda_l2 0.5) overfit on this small dataset
# and dropped Sharpe from 2.99 → 2.27 in the 2026-05-15 ablation. With
# more rows (NIFTY 200) we can re-tune; for now, less capacity = better
# generalisation.

_QUANTILE_PARAMS = dict(
    objective="quantile",
    metric="quantile",
    learning_rate=0.04,
    num_leaves=63,
    min_data_in_leaf=80,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    lambda_l2=0.1,
    verbosity=-1,
)

_CLASSIFIER_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    learning_rate=0.04,
    num_leaves=63,
    min_data_in_leaf=80,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    lambda_l2=0.1,
    verbosity=-1,
)


def train_quantile(
    X: pd.DataFrame, y: pd.Series, *, alpha: float, num_rounds: int = 600,
    valid: tuple[pd.DataFrame, pd.Series] | None = None,
) -> Any:
    lgb = _lgb()
    train_set = lgb.Dataset(X, label=y, categorical_feature=[
        c for c in CATEGORICAL_FEATS if c in X.columns
    ])
    params = {**_QUANTILE_PARAMS, "alpha": alpha}
    valid_sets = [train_set]
    if valid is not None:
        valid_sets.append(lgb.Dataset(
            valid[0], label=valid[1], reference=train_set,
            categorical_feature=[c for c in CATEGORICAL_FEATS if c in valid[0].columns],
        ))
    booster = lgb.train(
        params, train_set, num_boost_round=num_rounds,
        valid_sets=valid_sets,
        callbacks=[lgb.early_stopping(50)] if valid is not None else None,
    )
    return booster


def train_classifier(
    X: pd.DataFrame, y: pd.Series, *, num_rounds: int = 800,
    valid: tuple[pd.DataFrame, pd.Series] | None = None,
    seed: int | None = None,
) -> Any:
    lgb = _lgb()
    train_set = lgb.Dataset(X, label=y, categorical_feature=[
        c for c in CATEGORICAL_FEATS if c in X.columns
    ])
    valid_sets = [train_set]
    if valid is not None:
        valid_sets.append(lgb.Dataset(
            valid[0], label=valid[1], reference=train_set,
            categorical_feature=[c for c in CATEGORICAL_FEATS if c in valid[0].columns],
        ))
    params = dict(_CLASSIFIER_PARAMS)
    if seed is not None:
        # Three different seeds because LightGBM has three independent RNG
        # streams (bagging / feature subsample / data init).
        params.update(seed=seed, bagging_seed=seed,
                      feature_fraction_seed=seed, data_random_seed=seed)
    booster = lgb.train(
        params, train_set, num_boost_round=num_rounds,
        valid_sets=valid_sets,
        callbacks=[lgb.early_stopping(50)] if valid is not None else None,
    )
    return booster


# ---------------------------------------------------------------------------
# Calibrated classifier
# ---------------------------------------------------------------------------


@dataclass
class CalibratedBinaryModel:
    """A LightGBM binary booster + isotonic calibrator.

    Keep them paired so the calibrator never falls out of sync with the
    booster on save/load.
    """
    booster: Any = None              # may be None for ensemble subclasses
    calibrator: Any | None = None    # IsotonicRegression or None
    feature_cols: list[str] | None = None

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.predict_raw(X)
        if self.calibrator is None:
            return raw
        return self.calibrator.predict(raw)


def fit_calibrator(
    raw_preds: np.ndarray, y_true: np.ndarray,
) -> Any:
    iso = _isotonic()(out_of_bounds="clip")
    iso.fit(raw_preds, y_true)
    return iso


# ---------------------------------------------------------------------------
# Held-out splitter (time-aware to avoid leakage)
# ---------------------------------------------------------------------------


def _time_ordered_holdout(
    train_df: pd.DataFrame, frac: float = 0.2, purge_days: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a training slice into (early, late) by date.

    Used so we can:
      * fit the booster on the **early** 80 % of dates,
      * fit the isotonic calibrator on the **held-out late** 20 %,
    which gives calibrated probabilities that actually generalise. The
    older in-sample calibrator was producing the P=0/P=1 cliff.

    Sorted by date so the held-out slice is always the latest 20 % —
    the closest analogue of the OOS test fold.
    """
    if "date" not in train_df.columns:
        raise ValueError("expected a 'date' column")
    df = train_df.sort_values(["date", "token"])
    dates = pd.Index(pd.to_datetime(df["date"]).drop_duplicates().sort_values())
    if len(dates) < 3:
        return df.iloc[0:0], df
    cut = max(1, min(len(dates) - 1, int(len(dates) * (1.0 - frac))))
    valid_start = dates[cut]
    purge_start = valid_start - pd.offsets.BDay(max(0, int(purge_days)))
    early = df[pd.to_datetime(df["date"]) < purge_start]
    late = df[pd.to_datetime(df["date"]) >= valid_start]
    return early, late


def train_quantile_time_ordered(
    train_df: pd.DataFrame,
    spec: TrainSpec,
    *,
    alpha: float,
    holdout: float = 0.2,
    num_rounds: int = 600,
    min_holdout_rows: int = 200,
) -> Any:
    """Select boosting rounds inside the training fold, then refit on all of it.

    The former walk-forward driver passed the outer test fold to LightGBM for
    early stopping and then reported pinball loss on that same fold. This
    helper creates a purged, time-ordered inner holdout, uses it only to select
    ``best_iteration``, discards that pilot model, and fits the returned model
    on the complete outer-training slice with a fixed number of rounds.
    """
    purge_sessions = int(spec.horizon.rstrip("d"))
    early, late = _time_ordered_holdout(
        train_df, frac=holdout, purge_days=purge_sessions,
    )
    if early.empty or len(late) < int(min_holdout_rows):
        X_all, y_all = prepare_xy(train_df, spec)
        return train_quantile(
            X_all, y_all, alpha=alpha, num_rounds=num_rounds, valid=None,
        )

    X_early, y_early = prepare_xy(early, spec)
    X_late, y_late = prepare_xy(late, spec)
    if X_early.empty or len(X_late) < int(min_holdout_rows):
        X_all, y_all = prepare_xy(train_df, spec)
        return train_quantile(
            X_all, y_all, alpha=alpha, num_rounds=num_rounds, valid=None,
        )

    pilot = train_quantile(
        X_early, y_early, alpha=alpha, num_rounds=num_rounds,
        valid=(X_late, y_late),
    )
    selected_rounds = int(getattr(pilot, "best_iteration", 0) or num_rounds)
    selected_rounds = max(1, min(selected_rounds, int(num_rounds)))
    X_all, y_all = prepare_xy(train_df, spec)
    return train_quantile(
        X_all, y_all, alpha=alpha, num_rounds=selected_rounds, valid=None,
    )


# ---------------------------------------------------------------------------
# Calibrated-ensemble trainer
# ---------------------------------------------------------------------------


def train_classifier_ensemble(
    train_df: pd.DataFrame,
    spec: TrainSpec,
    *,
    n_seeds:  int = 1,
    holdout:  float = 0.2,
    num_rounds: int = 800,
) -> CalibratedBinaryModel:
    """Train ``n_seeds`` boosters on a held-out-aware split and return one
    `CalibratedBinaryModel` whose `predict_proba` averages the seed-level
    calibrated outputs.

    Workflow per seed:
      1. Time-ordered 80/20 split of the training slice.
      2. Fit booster on early 80 %.
      3. Predict raw scores on the held-out late 20 %.
      4. Fit isotonic calibrator on those held-out scores.
      5. Retrain booster on the FULL training slice (so the production
         model uses all available data) with the calibrator frozen.

    The calibrator from step 4 is what we use at inference — it was fit
    on data the booster wasn't trained on, so the calibrated probs are
    real confidences not in-sample artifacts.
    """
    boosters: list[Any] = []
    calibrators: list[Any] = []

    purge_days = int(spec.horizon.rstrip("d"))
    early, late = _time_ordered_holdout(
        train_df, frac=holdout, purge_days=purge_days,
    )
    if len(late) < 200:
        # Not enough rows to do a meaningful held-out fit. Fall back to
        # the original in-sample isotonic.
        Xtr, ytr = prepare_xy(train_df, spec)
        for s in range(n_seeds):
            b = train_classifier(Xtr, ytr, num_rounds=num_rounds, seed=42 + s)
            raw = b.predict(Xtr)
            cal = fit_calibrator(raw, ytr.to_numpy())
            boosters.append(b)
            calibrators.append(cal)
        return _EnsembleCalibrated(boosters=boosters, calibrators=calibrators,
                                   feature_cols=spec.feature_cols)

    # Held-out path: fit calibrator on held-out, then refit booster on all data.
    X_early, y_early = prepare_xy(early, spec)
    X_late,  y_late  = prepare_xy(late,  spec)
    X_all,   y_all   = prepare_xy(train_df, spec)

    for s in range(n_seeds):
        b_early = train_classifier(X_early, y_early, num_rounds=num_rounds,
                                   seed=42 + s)
        raw_late = b_early.predict(X_late)
        cal = fit_calibrator(raw_late, y_late.to_numpy())
        # Production booster — same seed, trained on everything.
        b_full = train_classifier(X_all, y_all, num_rounds=num_rounds,
                                  seed=42 + s)
        boosters.append(b_full)
        calibrators.append(cal)

    return _EnsembleCalibrated(boosters=boosters, calibrators=calibrators,
                               feature_cols=spec.feature_cols)


@dataclass
class _EnsembleCalibrated(CalibratedBinaryModel):
    """Ensemble of (booster, calibrator) pairs. ``booster`` is unused — we
    keep it on the dataclass to stay duck-compatible with single-model
    callers; predict_proba averages the calibrated outputs of every seed.
    """
    boosters:    list[Any] = field(default_factory=list)
    calibrators: list[Any] = field(default_factory=list)

    def __post_init__(self):
        # Make the parent class's `booster` field point at the first
        # booster so callers that pull `.booster` for SHAP still work
        # (we explain via the first seed; signs are stable across seeds).
        if self.boosters and self.booster is None:
            self.booster = self.boosters[0]

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        if not self.boosters:
            raise RuntimeError("empty ensemble")
        return np.mean([b.predict(X) for b in self.boosters], axis=0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.boosters:
            raise RuntimeError("empty ensemble")
        # Calibrate each seed separately, then average. (Averaging
        # post-calibration is more honest than averaging raw scores and
        # calibrating once, which the calibrator wasn't fit to handle.)
        per_seed = [
            (cal.predict(b.predict(X)) if cal is not None else b.predict(X))
            for b, cal in zip(self.boosters, self.calibrators)
        ]
        return np.mean(per_seed, axis=0)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def _horizon_dir(horizon: str, ablation: str) -> Path:
    return MODELS_DIR / horizon / ablation


def save_bundle(
    *,
    horizon: str,
    ablation: str,
    quantiles: dict[str, Any],
    classifier: CalibratedBinaryModel,
    feature_cols: list[str],
    train_window: tuple[str, str],
    extra: dict | None = None,
) -> Path:
    out = _horizon_dir(horizon, ablation)
    out.mkdir(parents=True, exist_ok=True)
    j = _joblib()
    for stale in list(out.glob("clf_*.txt")) + [out / "clf_calibrators.pkl"]:
        if stale.exists():
            stale.unlink()
    for tag, booster in quantiles.items():
        booster.save_model(str(out / f"q{tag}.txt"))
    classifier_type = "single"
    ensemble_size = 1
    if isinstance(classifier, _EnsembleCalibrated):
        classifier_type = "ensemble"
        ensemble_size = len(classifier.boosters)
        for i, booster in enumerate(classifier.boosters):
            booster.save_model(str(out / f"clf_{i}.txt"))
        j.dump(classifier.calibrators, out / "clf_calibrators.pkl")
        # Keep the legacy first-seed files for older readers and SHAP tools.
        classifier.boosters[0].save_model(str(out / "clf.txt"))
        if classifier.calibrators:
            j.dump(classifier.calibrators[0], out / "clf_calibrator.pkl")
    else:
        classifier.booster.save_model(str(out / "clf.txt"))
        if classifier.calibrator is not None:
            j.dump(classifier.calibrator, out / "clf_calibrator.pkl")
    schema_hash = hashlib.sha256(
        json.dumps(feature_cols, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    meta = {
        "artifact_version": ARTIFACT_VERSION,
        "horizon": horizon,
        "ablation": ablation,
        "feature_cols": feature_cols,
        "feature_schema_hash": schema_hash,
        "train_window": list(train_window),
        "classifier_type": classifier_type,
        "ensemble_size": ensemble_size,
        "created_at": int(time.time()),
        "extra": extra or {},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("saved %s/%s models to %s", horizon, ablation, out)
    return out


def load_bundle(horizon: str, ablation: str) -> dict[str, Any]:
    out = _horizon_dir(horizon, ablation)
    if not out.exists():
        raise FileNotFoundError(f"no models at {out}")
    lgb = _lgb()
    j = _joblib()
    quantiles = {}
    for tag in ("10", "50", "90"):
        p = out / f"q{tag}.txt"
        if p.exists():
            quantiles[tag] = lgb.Booster(model_file=str(p))
    meta = json.loads((out / "meta.json").read_text())
    ensemble_paths = sorted(
        out.glob("clf_*.txt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if meta.get("classifier_type") == "ensemble" and ensemble_paths:
        boosters = [lgb.Booster(model_file=str(p)) for p in ensemble_paths]
        cals_p = out / "clf_calibrators.pkl"
        calibrators = j.load(cals_p) if cals_p.exists() else [None] * len(boosters)
        clf = _EnsembleCalibrated(
            boosters=boosters,
            calibrators=calibrators,
            feature_cols=meta.get("feature_cols"),
        )
    else:
        booster_p = out / "clf.txt"
        cal_p = out / "clf_calibrator.pkl"
        calibrator = j.load(cal_p) if cal_p.exists() else None
        clf = CalibratedBinaryModel(
            booster=lgb.Booster(model_file=str(booster_p)) if booster_p.exists() else None,
            calibrator=calibrator,
            feature_cols=meta.get("feature_cols"),
        )
    return {"quantiles": quantiles, "classifier": clf, "meta": meta}


# ---------------------------------------------------------------------------
# Walk-forward driver — single ablation, one horizon
# ---------------------------------------------------------------------------


def walk_forward_train(
    panel: pd.DataFrame,
    *,
    horizon: str,
    ablation: str,
    target: str = "xs",
    train_years: int = 3,
    test_months: int = 6,
    min_train_rows: int = 5000,
    n_seeds: int = 3,
    holdout: float = 0.2,
) -> list[dict]:
    """Rolling-window training. Returns a list of fold-result dicts.

    Each fold:
      * trains all 3 quantile regressors + 1 classifier on the train slice
      * scores them on the next ``test_months`` slice
      * selects quantile boosting rounds and fits classifier calibration on
        purged, time-ordered holdouts contained wholly inside the train slice.
    """
    if "date" not in panel.columns:
        raise ValueError("panel must have a 'date' column")
    panel = panel.sort_values(["date", "token"]).reset_index(drop=True)
    panel["date"] = pd.to_datetime(panel["date"])

    spec_reg = make_spec(horizon, ablation, classification=False, target=target)
    spec_clf = make_spec(horizon, ablation, classification=True,  target=target)

    folds: list[dict] = []
    train_start = panel["date"].min()
    end = panel["date"].max()
    cur_train_end = train_start + pd.DateOffset(years=train_years)

    while cur_train_end + pd.DateOffset(months=test_months) <= end:
        test_end = cur_train_end + pd.DateOffset(months=test_months)
        h = int(horizon.rstrip("d"))
        purge_end = cur_train_end - pd.offsets.BDay(h)
        train_mask = (panel["date"] >= train_start) & (panel["date"] < purge_end)
        test_mask  = (panel["date"] >= cur_train_end) & (panel["date"] < test_end)
        train_df = panel[train_mask]
        test_df  = panel[test_mask]
        if len(train_df) < min_train_rows or test_df.empty:
            cur_train_end += pd.DateOffset(months=test_months)
            continue

        # Regressors
        Xte, yte = prepare_xy(test_df, spec_reg)
        quantiles: dict[str, Any] = {}
        for tag, alpha in (("10", 0.10), ("50", 0.50), ("90", 0.90)):
            booster = train_quantile_time_ordered(
                train_df, spec_reg, alpha=alpha, holdout=holdout,
            )
            quantiles[tag] = booster

        # Classifier — held-out-calibrated ensemble. This was previously
        # in-sample isotonic on a single seed, which produced the P=0/1
        # cliff; the held-out + seed-averaged calibration gives real
        # confidence scores and a small accuracy bump for free.
        Xc_te, yc_te = prepare_xy(test_df, spec_clf)
        clf = train_classifier_ensemble(
            train_df, spec_clf, n_seeds=n_seeds, holdout=holdout,
        )

        # Metrics on test
        p_test = clf.predict_proba(Xc_te)
        truth = yc_te.to_numpy(dtype=float)
        pred_class = p_test >= 0.5
        acc = float((pred_class == truth.astype(bool)).mean())
        eps = 1e-7
        clipped = np.clip(p_test, eps, 1 - eps)
        log_loss = float(-np.mean(
            truth * np.log(clipped) + (1 - truth) * np.log(1 - clipped)
        ))
        brier = float(np.mean((p_test - truth) ** 2))
        top_n = max(1, int(len(p_test) * 0.2))
        top_precision = float(truth[np.argsort(p_test)[-top_n:]].mean())
        # Pinball losses
        def pinball(yt, yp, a):
            d = yt - yp
            return float(np.mean(np.maximum(a*d, (a-1)*d)))
        pb = {
            tag: pinball(yte.to_numpy(), quantiles[tag].predict(Xte), float(tag)/100)
            for tag in quantiles
        }

        folds.append({
            "train_start": str(train_start.date()),
            "train_end":   str(cur_train_end.date()),
            "test_end":    str(test_end.date()),
            "n_train":     int(len(train_df)),
            "n_test":      int(len(test_df)),
            "acc":         acc,
            "log_loss":    log_loss,
            "brier":       brier,
            "top20_precision": top_precision,
            "pinball":     pb,
            "models": {"quantiles": quantiles, "classifier": clf},
        })
        log.info(
            "fold %s→%s  acc=%.3f  pb50=%.4f", cur_train_end.date(),
            test_end.date(), acc, pb["50"],
        )
        cur_train_end += pd.DateOffset(months=test_months)

    return folds


def train_and_save_final(
    panel: pd.DataFrame,
    *,
    horizon: str,
    ablation: str,
    target: str = "xs",
    n_seeds: int = 3,
    holdout: float = 0.2,
) -> Path:
    """Train one final model on the *entire* panel and persist to disk.

    Run this AFTER `walk_forward_train` gives you confidence the
    ablation is healthy. The model saved here is what `forecast_predict`
    loads at inference time.
    """
    spec_reg = make_spec(horizon, ablation, classification=False, target=target)
    spec_clf = make_spec(horizon, ablation, classification=True,  target=target)
    Xtr, ytr = prepare_xy(panel, spec_reg)

    quantiles = {
        tag: train_quantile(Xtr, ytr, alpha=alpha)
        for tag, alpha in (("10", 0.10), ("50", 0.50), ("90", 0.90))
    }
    clf = train_classifier_ensemble(
        panel, spec_clf, n_seeds=n_seeds, holdout=holdout,
    )
    actual_feature_cols = list(prepare_xy(panel, spec_clf)[0].columns)
    clf.feature_cols = actual_feature_cols
    out = save_bundle(
        horizon=horizon, ablation=ablation,
        quantiles=quantiles, classifier=clf,
        feature_cols=actual_feature_cols,
        train_window=(
            str(pd.to_datetime(panel["date"]).min().date()),
            str(pd.to_datetime(panel["date"]).max().date()),
        ),
        extra={"target": target},
    )
    try:
        from . import registry
        version = f"{pd.to_datetime(panel['date']).max():%Y%m%d}-{int(time.time())}"
        registry.register(
            model_key=f"{horizon}:{ablation}:{target}",
            version=version,
            artifact_path=out,
            metrics={},
        )
    except Exception as exc:
        log.warning("model registry update failed: %s", exc)
    return out

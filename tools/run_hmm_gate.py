"""HMM regime gate (paper Section 3.2 / Fig. 4). Design: docs/HMM_GATE_DESIGN.md.

Per fold, for each stock a 3-state Gaussian HMM is fitted on that stock's 1-day log
returns strictly before the fold's test window; every day up to the end of the test
window is then labelled from the *forward-filtered* state probabilities (no smoothing,
no Viterbi), so a label depends only on observations up to that day. The highest-variance
state is `chop`; of the other two, the one with the larger absolute mean is `trend`, the
other `meanrev`. Everything else (features, LightGBM ranker, purge, folds, seed, expert
floor, IC and paired inference) is identical to the main replay.

Usage:
    python tools/run_hmm_gate.py [--out outputs/hmm] [--max-folds N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import regenerate_mixture_ic_series as rep  # noqa: E402
import robust_inference as ri  # noqa: E402
from core.forecast import features, ranking, regime, regime_mixture  # noqa: E402

N_STATES = 3
MIN_OBS = 500
N_ITER = 100
TOL = 1e-3
SEED = 42


def _fit_and_label(returns_pct: np.ndarray, fit_mask: np.ndarray):
    """Fit on returns[fit_mask]; return regime labels (object array) for all days, or None."""
    from hmmlearn import _hmmc, hmm

    fit = returns_pct[fit_mask & np.isfinite(returns_pct)]
    if len(fit) < MIN_OBS:
        return None, "too_little_history"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = hmm.GaussianHMM(
                n_components=N_STATES, covariance_type="diag", n_iter=N_ITER,
                tol=TOL, random_state=SEED)
            model.fit(fit.reshape(-1, 1))
        var = np.array([np.atleast_2d(c)[0, 0] for c in model.covars_], dtype=float)
        mean = np.asarray(model.means_, dtype=float).ravel()
        chop_state = int(np.argmax(var))
        rest = [s for s in range(N_STATES) if s != chop_state]
        trend_state = rest[int(np.argmax(np.abs(mean[rest])))]
        names = {chop_state: "chop", trend_state: "trend"}
        for s in rest:
            names.setdefault(s, "meanrev")
        obs = np.where(np.isfinite(returns_pct), returns_pct, 0.0).reshape(-1, 1)
        log_frameprob = model._compute_log_likelihood(obs)
        _, fwd = _hmmc.forward_log(model.startprob_, model.transmat_, log_frameprob)
        state = np.argmax(fwd, axis=1)
        return np.array([names[int(s)] for s in state], dtype=object), "ok"
    except Exception as exc:  # noqa: BLE001 - counted and reported
        return None, f"fit_error:{type(exc).__name__}"


def label_fold(data: pd.DataFrame, tok_index: dict, fold_start, test_end) -> tuple[pd.DataFrame, dict]:
    """Return rows with date < test_end and a causal HMM `regime` column."""
    window = data[data["date"] < test_end].copy()
    window["regime"] = None
    stats = {"ok": 0, "too_little_history": 0, "fit_error": 0}
    for tok, idx in tok_index.items():
        g = window.loc[window.index.intersection(idx)]
        if g.empty:
            continue
        g = g.sort_values("date")
        r = g["logret_1"].to_numpy(dtype=float) * 100.0
        fit_mask = (g["date"] < fold_start).to_numpy()
        labels, status = _fit_and_label(r, fit_mask)
        key = "fit_error" if status.startswith("fit_error") else status
        stats[key] += 1
        if labels is not None:
            window.loc[g.index, "regime"] = labels
    return window, stats


def replay_hmm(panel: pd.DataFrame, *, config: regime_mixture.RegimeMixtureConfig,
               train_years: int = 3, test_months: int = 6, max_folds: int | None = None) -> dict:
    base_config = ranking.RankingConfig(**config.base_config)
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"])
    start, end = data["date"].min(), data["date"].max()
    fold_start = start + pd.DateOffset(years=train_years)
    tok_index = {tok: g.index for tok, g in data.groupby("token", sort=False)}

    per_date_rows: list[pd.DataFrame] = []
    folds: list[dict] = []
    fold_index = 0
    while fold_start + pd.DateOffset(months=test_months) <= end:
        test_end = fold_start + pd.DateOffset(months=test_months)
        probe_train = ranking._purged_before(data, fold_start)
        probe_test = data[(data["date"] >= fold_start) & (data["date"] < test_end)]
        if probe_train.empty or probe_test.empty:
            fold_start = test_end
            continue
        fold_index += 1
        if max_folds is not None and fold_index > max_folds:
            break
        t0 = time.time()
        window, fit_stats = label_fold(data, tok_index, fold_start, test_end)
        train = ranking._purged_before(window, fold_start)
        test = window[(window["date"] >= fold_start) & (window["date"] < test_end)]

        stable_cols = ranking.stable_feature_columns(train, ablation=base_config.ablation)
        pooled_artifact = ranking.train_ranker(train, base_config, feature_cols=stable_cols)
        pooled_scored = ranking.score(pooled_artifact, test)
        specialists, coverage = {}, {}
        for label in regime.REGIMES:
            rtrain = train[train["regime"] == label]
            coverage[label] = {"train_rows": int(len(rtrain)),
                               "used_specialist": bool(len(rtrain) >= config.min_regime_train_rows)}
            if coverage[label]["used_specialist"]:
                specialists[label] = ranking.train_ranker(rtrain, base_config, feature_cols=stable_cols)
        mixture_scored = regime_mixture._score_regime_routed(pooled_artifact, specialists, test)

        for label in regime.REGIMES:
            pic = rep._ic_frame(pooled_scored, label)
            mic = rep._ic_frame(mixture_scored, label)
            common = pic.index.intersection(mic.index)
            if len(common):
                per_date_rows.append(pd.DataFrame({
                    "date": pd.to_datetime(pd.Series(list(common))), "regime": label,
                    "fold": fold_index, "pooled_ic": pic.loc[common].to_numpy(),
                    "mixture_ic": mic.loc[common].to_numpy()}))
        share = test["regime"].value_counts(normalize=True).round(4).to_dict()
        folds.append({"fold": fold_index, "test_start": str(fold_start.date()),
                      "test_end": str(test_end.date()), "hmm_fits": fit_stats,
                      "test_regime_share": share, "specialist_coverage": coverage})
        print(f"  fold {fold_index:2d}  {fold_start.date()} -> {test_end.date()}  fits={fit_stats} "
              f"share={share}  ({time.time() - t0:.0f}s)", flush=True)
        fold_start = test_end
    return {"folds": folds, "per_date": pd.concat(per_date_rows, ignore_index=True)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/hmm")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--max-folds", type=int, default=None)
    args = ap.parse_args()

    published = json.loads(rep.PUBLISHED.read_text(encoding="utf-8"))
    config = regime_mixture.RegimeMixtureConfig(
        base_config=dict(published["config"]["base_config"]),
        min_regime_train_rows=int(published["config"]["min_regime_train_rows"]))
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    import hmmlearn
    print(f"hmmlearn {hmmlearn.__version__}; loading feature panel ...", flush=True)
    panel = features.load_features(start=args.start)
    result = replay_hmm(panel, config=config, max_folds=args.max_folds)

    csv_path = out_dir / "ic_by_date.csv"
    result["per_date"].sort_values(["regime", "date"]).to_csv(csv_path, index=False, date_format="%Y-%m-%d")
    inference = ri.run(csv_path, lags=5, mean_block=10.0, n_boot=10_000, seed=42)
    payload = {
        "design": "docs/HMM_GATE_DESIGN.md",
        "hmm": {"n_states": N_STATES, "min_obs": MIN_OBS, "n_iter": N_ITER, "tol": TOL,
                "random_state": SEED, "covariance_type": "diag", "hmmlearn": hmmlearn.__version__},
        "n_folds_run": len(result["folds"]),
        "folds": result["folds"],
        "inference": inference,
    }
    (out_dir / "hmm_gate_result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\npaired mixture - pooled (Newey-West L=5):")
    for r in ri.REGIMES:
        s = inference["regimes"][r]["paired_mixture_minus_pooled"]
        print(f"  {r:8s} diff={s['mean_ic']:+.4f}  t={s['t_newey_west']:+.2f}  p={s['p_newey_west']:.4f}"
              f"  CI=[{s['boot_ci95'][0]:+.4f}, {s['boot_ci95'][1]:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

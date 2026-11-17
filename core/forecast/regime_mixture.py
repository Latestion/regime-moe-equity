"""Regime-specialist mixture ranking model.

Every design in `selective.py` changes how a single, ordinary cross-sectional
ranker's *output* is used downstream (gated, blended, tilted into a low-vol
book) — the prediction algorithm itself never changed, and every one of those
designs failed to beat a trivial portfolio baseline once tested rigorously
(see `PREREGISTRATION_SELECTIVE.md` §16). This module instead changes the
*training* of the forecaster: rather than one model pooled across all market
regimes, it trains a separate specialist model per regime
(`core.forecast.regime.REGIMES`) and routes each out-of-sample prediction to
the specialist matching that day's regime, falling back to the pooled model
where a regime has too little training data to fit its own specialist.

This is evaluated purely on out-of-sample forecasting accuracy — regime-
stratified rank IC, Bonferroni-corrected, identical methodology to Stage 0/1
— never on a downstream portfolio Sharpe. That keeps the question clean:
does conditioning the *model* on regime produce a better forecast than
conditioning a downstream *decision* on regime, which is what every prior
design in this session tried and failed to do.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from . import backtest, ranking, regime, research_statistics, selective


@dataclass(frozen=True)
class RegimeMixtureConfig:
    base_config: dict
    min_regime_train_rows: int = 5000

    def __post_init__(self) -> None:
        if self.min_regime_train_rows < 1:
            raise ValueError("min_regime_train_rows must be positive")


def nested_walk_forward_regime_mixture(
    panel: pd.DataFrame,
    *,
    config: RegimeMixtureConfig,
    train_years: int = 3,
    test_months: int = 6,
) -> dict:
    """Fold structure mirrors Stage 0/1's `_ranking_oos_scores`: train strictly
    before each test window (purged), no inner-validation split needed since
    nothing here is fit or gated on validation data — both the pooled
    baseline and the per-regime specialists are trained the same way, only
    differing in which rows they see.
    """
    base_config = ranking.RankingConfig(**config.base_config)
    data = panel.copy()
    if "regime" not in data.columns:
        data = regime.attach_to_panel(data)
    data["date"] = pd.to_datetime(data["date"])
    start, end = data["date"].min(), data["date"].max()
    fold_start = start + pd.DateOffset(years=train_years)

    folds: list[dict] = []
    pooled_frames: list[pd.DataFrame] = []
    mixture_frames: list[pd.DataFrame] = []

    while fold_start + pd.DateOffset(months=test_months) <= end:
        test_end = fold_start + pd.DateOffset(months=test_months)
        train = ranking._purged_before(data, fold_start)
        test = data[(data["date"] >= fold_start) & (data["date"] < test_end)]
        if train.empty or test.empty:
            fold_start = test_end
            continue

        stable_cols = ranking.stable_feature_columns(train, ablation=base_config.ablation)
        pooled_artifact = ranking.train_ranker(train, base_config, feature_cols=stable_cols)
        pooled_scored = ranking.score(pooled_artifact, test)
        pooled_frames.append(pooled_scored)

        specialist_coverage: dict[str, dict] = {}
        specialists: dict[str, ranking.RankingArtifact] = {}
        for regime_label in regime.REGIMES:
            regime_train = train[train["regime"] == regime_label]
            specialist_coverage[regime_label] = {"train_rows": int(len(regime_train))}
            if len(regime_train) >= config.min_regime_train_rows:
                specialists[regime_label] = ranking.train_ranker(
                    regime_train, base_config, feature_cols=stable_cols,
                )
                specialist_coverage[regime_label]["used_specialist"] = True
            else:
                specialist_coverage[regime_label]["used_specialist"] = False

        mixture_scored = _score_regime_routed(pooled_artifact, specialists, test)
        mixture_frames.append(mixture_scored)

        folds.append({
            "train_end": str(pd.to_datetime(train["date"]).max().date()),
            "test_start": str(fold_start.date()),
            "test_end": str(test_end.date()),
            "specialist_coverage": specialist_coverage,
        })
        fold_start = test_end

    pooled_all = (
        pd.concat(pooled_frames, ignore_index=True) if pooled_frames else pd.DataFrame()
    )
    mixture_all = (
        pd.concat(mixture_frames, ignore_index=True) if mixture_frames else pd.DataFrame()
    )
    pooled_report = _regime_ic_report(pooled_all) if not pooled_all.empty else None
    mixture_report = _regime_ic_report(mixture_all) if not mixture_all.empty else None
    paired = _paired_ic_comparison(pooled_all, mixture_all)

    return {
        "config": {**asdict(config), "base_config": asdict(base_config)},
        "folds": folds,
        "pooled_baseline": pooled_report,
        "mixture": mixture_report,
        "paired_comparison": paired,
    }


def portfolio_walk_forward_regime_mixture(
    panel: pd.DataFrame,
    *,
    config: RegimeMixtureConfig,
    train_years: int = 3,
    test_months: int = 6,
) -> dict:
    """Same fold structure and model fitting as
    `nested_walk_forward_regime_mixture`, but simulates both the pooled
    baseline and the regime-routed mixture through the identical portfolio
    ledger (`ranking._simulate_scored`) per fold, exactly as
    `ranking.nested_walk_forward` does for its own oos_returns. This answers
    the question the rank-IC comparison above cannot: does the mixture
    model's better out-of-sample forecast actually convert into a better
    walk-forward net Sharpe, or does it evaporate once trade timing, top_k
    concentration and costs are applied (the same gap that sank every
    downstream construction on the pooled model in `selective.py`, per
    PREREGISTRATION_SELECTIVE.md §16)?
    """
    base_config = ranking.RankingConfig(**config.base_config)
    data = panel.copy()
    if "regime" not in data.columns:
        data = regime.attach_to_panel(data)
    data["date"] = pd.to_datetime(data["date"])
    start, end = data["date"].min(), data["date"].max()
    fold_start = start + pd.DateOffset(years=train_years)

    folds: list[dict] = []
    pooled_returns: list[pd.Series] = []
    mixture_returns: list[pd.Series] = []

    while fold_start + pd.DateOffset(months=test_months) <= end:
        test_end = fold_start + pd.DateOffset(months=test_months)
        train = ranking._purged_before(data, fold_start)
        test = data[(data["date"] >= fold_start) & (data["date"] < test_end)]
        if train.empty or test.empty:
            fold_start = test_end
            continue

        stable_cols = ranking.stable_feature_columns(train, ablation=base_config.ablation)
        pooled_artifact = ranking.train_ranker(train, base_config, feature_cols=stable_cols)
        pooled_scored = ranking.score(pooled_artifact, test)

        specialists: dict[str, ranking.RankingArtifact] = {}
        for regime_label in regime.REGIMES:
            regime_train = train[train["regime"] == regime_label]
            if len(regime_train) >= config.min_regime_train_rows:
                specialists[regime_label] = ranking.train_ranker(
                    regime_train, base_config, feature_cols=stable_cols,
                )
        mixture_scored = _score_regime_routed(pooled_artifact, specialists, test)

        pooled_result = ranking._simulate_scored(pooled_scored, base_config)
        mixture_result = ranking._simulate_scored(mixture_scored, base_config)
        pooled_returns.append(pooled_result["returns"])
        mixture_returns.append(mixture_result["returns"])

        folds.append({
            "test_start": str(fold_start.date()),
            "test_end": str(test_end.date()),
            "pooled_metrics": pooled_result["metrics"],
            "mixture_metrics": mixture_result["metrics"],
        })
        fold_start = test_end

    def _aggregate(pieces: list[pd.Series]) -> dict:
        combined = pd.concat(pieces).sort_index() if pieces else pd.Series(dtype=float)
        combined = combined[~combined.index.duplicated(keep="first")]
        equity = (1 + combined).cumprod()
        return backtest._metrics(combined, equity) if len(combined) else {}

    return {
        "config": {**asdict(config), "base_config": asdict(base_config)},
        "folds": folds,
        "pooled_aggregate": _aggregate(pooled_returns),
        "mixture_aggregate": _aggregate(mixture_returns),
    }


def nested_walk_forward_lowvol_tilt_mixture(
    panel: pd.DataFrame,
    *,
    config: RegimeMixtureConfig,
    model_weight: float = 0.5,
    top_k: int = 40,
    train_years: int = 3,
    test_months: int = 6,
) -> dict:
    """`selective.nested_walk_forward_lowvol_tilt`, but the forecast blended
    into the low-vol book comes from the regime-routed mixture model instead
    of a single pooled model.

    Motivation: `portfolio_walk_forward_regime_mixture` shows the mixture
    model has a significantly better out-of-sample rank-IC than the pooled
    model in `trend`/`chop`, but loses to the pooled model on raw top-k
    portfolio Sharpe. Every attempt to harvest the *pooled* model's forecast
    profitably failed once corrected for a leakage bug
    (`PREREGISTRATION_SELECTIVE.md` §16) — including this exact low-vol/
    forecast blend, which the no-model low-vol control beat outright. This
    re-runs that same, already-corrected construction with the stronger
    mixture forecast in place of the pooled one, to test whether the better
    IC — not just a different downstream trick — is what was missing.
    """
    base_config = ranking.RankingConfig(**config.base_config)
    data = panel.copy()
    if "regime" not in data.columns:
        data = regime.attach_to_panel(data)
    data["date"] = pd.to_datetime(data["date"])
    start, end = data["date"].min(), data["date"].max()
    fold_start = start + pd.DateOffset(years=train_years)
    tilt_config = replace(base_config, top_k=top_k, rank_buffer=0, weighting="inverse_vol")

    folds: list[dict] = []
    oos_returns: list[pd.Series] = []

    while fold_start + pd.DateOffset(months=test_months) <= end:
        test_end = fold_start + pd.DateOffset(months=test_months)
        train = ranking._purged_before(data, fold_start)
        test = data[(data["date"] >= fold_start) & (data["date"] < test_end)]
        if train.empty or test.empty:
            fold_start = test_end
            continue

        stable_cols = ranking.stable_feature_columns(train, ablation=base_config.ablation)
        pooled_artifact = ranking.train_ranker(train, base_config, feature_cols=stable_cols)
        specialists: dict[str, ranking.RankingArtifact] = {}
        for regime_label in regime.REGIMES:
            regime_train = train[train["regime"] == regime_label]
            if len(regime_train) >= config.min_regime_train_rows:
                specialists[regime_label] = ranking.train_ranker(
                    regime_train, base_config, feature_cols=stable_cols,
                )
        mixture_scored = _score_regime_routed(pooled_artifact, specialists, test)
        blended = selective.blend_lowvol_forecast_score(mixture_scored, model_weight=model_weight)

        result = ranking._simulate_scored(blended, tilt_config)
        folds.append({
            "test_start": str(fold_start.date()),
            "test_end": str(test_end.date()),
            "metrics": result["metrics"],
        })
        oos_returns.append(result["returns"])
        fold_start = test_end

    combined = pd.concat(oos_returns).sort_index() if oos_returns else pd.Series(dtype=float)
    combined = combined[~combined.index.duplicated(keep="first")]
    equity = (1 + combined).cumprod()
    aggregate = backtest._metrics(combined, equity) if len(combined) else {}
    aggregate["folds"] = [fold["metrics"] for fold in folds]
    return {
        "config": {
            **asdict(config), "base_config": asdict(base_config),
            "model_weight": model_weight, "top_k": top_k,
        },
        "folds": folds,
        "aggregate": aggregate,
    }


DEFAULT_REGIME_CONFIG_OVERRIDES: dict[str, dict] = {
    # trend: mixture's strongest edge (paired t=2.60, IR=2.04) — let winners
    # run with a tighter, less-churned book.
    "trend": {"top_k": 10, "rank_buffer": 8, "min_holding_days": 15},
    # chop: mixture makes this newly significant but the weakest of the three
    # (IR=0.88) — a wider, faster-turning book spreads a thinner edge.
    "chop": {"top_k": 25, "rank_buffer": 3, "min_holding_days": 5},
    # meanrev: mixture ties the pooled model here (no improvement) — leave
    # construction at the unmodified base-config defaults.
    "meanrev": {"top_k": 15, "rank_buffer": 5, "min_holding_days": 10},
}


def nested_walk_forward_regime_conditional_construction(
    panel: pd.DataFrame,
    *,
    config: RegimeMixtureConfig,
    regime_config_overrides: dict[str, dict] | None = None,
    train_years: int = 3,
    test_months: int = 6,
) -> dict:
    """Test the one idea flagged at the end of PREREGISTRATION_SELECTIVE.md §17:
    every prior construction (raw top-k, low-vol/forecast blend) fed the
    mixture model's regime-varying forecast through ONE fixed-parameter
    portfolio construction built around a single pooled model's statistical
    properties. This instead lets top_k/rank_buffer/min_holding_days vary by
    the day's dominant regime (the regime most of that day's cross-section
    carries — regime is per-token, but one regime typically dominates >90%
    of names on a given day, per §12's holdout observation), sized against
    each regime's own measured mixture-model edge from §17 Step 1: a tight,
    sticky book where the edge is strongest (`trend`), a wider, faster-turning
    book where it is weakest but still real (`chop`), and the unmodified
    baseline where the mixture ties the pooled model (`meanrev`).
    """
    overrides = regime_config_overrides or DEFAULT_REGIME_CONFIG_OVERRIDES
    base_config = ranking.RankingConfig(**config.base_config)
    regime_configs = {
        label: replace(base_config, **params) for label, params in overrides.items()
    }
    data = panel.copy()
    if "regime" not in data.columns:
        data = regime.attach_to_panel(data)
    data["date"] = pd.to_datetime(data["date"])
    start, end = data["date"].min(), data["date"].max()
    fold_start = start + pd.DateOffset(years=train_years)

    folds: list[dict] = []
    oos_returns: list[pd.Series] = []

    while fold_start + pd.DateOffset(months=test_months) <= end:
        test_end = fold_start + pd.DateOffset(months=test_months)
        train = ranking._purged_before(data, fold_start)
        test = data[(data["date"] >= fold_start) & (data["date"] < test_end)]
        if train.empty or test.empty:
            fold_start = test_end
            continue

        stable_cols = ranking.stable_feature_columns(train, ablation=base_config.ablation)
        pooled_artifact = ranking.train_ranker(train, base_config, feature_cols=stable_cols)
        specialists: dict[str, ranking.RankingArtifact] = {}
        for regime_label in regime.REGIMES:
            regime_train = train[train["regime"] == regime_label]
            if len(regime_train) >= config.min_regime_train_rows:
                specialists[regime_label] = ranking.train_ranker(
                    regime_train, base_config, feature_cols=stable_cols,
                )
        mixture_scored = _score_regime_routed(pooled_artifact, specialists, test)

        returns, _, _ = _simulate_regime_conditional_portfolio(
            mixture_scored, regime_configs, default_config=base_config,
        )
        folds.append({
            "test_start": str(fold_start.date()),
            "test_end": str(test_end.date()),
        })
        oos_returns.append(returns)
        fold_start = test_end

    combined = pd.concat(oos_returns).sort_index() if oos_returns else pd.Series(dtype=float)
    combined = combined[~combined.index.duplicated(keep="first")]
    equity = (1 + combined).cumprod()
    aggregate = backtest._metrics(combined, equity) if len(combined) else {}
    return {
        "config": {**asdict(config), "base_config": asdict(base_config)},
        "regime_config_overrides": overrides,
        "folds": folds,
        "aggregate": aggregate,
    }


def _simulate_regime_conditional_portfolio(
    scored: pd.DataFrame,
    regime_configs: dict[str, ranking.RankingConfig],
    *,
    default_config: ranking.RankingConfig,
) -> tuple[pd.Series, pd.Series, list[dict]]:
    """`ranking._simulate_buffered_portfolio`, but at each rebalance the
    top_k/rank_buffer/min_holding_days used come from whichever
    `regime_configs` entry matches that day's dominant regime (the most
    common per-token regime label in that day's cross-section), instead of
    one fixed config for the whole backtest. Cost/weighting/turnover-cap
    parameters are taken from `default_config` and held constant across
    regimes, since only the selection/churn policy is the thing being
    tested here.
    """
    frame = scored.copy()
    if frame.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), []
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["token", "date"])
    frame["_execution_ret"] = frame.groupby("token")["ret_1d"].shift(-1)
    days = sorted(frame["date"].drop_duplicates())
    step = max(1, int(default_config.rebalance_every))
    cost_rate = (default_config.cost_bps + default_config.slippage_bps) * 1e-4
    weights: dict[str, float] = {}
    entered: dict[str, int] = {}
    daily_rows: list[dict] = []
    trades: list[dict] = []

    for i, date in enumerate(days):
        day = frame[frame["date"] == date].dropna(subset=["score"])
        turnover = 0.0
        rebalanced = (i % step) == 0
        if rebalanced and not day.empty:
            dominant_regime = (
                day["regime"].astype(str).mode().iloc[0]
                if "regime" in day and not day["regime"].dropna().empty
                else None
            )
            active_config = regime_configs.get(dominant_regime, default_config)

            ranked = day.sort_values("score", ascending=False).reset_index(drop=True)
            rank_by_token = {
                str(row["token"]): rank + 1
                for rank, (_, row) in enumerate(ranked.iterrows())
            }
            keep_limit = int(active_config.top_k) + int(active_config.rank_buffer)
            retained = [
                token for token in weights
                if (
                    rank_by_token.get(token, 10**9) <= keep_limit
                    or i - entered.get(token, -10**9) < int(active_config.min_holding_days)
                )
            ]
            target_tokens = list(dict.fromkeys(retained))
            for token in ranked["token"].astype(str):
                if len(target_tokens) >= int(active_config.top_k):
                    break
                if token not in target_tokens:
                    target_tokens.append(token)
            selected = ranked[
                ranked["token"].astype(str).isin(target_tokens)
            ].head(int(active_config.top_k))
            target = ranking._constrained_weights(selected, active_config)
            desired_turnover = sum(
                abs(target.get(token, 0.0) - weights.get(token, 0.0))
                for token in set(target) | set(weights)
            )
            max_turnover = max(0.0, float(default_config.max_turnover))
            if desired_turnover > max_turnover and desired_turnover > 0:
                fraction = max_turnover / desired_turnover
                target = {
                    token: weights.get(token, 0.0)
                    + fraction * (target.get(token, 0.0) - weights.get(token, 0.0))
                    for token in set(target) | set(weights)
                }
            turnover = sum(
                abs(target.get(token, 0.0) - weights.get(token, 0.0))
                for token in set(target) | set(weights)
            )
            for token in target:
                if token not in weights or weights.get(token, 0.0) <= 1e-9:
                    entered[token] = i
            weights = {
                token: weight for token, weight in target.items()
                if weight > 1e-6
            }
            trades.append({
                "date": str(pd.Timestamp(date).date()),
                "dominant_regime": dominant_regime,
                "weights": weights.copy(),
                "turnover": turnover,
                "estimated_cost": turnover * cost_rate,
            })

        realised = {
            str(row["token"]): float(np.expm1(row["_execution_ret"]))
            for _, row in day.dropna(subset=["_execution_ret"]).iterrows()
        }
        gross = sum(
            weight * realised.get(token, 0.0)
            for token, weight in weights.items()
        )
        daily_rows.append({
            "date": date,
            "pnl": gross - turnover * cost_rate,
        })

    if not daily_rows:
        return pd.Series(dtype=float), pd.Series(dtype=float), trades
    daily = pd.DataFrame(daily_rows).set_index("date")["pnl"].sort_index()
    equity = (1.0 + daily).cumprod()
    return daily, equity, trades


def _score_regime_routed(
    pooled_artifact: ranking.RankingArtifact,
    specialists: dict[str, ranking.RankingArtifact],
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Route each row's *raw* prediction to the specialist for its own
    regime, but rank it into a percentile against the FULL day's
    cross-section — never against only same-regime peers.

    `regime.classify_close` runs per-token, so `regime` is a per-stock, per-
    date label, not a market-wide one: on any given day different names can
    carry different regime labels. Calling `ranking.score()` separately on
    each regime-filtered subset (an earlier version of this function did
    this) would rank each name only against its same-regime peers that day —
    a smaller, variable-size, differently-composed group than the pooled
    baseline's full-universe ranking, making the two scores incomparable and
    confounding "the specialist model is better" with "ranking against a
    smaller/different peer group changes the numbers." This mirrors
    `ranking.score`'s own rank/blend logic exactly, just with `ml_score`
    assembled from per-row-routed predictions first.
    """
    out = test.reset_index(drop=True).copy()
    raw = pd.Series(index=out.index, dtype=float)
    for regime_label, group in out.groupby("regime"):
        artifact = specialists.get(str(regime_label), pooled_artifact)
        raw.loc[group.index] = artifact.predict(group)
    out["ml_score"] = raw.astype(float)
    out["ml_rank"] = out.groupby("date")["ml_score"].rank(pct=True)
    if "logret_20" in out:
        out["momentum_rank"] = out.groupby("date")["logret_20"].rank(pct=True)
    else:
        out["momentum_rank"] = 0.5
    weight = float(pooled_artifact.config.get("momentum_weight", 0))
    out["score"] = (1 - weight) * out["ml_rank"] + weight * out["momentum_rank"]
    return out


def _regime_ic_report(scored: pd.DataFrame) -> dict | None:
    clean = scored.dropna(subset=["score", "xs_excess_5d", "regime"])
    ic_by_regime = {}
    for regime_label, group in clean.groupby("regime"):
        try:
            ic_series = research_statistics.rank_ic_series(
                group["score"], group["xs_excess_5d"], group["date"],
            )
        except ValueError:
            continue
        if len(ic_series) < 2:
            continue
        ic_by_regime[str(regime_label)] = ic_series.to_numpy()
    if not ic_by_regime:
        return None
    return research_statistics.ic_significance_multi_regime(ic_by_regime)


def _paired_ic_comparison(pooled: pd.DataFrame, mixture: pd.DataFrame) -> dict:
    """Per-regime paired t-test on (mixture_ic[date] - pooled_ic[date]) —
    the honest test of whether the mixture model beats the pooled baseline,
    not just whether either is independently significant on its own."""
    result: dict[str, dict] = {}
    if pooled.empty or mixture.empty:
        return result
    for regime_label in regime.REGIMES:
        pooled_group = pooled[pooled["regime"].astype(str) == regime_label].dropna(
            subset=["score", "xs_excess_5d"],
        )
        mixture_group = mixture[mixture["regime"].astype(str) == regime_label].dropna(
            subset=["score", "xs_excess_5d"],
        )
        if pooled_group.empty or mixture_group.empty:
            continue
        try:
            pooled_ic = research_statistics.rank_ic_series(
                pooled_group["score"], pooled_group["xs_excess_5d"], pooled_group["date"],
            )
            mixture_ic = research_statistics.rank_ic_series(
                mixture_group["score"], mixture_group["xs_excess_5d"], mixture_group["date"],
            )
        except ValueError:
            continue
        common = pooled_ic.index.intersection(mixture_ic.index)
        if len(common) < 2:
            continue
        diff = (mixture_ic.loc[common] - pooled_ic.loc[common]).to_numpy()
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))
        n = len(diff)
        t_stat = (
            mean_diff / (std_diff / np.sqrt(n)) if std_diff > 0 else float("nan")
        )
        result[regime_label] = {
            "n_common_dates": int(n),
            "mean_ic_diff": mean_diff,
            "paired_t_stat": t_stat,
            "pooled_mean_ic": float(pooled_ic.loc[common].mean()),
            "mixture_mean_ic": float(mixture_ic.loc[common].mean()),
        }
    return result

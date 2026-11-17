"""Regime-gated selective forecaster.

Stage 0/1 (`PREREGISTRATION_SELECTIVE.md`) found that a plain cross-sectional
ranking model has statistically significant rank IC in the `trend` and
`meanrev` regimes but not `chop`, on the (proxy, survivorship-biased)
`yahoo_angel200` development panel. This module gates the model's positions on
that pre-registered, per-fold-refit skill test: it only trades a regime once
that regime's inner-validation IC clears a Bonferroni-corrected significance
threshold. Two abstain modes are supported: `"flat"` (no position when the
gate is closed) and `"passive"` (hold the equal-weight universe when the gate
is closed, trade the active selection when open). `"flat"` isolates the
"skill when active" claim cleanly but forfeits the equity risk premium on
abstain days; `"passive"` keeps market exposure at all times and only turns
the active tilt on/off, which is the standard real-world design for an
alpha-overlay with proven, regime-conditional skill.

Both `ranking._simulate_buffered_portfolio` and `momentum.simulate` already
treat a NaN score as "no position" for that name that day, so gating is
implemented as NaN-ing out scores for non-significant-regime rows before
handing the frame to the existing, already-tested simulator — no separate
cost/turnover accounting is written here.

Every regime-*timing* gate tried here (`nested_walk_forward_selective`, both
abstain modes, several inner-validation windows) failed a diagnostic check:
none beat simply holding a diversified low-volatility book at all times,
which uses none of the Stage 0/1 forecasting-skill finding at all. See
`PREREGISTRATION_SELECTIVE.md` §13-15. `LowVolTiltConfig` /
`nested_walk_forward_lowvol_tilt` is a different design that abandons timing
entirely: always trade, and use the proven forecasting skill only to tilt
*which* low-vol names are preferred. This isolated the model's own
contribution from the low-vol factor effect (a same-`top_k` control test) and
held up across three distinct market eras spanning 2013-2025, unlike any
gated variant.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from . import backtest, ranking, regime, research_statistics

SELECTIVE_ID = "selective_regime_gated"
LOWVOL_TILT_ID = "lowvol_forecast_tilt"


@dataclass(frozen=True)
class SelectiveConfig:
    base_config: dict
    ic_alpha: float = 0.05
    ic_correction: str = "bonferroni"
    min_regime_obs: int = 30
    abstain_mode: str = "flat"

    def __post_init__(self) -> None:
        if self.abstain_mode not in {"flat", "passive"}:
            raise ValueError(
                "abstain_mode must be 'flat' (no position when the gate is "
                "closed) or 'passive' (hold the equal-weight universe when "
                "the gate is closed, trade the active selection when open)"
            )


@dataclass(frozen=True)
class SelectiveGate:
    """Frozen at fit time. `apply_gate` is a pure function of this object."""
    significant_regimes: tuple[str, ...]
    regime_stats: dict
    fit_start: str
    fit_end: str
    fit_n_rows: int


def fit_gate(
    validation_scored: pd.DataFrame,
    *,
    ic_alpha: float = 0.05,
    ic_correction: str = "bonferroni",
    min_regime_obs: int = 30,
) -> SelectiveGate:
    """Freeze which regimes may trade, from inner-validation scores only.

    ``validation_scored`` must carry `score`, `xs_excess_5d`, `regime` and
    `date` columns and must contain no outer-test-fold row — the caller owns
    that split. This function only ever sees what it is given, so it cannot
    leak by construction; it must be called exactly once per fold, before the
    base model is refit on train+validation and before that refit artifact
    ever scores the outer-test fold.
    """
    fit_start = (
        str(pd.to_datetime(validation_scored["date"]).min().date())
        if not validation_scored.empty else ""
    )
    fit_end = (
        str(pd.to_datetime(validation_scored["date"]).max().date())
        if not validation_scored.empty else ""
    )
    clean = validation_scored.dropna(subset=["score", "xs_excess_5d", "regime"])
    ic_by_regime: dict[str, np.ndarray] = {}
    for regime_label, group in clean.groupby("regime"):
        if len(group) < min_regime_obs:
            continue
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
        return SelectiveGate(
            significant_regimes=(), regime_stats={},
            fit_start=fit_start, fit_end=fit_end,
            fit_n_rows=int(len(validation_scored)),
        )
    stats = research_statistics.ic_significance_multi_regime(
        ic_by_regime, alpha=ic_alpha, method=ic_correction,
    )
    return SelectiveGate(
        significant_regimes=tuple(stats["significant_regimes"]),
        regime_stats=stats,
        fit_start=fit_start, fit_end=fit_end,
        fit_n_rows=int(len(validation_scored)),
    )


def apply_gate(scored: pd.DataFrame, gate: SelectiveGate) -> pd.DataFrame:
    """Return a copy with `score` NaN-ed outside the gate's proven regimes."""
    out = scored.copy()
    if not gate.significant_regimes:
        out["score"] = np.nan
        return out
    allowed = set(gate.significant_regimes)
    blocked = ~out["regime"].astype(str).isin(allowed)
    out.loc[blocked, "score"] = np.nan
    return out


def nested_walk_forward_selective(
    panel: pd.DataFrame,
    *,
    config: SelectiveConfig,
    train_years: int = 3,
    test_months: int = 6,
    validation_months: int = 6,
    accumulate_gate_evidence: bool = False,
) -> dict:
    """Same fold structure as ranking.nested_walk_forward, single fixed base
    config (Stage 0/1 already selected `RankingConfig(ablation="price_only")`
    as the only base model carrying provable skill — this is not a fresh
    hyperparameter search), gate fit on inner-validation only and frozen
    before it ever sees the outer-test fold.

    ``validation_months`` controls how much history the gate's significance
    test is fit on (default 6, matching the base ranking model's own
    inner-validation window). ``accumulate_gate_evidence=True`` additionally
    pools every *prior* fold's own realized out-of-sample test-period scores
    into the current fold's gate-fit sample (still causal: only folds whose
    test window is strictly before the current one are ever included). This
    trades gate reactivity for a far larger, more stable sample size, closer
    to the pooled-history sample size Stage 0/1's original significance test
    ran on, at the cost of being slower to notice a regime's skill changing.
    """
    base_config = ranking.RankingConfig(**config.base_config)
    data = panel.copy()
    if "regime" not in data.columns:
        data = regime.attach_to_panel(data)
    data["date"] = pd.to_datetime(data["date"])
    start, end = data["date"].min(), data["date"].max()
    fold_start = start + pd.DateOffset(years=train_years)

    folds: list[dict] = []
    oos_returns: list[pd.Series] = []
    gated_test_frames: list[pd.DataFrame] = []
    accumulated_evidence: list[pd.DataFrame] = []

    while fold_start + pd.DateOffset(months=test_months) <= end:
        test_end = fold_start + pd.DateOffset(months=test_months)
        validation_start = fold_start - pd.DateOffset(months=validation_months)
        train = ranking._purged_before(data, validation_start)
        validation = ranking._purged_window(data, validation_start, fold_start)
        test = data[(data["date"] >= fold_start) & (data["date"] < test_end)]
        if train.empty or validation.empty or test.empty:
            fold_start = test_end
            continue

        # Feature selection must use only this fold's own training data —
        # computing it once on the whole panel (as an earlier version of this
        # function did) let drift filtering near the end of the full panel
        # (which sits at the edge of the holdout) leak into every fold's
        # feature set, including folds trained a decade earlier. See the
        # code-review note in PREREGISTRATION_SELECTIVE.md.
        stable_cols = ranking.stable_feature_columns(train, ablation=base_config.ablation)
        artifact = ranking.train_ranker(train, base_config, feature_cols=stable_cols)
        validation_scored = ranking.score(artifact, validation)
        gate_fit_input = (
            pd.concat([validation_scored, *accumulated_evidence], ignore_index=True)
            if accumulate_gate_evidence and accumulated_evidence
            else validation_scored
        )
        gate = fit_gate(
            gate_fit_input,
            ic_alpha=config.ic_alpha, ic_correction=config.ic_correction,
            min_regime_obs=config.min_regime_obs,
        )

        refit = ranking._purged_before(data, fold_start)
        refit_artifact = ranking.train_ranker(refit, base_config, feature_cols=stable_cols)
        test_scored = ranking.score(refit_artifact, test)
        gated_test = apply_gate(test_scored, gate)

        result = ranking._simulate_scored(gated_test, base_config)
        active_returns = result["returns"]
        active_flag_by_date = _daily_gate_open_flag(gated_test)
        active_turnover_by_date = {
            pd.Timestamp(trade["date"]): float(trade.get("turnover", 0.0))
            for trade in result["trades"]
        }
        active_cost_by_date = {
            pd.Timestamp(trade["date"]): float(trade.get("estimated_cost", 0.0))
            for trade in result["trades"]
        }
        if config.abstain_mode == "passive" and not active_returns.empty:
            passive_result = _risk_matched_passive_result(test, base_config)
            passive_returns = passive_result["returns"]
            passive_turnover_by_date = {
                pd.Timestamp(trade["date"]): float(trade.get("turnover", 0.0))
                for trade in passive_result["trades"]
            }
            passive_cost_by_date = {
                pd.Timestamp(trade["date"]): float(trade.get("estimated_cost", 0.0))
                for trade in passive_result["trades"]
            }
            blended = active_returns.copy()
            blended_turnover: dict[pd.Timestamp, float] = {}
            blended_cost = 0.0
            for date in blended.index:
                if active_flag_by_date.get(date, False):
                    blended_turnover[date] = active_turnover_by_date.get(date, 0.0)
                    blended_cost += active_cost_by_date.get(date, 0.0)
                else:
                    blended.loc[date] = float(passive_returns.get(date, 0.0))
                    blended_turnover[date] = passive_turnover_by_date.get(date, 0.0)
                    blended_cost += passive_cost_by_date.get(date, 0.0)
            fold_returns = blended
        else:
            blended_turnover = active_turnover_by_date
            blended_cost = float(sum(active_cost_by_date.values()))
            fold_returns = active_returns
        fold_equity = (1.0 + fold_returns).cumprod()
        metrics = backtest._metrics(fold_returns, fold_equity) if len(fold_returns) else {}
        metrics["avg_turnover"] = (
            float(np.mean(list(blended_turnover.values()))) if blended_turnover else 0.0
        )
        metrics["total_cost"] = float(blended_cost)
        ranking._attach_signal_quality(metrics, gated_test, base_config)
        metrics["pct_days_active"] = (
            float(
                gated_test.assign(_active=gated_test["score"].notna())
                .groupby("date")["_active"].any().mean()
            ) if len(gated_test) else 0.0
        )
        folds.append({
            "train_end": str(pd.to_datetime(train["date"]).max().date()),
            "validation_start": str(validation_start.date()),
            "validation_end": str(pd.to_datetime(validation["date"]).max().date()),
            "test_start": str(fold_start.date()),
            "test_end": str(test_end.date()),
            "gate": {
                "significant_regimes": list(gate.significant_regimes),
                "fit_start": gate.fit_start,
                "fit_end": gate.fit_end,
                "fit_n_rows": gate.fit_n_rows,
            },
            "metrics": metrics,
        })
        gated_test_frames.append(gated_test)
        oos_returns.append(fold_returns)
        if accumulate_gate_evidence:
            accumulated_evidence.append(test_scored)
        fold_start = test_end

    combined = pd.concat(oos_returns).sort_index() if oos_returns else pd.Series(dtype=float)
    combined = combined[~combined.index.duplicated(keep="first")]
    equity = (1 + combined).cumprod()
    aggregate = backtest._metrics(combined, equity) if len(combined) else {}
    aggregate["folds"] = [fold["metrics"] for fold in folds]
    aggregate["positive_rank_ic_folds"] = sum(
        float(fold["metrics"].get("rank_ic", 0)) > 0 for fold in folds
    )
    aggregate["rank_ic"] = float(np.mean([
        fold["metrics"].get("rank_ic", 0) for fold in folds
    ])) if folds else 0.0
    aggregate["pct_days_active"] = float(np.mean([
        fold["metrics"].get("pct_days_active", 0) for fold in folds
    ])) if folds else 0.0
    aggregate["avg_turnover"] = float(np.mean([
        fold["metrics"].get("avg_turnover", 0) for fold in folds
    ])) if folds else 0.0
    aggregate["total_cost"] = float(np.sum([
        fold["metrics"].get("total_cost", 0) for fold in folds
    ])) if folds else 0.0
    aggregate["sharpe_active_only"] = (
        _sharpe_active_only(combined, gated_test_frames) if folds else None
    )
    # Not applicable: every fold uses the same fixed, pre-selected base
    # config, so there is no config-neighbourhood to test for stability.
    aggregate["neighbour_stable"] = None
    if gated_test_frames and folds:
        all_gated = pd.concat(gated_test_frames, ignore_index=True)
        frozen = base_config
        double_cost = ranking._simulate_scored(
            all_gated,
            replace(frozen, cost_bps=frozen.cost_bps * 2, slippage_bps=frozen.slippage_bps * 2),
        )
        aggregate["stress"] = {
            "double_cost_sharpe": double_cost["metrics"].get("sharpe"),
        }
    else:
        aggregate["stress"] = {}
    return {
        "config": {**asdict(config), "base_config": asdict(base_config)},
        "folds": folds,
        "aggregate": aggregate,
        "daily_returns": combined,
    }


def run_sealed_holdout(
    dev_panel: pd.DataFrame,
    holdout_panel: pd.DataFrame,
    *,
    config: SelectiveConfig,
    validation_months: int = 6,
) -> dict:
    """One frozen confirmatory run against a sealed holdout — opened once.

    This is deliberately NOT `nested_walk_forward_selective` run over
    dev+holdout: that would create a rolling series of folds, most of which
    would still be inside development data. Instead this is exactly one fold,
    anchored at the holdout boundary: train on all development data before
    the inner-validation window, fit the gate on the last
    ``validation_months`` of development only, refit on all of development,
    then score and simulate on the holdout — and only the holdout — once.
    Every step reuses the same helpers (`ranking._purged_before`,
    `ranking._purged_window`, `fit_gate`, `apply_gate`,
    `_risk_matched_passive_result`) as the development-stage walk-forward, so
    this is the identical protocol, not a different one built for the
    occasion.
    """
    base_config = ranking.RankingConfig(**config.base_config)
    dev = dev_panel.copy()
    if "regime" not in dev.columns:
        dev = regime.attach_to_panel(dev)
    dev["date"] = pd.to_datetime(dev["date"])
    hold = holdout_panel.copy()
    if "regime" not in hold.columns:
        hold = regime.attach_to_panel(hold)
    hold["date"] = pd.to_datetime(hold["date"])

    holdout_start = hold["date"].min()
    holdout_end = hold["date"].max() + pd.Timedelta(days=1)
    validation_start = holdout_start - pd.DateOffset(months=validation_months)

    train = ranking._purged_before(dev, validation_start)
    validation = ranking._purged_window(dev, validation_start, holdout_start)
    if train.empty or validation.empty or hold.empty:
        raise RuntimeError(
            "sealed holdout run aborted: empty train/validation/test split "
            "— this must never silently produce a partial or fabricated result"
        )

    # Same fix as nested_walk_forward_selective: derive the feature set from
    # this run's own training data only, not the full dev+holdout panel.
    stable_cols = ranking.stable_feature_columns(train, ablation=base_config.ablation)
    artifact = ranking.train_ranker(train, base_config, feature_cols=stable_cols)
    validation_scored = ranking.score(artifact, validation)
    gate = fit_gate(
        validation_scored,
        ic_alpha=config.ic_alpha, ic_correction=config.ic_correction,
        min_regime_obs=config.min_regime_obs,
    )

    refit = ranking._purged_before(dev, holdout_start)
    refit_artifact = ranking.train_ranker(refit, base_config, feature_cols=stable_cols)
    test_scored = ranking.score(refit_artifact, hold)
    gated_test = apply_gate(test_scored, gate)

    result = ranking._simulate_scored(gated_test, base_config)
    active_returns = result["returns"]
    active_flag_by_date = _daily_gate_open_flag(gated_test)
    active_turnover_by_date = {
        pd.Timestamp(trade["date"]): float(trade.get("turnover", 0.0))
        for trade in result["trades"]
    }
    active_cost_by_date = {
        pd.Timestamp(trade["date"]): float(trade.get("estimated_cost", 0.0))
        for trade in result["trades"]
    }

    if config.abstain_mode == "passive" and not active_returns.empty:
        passive_result = _risk_matched_passive_result(hold, base_config)
        passive_returns = passive_result["returns"]
        passive_turnover_by_date = {
            pd.Timestamp(trade["date"]): float(trade.get("turnover", 0.0))
            for trade in passive_result["trades"]
        }
        passive_cost_by_date = {
            pd.Timestamp(trade["date"]): float(trade.get("estimated_cost", 0.0))
            for trade in passive_result["trades"]
        }
        blended = active_returns.copy()
        blended_turnover: dict[pd.Timestamp, float] = {}
        blended_cost = 0.0
        for date in blended.index:
            if active_flag_by_date.get(date, False):
                blended_turnover[date] = active_turnover_by_date.get(date, 0.0)
                blended_cost += active_cost_by_date.get(date, 0.0)
            else:
                blended.loc[date] = float(passive_returns.get(date, 0.0))
                blended_turnover[date] = passive_turnover_by_date.get(date, 0.0)
                blended_cost += passive_cost_by_date.get(date, 0.0)
        holdout_returns = blended
    else:
        blended_turnover = active_turnover_by_date
        blended_cost = float(sum(active_cost_by_date.values()))
        holdout_returns = active_returns

    holdout_equity = (1.0 + holdout_returns).cumprod()
    metrics = backtest._metrics(holdout_returns, holdout_equity) if len(holdout_returns) else {}
    metrics["avg_turnover"] = (
        float(np.mean(list(blended_turnover.values()))) if blended_turnover else 0.0
    )
    metrics["total_cost"] = float(blended_cost)
    ranking._attach_signal_quality(metrics, gated_test, base_config)
    metrics["pct_days_active"] = (
        float(
            gated_test.assign(_active=gated_test["score"].notna())
            .groupby("date")["_active"].any().mean()
        ) if len(gated_test) else 0.0
    )

    always_on_result = ranking._simulate_scored(test_scored, base_config)
    always_on_equity = (1.0 + always_on_result["returns"]).cumprod()
    always_on_metrics = (
        backtest._metrics(always_on_result["returns"], always_on_equity)
        if len(always_on_result["returns"]) else {}
    )
    always_on_metrics["avg_turnover"] = float(np.mean([
        trade.get("turnover", 0) for trade in always_on_result["trades"]
    ])) if always_on_result["trades"] else 0.0
    always_on_metrics["total_cost"] = float(sum(
        trade.get("estimated_cost", 0) for trade in always_on_result["trades"]
    ))

    return {
        "holdout_opened": True,
        "config": {**asdict(config), "base_config": asdict(base_config)},
        "train_end": str(pd.to_datetime(train["date"]).max().date()),
        "validation_start": str(validation_start.date()),
        "validation_end": str(pd.to_datetime(validation["date"]).max().date()),
        "holdout_start": str(holdout_start.date()),
        "holdout_end": str((holdout_end - pd.Timedelta(days=1)).date()),
        "gate": {
            "significant_regimes": list(gate.significant_regimes),
            "regime_stats": gate.regime_stats,
            "fit_start": gate.fit_start,
            "fit_end": gate.fit_end,
            "fit_n_rows": gate.fit_n_rows,
        },
        "gated_metrics": metrics,
        "always_on_metrics": always_on_metrics,
    }


@dataclass(frozen=True)
class LowVolTiltConfig:
    """A single always-on book: rank by a blend of low-volatility and the
    base ranking model's forecast, not a timing switch between two books.

    Motivated by a diagnostic finding (`PREREGISTRATION_SELECTIVE.md` §13-15):
    every regime-*timing* gate tried (flat-abstain, passive-abstain at several
    validation windows) failed to beat simply holding a diversified low-vol
    book at all times — the proven Stage 0/1 forecasting skill was being
    diluted, not exploited, by switching in and out of it. This design instead
    always trades, and uses the proven skill only to refine *which* low-vol
    names are preferred, isolating the model's contribution from the
    low-volatility factor effect rather than competing with it.
    """
    base_config: dict
    model_weight: float = 0.5
    top_k: int = 40

    def __post_init__(self) -> None:
        if not 0.0 <= self.model_weight <= 1.0:
            raise ValueError("model_weight must be between 0 and 1")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")


def nested_walk_forward_lowvol_tilt(
    panel: pd.DataFrame,
    *,
    config: LowVolTiltConfig,
    train_years: int = 3,
    test_months: int = 6,
) -> dict:
    """Walk-forward for the low-vol + forecast-tilt design (see
    `LowVolTiltConfig`). No inner-validation fold is needed here — there is no
    gate to fit, so the base model is simply refit on all data strictly before
    each test window (identical purge discipline to every other walk-forward
    in this module) and used directly.
    """
    base_config = ranking.RankingConfig(**config.base_config)
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"])
    start, end = data["date"].min(), data["date"].max()
    fold_start = start + pd.DateOffset(years=train_years)
    tilt_config = replace(
        base_config, top_k=config.top_k, rank_buffer=0, weighting="inverse_vol",
    )

    folds: list[dict] = []
    oos_returns: list[pd.Series] = []

    while fold_start + pd.DateOffset(months=test_months) <= end:
        test_end = fold_start + pd.DateOffset(months=test_months)
        train = ranking._purged_before(data, fold_start)
        test = data[(data["date"] >= fold_start) & (data["date"] < test_end)]
        if train.empty or test.empty:
            fold_start = test_end
            continue

        # Same leakage fix as nested_walk_forward_selective: feature
        # selection must come from this fold's own training data only.
        stable_cols = ranking.stable_feature_columns(train, ablation=base_config.ablation)
        artifact = ranking.train_ranker(train, base_config, feature_cols=stable_cols)
        scored = ranking.score(artifact, test)
        blended = blend_lowvol_forecast_score(scored, model_weight=config.model_weight)

        result = ranking._simulate_scored(blended, tilt_config)
        metrics = result["metrics"]
        ranking._attach_signal_quality(metrics, blended, tilt_config)
        folds.append({
            "train_end": str(pd.to_datetime(train["date"]).max().date()),
            "test_start": str(fold_start.date()),
            "test_end": str(test_end.date()),
            "metrics": metrics,
        })
        oos_returns.append(result["returns"])
        fold_start = test_end

    combined = pd.concat(oos_returns).sort_index() if oos_returns else pd.Series(dtype=float)
    combined = combined[~combined.index.duplicated(keep="first")]
    equity = (1 + combined).cumprod()
    aggregate = backtest._metrics(combined, equity) if len(combined) else {}
    aggregate["folds"] = [fold["metrics"] for fold in folds]
    aggregate["rank_ic"] = float(np.mean([
        fold["metrics"].get("rank_ic", 0) for fold in folds
    ])) if folds else 0.0
    return {
        "config": {**asdict(config), "base_config": asdict(base_config)},
        "folds": folds,
        "aggregate": aggregate,
        "daily_returns": combined,
    }


def blend_lowvol_forecast_score(
    scored: pd.DataFrame, *, model_weight: float,
) -> pd.DataFrame:
    """Replace `score` with a per-date-rank blend of (1-model_weight) low-vol
    rank and model_weight forecast rank. Both inputs are already-causal
    (`rv_20` from the feature panel, `score` from a model trained strictly
    before the date being scored) — this only recombines them, no new
    information enters."""
    out = scored.copy()
    if out.empty:
        return out
    lowvol_rank = out.groupby("date")["rv_20"].rank(pct=True, ascending=False)
    model_rank = out.groupby("date")["score"].rank(pct=True)
    out["score"] = (1.0 - model_weight) * lowvol_rank + model_weight * model_rank
    return out


def blend_lowvol_forecast_score_conditional(
    scored: pd.DataFrame, *, gate: SelectiveGate, model_weight: float,
) -> pd.DataFrame:
    """Same blend as `blend_lowvol_forecast_score`, but the model's blend
    weight only applies on rows whose regime is in ``gate.significant_regimes``
    — everywhere else the row gets pure low-vol rank (weight 0), never NaN.

    This is deliberately NOT `apply_gate` + `blend_lowvol_forecast_score`: the
    portfolio always holds a full `top_k` book every day (there is no
    abstention, no flat days, no stale-weight carry-over from
    `_simulate_buffered_portfolio`'s rebalance-skip behaviour). Only *which*
    names are selected shifts with the regime — never *whether* to trade.
    """
    out = scored.copy()
    if out.empty:
        return out
    lowvol_rank = out.groupby("date")["rv_20"].rank(pct=True, ascending=False)
    model_rank = out.groupby("date")["score"].rank(pct=True)
    allowed = set(gate.significant_regimes)
    row_weight = np.where(
        out["regime"].astype(str).isin(allowed), float(model_weight), 0.0,
    )
    out["score"] = (1.0 - row_weight) * lowvol_rank + row_weight * model_rank
    return out


@dataclass(frozen=True)
class RegimeConditionalTiltConfig:
    """Always-on low-vol book (like `LowVolTiltConfig`), but the model's
    blend weight is only switched on for rows in a regime that Stage 0/1
    proved has significant forecasting skill — pure low-vol rank elsewhere.
    Unlike `SelectiveConfig`, this never abstains from trading; it only
    changes which score ranks the always-held book.
    """
    base_config: dict
    model_weight: float = 0.5
    top_k: int = 40
    ic_alpha: float = 0.05
    ic_correction: str = "bonferroni"
    min_regime_obs: int = 30

    def __post_init__(self) -> None:
        if not 0.0 <= self.model_weight <= 1.0:
            raise ValueError("model_weight must be between 0 and 1")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")


def nested_walk_forward_regime_conditional_tilt(
    panel: pd.DataFrame,
    *,
    config: RegimeConditionalTiltConfig,
    train_years: int = 3,
    test_months: int = 6,
    validation_months: int = 6,
) -> dict:
    """Walk-forward for the regime-conditional tilt design.

    Same fold structure as `nested_walk_forward_selective` (train / inner-
    validation to fit which regimes carry proven skill / refit on
    train+validation / test), but the gate only ever changes the blend
    weight passed to `blend_lowvol_forecast_score_conditional` — it never
    NaNs a score or skips a rebalance, so the always-on book and its returns
    are unaffected by `_simulate_buffered_portfolio`'s stale-weight-on-empty-
    day behaviour that undermined the flat-abstain designs.
    """
    base_config = ranking.RankingConfig(**config.base_config)
    data = panel.copy()
    if "regime" not in data.columns:
        data = regime.attach_to_panel(data)
    data["date"] = pd.to_datetime(data["date"])
    start, end = data["date"].min(), data["date"].max()
    fold_start = start + pd.DateOffset(years=train_years)
    tilt_config = replace(
        base_config, top_k=config.top_k, rank_buffer=0, weighting="inverse_vol",
    )

    folds: list[dict] = []
    oos_returns: list[pd.Series] = []

    while fold_start + pd.DateOffset(months=test_months) <= end:
        test_end = fold_start + pd.DateOffset(months=test_months)
        validation_start = fold_start - pd.DateOffset(months=validation_months)
        train = ranking._purged_before(data, validation_start)
        validation = ranking._purged_window(data, validation_start, fold_start)
        test = data[(data["date"] >= fold_start) & (data["date"] < test_end)]
        if train.empty or validation.empty or test.empty:
            fold_start = test_end
            continue

        stable_cols = ranking.stable_feature_columns(train, ablation=base_config.ablation)
        artifact = ranking.train_ranker(train, base_config, feature_cols=stable_cols)
        validation_scored = ranking.score(artifact, validation)
        gate = fit_gate(
            validation_scored,
            ic_alpha=config.ic_alpha, ic_correction=config.ic_correction,
            min_regime_obs=config.min_regime_obs,
        )

        refit = ranking._purged_before(data, fold_start)
        refit_artifact = ranking.train_ranker(refit, base_config, feature_cols=stable_cols)
        test_scored = ranking.score(refit_artifact, test)
        blended = blend_lowvol_forecast_score_conditional(
            test_scored, gate=gate, model_weight=config.model_weight,
        )

        result = ranking._simulate_scored(blended, tilt_config)
        metrics = result["metrics"]
        ranking._attach_signal_quality(metrics, blended, tilt_config)
        folds.append({
            "train_end": str(pd.to_datetime(train["date"]).max().date()),
            "validation_start": str(validation_start.date()),
            "validation_end": str(pd.to_datetime(validation["date"]).max().date()),
            "test_start": str(fold_start.date()),
            "test_end": str(test_end.date()),
            "gate": {
                "significant_regimes": list(gate.significant_regimes),
                "fit_start": gate.fit_start,
                "fit_end": gate.fit_end,
                "fit_n_rows": gate.fit_n_rows,
            },
            "metrics": metrics,
        })
        oos_returns.append(result["returns"])
        fold_start = test_end

    combined = pd.concat(oos_returns).sort_index() if oos_returns else pd.Series(dtype=float)
    combined = combined[~combined.index.duplicated(keep="first")]
    equity = (1 + combined).cumprod()
    aggregate = backtest._metrics(combined, equity) if len(combined) else {}
    aggregate["folds"] = [fold["metrics"] for fold in folds]
    aggregate["rank_ic"] = float(np.mean([
        fold["metrics"].get("rank_ic", 0) for fold in folds
    ])) if folds else 0.0
    return {
        "config": {**asdict(config), "base_config": asdict(base_config)},
        "folds": folds,
        "aggregate": aggregate,
        "daily_returns": combined,
    }


def _daily_gate_open_flag(gated_test: pd.DataFrame) -> dict[pd.Timestamp, bool]:
    """True for each date the gate was open (any non-NaN score that day).

    NOT derived from the simulator's sparse `trades` list — that list only
    gets an entry on actual rebalance checkpoints (every `rebalance_every`
    days), so using it as a daily "is the gate open" signal silently treats
    every non-rebalance day as closed regardless of true regime, which
    corrupts the active/passive blend on the ~90% of days between rebalances.
    This reads the gate's true daily status directly from the gated score
    frame instead, independent of the base model's own rebalance cadence.
    """
    if gated_test.empty:
        return {}
    flags = (
        gated_test.assign(_open=gated_test["score"].notna())
        .groupby("date")["_open"].any()
    )
    return {pd.Timestamp(date): bool(value) for date, value in flags.items()}


def _risk_matched_passive_result(
    test: pd.DataFrame, active_config: "ranking.RankingConfig",
) -> dict:
    """A diversified, low-volatility-tilted book, costed and turnover-capped
    the same way as the active book, used as the 'passive' abstain leg.

    A naive full-universe equal-weight book (`backtest._benchmark_buy_hold`)
    carries far more risk than the active, risk-controlled top-k selection —
    it includes every small/volatile name with no diversification cap, which
    makes it a poor stand-in for "the model has no edge today, do something
    safe." This instead reuses the exact same tested simulator
    (`ranking._simulate_scored`) with `-rv_20` (lower realised vol ranks
    higher) as the score and a much broader top_k, so the two legs share a
    weighting scheme, per-name/sector caps, and cost model — the only thing
    that differs is which names are picked.
    """
    if "rv_20" not in test.columns or test.empty:
        return {"returns": pd.Series(dtype=float), "trades": []}
    passive_frame = test.copy()
    passive_frame["score"] = -passive_frame["rv_20"].astype(float)
    passive_config = replace(
        active_config,
        top_k=min(60, max(10, int(active_config.top_k) * 4)),
        rank_buffer=0,
        weighting="inverse_vol",
    )
    return ranking._simulate_scored(passive_frame, passive_config)


def _sharpe_active_only(
    combined_returns: pd.Series, gated_test_frames: list[pd.DataFrame],
) -> float | None:
    """Sharpe computed only over sessions where the gate was active (any
    non-NaN score that day), i.e. excluding forced-flat abstention days from
    the denominator. Reported alongside, never substituted for, the headline
    Sharpe (which includes abstention days at zero return)."""
    if not gated_test_frames:
        return None
    active_dates = set()
    for frame in gated_test_frames:
        if frame.empty:
            continue
        active = frame.assign(_active=frame["score"].notna())
        active_dates.update(
            pd.to_datetime(active.loc[active["_active"], "date"]).dt.normalize()
        )
    if not active_dates:
        return None
    active_returns = combined_returns[combined_returns.index.normalize().isin(active_dates)]
    if len(active_returns) < 2 or float(active_returns.std(ddof=1)) == 0:
        return None
    return float(
        active_returns.mean() / active_returns.std(ddof=1) * np.sqrt(252)
    )

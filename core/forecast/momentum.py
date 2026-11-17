"""Nested walk-forward momentum optimizer with turnover-aware portfolios."""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import backtest, strategies


@dataclass(frozen=True)
class MomentumConfig:
    horizons: tuple[int, ...] = (20, 60, 120, 252)
    rebalance_every: int = 10
    top_k: int = 15
    weighting: str = "inverse_vol"
    rank_buffer: int = 5
    min_holding_days: int = 10
    max_name_weight: float = 0.10
    max_sector_weight: float = 0.30
    max_turnover: float = 0.60
    sector_relative: bool = True
    high_anchor: bool = True
    market_risk_filter: bool = True
    cost_bps: float = 10.0
    slippage_bps: float = 5.0
    execution_delay: int = 0


def parameter_grid() -> list[MomentumConfig]:
    return [
        MomentumConfig(
            horizons=horizons,
            rebalance_every=rebalance,
            top_k=top_k,
            weighting=weighting,
        )
        for horizons, rebalance, top_k, weighting in itertools.product(
            ((20,), (60,), (20, 60, 120), (20, 60, 120, 252)),
            (5, 10, 20),
            (10, 15, 20),
            ("equal", "inverse_vol", "vol_target"),
        )
    ]


def score_panel(panel: pd.DataFrame, config: MomentumConfig) -> pd.DataFrame:
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    components = []
    columns = {20: "logret_20", 60: "logret_60", 120: "mom_3m", 252: "mom_12_1"}
    for horizon in config.horizons:
        col = columns[horizon]
        if col in frame:
            components.append(frame.groupby("date")[col].rank(pct=True))
    if not components:
        raise ValueError("panel has no requested momentum features")
    score = pd.concat(components, axis=1).mean(axis=1)
    if config.sector_relative and "cs_sector_rank" in frame:
        score = 0.75 * score + 0.25 * frame["cs_sector_rank"]
    if config.high_anchor and "dist_52w_high" in frame:
        score = 0.85 * score + 0.15 * frame.groupby("date")[
            "dist_52w_high"
        ].rank(pct=True)
    frame["score"] = score
    return frame


def simulate(panel: pd.DataFrame, config: MomentumConfig) -> dict[str, Any]:
    frame = score_panel(panel, config)
    frame = frame.sort_values(["token", "date"])
    frame["_execution_ret"] = frame.groupby("token")["ret_1d"].shift(
        -(1 + int(config.execution_delay))
    )
    days = sorted(frame["date"].drop_duplicates())
    cost_rate = (config.cost_bps + config.slippage_bps) * 1e-4
    weights: dict[str, float] = {}
    entered: dict[str, int] = {}
    daily_rows = []
    trades = []

    for i, date in enumerate(days):
        day = frame[frame["date"] == date].dropna(
            subset=["score", "_execution_ret"]
        )
        turnover = 0.0
        if i % config.rebalance_every == 0 and not day.empty:
            ranked = day.sort_values("score", ascending=False).reset_index(drop=True)
            rank_by_token = {
                str(row["token"]): rank + 1 for rank, (_, row) in enumerate(ranked.iterrows())
            }
            keep_limit = config.top_k + config.rank_buffer
            retained = {
                token for token in weights
                if rank_by_token.get(token, 10**9) <= keep_limit
                or i - entered.get(token, -10**9) < config.min_holding_days
            }
            target_tokens = list(retained)
            for token in ranked["token"].astype(str):
                if len(target_tokens) >= config.top_k:
                    break
                if token not in target_tokens:
                    target_tokens.append(token)
            selected = ranked[ranked["token"].astype(str).isin(target_tokens)].head(
                config.top_k
            )
            target = _constrained_weights(selected, config)
            if config.market_risk_filter:
                breadth = float(day["market_breadth_up"].iloc[0]) if "market_breadth_up" in day else 0.5
                market_ret = float(day["market_mean_return"].iloc[0]) if "market_mean_return" in day else 0
                if breadth < 0.35 and market_ret < 0:
                    target = {token: weight * 0.5 for token, weight in target.items()}

            desired_turnover = sum(
                abs(target.get(token, 0) - weights.get(token, 0))
                for token in set(target) | set(weights)
            )
            if desired_turnover > config.max_turnover and desired_turnover > 0:
                fraction = config.max_turnover / desired_turnover
                target = {
                    token: weights.get(token, 0)
                    + fraction * (target.get(token, 0) - weights.get(token, 0))
                    for token in set(target) | set(weights)
                }
            turnover = sum(
                abs(target.get(token, 0) - weights.get(token, 0))
                for token in set(target) | set(weights)
            )
            for token in target:
                if token not in weights or weights.get(token, 0) == 0:
                    entered[token] = i
            weights = {token: weight for token, weight in target.items() if weight > 1e-6}
            trades.append({
                "date": str(pd.Timestamp(date).date()),
                "weights": weights.copy(),
                "turnover": turnover,
                "estimated_cost": turnover * cost_rate,
            })

        returns = {
            str(row["token"]): float(np.expm1(row["_execution_ret"]))
            for _, row in day.iterrows()
        }
        gross = sum(weight * returns.get(token, 0) for token, weight in weights.items())
        daily_rows.append({
            "date": date,
            "pnl": gross - turnover * cost_rate,
            "gross": gross,
            "turnover": turnover,
        })

    daily = pd.DataFrame(daily_rows).set_index("date")["pnl"].sort_index()
    equity = (1 + daily).cumprod()
    metrics = backtest._metrics(daily, equity)
    metrics["avg_turnover"] = float(np.mean([
        trade["turnover"] for trade in trades
    ])) if trades else 0.0
    metrics["total_cost"] = float(sum(trade["estimated_cost"] for trade in trades))
    metrics["trade_count"] = len(trades)
    return {
        "metrics": metrics,
        "returns": daily,
        "equity": equity,
        "trades": trades,
    }


def nested_walk_forward(
    panel: pd.DataFrame,
    *,
    configs: list[MomentumConfig] | None = None,
    outer_test_months: int = 6,
    inner_validation_months: int = 6,
    train_years: int = 3,
) -> dict:
    configs = configs or parameter_grid()
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"])
    start, end = data["date"].min(), data["date"].max()
    outer_start = start + pd.DateOffset(years=train_years)
    folds = []
    all_oos_returns = []

    while outer_start + pd.DateOffset(months=outer_test_months) <= end:
        outer_end = outer_start + pd.DateOffset(months=outer_test_months)
        inner_end = outer_start
        inner_start = inner_end - pd.DateOffset(months=inner_validation_months)
        train = data[data["date"] < inner_start]
        validation = data[(data["date"] >= inner_start) & (data["date"] < inner_end)]
        test = data[(data["date"] >= outer_start) & (data["date"] < outer_end)]
        if train.empty or validation.empty or test.empty:
            outer_start = outer_end
            continue

        scored = []
        tuning = pd.concat([train, validation], ignore_index=True)
        for config in configs:
            result = simulate(tuning[tuning["date"] >= inner_start], config)
            metrics = result["metrics"]
            scored.append((strategies.objective_score(metrics), config, metrics))
        scored.sort(key=lambda item: item[0], reverse=True)
        _, winner, validation_metrics = scored[0]
        test_result = simulate(test, winner)
        all_oos_returns.append(test_result["returns"])
        folds.append({
            "train_end": str(inner_start.date()),
            "validation_end": str(inner_end.date()),
            "test_end": str(outer_end.date()),
            "config": asdict(winner),
            "validation": validation_metrics,
            "metrics": test_result["metrics"],
        })
        outer_start = outer_end

    combined = pd.concat(all_oos_returns).sort_index() if all_oos_returns else pd.Series(dtype=float)
    combined = combined[~combined.index.duplicated(keep="first")]
    equity = (1 + combined).cumprod()
    aggregate = backtest._metrics(combined, equity) if len(combined) else {}
    if folds:
        configs_json = [json.dumps(fold["config"], sort_keys=True) for fold in folds]
        champion_json = max(set(configs_json), key=configs_json.count)
        champion_config = json.loads(champion_json)
    else:
        champion_config = asdict(MomentumConfig())
    aggregate["folds"] = [fold["metrics"] for fold in folds]
    _attach_fold_aggregate_metrics(aggregate)
    aggregate["positive_sharpe_folds"] = sum(
        float(fold["metrics"].get("sharpe", -99)) > 0 for fold in folds
    )
    aggregate["neighbour_stable"] = _neighbour_stability(folds)
    if len(combined):
        oos_start, oos_end = combined.index.min(), combined.index.max()
        oos_panel = data[
            (data["date"] >= oos_start) & (data["date"] <= oos_end)
        ]
        frozen = MomentumConfig(**champion_config)
        double_cost = simulate(
            oos_panel,
            replace(
                frozen,
                cost_bps=frozen.cost_bps * 2,
                slippage_bps=frozen.slippage_bps * 2,
            ),
        )
        delayed = simulate(
            oos_panel, replace(frozen, execution_delay=1)
        )
        aggregate["stress"] = {
            "double_cost_sharpe": double_cost["metrics"].get("sharpe"),
            "delay_sharpe": delayed["metrics"].get("sharpe"),
        }
        bh_returns, bh_equity = backtest._benchmark_buy_hold(oos_panel, h=5)
        aggregate["buy_hold"] = backtest._metrics(bh_returns, bh_equity)
    return {
        "champion_config": champion_config,
        "folds": folds,
        "aggregate": aggregate,
    }


def register_result(result: dict, *, universe: str = "nifty200") -> dict:
    version = f"{pd.Timestamp.utcnow():%Y%m%d%H%M%S}"
    strategies.register_strategy(
        strategy_id=strategies.MOMENTUM_ID,
        version=version,
        name="Momentum Champion",
        status="challenger",
        universe=universe,
        signal_type="target_weight",
        config=result["champion_config"],
        metrics=result["aggregate"],
    )
    return strategies.get_strategy(strategies.MOMENTUM_ID, version)


def _attach_fold_aggregate_metrics(aggregate: dict) -> None:
    folds = aggregate.get("folds") or []
    for name in ("avg_turnover", "total_cost", "trade_count"):
        values = [
            float(fold[name]) for fold in folds
            if fold.get(name) is not None
        ]
        if not values:
            continue
        if name in {"total_cost", "trade_count"}:
            aggregate[name] = float(sum(values))
        else:
            aggregate[name] = float(np.mean(values))


def _constrained_weights(
    selected: pd.DataFrame, config: MomentumConfig,
) -> dict[str, float]:
    if selected.empty:
        return {}
    if config.weighting in {"inverse_vol", "vol_target"} and "rv_20" in selected:
        raw = 1 / selected["rv_20"].astype(float).clip(lower=0.05)
    else:
        raw = pd.Series(1.0, index=selected.index)
    raw = raw / raw.sum()
    rows = []
    for idx, row in selected.iterrows():
        rows.append({
            "token": str(row["token"]),
            "sector": str(row.get("sector") or "Unknown"),
            "weight": min(float(raw.loc[idx]), config.max_name_weight),
        })
    sector_used: dict[str, float] = {}
    result = {}
    for row in sorted(rows, key=lambda item: item["weight"], reverse=True):
        available = max(0.0, config.max_sector_weight - sector_used.get(row["sector"], 0))
        weight = min(row["weight"], available)
        if weight > 0:
            result[row["token"]] = weight
            sector_used[row["sector"]] = sector_used.get(row["sector"], 0) + weight
    return result


def _neighbour_stability(folds: list[dict]) -> bool:
    if len(folds) < 3:
        return False
    sharpes = [float(fold["metrics"].get("sharpe", -99)) for fold in folds]
    return sum(value > 0 for value in sharpes) >= math.ceil(len(sharpes) * 0.6)

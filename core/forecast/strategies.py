"""Generic strategy registry and signal interface."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..db import get_conn, lock


RETIRED_PRISM_ID = "prism_binary_5d"
MOMENTUM_ID = "momentum_champion"
RANKING_ID = "ranking_challenger"
SELECTIVE_ID = "selective_regime_gated"
LOWVOL_TILT_ID = "lowvol_forecast_tilt"


def buy_hold_id(universe: str) -> str:
    return f"buy_hold_{universe.lower()}"


@dataclass(frozen=True)
class StrategySignal:
    strategy_id: str
    strategy_version: str
    token: str
    symbol: str
    exchange: str
    asof: str
    score: float
    rank: int
    target_weight: float
    expected_return: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def register_strategy(
    *,
    strategy_id: str,
    version: str,
    name: str,
    status: str,
    universe: str,
    signal_type: str,
    config: dict,
    metrics: dict | None = None,
    artifact_path: str | Path | None = None,
) -> None:
    if status not in {"benchmark", "challenger", "champion", "retired", "experimental"}:
        raise ValueError(f"unsupported strategy status {status!r}")
    with lock():
        get_conn().execute(
            """INSERT INTO strategy_registry(
                 strategy_id,version,name,status,universe,signal_type,
                 artifact_path,config_json,metrics_json,created_at
               ) VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(strategy_id,version) DO UPDATE SET
                 name=excluded.name,status=excluded.status,
                 universe=excluded.universe,signal_type=excluded.signal_type,
                 artifact_path=excluded.artifact_path,
                 config_json=excluded.config_json,
                 metrics_json=excluded.metrics_json""",
            (
                strategy_id, version, name, status, universe, signal_type,
                str(artifact_path) if artifact_path else None,
                json.dumps(config), json.dumps(metrics or {}), int(time.time()),
            ),
        )


def get_strategy(strategy_id: str, version: str | None = None) -> dict | None:
    if version:
        row = get_conn().execute(
            """SELECT * FROM strategy_registry
               WHERE strategy_id=? AND version=?""",
            (strategy_id, version),
        ).fetchone()
        return _decode(row) if row else None
    else:
        rows = get_conn().execute(
            """SELECT * FROM strategy_registry WHERE strategy_id=?""",
            (strategy_id,),
        ).fetchall()
        decoded = [_decode(row) for row in rows]
        if not decoded:
            return None
        decoded = _best_current_per_strategy_universe(decoded)
        decoded.sort(
            key=lambda row: (
                row["status"] == "champion",
                objective_score(row["metrics"]),
                int(row.get("created_at") or 0),
            ),
            reverse=True,
        )
        return decoded[0]


def list_strategies() -> list[dict]:
    rows = get_conn().execute(
        """SELECT * FROM strategy_registry
           ORDER BY (status='champion') DESC,(status='challenger') DESC,
                    created_at DESC"""
    ).fetchall()
    return [_decode(row) for row in rows]


def leaderboard() -> list[dict]:
    rows = _best_current_per_strategy_universe(list_strategies())
    for item in rows:
        metrics = item["metrics"]
        item["objective_score"] = objective_score(metrics)
    return sorted(rows, key=lambda row: row["objective_score"], reverse=True)


def _best_current_per_strategy_universe(rows: list[dict]) -> list[dict]:
    latest_signature: dict[tuple[str, str], tuple] = {}
    latest_created: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row["strategy_id"]), str(row["universe"]))
        created = int(row.get("created_at") or 0)
        if created >= latest_created.get(key, -1):
            latest_created[key] = created
            latest_signature[key] = _metrics_signature(row.get("metrics") or {})

    best: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (str(row["strategy_id"]), str(row["universe"]))
        if _metrics_signature(row.get("metrics") or {}) != latest_signature.get(key):
            continue
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        if row["status"] == "champion" and current["status"] != "champion":
            best[key] = row
            continue
        if current["status"] == "champion" and row["status"] != "champion":
            continue
        row_score = objective_score(row.get("metrics") or {})
        current_score = objective_score(current.get("metrics") or {})
        if (
            row_score > current_score
            or (
                row_score == current_score
                and int(row.get("created_at") or 0)
                > int(current.get("created_at") or 0)
            )
        ):
            best[key] = row
    return list(best.values())


def _metrics_signature(metrics: dict) -> tuple:
    """Identify the data/evaluation cohort without needing a separate artifact.

    Older registry entries may have been produced from stale feature panels.
    Until every artifact has an explicit dataset version, the evaluation window
    is the safest available compatibility key.
    """
    return (
        metrics.get("n_days"),
        metrics.get("years"),
        (metrics.get("buy_hold") or {}).get("n_days"),
    )


def objective_score(metrics: dict) -> float:
    """Multi-objective score; hard promotion rules remain separate."""
    if not metrics:
        return float("-inf")
    return float(
        3.0 * float(metrics.get("sharpe", -10))
        + 0.8 * float(metrics.get("sortino", -10))
        + 0.5 * float(metrics.get("calmar", -10))
        + 2.0 * float(metrics.get("cagr", -1))
        - 1.5 * abs(float(metrics.get("max_drawdown", -1)))
        - 0.3 * float(metrics.get("avg_turnover", 10))
    )


def historical_promotion_check(
    candidate: dict,
    *,
    momentum: dict,
    buy_hold: dict,
) -> dict:
    folds = candidate.get("folds") or []
    positive_folds = sum(float(f.get("sharpe", -99)) > 0 for f in folds)
    stress = candidate.get("stress") or {}
    failures = []
    if float(candidate.get("sharpe", -99)) <= 0:
        failures.append("non_positive_sharpe")
    if len(folds) < 5 or positive_folds < 4:
        failures.append("fewer_than_four_positive_folds")
    for metric in ("sharpe", "sortino", "calmar"):
        value = float(candidate.get(metric, -99))
        if value <= float(momentum.get(metric, -99)):
            failures.append(f"did_not_beat_momentum_{metric}")
        if value <= float(buy_hold.get(metric, -99)):
            failures.append(f"did_not_beat_buy_hold_{metric}")
    best_cagr = max(float(momentum.get("cagr", -1)), float(buy_hold.get("cagr", -1)))
    if float(candidate.get("cagr", -1)) < best_cagr - 0.02:
        failures.append("cagr_more_than_2pp_below_best_baseline")
    if float(candidate.get("max_drawdown", -1)) <= float(buy_hold.get("max_drawdown", -1)):
        failures.append("drawdown_not_better_than_buy_hold")
    if float(candidate.get("rank_ic", -1)) <= 0:
        failures.append("non_positive_rank_ic")
    if int(candidate.get("positive_rank_ic_folds", 0)) <= len(folds) / 2:
        failures.append("rank_ic_not_positive_in_most_folds")
    if float(stress.get("double_cost_sharpe", -99)) <= 0:
        failures.append("fails_double_cost_stress")
    if float(stress.get("delay_sharpe", -99)) <= 0:
        failures.append("fails_execution_delay_stress")
    if not bool(candidate.get("neighbour_stable", False)):
        failures.append("parameter_neighbours_unstable")
    failures.extend(multiplicity_promotion_failures(candidate))
    return {
        "passed": not failures,
        "failures": failures,
        "positive_folds": positive_folds,
        "fold_count": len(folds),
    }


def multiplicity_promotion_failures(metrics: dict) -> list[str]:
    """Fail-closed publication/automation gates for model-search inflation."""
    failures: list[str] = []
    dsr = metrics.get("dsr_raw")
    pbo = metrics.get("pbo")
    trial_count = metrics.get("raw_trial_count")
    if dsr is None:
        failures.append("deflated_sharpe_missing")
    elif float(dsr) < 0.95:
        failures.append("deflated_sharpe_below_95pct")
    if pbo is None:
        failures.append("pbo_missing")
    elif float(pbo) > 0.10:
        failures.append("pbo_above_10pct")
    if trial_count is None:
        failures.append("experiment_trial_count_missing")
    elif int(trial_count) < 10:
        failures.append("too_few_trials_for_pbo")
    return failures


def momentum_signals(
    panel: pd.DataFrame,
    *,
    strategy: dict,
    asof: str | None = None,
) -> list[StrategySignal]:
    config = strategy["config"]
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    asof_ts = pd.to_datetime(asof).normalize() if asof else frame["date"].max()
    latest = frame[frame["date"] <= asof_ts].sort_values("date").groupby(
        "token", as_index=False
    ).tail(1)
    score = momentum_score(latest, config)
    latest = latest.assign(_score=score).dropna(subset=["_score", "close"])
    if "volume" in latest.columns and config.get("min_dollar_volume"):
        latest = latest[
            latest["close"] * latest["volume"]
            >= float(config["min_dollar_volume"])
        ]
    latest = latest.sort_values("_score", ascending=False)
    top_k = int(config.get("top_k", 15))
    selected = latest.head(top_k)
    weights = _weights(selected, config)
    version = strategy["version"]
    signals = []
    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        token = str(row["token"])
        signals.append(StrategySignal(
            strategy_id=strategy["strategy_id"],
            strategy_version=version,
            token=token,
            symbol=str(row.get("symbol") or token),
            exchange=str(row.get("exchange") or "NSE"),
            asof=str(pd.Timestamp(row["date"]).date()),
            score=float(row["_score"]),
            rank=rank,
            target_weight=float(weights.get(token, 0)),
            expected_return=None,
            metadata={"sector": row.get("sector") or ""},
        ))
    return signals


def momentum_score(frame: pd.DataFrame, config: dict) -> pd.Series:
    horizons = config.get("horizons", [20, 60, 120, 252])
    columns = {
        20: "logret_20",
        60: "logret_60",
        120: "mom_3m",
        252: "mom_12_1",
    }
    components = []
    for horizon in horizons:
        col = columns.get(int(horizon))
        if col in frame.columns:
            components.append(frame[col].rank(pct=True))
    if not components:
        return pd.Series(np.nan, index=frame.index)
    score = pd.concat(components, axis=1).mean(axis=1)
    if config.get("sector_relative") and "cs_sector_rank" in frame:
        score = 0.75 * score + 0.25 * frame["cs_sector_rank"]
    if config.get("high_anchor") and "dist_52w_high" in frame:
        score = 0.85 * score + 0.15 * frame["dist_52w_high"].rank(pct=True)
    return score


def _weights(selected: pd.DataFrame, config: dict) -> dict[str, float]:
    if selected.empty:
        return {}
    kind = config.get("weighting", "equal")
    if kind in {"inverse_vol", "vol_target"} and "rv_20" in selected:
        raw = 1 / selected["rv_20"].astype(float).clip(lower=0.05)
    else:
        raw = pd.Series(1.0, index=selected.index)
    raw = raw / raw.sum()
    cap = float(config.get("max_name_weight", 0.10))
    weights = {
        str(selected.loc[idx, "token"]): min(float(raw.loc[idx]), cap)
        for idx in selected.index
    }
    return weights


def seed_builtin_strategies() -> None:
    _migrate_legacy_proxy_benchmarks()
    if not get_strategy(RETIRED_PRISM_ID):
        register_strategy(
            strategy_id=RETIRED_PRISM_ID, version="2", name="PRISM Binary 5D",
            status="retired", universe="nifty50", signal_type="binary_rank",
            config={"horizon": "5d", "automated_orders": False},
            metrics={"sharpe": -0.042, "rank_ic": 0.0046},
        )
    if not get_strategy(buy_hold_id("nifty200")):
        register_strategy(
            strategy_id=buy_hold_id("nifty200"), version="1",
            name="NIFTY 200 Buy & Hold", status="benchmark",
            universe="nifty200", signal_type="benchmark",
            config={"automated_orders": False},
        )


def _migrate_legacy_proxy_benchmarks() -> None:
    rows = get_conn().execute(
        """SELECT strategy_id,version,universe FROM strategy_registry
           WHERE strategy_id='buy_hold_nifty200' AND universe!='nifty200'"""
    ).fetchall()
    if not rows:
        return
    with lock():
        for row in rows:
            target_id = buy_hold_id(row["universe"])
            existing = get_conn().execute(
                """SELECT 1 FROM strategy_registry
                   WHERE strategy_id=? AND version=?""",
                (target_id, row["version"]),
            ).fetchone()
            if existing:
                get_conn().execute(
                    """DELETE FROM strategy_registry
                       WHERE strategy_id=? AND version=?""",
                    (row["strategy_id"], row["version"]),
                )
            else:
                get_conn().execute(
                    """UPDATE strategy_registry
                       SET strategy_id=?, name=?
                       WHERE strategy_id=? AND version=?""",
                    (
                        target_id,
                        f"{str(row['universe']).upper()} Buy & Hold",
                        row["strategy_id"],
                        row["version"],
                    ),
                )


def _decode(row) -> dict:
    item = dict(row)
    item["config"] = json.loads(item.pop("config_json") or "{}")
    item["metrics"] = json.loads(item.pop("metrics_json") or "{}")
    return item

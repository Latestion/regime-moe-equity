"""Research workflows for universe data, momentum, ranking and promotion."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .. import ticker_meta
from . import (
    data_quality, features, momentum, ranking, regime, research_sources,
    research_statistics, strategies,
)

log = logging.getLogger("forecast.research")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    imp = sub.add_parser("import")
    imp.add_argument("--membership")
    imp.add_argument("--symbol-history")
    imp.add_argument("--corporate-actions")
    imp.add_argument("--eod")
    imp.add_argument("--macro")
    imp.add_argument("--fundamentals")
    imp.add_argument("--universe", default="nifty200")

    val = sub.add_parser("validate")
    val.add_argument("--universe", default="nifty200")
    val.add_argument("--start", default="2020-01-01")
    val.add_argument("--allow-proxy", action="store_true")

    build = sub.add_parser("build-panel")
    build.add_argument("--universe", default="nifty200")
    build.add_argument("--start", default="2015-01-01")
    build.add_argument("--allow-proxy", action="store_true")

    bootstrap = sub.add_parser("bootstrap")
    bootstrap.add_argument("--universe", default="nifty200")
    bootstrap.add_argument("--start", default="2015-01-01")
    bootstrap.add_argument(
        "--proxy-current", action="store_true",
        help="Use a labelled Yahoo/Angel current large-cap proxy, not NIFTY 200",
    )
    bootstrap.add_argument("--limit", type=int)
    bootstrap.add_argument("--out", default="data/research_bootstrap_report.json")

    mom = sub.add_parser("momentum")
    mom.add_argument("--universe", default="nifty200")
    mom.add_argument("--start", default="2020-01-01")
    mom.add_argument("--quick", action="store_true")
    mom.add_argument("--allow-proxy", action="store_true")
    mom.add_argument("--out", default="data/momentum_research.json")

    rank = sub.add_parser("ranking")
    rank.add_argument("--universe", default="nifty200")
    rank.add_argument("--start", default="2020-01-01")
    rank.add_argument("--quick", action="store_true")
    rank.add_argument("--allow-proxy", action="store_true")
    rank.add_argument(
        "--ablations", nargs="+", default=["full_research_model"],
        choices=list(ranking.RANK_ABLATIONS) + ["full"],
    )
    rank.add_argument("--out", default="data/ranking_research.json")

    feas = sub.add_parser("selective-feasibility")
    feas.add_argument("--universe", default="nifty200")
    feas.add_argument("--start", default="2020-01-01")
    feas.add_argument("--allow-proxy", action="store_true")
    feas.add_argument(
        "--ranking-ablation", default="price_only",
        choices=list(ranking.RANK_ABLATIONS),
    )
    feas.add_argument("--out", default="data/selective_feasibility.json")

    sub.add_parser("leaderboard")

    promote = sub.add_parser("promote")
    promote.add_argument("strategy_id")
    promote.add_argument("--version")

    shadow = sub.add_parser("shadow-report")
    shadow.add_argument("strategy_id")

    shadow_run = sub.add_parser("shadow")
    shadow_run.add_argument("--strategy", default=strategies.RANKING_ID)
    shadow_run.add_argument("--asof")
    shadow_run.add_argument("--start", default="2020-01-01")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s | %(message)s",
    )
    strategies.seed_builtin_strategies()

    if args.command == "import":
        counts = {}
        if args.membership:
            counts["membership"] = data_quality.import_membership_csv(
                args.membership, universe=args.universe,
            )
        if args.symbol_history:
            counts["symbol_history"] = data_quality.import_symbol_history_csv(
                args.symbol_history,
            )
        if args.corporate_actions:
            counts["corporate_actions"] = data_quality.import_corporate_actions_csv(
                args.corporate_actions,
            )
        if args.eod:
            counts["eod"] = data_quality.import_eod_csv(args.eod)
        if args.macro:
            counts["macro"] = data_quality.import_macro_csv(args.macro)
        if args.fundamentals:
            counts["fundamentals"] = data_quality.import_fundamentals_csv(
                args.fundamentals,
            )
        print(json.dumps(counts, indent=2))
        return 0

    if args.command == "leaderboard":
        print(json.dumps(strategies.leaderboard(), indent=2))
        return 0

    if args.command == "promote":
        result = promote_strategy(args.strategy_id, args.version)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "shadow-report":
        print(json.dumps(shadow_report(args.strategy_id), indent=2))
        return 0
    if args.command == "shadow":
        panel = features.load_features(start=args.start)
        if panel.empty:
            raise RuntimeError("feature panel is empty")
        result = write_shadow_signals(
            args.strategy, panel=panel, asof=args.asof,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "bootstrap":
        result = research_sources.bootstrap(
            universe=args.universe,
            start=args.start,
            proxy_current=args.proxy_current,
            limit=args.limit,
        )
        _write(args.out, result)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "build-panel":
        result = build_panel(
            universe=args.universe,
            start=args.start,
            allow_proxy=args.allow_proxy,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0

    panel = features.load_features(start=args.start)
    if panel.empty:
        raise RuntimeError("feature panel is empty")
    validation = data_quality.validate_research_data(
        panel, universe=args.universe,
        allow_current_proxy=getattr(args, "allow_proxy", False),
    )
    if args.command == "validate":
        print(json.dumps(validation, indent=2))
        return 0 if validation["passed"] else 2
    if not validation["passed"]:
        raise RuntimeError(
            "research data gate failed: " + ", ".join(validation["failures"])
        )

    if args.command == "momentum":
        configs = [
            momentum.MomentumConfig(),
            momentum.MomentumConfig(
                horizons=(20, 60, 120), rebalance_every=20, top_k=20,
            ),
        ] if args.quick else None
        result = momentum.nested_walk_forward(panel, configs=configs)
        buy_hold_metrics = result["aggregate"].get("buy_hold") or {}
        strategies.register_strategy(
            strategy_id=strategies.buy_hold_id(args.universe),
            version="1",
            name=f"{args.universe.upper()} Buy & Hold",
            status="benchmark",
            universe=args.universe,
            signal_type="benchmark",
            config={"automated_orders": False},
            metrics=buy_hold_metrics,
        )
        registered = momentum.register_result(result, universe=args.universe)
        payload = {**result, "registered": registered}
        _write(args.out, payload)
        print(json.dumps({
            "strategy": registered,
            "aggregate": result["aggregate"],
        }, indent=2))
        return 0

    if args.command == "selective-feasibility":
        result = selective_feasibility(
            panel, ranking_ablation=args.ranking_ablation,
        )
        _write(args.out, result)
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.command == "ranking":
        configs = [
            ranking.RankingConfig(model_type="lambdarank"),
            ranking.RankingConfig(model_type="linear", momentum_weight=0.25),
        ] if args.quick else None
        payload = {"ablations": {}, "validation": validation}
        registered = None
        for ablation in [_normalise_ablation(item) for item in args.ablations]:
            result = ranking.nested_walk_forward(
                panel, configs=configs, ablation=ablation,
            )
            config = ranking.RankingConfig(**result["champion_config"])
            registered = ranking.train_final(
                panel, config=config, universe=args.universe,
                metrics=result["aggregate"], validation=validation,
            )
            payload["ablations"][ablation] = {**result, "registered": registered}
        _write(args.out, payload)
        print(json.dumps({
            "strategy": registered,
            "ablations": {
                key: value["aggregate"] for key, value in payload["ablations"].items()
            },
        }, indent=2))
        return 0
    return 2


def promote_strategy(strategy_id: str, version: str | None = None) -> dict:
    candidate = strategies.get_strategy(strategy_id, version)
    if not candidate:
        raise KeyError("strategy not found")
    if candidate["status"] in {"retired", "benchmark"}:
        raise ValueError("retired/benchmark strategies cannot be promoted")
    if candidate["universe"] != "nifty200":
        return {
            "promoted": False,
            "check": {
                "passed": False,
                "failures": ["exploratory_universe_not_promotable"],
                "message": (
                    "only point-in-time nifty200 research can be promoted; "
                    f"{candidate['universe']!r} is exploratory"
                ),
            },
            "strategy": candidate,
        }
    if not candidate.get("config", {}).get("point_in_time_passed"):
        return {
            "promoted": False,
            "check": {
                "passed": False,
                "failures": ["point_in_time_data_gate_not_passed"],
                "message": "strategy artifact was not trained on validated point-in-time data",
            },
            "strategy": candidate,
        }
    board = strategies.list_strategies()
    momentum_row = next(
        (
            row for row in board
            if row["strategy_id"] == strategies.MOMENTUM_ID
            and row["universe"] == candidate["universe"]
        ),
        None,
    )
    buy_hold = next(
        (
            row for row in board
            if row["strategy_id"] == strategies.buy_hold_id(candidate["universe"])
            and row["universe"] == candidate["universe"]
        ),
        None,
    )
    if strategy_id == strategies.MOMENTUM_ID:
        metrics = candidate["metrics"]
        failures = []
        folds = metrics.get("folds") or []
        stress = metrics.get("stress") or {}
        if float(metrics.get("sharpe", -99)) <= 0:
            failures.append("non_positive_sharpe")
        if len(folds) < 5 or sum(
            float(fold.get("sharpe", -99)) > 0 for fold in folds
        ) < 4:
            failures.append("fold_stability_failed")
        if not metrics.get("neighbour_stable"):
            failures.append("parameter_neighbours_unstable")
        if not buy_hold or not buy_hold.get("metrics"):
            failures.append("buy_hold_baseline_missing")
        else:
            baseline = buy_hold["metrics"]
            for metric in ("sharpe", "sortino", "calmar"):
                if float(metrics.get(metric, -99)) <= float(
                    baseline.get(metric, -99)
                ):
                    failures.append(f"did_not_beat_buy_hold_{metric}")
            if float(metrics.get("max_drawdown", -1)) <= float(
                baseline.get("max_drawdown", -1)
            ):
                failures.append("drawdown_not_better_than_buy_hold")
        if float(stress.get("double_cost_sharpe", -99)) <= 0:
            failures.append("fails_double_cost_stress")
        if float(stress.get("delay_sharpe", -99)) <= 0:
            failures.append("fails_execution_delay_stress")
        failures.extend(strategies.multiplicity_promotion_failures(metrics))
        check = {"passed": not failures, "failures": failures}
    else:
        if not momentum_row or not buy_hold:
            raise ValueError("momentum and buy-hold baselines must be registered")
        check = strategies.historical_promotion_check(
            candidate["metrics"],
            momentum=momentum_row["metrics"],
            buy_hold=buy_hold["metrics"],
        )
    if not check["passed"]:
        return {"promoted": False, "check": check, "strategy": candidate}
    now = int(pd.Timestamp.utcnow().timestamp())
    from ..db import get_conn, lock
    with lock():
        get_conn().execute(
            "UPDATE strategy_registry SET status='challenger' WHERE status='champion'"
        )
        get_conn().execute(
            """UPDATE strategy_registry SET status='champion',promoted_at=?
               WHERE strategy_id=? AND version=?""",
            (now, candidate["strategy_id"], candidate["version"]),
        )
    return {
        "promoted": True,
        "check": check,
        "strategy": strategies.get_strategy(
            candidate["strategy_id"], candidate["version"],
        ),
    }


def shadow_report(strategy_id: str) -> dict:
    from ..db import get_conn
    rows = get_conn().execute(
        """SELECT * FROM shadow_performance WHERE strategy_id=?
           ORDER BY trade_date""",
        (strategy_id,),
    ).fetchall()
    closed = [dict(row) for row in rows if row["status"] == "closed"]
    returns = []
    for row in closed:
        realised = json.loads(row["realised_json"] or "{}")
        if realised.get("return") is not None:
            returns.append(float(realised["return"]))
    return {
        "strategy_id": strategy_id,
        "days": len(closed),
        "mean_return": float(pd.Series(returns).mean()) if returns else None,
        "positive_days": int(sum(value > 0 for value in returns)),
        "ready_for_approval": len(closed) >= 60,
    }


def selective_feasibility(
    panel: pd.DataFrame,
    *,
    ranking_ablation: str = "price_only",
    train_years: int = 3,
    test_months: int = 6,
) -> dict:
    """Stage 0 cheap feasibility check for the regime-gated selective forecaster.

    Tags the panel with a causal regime label, scores it with two independent
    *existing* base models (momentum's rule-based score_panel, and a single
    fixed-config ranking model evaluated strictly out-of-sample), and reports
    per-regime rank-IC significance for each. This is a diagnostic only: it
    does not fit or freeze anything, and it must be read against the Stage 0
    kill/go conditions in PREREGISTRATION_SELECTIVE.md before any further work
    is done.
    """
    tagged = regime.attach_to_panel(panel)
    tagged = tagged.dropna(subset=["regime", "xs_excess_5d"])
    if tagged.empty:
        raise RuntimeError("panel has no regime-tagged, labelled rows")

    momentum_scored = momentum.score_panel(tagged, momentum.MomentumConfig())
    momentum_result = _regime_ic_report(momentum_scored)

    ranking_config = ranking.RankingConfig(ablation=ranking_ablation)
    ranking_scored = _ranking_oos_scores(
        tagged, ranking_config, train_years=train_years, test_months=test_months,
    )
    ranking_result = (
        _regime_ic_report(ranking_scored) if not ranking_scored.empty else None
    )

    return {
        "panel_start": str(pd.to_datetime(tagged["date"]).min().date()),
        "panel_end": str(pd.to_datetime(tagged["date"]).max().date()),
        "n_rows": int(len(tagged)),
        "n_tokens": int(tagged["token"].nunique()),
        "momentum": momentum_result,
        "ranking": {
            "ablation": ranking_ablation,
            "config": asdict(ranking_config),
            "result": ranking_result,
        },
    }


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


def _ranking_oos_scores(
    panel: pd.DataFrame,
    config: "ranking.RankingConfig",
    *,
    train_years: int,
    test_months: int,
) -> pd.DataFrame:
    """Out-of-sample ranking scores only, one fixed config, no tuning.

    Mirrors ranking.nested_walk_forward's train/test split (purged before the
    inner-validation boundary) but skips the inner-validation config search,
    since Stage 0 is a cheap feasibility check, not a champion search.
    """
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"])
    start, end = data["date"].min(), data["date"].max()
    fold_start = start + pd.DateOffset(years=train_years)
    frames: list[pd.DataFrame] = []
    while fold_start + pd.DateOffset(months=test_months) <= end:
        test_end = fold_start + pd.DateOffset(months=test_months)
        validation_start = fold_start - pd.DateOffset(months=6)
        train = ranking._purged_before(data, validation_start)
        test = data[(data["date"] >= fold_start) & (data["date"] < test_end)]
        if not train.empty and not test.empty:
            # Feature selection must use only this fold's own training data —
            # computing it once on the whole panel let drift filtering near
            # the end of the full panel leak into every fold, including ones
            # trained a decade earlier. Same fix as selective.py.
            stable_cols = ranking.stable_feature_columns(train, ablation=config.ablation)
            artifact = ranking.train_ranker(train, config, feature_cols=stable_cols)
            frames.append(ranking.score(artifact, test))
        fold_start = test_end
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_panel(
    *, universe: str = "nifty200", start: str = "2015-01-01",
    allow_proxy: bool = False,
) -> dict:
    refs = research_sources.historical_members(universe)
    if not refs and allow_proxy:
        refs = research_sources.historical_members("yahoo_angel200")
        universe = "yahoo_angel200"
    if not refs:
        raise RuntimeError(
            "no universe membership available; import point-in-time membership "
            "or run bootstrap with --proxy-current for exploratory work"
        )

    def candles_for(ref):
        return data_quality.load_eod(str(ref["token"]), start=start)

    panel_refs = []
    for ref in refs:
        meta = ticker_meta.get(ref.token) or {}
        panel_refs.append({
            "token": ref.token,
            "symbol": ref.symbol,
            "exchange": ref.exchange,
            "sector": meta.get("sector") or "",
            "industry": meta.get("industry") or "",
        })
    panel = features.build_universe_features(
        panel_refs,
        candles_for,
        universe=universe,
    )
    validation = data_quality.validate_research_data(
        panel, universe=universe, allow_current_proxy=allow_proxy,
    )
    return {
        "universe": universe,
        "members": len(refs),
        "rows": int(len(panel)),
        "partitions_written": not panel.empty,
        "validation": validation,
    }


def write_shadow_signals(
    strategy_id: str,
    *,
    panel: pd.DataFrame,
    asof: str | None = None,
) -> dict:
    import time
    from ..db import get_conn, lock

    strategy = strategies.get_strategy(strategy_id)
    if not strategy:
        raise KeyError("strategy not found")
    if strategy["status"] == "retired":
        raise ValueError("retired strategies cannot shadow trade")
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    asof_ts = pd.to_datetime(asof).normalize() if asof else frame["date"].max()
    if strategy_id == strategies.RANKING_ID:
        if not strategy.get("artifact_path"):
            raise RuntimeError("ranking strategy has no artifact")
        artifact = ranking.load_artifact(strategy["artifact_path"])
        latest = frame[frame["date"] <= asof_ts].sort_values("date").groupby(
            "token", as_index=False,
        ).tail(1)
        scored = ranking.score(artifact, latest).sort_values(
            "score", ascending=False,
        )
        top_k = int(strategy["config"].get("top_k", 15))
        selected = scored.head(top_k)
        weights = ranking._constrained_weights(
            selected, ranking.RankingConfig(**{
                key: value for key, value in strategy["config"].items()
                if key in ranking.RankingConfig.__dataclass_fields__
            }),
        )
        rows = _signal_rows_from_scored(
            selected, weights, strategy=strategy,
        )
    elif strategy_id == strategies.MOMENTUM_ID:
        signals = strategies.momentum_signals(
            frame, strategy=strategy, asof=str(asof_ts.date()),
        )
        rows = [asdict(signal) for signal in signals]
    else:
        raise ValueError("benchmark strategies cannot shadow trade")

    trade_date = str(asof_ts.date())
    expected = {
        "strategy_id": strategy["strategy_id"],
        "strategy_version": strategy["version"],
        "trade_date": trade_date,
        "signals": rows,
        "mode": "shadow",
        "places_orders": False,
    }
    now = int(time.time())
    with lock():
        get_conn().execute(
            """INSERT INTO shadow_performance(
                 strategy_id,strategy_version,trade_date,expected_json,
                 realised_json,status,created_at,updated_at
               ) VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(strategy_id,strategy_version,trade_date)
               DO UPDATE SET expected_json=excluded.expected_json,
                             status='open',updated_at=excluded.updated_at""",
            (
                strategy["strategy_id"], strategy["version"], trade_date,
                json.dumps(expected, default=str), None, "open", now, now,
            ),
        )
    return {
        "strategy": strategy,
        "trade_date": trade_date,
        "signals": rows,
        "stored": True,
        "places_orders": False,
    }


def _signal_rows_from_scored(
    selected: pd.DataFrame, weights: dict[str, float], *, strategy: dict,
) -> list[dict]:
    rows = []
    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        token = str(row["token"])
        rows.append(asdict(strategies.StrategySignal(
            strategy_id=strategy["strategy_id"],
            strategy_version=strategy["version"],
            token=token,
            symbol=str(row.get("symbol") or token),
            exchange=str(row.get("exchange") or "NSE"),
            asof=str(pd.Timestamp(row["date"]).date()),
            score=float(row["score"]),
            rank=rank,
            target_weight=float(weights.get(token, 0.0)),
            expected_return=(
                float(row["xs_excess_5d"]) if pd.notna(row.get("xs_excess_5d")) else None
            ),
            metadata={
                "sector": row.get("sector") or "",
                "ml_rank": float(row.get("ml_rank") or 0.0),
                "rank_percentile": float(row.get("score") or 0.0),
            },
        )))
    return rows


def _normalise_ablation(value: str) -> str:
    return "full_research_model" if value == "full" else value


def _write(path: str, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    raise SystemExit(main())

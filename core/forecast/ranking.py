"""Cross-sectional ranking challengers and artifact-v3 persistence."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import backtest, data_quality, flow_features, models, monitor, strategies


ROOT = Path(__file__).parent.parent.parent
RANKING_DIR = ROOT / "data" / "ranking_models"
ARTIFACT_VERSION = 3
log = logging.getLogger("forecast.ranking")

PRICE_RANK_FEATURES = list(dict.fromkeys(
    models.PRICE_FEATS + [
        "mom_12_1", "mom_6m", "mom_3m", "reversal_1m",
        "dist_52w_high", "dist_52w_low", "drawdown_dx",
        "beta_60", "idio_vol_60", "lowvol_rank",
    ]
))

FUNDAMENTAL_RANK_FEATURES = [
    "fund_pe_ttm", "fund_pb", "fund_roe", "fund_roa",
    "fund_operating_margin", "fund_debt_to_equity",
    "fund_revenue_growth_qoq", "fund_eps_growth_qoq",
    "fund_ebitda_growth_qoq", "fund_net_income_growth_qoq",
    "fund_fcf_yield", "fund_fcf_growth_qoq",
    "fund_days_since", "fund_is_stale", "fund_has_recent",
]

FILING_RANK_FEATURES = list(models.FILING_FEATS)

GRAPH_RANK_FEATURES = list(dict.fromkeys(models.GRAPH_FEATS + [
    "graph_avg_confidence", "graph_filing_share",
    "graph_neighbour_shock", "graph_centrality_in",
]))

FLOW_RANK_FEATURES = list(flow_features.FLOW_FEATURES)

RANK_FEATURE_GROUPS: dict[str, list[str]] = {
    "price": PRICE_RANK_FEATURES,
    "fundamentals": FUNDAMENTAL_RANK_FEATURES,
    "filings": FILING_RANK_FEATURES,
    "graph": GRAPH_RANK_FEATURES,
}

RANK_ABLATIONS: dict[str, list[str]] = {
    "price_only": PRICE_RANK_FEATURES,
    "price_plus_fundamentals": PRICE_RANK_FEATURES + FUNDAMENTAL_RANK_FEATURES,
    "price_plus_filings": PRICE_RANK_FEATURES + FILING_RANK_FEATURES,
    "price_plus_graph": PRICE_RANK_FEATURES + GRAPH_RANK_FEATURES,
    # PREREGISTRATION_FLOW.md: the 24 price features plus six delivery/insider columns.
    "price_plus_flow": PRICE_RANK_FEATURES + FLOW_RANK_FEATURES,
    "full_research_model": (
        PRICE_RANK_FEATURES + FUNDAMENTAL_RANK_FEATURES
        + FILING_RANK_FEATURES + GRAPH_RANK_FEATURES
    ),
}

RANK_FEATURES = list(dict.fromkeys(RANK_ABLATIONS["full_research_model"]))
RANK_LABEL_HORIZON = 5


@dataclass(frozen=True)
class RankingConfig:
    ablation: str = "full_research_model"
    model_type: str = "lambdarank"  # lambdarank | linear | return_regression | catboost
    top_k: int = 15
    rebalance_every: int = 10
    weighting: str = "inverse_vol"
    momentum_weight: float = 0.0
    max_name_weight: float = 0.10
    max_sector_weight: float = 0.30
    rank_buffer: int = 5
    min_holding_days: int = 10
    max_turnover: float = 0.60
    cost_bps: float = 10.0
    slippage_bps: float = 5.0
    seed: int = 42


@dataclass
class RankingArtifact:
    model_type: str
    model: Any
    feature_cols: list[str]
    config: dict
    categorical_maps: dict[str, list[str]]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        X = prepare_features(
            frame, self.feature_cols, categorical_maps=self.categorical_maps,
        )
        return np.asarray(self.model.predict(X), dtype=float)


def prepare_features(
    frame: pd.DataFrame,
    feature_cols: list[str] | None = None,
    *,
    categorical_maps: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    cols = [col for col in (feature_cols or RANK_FEATURES) if col in frame.columns]
    X = frame[cols].copy()
    for col in X.columns:
        if col in models.CATEGORICAL_FEATS or X[col].dtype == object:
            categories = (categorical_maps or {}).get(col)
            X[col] = pd.Categorical(X[col], categories=categories).codes.astype(float)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def train_ranker(
    train: pd.DataFrame,
    config: RankingConfig,
    *,
    feature_cols: list[str] | None = None,
) -> RankingArtifact:
    data = train.dropna(subset=["xs_rank_5d"]).sort_values(["date", "token"])
    candidates = feature_cols or ablation_feature_columns(config.ablation)
    cols = [col for col in candidates if col in data.columns]
    categorical_maps = {
        col: sorted(data[col].fillna("").astype(str).unique().tolist())
        for col in cols if col in models.CATEGORICAL_FEATS or data[col].dtype == object
    }
    X = prepare_features(data, cols, categorical_maps=categorical_maps)
    relevance = np.clip(
        np.floor(data["xs_rank_5d"].astype(float) * 9), 0, 9,
    ).astype(int)
    if config.model_type == "linear":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=10.0)
        model.fit(X, data["xs_excess_5d"].astype(float))
    elif config.model_type == "catboost":
        from catboost import CatBoostRanker
        model = CatBoostRanker(
            loss_function="YetiRank",
            iterations=400,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=5.0,
            random_seed=config.seed,
            verbose=False,
        )
        group_id = pd.factorize(data["date"], sort=True)[0]
        model.fit(X, relevance, group_id=group_id)
    elif config.model_type == "return_regression":
        lgb = models._lgb()
        dataset = lgb.Dataset(X, label=data["xs_excess_5d"].astype(float))
        model = lgb.train({
            "objective": "huber",
            "metric": "l2",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_data_in_leaf": 100,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "seed": config.seed,
        }, dataset, num_boost_round=400)
    else:
        lgb = models._lgb()
        groups = data.groupby("date", sort=True).size().tolist()
        dataset = lgb.Dataset(X, label=relevance, group=groups)
        model = lgb.train({
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5, 10, 20],
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_data_in_leaf": 100,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "seed": config.seed,
        }, dataset, num_boost_round=500)
    return RankingArtifact(
        model_type=config.model_type,
        model=model,
        feature_cols=cols,
        config=asdict(config),
        categorical_maps=categorical_maps,
    )


def score(artifact: RankingArtifact, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    raw = artifact.predict(out)
    out["ml_score"] = raw
    out["ml_rank"] = out.groupby("date")["ml_score"].rank(pct=True)
    if "logret_20" in out:
        out["momentum_rank"] = out.groupby("date")["logret_20"].rank(pct=True)
    else:
        out["momentum_rank"] = 0.5
    weight = float(artifact.config.get("momentum_weight", 0))
    out["score"] = (
        (1 - weight) * out["ml_rank"] + weight * out["momentum_rank"]
    )
    return out


def _purged_before(
    frame: pd.DataFrame,
    boundary: pd.Timestamp,
    *,
    horizon_sessions: int = RANK_LABEL_HORIZON,
) -> pd.DataFrame:
    """Rows strictly before ``boundary`` with overlapping labels removed.

    The ranking label uses a forward ``horizon_sessions`` return. Therefore
    the final ``horizon_sessions`` observed market dates before a new fold can
    contain prices from that fold and must not be used for fitting or feature
    selection. Observed dates are used instead of generic business-day offsets
    so exchange holidays do not weaken the purge.
    """
    boundary = pd.Timestamp(boundary)
    source = frame[pd.to_datetime(frame["date"]) < boundary]
    if source.empty or horizon_sessions <= 0:
        return source
    dates = pd.Index(
        pd.to_datetime(source["date"]).drop_duplicates().sort_values(),
    )
    if len(dates) <= horizon_sessions:
        return source.iloc[0:0]
    cutoff = dates[-int(horizon_sessions)]
    return source[pd.to_datetime(source["date"]) < cutoff]


def _purged_window(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    horizon_sessions: int = RANK_LABEL_HORIZON,
) -> pd.DataFrame:
    """Half-open date window whose tail cannot label-leak into ``end``."""
    dated = frame.copy()
    dates = pd.to_datetime(dated["date"])
    window = dated[(dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))]
    if window.empty or horizon_sessions <= 0:
        return window
    observed = pd.Index(
        pd.to_datetime(window["date"]).drop_duplicates().sort_values(),
    )
    if len(observed) <= horizon_sessions:
        return window.iloc[0:0]
    cutoff = observed[-int(horizon_sessions)]
    return window[pd.to_datetime(window["date"]) < cutoff]


def nested_walk_forward(
    panel: pd.DataFrame,
    *,
    configs: list[RankingConfig] | None = None,
    ablation: str = "full_research_model",
    train_years: int = 3,
    test_months: int = 6,
) -> dict:
    configs = configs or parameter_grid(ablation=ablation)
    configs = [config_with_ablation(config, ablation) for config in configs]
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"])
    start, end = data["date"].min(), data["date"].max()
    fold_start = start + pd.DateOffset(years=train_years)
    planned_folds = []
    cursor = fold_start
    while cursor + pd.DateOffset(months=test_months) <= end:
        planned_folds.append(cursor)
        cursor = cursor + pd.DateOffset(months=test_months)
    total_candidate_fits = len(planned_folds) * len(configs)
    completed_candidate_fits = 0
    progress_started = time.perf_counter()
    if total_candidate_fits:
        log.info(
            "ranking search: %d folds × %d configs = %d candidate fits",
            len(planned_folds), len(configs), total_candidate_fits,
        )
    folds = []
    oos_returns = []
    selected_configs = []

    fold_number = 0
    while fold_start + pd.DateOffset(months=test_months) <= end:
        test_end = fold_start + pd.DateOffset(months=test_months)
        validation_start = fold_start - pd.DateOffset(months=6)
        train = _purged_before(data, validation_start)
        validation = _purged_window(data, validation_start, fold_start)
        test = data[(data["date"] >= fold_start) & (data["date"] < test_end)]
        if train.empty or validation.empty or test.empty:
            fold_start = test_end
            continue
        fold_number += 1
        log.info(
            "ranking fold %d/%d: train<%s validation=%s→%s test=%s→%s",
            fold_number, len(planned_folds),
            validation_start.date(), validation_start.date(), fold_start.date(),
            fold_start.date(), test_end.date(),
        )

        candidates = []
        stable_cols = stable_feature_columns(train, ablation=ablation)
        for config_index, config in enumerate(configs, start=1):
            artifact = train_ranker(train, config, feature_cols=stable_cols)
            validation_scored = score(artifact, validation)
            result = _simulate_scored(validation_scored, config)
            _attach_signal_quality(result["metrics"], validation_scored, config)
            candidates.append((
                ranking_selection_score(result["metrics"]),
                config, result["metrics"],
            ))
            completed_candidate_fits += 1
            if (
                config_index == 1
                or config_index == len(configs)
                or config_index % max(1, len(configs) // 4) == 0
            ):
                log.info(
                    "ranking progress: fold %d/%d config %d/%d, %d/%d fits done, ETA %s",
                    fold_number, len(planned_folds), config_index, len(configs),
                    completed_candidate_fits, total_candidate_fits,
                    _eta(progress_started, completed_candidate_fits, total_candidate_fits),
                )
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, winner, validation_metrics = candidates[0]
        log.info(
            "ranking fold %d/%d winner: %s top_k=%d rebalance=%d mom_w=%.2f "
            "validation Sharpe=%.3f rankIC=%.4f topK=%.4f",
            fold_number, len(planned_folds), winner.model_type, winner.top_k,
            winner.rebalance_every, winner.momentum_weight,
            float(validation_metrics.get("sharpe", 0.0) or 0.0),
            float(validation_metrics.get("rank_ic", 0.0) or 0.0),
            float(validation_metrics.get("top_k_excess_return", 0.0) or 0.0),
        )
        diagnostic_artifact = train_ranker(
            train, winner, feature_cols=stable_cols,
        )
        permutation = permutation_importance(
            diagnostic_artifact, validation, seed=winner.seed,
        )
        selected_cols = [
            col for col in stable_cols
            if permutation.get(col, 0.0) >= -0.001
        ] or stable_cols
        refit = _purged_before(data, fold_start)
        artifact = train_ranker(refit, winner, feature_cols=selected_cols)
        test_scored = score(artifact, test)
        result = _simulate_scored(test_scored, winner)
        metrics = result["metrics"]
        _attach_signal_quality(metrics, test_scored, winner)
        folds.append({
            "train_end": str(pd.to_datetime(train["date"]).max().date()),
            "validation_start": str(validation_start.date()),
            "validation_end": str(pd.to_datetime(validation["date"]).max().date()),
            "test_start": str(fold_start.date()),
            "test_end": str(test_end.date()),
            "purge_sessions": RANK_LABEL_HORIZON,
            "config": asdict(winner),
            "validation": validation_metrics,
            "metrics": metrics,
            "feature_cols": selected_cols,
            "permutation_importance": permutation,
        })
        selected_configs.append(winner)
        oos_returns.append(result["returns"])
        log.info(
            "ranking fold %d/%d test: Sharpe=%.3f Calmar=%.3f MaxDD=%.1f%% "
            "rankIC=%.4f ETA %s",
            fold_number, len(planned_folds),
            float(metrics.get("sharpe", 0.0) or 0.0),
            float(metrics.get("calmar", 0.0) or 0.0),
            100.0 * float(metrics.get("max_drawdown", 0.0) or 0.0),
            float(metrics.get("rank_ic", 0.0) or 0.0),
            _eta(progress_started, completed_candidate_fits, total_candidate_fits),
        )
        fold_start = test_end

    combined = pd.concat(oos_returns).sort_index() if oos_returns else pd.Series(dtype=float)
    combined = combined[~combined.index.duplicated(keep="first")]
    equity = (1 + combined).cumprod()
    aggregate = backtest._metrics(combined, equity) if len(combined) else {}
    aggregate["folds"] = [fold["metrics"] for fold in folds]
    _attach_fold_aggregate_metrics(aggregate)
    aggregate["positive_rank_ic_folds"] = sum(
        float(fold["metrics"].get("rank_ic", 0)) > 0 for fold in folds
    )
    aggregate["rank_ic"] = float(np.mean([
        fold["metrics"].get("rank_ic", 0) for fold in folds
    ])) if folds else 0.0
    aggregate["top_k_excess_return"] = float(np.mean([
        fold["metrics"].get("top_k_excess_return", 0) for fold in folds
    ])) if folds else 0.0
    winner = _modal_config(selected_configs)
    aggregate["neighbour_stable"] = _neighbour_stable(folds)
    aggregate["stress"] = _stress_summary(data, winner) if folds else {}
    log.info(
        "ranking complete: Sharpe=%.3f Sortino=%.3f Calmar=%.3f CAGR=%.1f%% "
        "MaxDD=%.1f%% rankIC=%.4f positiveRankIC=%d/%d",
        float(aggregate.get("sharpe", 0.0) or 0.0),
        float(aggregate.get("sortino", 0.0) or 0.0),
        float(aggregate.get("calmar", 0.0) or 0.0),
        100.0 * float(aggregate.get("cagr", 0.0) or 0.0),
        100.0 * float(aggregate.get("max_drawdown", 0.0) or 0.0),
        float(aggregate.get("rank_ic", 0.0) or 0.0),
        int(aggregate.get("positive_rank_ic_folds", 0) or 0),
        len(folds),
    )
    aggregate["feature_group_metrics"] = feature_group_metrics(data, ablation=ablation)
    return {
        "champion_config": asdict(winner),
        "folds": folds,
        "aggregate": aggregate,
        "ablation": ablation,
    }


def _eta(started: float, completed: int, total: int) -> str:
    if completed <= 0 or total <= 0:
        return "unknown"
    elapsed = max(0.0, time.perf_counter() - started)
    remaining = elapsed * max(0, total - completed) / completed
    return _format_duration(remaining)


def _format_duration(seconds: float) -> str:
    seconds = int(max(0, round(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {sec:02d}s"
    return f"{sec}s"


def ranking_selection_score(metrics: dict) -> float:
    """Validation objective for ranking challengers.

    Portfolio metrics alone can pick a lucky, high-churn score. This score
    keeps the normal multi-objective portfolio view, then requires evidence
    that the cross-sectional ranking is actually predictive.
    """
    base = strategies.objective_score(metrics)
    rank_ic = float(metrics.get("rank_ic", 0.0) or 0.0)
    top_k_excess = float(metrics.get("top_k_excess_return", 0.0) or 0.0)
    turnover = float(metrics.get("avg_turnover", 0.0) or 0.0)
    drawdown = abs(float(metrics.get("max_drawdown", 0.0) or 0.0))
    signal_bonus = 30.0 * rank_ic + 80.0 * top_k_excess
    signal_penalty = 2.0 * max(0.0, -rank_ic) + 0.7 * turnover + 1.0 * drawdown
    return float(base + signal_bonus - signal_penalty)


def train_final(
    panel: pd.DataFrame,
    *,
    config: RankingConfig,
    universe: str = "nifty200",
    metrics: dict | None = None,
    validation: dict | None = None,
) -> dict:
    artifact = train_ranker(
        panel, config, feature_cols=stable_feature_columns(
            panel, ablation=config.ablation,
        ),
    )
    version = f"{pd.Timestamp.utcnow():%Y%m%d%H%M%S}"
    out = RANKING_DIR / version
    out.mkdir(parents=True, exist_ok=True)
    joblib = models._joblib()
    if artifact.model_type in {"lambdarank", "return_regression"}:
        artifact.model.save_model(str(out / "model.txt"))
    elif artifact.model_type == "catboost":
        artifact.model.save_model(str(out / "model.cbm"))
    else:
        joblib.dump(artifact.model, out / "model.pkl")
    schema_hash = hashlib.sha256(
        json.dumps(artifact.feature_cols).encode()
    ).hexdigest()
    meta = {
        "artifact_version": ARTIFACT_VERSION,
        "strategy_id": strategies.RANKING_ID,
        "version": version,
        "dataset_version": dataset_version(panel, universe=universe),
        "universe_mode": universe_mode(universe),
        "ablation": config.ablation,
        "model_type": artifact.model_type,
        "feature_cols": artifact.feature_cols,
        "feature_schema_hash": schema_hash,
        "feature_group_metrics": feature_group_metrics(
            panel, ablation=config.ablation, feature_cols=artifact.feature_cols,
        ),
        "point_in_time_passed": bool((validation or {}).get("passed")),
        "shadow_eligible": bool(
            (validation or {}).get("passed")
            and (metrics or {}).get("sharpe") is not None
            and float((metrics or {}).get("sharpe") or 0.0) > 0
        ),
        "categorical_maps": artifact.categorical_maps,
        "config": artifact.config,
        "universe": universe,
        "train_window": [
            str(pd.to_datetime(panel["date"]).min().date()),
            str(pd.to_datetime(panel["date"]).max().date()),
        ],
        "created_at": int(time.time()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    strategies.register_strategy(
        strategy_id=strategies.RANKING_ID,
        version=version,
        name="Cross-Sectional Ranking Challenger",
        status="challenger",
        universe=universe,
        signal_type="target_weight",
        config={
            **artifact.config,
            "dataset_version": meta["dataset_version"],
            "universe_mode": meta["universe_mode"],
            "feature_schema_hash": schema_hash,
            "point_in_time_passed": meta["point_in_time_passed"],
            "shadow_eligible": meta["shadow_eligible"],
        },
        metrics={
            **(metrics or {}),
            "feature_group_metrics": meta["feature_group_metrics"],
        },
        artifact_path=out,
    )
    return strategies.get_strategy(strategies.RANKING_ID, version)


def parameter_grid(*, ablation: str = "full_research_model") -> list[RankingConfig]:
    """Conservative ranking grid.

    The model search must compete with momentum as an investable portfolio,
    not as a high-churn score oracle. Keep the grid focused on slower,
    buffered books so the selected challenger is not a turnover mirage.
    """
    return [
        RankingConfig(
            ablation=ablation,
            model_type=model_type,
            momentum_weight=momentum_weight,
            top_k=top_k,
            rebalance_every=rebalance_every,
            seed=42,
        )
        for model_type in available_model_types()
        for momentum_weight in (0.0, 0.25, 0.5, 0.75)
        for top_k in (10, 15, 20)
        for rebalance_every in (10, 20)
    ]


def load_artifact(path: str | Path) -> RankingArtifact:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    if int(meta.get("artifact_version", 0)) != ARTIFACT_VERSION:
        raise ValueError("ranking artifact is not version 3")
    if meta["model_type"] in {"lambdarank", "return_regression"}:
        model = models._lgb().Booster(model_file=str(path / "model.txt"))
    elif meta["model_type"] == "catboost":
        from catboost import CatBoostRanker
        model = CatBoostRanker()
        model.load_model(str(path / "model.cbm"))
    else:
        model = models._joblib().load(path / "model.pkl")
    return RankingArtifact(
        model_type=meta["model_type"],
        model=model,
        feature_cols=meta["feature_cols"],
        config=meta["config"],
        categorical_maps=meta.get("categorical_maps") or {},
    )


def _simulate_scored(frame: pd.DataFrame, config: RankingConfig) -> dict:
    returns, equity, trades = _simulate_buffered_portfolio(frame, config)
    metrics = backtest._metrics(returns, equity)
    metrics["avg_turnover"] = float(np.mean([
        trade.get("turnover", 0) for trade in trades
    ])) if trades else 0.0
    metrics["total_cost"] = float(sum(
        trade.get("estimated_cost", 0) for trade in trades
    ))
    metrics["trade_count"] = len(trades)
    return {"returns": returns, "equity": equity, "trades": trades, "metrics": metrics}


def _simulate_buffered_portfolio(
    scored: pd.DataFrame,
    config: RankingConfig,
) -> tuple[pd.Series, pd.Series, list[dict]]:
    """Turnover-aware ranking ledger using actual next-session returns.

    Plain top-K ranking looked good on CAGR but churned too much and suffered
    large drawdowns. This ledger mirrors the momentum champion: retain names
    inside a rank buffer, respect minimum holding days, cap changed notional,
    charge costs only on turnover, and leave cash when constraints bind.
    """
    frame = scored.copy()
    if frame.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), []
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["token", "date"])
    frame["_execution_ret"] = frame.groupby("token")["ret_1d"].shift(-1)
    days = sorted(frame["date"].drop_duplicates())
    step = max(1, int(config.rebalance_every))
    cost_rate = (config.cost_bps + config.slippage_bps) * 1e-4
    weights: dict[str, float] = {}
    entered: dict[str, int] = {}
    daily_rows: list[dict[str, Any]] = []
    trades: list[dict] = []

    for i, date in enumerate(days):
        day = frame[frame["date"] == date].dropna(subset=["score"])
        turnover = 0.0
        rebalanced = (i % step) == 0
        if rebalanced and not day.empty:
            ranked = day.sort_values("score", ascending=False).reset_index(drop=True)
            rank_by_token = {
                str(row["token"]): rank + 1
                for rank, (_, row) in enumerate(ranked.iterrows())
            }
            keep_limit = int(config.top_k) + int(config.rank_buffer)
            retained = [
                token for token in weights
                if (
                    rank_by_token.get(token, 10**9) <= keep_limit
                    or i - entered.get(token, -10**9) < int(config.min_holding_days)
                )
            ]
            target_tokens = list(dict.fromkeys(retained))
            for token in ranked["token"].astype(str):
                if len(target_tokens) >= int(config.top_k):
                    break
                if token not in target_tokens:
                    target_tokens.append(token)
            selected = ranked[
                ranked["token"].astype(str).isin(target_tokens)
            ].head(int(config.top_k))
            target = _constrained_weights(selected, config)
            desired_turnover = sum(
                abs(target.get(token, 0.0) - weights.get(token, 0.0))
                for token in set(target) | set(weights)
            )
            max_turnover = max(0.0, float(config.max_turnover))
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
                "weights": weights.copy(),
                "long_tokens": list(weights),
                "turnover": turnover,
                "estimated_cost": turnover * cost_rate,
                "active": bool(weights),
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
            "gross": gross,
            "turnover": turnover,
            "rebalanced": rebalanced,
        })

    if not daily_rows:
        return pd.Series(dtype=float), pd.Series(dtype=float), trades
    daily = pd.DataFrame(daily_rows).set_index("date")["pnl"].sort_index()
    equity = (1.0 + daily).cumprod()
    return daily, equity, trades


def _constrained_weights(
    selected: pd.DataFrame,
    config: RankingConfig,
) -> dict[str, float]:
    if selected.empty:
        return {}
    if config.weighting in {"inverse_vol", "vol_target"} and "rv_20" in selected:
        raw = 1.0 / selected["rv_20"].astype(float).clip(lower=0.05)
    else:
        raw = pd.Series(1.0, index=selected.index)
    raw = raw / raw.sum()
    rows = []
    for idx, row in selected.iterrows():
        rows.append({
            "token": str(row["token"]),
            "sector": str(row.get("sector") or "Unknown"),
            "weight": min(float(raw.loc[idx]), float(config.max_name_weight)),
        })
    sector_used: dict[str, float] = {}
    result: dict[str, float] = {}
    for row in sorted(rows, key=lambda item: item["weight"], reverse=True):
        available = max(
            0.0,
            float(config.max_sector_weight) - sector_used.get(row["sector"], 0.0),
        )
        weight = min(float(row["weight"]), available)
        if weight > 0:
            result[row["token"]] = weight
            sector_used[row["sector"]] = sector_used.get(row["sector"], 0.0) + weight
    return result


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


def available_model_types() -> tuple[str, ...]:
    types = ["lambdarank", "linear", "return_regression"]
    try:
        import catboost  # noqa: F401
    except ImportError:
        pass
    else:
        types.append("catboost")
    return tuple(types)


def stable_feature_columns(
    frame: pd.DataFrame, *, ablation: str = "full_research_model",
) -> list[str]:
    candidates = [col for col in ablation_feature_columns(ablation) if col in frame.columns]
    candidates = [
        col for col in candidates
        if frame[col].notna().mean() >= 0.60 and frame[col].nunique(dropna=True) > 1
    ]
    drift = monitor.drift_report(frame, candidates)
    unstable = {
        row["feature"] for row in drift.get("features", [])
        if row.get("severity") == "high"
    }
    return [col for col in candidates if col not in unstable]


def ablation_feature_columns(ablation: str) -> list[str]:
    if ablation not in RANK_ABLATIONS:
        raise ValueError(
            f"unknown ranking ablation {ablation!r}; want one of {list(RANK_ABLATIONS)}"
        )
    return list(dict.fromkeys(RANK_ABLATIONS[ablation]))


def config_with_ablation(config: RankingConfig, ablation: str) -> RankingConfig:
    if config.ablation == ablation:
        return config
    return RankingConfig(**{**asdict(config), "ablation": ablation})


def feature_group_metrics(
    panel: pd.DataFrame,
    *,
    ablation: str = "full_research_model",
    feature_cols: list[str] | None = None,
) -> dict:
    cols = set(feature_cols or ablation_feature_columns(ablation))
    metrics: dict[str, dict] = {}
    for group, group_cols in RANK_FEATURE_GROUPS.items():
        used = [col for col in group_cols if col in cols and col in panel.columns]
        if not used:
            metrics[group] = {"used": 0, "coverage": 0.0}
            continue
        coverage = float(panel[used].notna().mean().mean())
        metrics[group] = {
            "used": len(used),
            "coverage": round(coverage, 4),
        }
    return metrics


def dataset_version(panel: pd.DataFrame, *, universe: str) -> str:
    if panel.empty:
        return hashlib.sha256(f"{universe}:empty".encode()).hexdigest()[:16]
    payload = {
        "universe": universe.lower(),
        "start": str(pd.to_datetime(panel["date"]).min().date()),
        "end": str(pd.to_datetime(panel["date"]).max().date()),
        "tokens": sorted(panel["token"].astype(str).unique().tolist()),
        "rows": int(len(panel)),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode(),
    ).hexdigest()[:16]


def universe_mode(universe: str) -> str:
    rows = data_quality.get_conn().execute(
        """SELECT source FROM universe_membership WHERE universe=?""",
        (universe.lower(),),
    ).fetchall()
    if not rows:
        return "missing_membership"
    if any("proxy" in str(row["source"] or "").lower() for row in rows):
        return "current_proxy_only"
    return "point_in_time"


def permutation_importance(
    artifact: RankingArtifact,
    validation: pd.DataFrame,
    *,
    seed: int = 42,
) -> dict[str, float]:
    if validation.empty:
        return {}
    scored = score(artifact, validation)
    baseline = _mean_rank_ic(scored)
    rng = np.random.default_rng(seed)
    result = {}
    # Cap diagnostics so nested research remains practical on Windows.
    for col in artifact.feature_cols[:40]:
        permuted = validation.copy()
        permuted[col] = rng.permutation(permuted[col].to_numpy())
        result[col] = float(baseline - _mean_rank_ic(score(artifact, permuted)))
    return result


def _mean_rank_ic(frame: pd.DataFrame) -> float:
    values = frame.groupby("date").apply(
        lambda group: group["score"].corr(
            group["xs_excess_5d"], method="spearman",
        )
    ).dropna()
    return float(values.mean()) if len(values) else 0.0


def _top_k_excess(frame: pd.DataFrame, top_k: int) -> float:
    values = []
    for _, group in frame.groupby("date"):
        selected = group.nlargest(top_k, "score")
        if len(selected):
            values.append(float(selected["xs_excess_5d"].mean()))
    return float(np.mean(values)) if values else 0.0


def _attach_signal_quality(
    metrics: dict,
    scored: pd.DataFrame,
    config: RankingConfig,
) -> None:
    ic = scored.groupby("date").apply(
        lambda group: group["score"].corr(
            group["xs_excess_5d"], method="spearman",
        )
    ).dropna()
    metrics["rank_ic"] = float(ic.mean()) if len(ic) else 0.0
    metrics["top_k_excess_return"] = _top_k_excess(scored, config.top_k)


def _modal_config(configs: list[RankingConfig]) -> RankingConfig:
    if not configs:
        return RankingConfig()
    encoded = [json.dumps(asdict(config), sort_keys=True) for config in configs]
    counts = {value: encoded.count(value) for value in set(encoded)}
    # Fold winners often tie one vote each. Set iteration made that case
    # process-random, so identical research runs could save different final
    # strategies. Prefer highest frequency, then a stable lexical tie-break.
    winner = min(counts, key=lambda value: (-counts[value], value))
    return RankingConfig(**json.loads(winner))


def _neighbour_stable(folds: list[dict]) -> bool:
    if len(folds) < 3:
        return False
    positive = sum(
        float(fold["metrics"].get("sharpe", -99)) > 0 for fold in folds
    )
    encoded = [
        json.dumps(fold.get("config") or {}, sort_keys=True)
        for fold in folds
    ]
    modal_count = max(encoded.count(value) for value in set(encoded))
    # This is a conservative stability proxy: performance must retain its sign
    # in all but one fold and at least half the folds must independently choose
    # the same configuration. A collection of unrelated one-off winners is not
    # evidence of a stable parameter neighbourhood.
    return (
        positive >= max(1, len(folds) - 1)
        and modal_count >= math.ceil(len(folds) / 2)
    )


def _stress_summary(panel: pd.DataFrame, config: RankingConfig) -> dict:
    # Train on the early 80%, stress on the last 20%; diagnostic only.
    dates = sorted(pd.to_datetime(panel["date"]).unique())
    if len(dates) < 50:
        return {}
    cut = dates[int(len(dates) * 0.8)]
    artifact = train_ranker(_purged_before(panel, pd.Timestamp(cut)), config)
    test = score(artifact, panel[pd.to_datetime(panel["date"]) >= cut])
    double_cfg = RankingConfig(**{
        **asdict(config),
        "cost_bps": config.cost_bps * 2,
        "slippage_bps": config.slippage_bps * 2,
    })
    double_result = _simulate_scored(test, double_cfg)
    delayed = test.sort_values(["token", "date"]).copy()
    delayed["score"] = delayed.groupby("token")["score"].shift(1)
    delay_result = _simulate_scored(delayed, config)
    return {
        "double_cost_sharpe": double_result["metrics"].get("sharpe", 0),
        "delay_sharpe": delay_result["metrics"].get("sharpe", 0),
    }

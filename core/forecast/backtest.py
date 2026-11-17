"""Walk-forward backtest engine for PRISM.

Converts the trained classifier's per-(date, ticker) probabilities into a
realistic equity curve with all the metrics a professor will demand:
Sharpe, Sortino, Calmar, max drawdown, win rate, hit rate, exposure.

Trading model
-------------

Default policy: **cross-sectional long-only top-K, ``H``-day
holding period.** On every trading day D:

  * The classifier scores every ticker in the universe with data up to D.
  * Rank tickers by `P(outperform)`.
  * Long the top ``k`` tickers.
  * Hold ``H`` days (matches the prediction horizon).
  * Equal-weight the selected names.
  * Apply 5 bps slippage + 10 bps cost at every rebalance.

The point-in-time guarantee: the predictor at date D uses **only** the
walk-forward model trained on dates `< D - H` (so the label
``ret_<H>d`` for any training row was fully realised before D-H, no
leakage into the test).

Why long-only by default?
  The walk-forward tests show the model ranks likely outperformers better
  than likely underperformers. The bottom-ranked book is not reliable enough
  to short, while the top-ranked long basket improves Sharpe and drawdown
  versus the old long-short translation.

Comparators always evaluated alongside:
  * **buy_hold_nifty** — flat-weighted basket of the entire universe
  * **prism_<ablation>** — one curve per ablation rung evaluated

This makes the **ablation ladder** explicit on the equity-curve chart.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..db import get_conn, lock
from . import features, models, regime

log = logging.getLogger("forecast.backtest")

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    horizon: str            = "5d"     # holding period in trading days (1d or 5d)
    target:  str            = "xs"     # 'xs' (default) or 'abs'
    top_k:   int            = 10       # longs per rebalance
    long_short: bool        = False    # opt in only when testing the short leg
    cost_bps:   float       = 10.0     # one-way cost (bps of notional)
    slippage_bps: float     = 5.0      # one-way slippage (bps)
    rebalance_every: int | None = None # days between rebalances. None = match
                                       # horizon (the natural setting — exactly
                                       # one position alive at a time). Setting
                                       # to 1 with horizon>1 creates overlapping
                                       # bets, multiplies cost bleed.
    min_conviction: float   = 0.0      # only trade when |score - 0.5| >= this.
                                       # 0.0 = no filter (legacy). 0.03-0.08 is
                                       # the meaningful range for our calibrated
                                       # xs classifier; higher = fewer, more
                                       # confident bets.
    train_years: int        = 3
    test_months: int        = 6
    min_train_rows: int     = 5000
    weighting: str          = "equal"  # equal | inverse_vol
    max_name_weight: float  = 0.20


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _annualised_sharpe(returns: pd.Series) -> float:
    if len(returns) < 2 or returns.std(ddof=0) == 0:
        return 0.0
    return float(returns.mean() / returns.std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR))


def _annualised_sortino(returns: pd.Series) -> float:
    if len(returns) < 2:
        return 0.0
    downside = returns.where(returns < 0, 0.0)
    dd = downside.std(ddof=0)
    if dd == 0:
        return 0.0
    return float(returns.mean() / dd * math.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(equity: pd.Series) -> tuple[float, str | None, str | None]:
    if equity.empty:
        return 0.0, None, None
    # Include initial capital 1.0. Without this floor a loss on the first OOS
    # day becomes the initial peak and disappears from max drawdown.
    peak = equity.cummax().clip(lower=1.0)
    dd   = equity / peak - 1.0
    trough_idx = dd.idxmin()
    if pd.isna(trough_idx):
        return 0.0, None, None
    before = equity.loc[:trough_idx]
    peak_idx = before.idxmax() if float(before.max()) > 1.0 else None
    return (
        float(dd.min()),
        str(peak_idx.date()) if peak_idx is not None else None,
        str(trough_idx.date()),
    )


def _calmar(equity: pd.Series, returns: pd.Series) -> float:
    if equity.empty or len(returns) < 2:
        return 0.0
    years = len(returns) / TRADING_DAYS_PER_YEAR
    if years == 0 or equity.iloc[-1] == 1.0:
        return 0.0
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0
    mdd, _, _ = _max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return float(cagr / abs(mdd))


def _metrics(returns: pd.Series, equity: pd.Series) -> dict[str, Any]:
    mdd, mdd_peak, mdd_trough = _max_drawdown(equity)
    n = len(returns)
    years = n / TRADING_DAYS_PER_YEAR if n else 0.0
    total_return = float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0
    cagr = (
        float((1.0 + total_return) ** (1.0 / years) - 1.0)
        if years > 0 and total_return > -1 else 0.0
    )
    return {
        "n_days":       int(n),
        "years":        round(years, 3),
        "total_return": round(total_return, 4),
        "cagr":         round(cagr, 4),
        "ann_vol":      round(float(returns.std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)), 4),
        "sharpe":       round(_annualised_sharpe(returns), 3),
        "sortino":      round(_annualised_sortino(returns), 3),
        "calmar":       round(_calmar(equity, returns), 3),
        "max_drawdown": round(mdd, 4),
        "mdd_peak":     mdd_peak,
        "mdd_trough":   mdd_trough,
        "win_rate":     round(float((returns > 0).mean()), 3) if n else 0.0,
        "hit_rate":     round(float((returns > 0).sum() / max(1, (returns != 0).sum())), 3),
        "exposure":     round(float((returns != 0).mean()), 3) if n else 0.0,
    }


# ---------------------------------------------------------------------------
# Strategy simulation
# ---------------------------------------------------------------------------


def _next_h_day_close_return(panel: pd.DataFrame, *, h: int) -> pd.Series:
    """Per (token, date) return realised over the next ``h`` trading days,
    computed from `ret_<h>d` already present in the feature panel."""
    col = f"ret_{h}d"
    if col not in panel.columns:
        raise ValueError(f"panel is missing forward-return column {col!r}")
    return panel[col]


def _simulate_long_short(
    scored: pd.DataFrame,
    *,
    cfg: BacktestConfig,
    h: int,
) -> tuple[pd.Series, pd.Series, list[dict]]:
    """Daily-rebalanced LS strategy. Returns (daily_pnl, equity_curve, trade_log).

    `scored` columns required: date, token, score, ret_h.
    Each day, sort by score, long top_k / short bottom_k (if long_short else
    top_k only). Costs assessed per leg per rebalance.
    """
    cost_total = (cfg.cost_bps + cfg.slippage_bps) * 1e-4  # one-way fraction
    daily_rows: list[dict] = []
    trades: list[dict] = []
    days = sorted(scored["date"].unique())
    # Step = how many days between rebalances. Default matches the holding
    # horizon so positions don't overlap and we don't pay 5× the cost.
    step = cfg.rebalance_every if cfg.rebalance_every is not None else h
    step = max(1, int(step))
    i = 0
    while i < len(days):
        d = days[i]
        slice_d = scored[scored["date"] == d].dropna(subset=["score", "ret_h"])
        if slice_d.empty or len(slice_d) < 2 * cfg.top_k:
            i += step
            continue
        slice_d = slice_d.sort_values("score", ascending=False)
        if cfg.min_conviction > 0:
            longs_pool  = slice_d[slice_d["score"] >= 0.5 + cfg.min_conviction]
            shorts_pool = slice_d[slice_d["score"] <= 0.5 - cfg.min_conviction]
            longs  = longs_pool.head(cfg.top_k)
            shorts = shorts_pool.tail(cfg.top_k) if cfg.long_short else None
            # If neither side has any conviction-passing names, skip the day.
            if longs.empty and (shorts is None or shorts.empty):
                i += step
                continue
        else:
            longs  = slice_d.head(cfg.top_k)
            shorts = slice_d.tail(cfg.top_k) if cfg.long_short else None

        # Per-name realised return over the holding period h.
        long_ret_arr = longs["ret_h"].to_numpy() if len(longs) else np.array([])
        long_avg_log = float(long_ret_arr.mean()) if len(long_ret_arr) else 0.0
        if cfg.long_short and shorts is not None and not shorts.empty:
            short_ret_arr = shorts["ret_h"].to_numpy()
            short_avg_log = float(short_ret_arr.mean())
            gross_log = long_avg_log - short_avg_log
            n_legs = 2 if len(long_ret_arr) else 1
        else:
            short_ret_arr = np.array([])
            gross_log = long_avg_log
            n_legs = 1 if len(long_ret_arr) else 0
        if n_legs == 0:
            i += step
            continue
        # Spread the realised h-day log return over the holding window's
        # `step` plotted days so the equity curve is smooth. Costs paid
        # once per rebalance.
        plot_days = min(step, h, len(days) - i)
        gross_daily = gross_log / plot_days if plot_days else 0.0
        cost_charge = n_legs * cost_total
        for j in range(plot_days):
            target_date = days[i + j]
            pnl = gross_daily - (cost_charge if j == 0 else 0.0)
            daily_rows.append({"date": target_date, "pnl": pnl, "rebal": (j == 0)})

        trades.append({
            "date": str(pd.Timestamp(d).date()),
            "h":    h,
            "long_tokens":  longs["token"].tolist() if len(longs) else [],
            "short_tokens": shorts["token"].tolist() if (shorts is not None and not shorts.empty) else [],
            "long_ret":     long_avg_log,
            "short_ret":    float(short_ret_arr.mean()) if short_ret_arr.size else 0.0,
            "gross_log":    gross_log,
        })
        i += step

    if not daily_rows:
        return pd.Series(dtype=float), pd.Series(dtype=float), trades

    by_date = pd.DataFrame(daily_rows).groupby("date")["pnl"].sum().sort_index()
    by_date.index = pd.to_datetime(by_date.index)
    equity = (1.0 + by_date).cumprod()
    return by_date, equity, trades


def _simulate_event_driven(
    scored: pd.DataFrame,
    *,
    cfg: BacktestConfig,
    h: int,
) -> tuple[pd.Series, pd.Series, list[dict]]:
    """Target-weight ledger using the actual next-session return per name.

    Rebalances replace the target portfolio, so overlapping forecasts cannot
    multiply exposure. Transaction costs are charged on changed notional.
    """
    cost_rate = (cfg.cost_bps + cfg.slippage_bps) * 1e-4
    frame = scored.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["token", "date"])
    frame["_execution_ret"] = frame.groupby("token")["ret_1d"].shift(-1)
    days = sorted(frame["date"].drop_duplicates())
    step = max(1, int(cfg.rebalance_every or h))
    weights: dict[str, float] = {}
    daily_rows: list[dict[str, Any]] = []
    trades: list[dict] = []

    def group_weights(group: pd.DataFrame, gross: float) -> dict[str, float]:
        if group.empty:
            return {}
        if cfg.weighting == "inverse_vol" and "rv_20" in group.columns:
            raw = 1.0 / group["rv_20"].astype(float).clip(lower=0.05)
            raw = raw / raw.sum()
        else:
            raw = pd.Series(1.0 / len(group), index=group.index)
        result = {
            str(group.loc[idx, "token"]): float(raw.loc[idx] * gross)
            for idx in group.index
        }
        cap = max(0.0, float(cfg.max_name_weight))
        if cap:
            result = {
                token: float(np.sign(weight) * min(abs(weight), cap))
                for token, weight in result.items()
            }
        return result

    for i, date in enumerate(days):
        day = frame[frame["date"] == date]
        turnover = 0.0
        rebalanced = (i % step) == 0
        if rebalanced:
            candidates = day.dropna(subset=["score"]).sort_values(
                "score", ascending=False,
            )
            long_pool = candidates
            if cfg.min_conviction > 0:
                long_pool = long_pool[
                    long_pool["score"] >= 0.5 + cfg.min_conviction
                ]
            longs = long_pool.head(cfg.top_k)
            shorts = candidates.iloc[0:0]
            target = group_weights(longs, 0.5 if cfg.long_short else 1.0)
            if cfg.long_short:
                short_pool = candidates[
                    candidates["score"] <= 0.5 - cfg.min_conviction
                ]
                shorts = short_pool.tail(cfg.top_k)
                for token, weight in group_weights(shorts, -0.5).items():
                    if token not in target:
                        target[token] = weight
            turnover = sum(
                abs(target.get(token, 0.0) - weights.get(token, 0.0))
                for token in set(target) | set(weights)
            )
            weights = target
            trades.append({
                "date": str(pd.Timestamp(date).date()),
                "h": h,
                "weights": weights.copy(),
                "long_tokens": [str(v) for v in longs["token"].tolist()],
                "short_tokens": [str(v) for v in shorts["token"].tolist()],
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


# ---------------------------------------------------------------------------
# Walk-forward evaluator
# ---------------------------------------------------------------------------


def _score_oos(
    panel: pd.DataFrame,
    *,
    horizon: str,
    ablation: str,
    target: str,
    train_years: int,
    test_months: int,
    min_train_rows: int,
    use_regime_gate: bool = False,
) -> pd.DataFrame:
    """Walk-forward inference. Returns a (date, token, score, ret_h) frame
    where `score = P(outperform)` from a calibrated classifier trained on
    data strictly before the test slice.
    """
    panel = panel.sort_values(["date", "token"]).reset_index(drop=True)
    panel["date"] = pd.to_datetime(panel["date"])

    spec = models.make_spec(horizon, ablation, classification=True, target=target)
    h = int(horizon.rstrip("d"))
    fwd_col = f"ret_{h}d"
    if fwd_col not in panel.columns:
        raise ValueError(f"panel missing {fwd_col}")

    chunks: list[pd.DataFrame] = []
    train_start = panel["date"].min()
    end = panel["date"].max()
    cur_train_end = train_start + pd.DateOffset(years=train_years)

    while cur_train_end + pd.DateOffset(months=test_months) <= end:
        test_end = cur_train_end + pd.DateOffset(months=test_months)
        purge_end = cur_train_end - pd.offsets.BDay(h)
        train_mask = (panel["date"] >= train_start) & (panel["date"] < purge_end)
        test_mask  = (panel["date"] >= cur_train_end) & (panel["date"] < test_end)
        train_df = panel[train_mask]
        test_df  = panel[test_mask]
        if len(train_df) < min_train_rows or test_df.empty:
            cur_train_end += pd.DateOffset(months=test_months)
            continue

        if use_regime_gate:
            from . import stack
            clf = stack.train_regime_gated(train_df, spec, n_seeds=1)
        else:
            clf = models.train_classifier_ensemble(
                train_df, spec, n_seeds=1, holdout=0.2,
            )
        # Predict on test slice — keep date/token/score/ret_h.
        Xte, yte = models.prepare_xy(test_df, spec)
        # prepare_xy drops the label, but the test df has the forward return
        # under fwd_col — rejoin by aligning indices.
        keep_cols = ["date", "token", fwd_col, "ret_1d"]
        if "logret_20" in test_df.columns:
            keep_cols.append("logret_20")
        if "rv_20" in test_df.columns:
            keep_cols.append("rv_20")
        scored = test_df.loc[Xte.index, keep_cols].copy()
        scored["score"] = clf.predict_proba(Xte)
        scored["truth"] = yte.to_numpy(dtype=float)
        scored = scored.rename(columns={fwd_col: "ret_h"})
        chunks.append(scored)
        log.info(
            "scored fold %s→%s  rows=%d",
            cur_train_end.date(), test_end.date(), len(scored),
        )
        cur_train_end += pd.DateOffset(months=test_months)

    if not chunks:
        return pd.DataFrame(columns=["date", "token", "score", "ret_h"])
    return pd.concat(chunks, ignore_index=True)


def _benchmark_buy_hold(panel: pd.DataFrame, *, h: int) -> tuple[pd.Series, pd.Series]:
    """Universe-flat-weighted buy-and-hold daily return series.

    Equal-weights every ticker present each day, computes the mean 1-day
    log return, compounds. This is the 'NIFTY-equal-weight' benchmark and
    is the right comparator for a long-only translation of the strategy.
    """
    if "logret_1" not in panel.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    daily = (
        panel.dropna(subset=["logret_1"])
             .groupby("date")["logret_1"].mean()
             .sort_index()
    )
    daily.index = pd.to_datetime(daily.index)
    # log-return per day → simple-return for equity compounding
    simple = np.expm1(daily)
    equity = (1.0 + simple).cumprod()
    return simple, equity


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


@dataclass
class AblationRun:
    name: str
    daily_returns: pd.Series
    equity:        pd.Series
    metrics:       dict[str, Any]
    trades:        list[dict] = field(default_factory=list)


def run_backtest(
    panel: pd.DataFrame,
    *,
    ablations: Iterable[str] = ("price_only", "plus_sentiment", "plus_filings", "full"),
    cfg: BacktestConfig = BacktestConfig(),
    label: str | None = None,
    use_regime_gate: bool = False,
) -> dict[str, Any]:
    """Run the ablation ladder + buy-and-hold benchmark on a feature panel
    and return a JSON-serialisable result dict the API/UI can render.
    """
    panel = regime.attach_to_panel(panel, persist_too=False)
    h = int(cfg.horizon.rstrip("d"))
    results: list[AblationRun] = []
    summary: dict[str, dict] = {}
    stress_summary: dict[str, dict] = {}
    scored_dates: pd.DatetimeIndex | None = None

    # PRISM ablation rungs.
    for ab in ablations:
        log.info("backtest: ablation=%s horizon=%s target=%s", ab, cfg.horizon, cfg.target)
        try:
            scored = _score_oos(
                panel, horizon=cfg.horizon, ablation=ab, target=cfg.target,
                train_years=cfg.train_years, test_months=cfg.test_months,
                min_train_rows=cfg.min_train_rows,
                use_regime_gate=use_regime_gate,
            )
        except Exception as e:
            log.warning("ablation %s failed: %s", ab, e)
            continue
        if scored.empty:
            log.warning("ablation %s produced no scored rows", ab)
            continue
        if scored_dates is None:
            scored_dates = pd.DatetimeIndex(pd.to_datetime(scored["date"]).unique())
        rets, equity, trades = _simulate_event_driven(scored, cfg=cfg, h=h)
        metrics = _metrics(rets, equity)
        metrics["avg_turnover"] = round(float(np.mean(
            [trade.get("turnover", 0.0) for trade in trades]
        )), 4) if trades else 0.0
        metrics["total_cost"] = round(float(sum(
            trade.get("estimated_cost", 0.0) for trade in trades
        )), 4)
        truth = scored["truth"].to_numpy(dtype=float)
        probs = scored["score"].to_numpy(dtype=float)
        metrics["accuracy"] = round(float(
            ((probs >= 0.5) == truth.astype(bool)).mean()
        ), 4)
        metrics["brier"] = round(float(np.mean((probs - truth) ** 2)), 6)
        eps = 1e-7
        clipped = np.clip(probs, eps, 1 - eps)
        metrics["log_loss"] = round(float(-np.mean(
            truth * np.log(clipped) + (1 - truth) * np.log(1 - clipped)
        )), 6)
        daily_ic = scored.groupby("date").apply(
            lambda group: group["score"].corr(group["ret_h"], method="spearman"),
        ).dropna()
        metrics["rank_ic"] = round(float(daily_ic.mean()), 4) if len(daily_ic) else 0.0
        results.append(AblationRun(
            name=f"prism_{ab}", daily_returns=rets, equity=equity,
            metrics=metrics, trades=trades,
        ))
        summary[f"prism_{ab}"] = metrics
        doubled_cfg = replace(
            cfg, cost_bps=cfg.cost_bps * 2, slippage_bps=cfg.slippage_bps * 2,
        )
        stress_rets, stress_equity, _ = _simulate_event_driven(
            scored, cfg=doubled_cfg, h=h,
        )
        delayed = scored.sort_values(["token", "date"]).copy()
        delayed["score"] = delayed.groupby("token")["score"].shift(1)
        delay_rets, delay_equity, _ = _simulate_event_driven(
            delayed, cfg=cfg, h=h,
        )
        stress_summary[f"prism_{ab}"] = {
            "double_costs": _metrics(stress_rets, stress_equity),
            "one_session_signal_delay": _metrics(delay_rets, delay_equity),
        }

    # Deterministic factor baselines on the exact OOS date region.
    if scored_dates is not None:
        baseline_panel = panel[pd.to_datetime(panel["date"]).isin(scored_dates)]
        for name, feature, ascending in (
            ("momentum_20d", "logret_20", True),
            ("low_volatility", "rv_20", False),
        ):
            if feature not in baseline_panel.columns:
                continue
            baseline_cols = list(dict.fromkeys(
                ["date", "token", feature, "ret_1d", "rv_20"]
            ))
            baseline = baseline_panel[baseline_cols].copy()
            baseline["score"] = baseline.groupby("date")[feature].rank(
                pct=True, ascending=ascending,
            )
            rets, equity, trades = _simulate_event_driven(
                baseline, cfg=cfg, h=h,
            )
            metrics = _metrics(rets, equity)
            metrics["avg_turnover"] = round(float(np.mean(
                [trade.get("turnover", 0.0) for trade in trades]
            )), 4) if trades else 0.0
            results.append(AblationRun(
                name=name, daily_returns=rets, equity=equity,
                metrics=metrics, trades=trades,
            ))
            summary[name] = metrics

    # Buy-and-hold benchmark on the *test region only* (so the curves are
    # directly comparable to the PRISM lines which start at first test day).
    non_empty = [r.daily_returns for r in results if not r.daily_returns.empty]
    if non_empty:
        test_dates = pd.concat(non_empty, axis=0).index
        first_date = pd.to_datetime(test_dates).min()
        last_date = pd.to_datetime(test_dates).max()
        panel_dates = pd.to_datetime(panel["date"])
        bh_panel = panel[
            (panel_dates >= first_date) & (panel_dates <= last_date)
        ]
    else:
        log.warning(
            "no ablation produced output — check that train_years + "
            "test_months fits inside the date range of the feature panel"
        )
        bh_panel = panel.iloc[0:0]
    bh_returns, bh_equity = _benchmark_buy_hold(bh_panel, h=h)
    bh_metrics = _metrics(bh_returns, bh_equity) if not bh_returns.empty else {}
    results.append(AblationRun(
        name="buy_hold_universe",
        daily_returns=bh_returns, equity=bh_equity,
        metrics=bh_metrics,
    ))
    if bh_metrics:
        summary["buy_hold_universe"] = bh_metrics

    # Serialise.
    payload = {
        "label":   label or f"prism_{cfg.horizon}_{cfg.target}",
        "config":  asdict(cfg),
        "summary": summary,
        "stress": stress_summary,
        "equity_curves": {
            r.name: {
                "dates":   [str(d.date()) for d in r.equity.index],
                "equity":  [float(v) for v in r.equity.values],
            } for r in results if not r.equity.empty
        },
        "trade_log": {
            r.name: r.trades for r in results if r.trades
        },
        "created_at": int(time.time()),
    }
    return payload


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_run(payload: dict) -> int:
    """Persist a backtest payload to the `backtests` table; return new id."""
    with lock():
        cur = get_conn().execute(
            """INSERT INTO backtests
               (label, params_json, metrics_json, equity_curve_json,
                trades_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                payload.get("label") or "",
                json.dumps({
                    "config": payload.get("config") or {},
                    "stress": payload.get("stress") or {},
                }),
                json.dumps(payload.get("summary") or {}),
                json.dumps(payload.get("equity_curves") or {}),
                json.dumps(payload.get("trade_log") or {}),
                int(payload.get("created_at") or time.time()),
            ),
        )
        return int(cur.lastrowid)


def load_run(run_id: int) -> dict | None:
    row = get_conn().execute(
        "SELECT * FROM backtests WHERE id = ?", (int(run_id),),
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    params = json.loads(d["params_json"] or "{}")
    if "config" in params:
        config = params.get("config") or {}
        stress = params.get("stress") or {}
    else:
        config = params
        stress = {}
    return {
        "id":            d["id"],
        "label":         d["label"],
        "config":        config,
        "stress":        stress,
        "summary":       json.loads(d["metrics_json"] or "{}"),
        "equity_curves": json.loads(d["equity_curve_json"] or "{}"),
        "trade_log":     json.loads(d["trades_json"] or "{}"),
        "created_at":    d["created_at"],
    }


def list_runs(limit: int = 25) -> list[dict]:
    rows = get_conn().execute(
        """SELECT id, label, metrics_json, created_at
           FROM backtests ORDER BY created_at DESC LIMIT ?""",
        (int(limit),),
    ).fetchall()
    return [
        {"id": r["id"], "label": r["label"],
         "summary": json.loads(r["metrics_json"] or "{}"),
         "created_at": r["created_at"]}
        for r in rows
    ]

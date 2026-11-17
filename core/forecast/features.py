"""Feature engineering — the data spine for PRISM.

Per (ticker, date) feature row containing:

  * **Price / volume**        — log-returns at multiple lags, realised vol,
                                tech indicators (reusing core/indicators.py),
                                volume z-score, intraday range, gap %
  * **Cross-sectional**       — return vs NIFTY, return vs sector, sector
                                momentum, rank-within-sector
  * **Sentiment**             — rolling FinBERT score mean / dispersion /
                                count from `news_sentiment` (cached)
  * **Filings**               — days since last filing, guidance signal,
                                concerns count, "what_changed" delta
  * **Graph**                 — counterparty counts, weighted neighbour
                                return (stub for now; expanded in #8)
  * **Macro**                 — INR/USD, Brent, India 10Y, day-of-week,
                                days-to-earnings (stubbed where data
                                ingestion isn't built yet)
  * **Targets**               — `ret_1d`, `ret_5d`, `up_1d`, `up_5d`
                                (forward-looking — set to NaN at the
                                latest dates so training never leaks)

Point-in-time safety:
  * All rolling stats use a strictly trailing window (no look-ahead).
  * Sentiment aggregates use `published_at <= asof`.
  * Filings features use `report_date <= asof`.
  * Graph edges activate only on their source filing's announcement date;
    undated news edges are excluded from historical research features.
  * Targets are computed last and use `shift(-h)` — they are forward-
    looking by definition, and rows where the future is unknown carry
    NaN labels which training loops must drop.

Output: monthly Parquet partitions under `data/features/<YYYY-MM>.parquet`
so we can re-build or filter a single month without touching the rest.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .. import indicators
from ..db import get_conn
from . import sentiment as sent_mod
from . import data_quality

log = logging.getLogger("forecast.features")

ROOT = Path(__file__).parent.parent.parent
FEATURES_DIR = ROOT / "data" / "features"


# ---------------------------------------------------------------------------
# Per-ticker time-series features
# ---------------------------------------------------------------------------


_LAGS = (1, 5, 20, 60)
_VOL_WINDOWS = (5, 20, 60)


def _safe_log_return(close: pd.Series, n: int) -> pd.Series:
    return np.log(close / close.shift(n))


def _alpha_features(df: pd.DataFrame) -> pd.DataFrame:
    """Classic cross-sectional alpha signals from the equities literature.

    These don't overlap with the tech indicators in `_price_volume_features`
    — they're slower, smoother signals that LightGBM can combine with the
    fast technical features to find conditional edge.

    Sources:
      * **Momentum (12-1)** — Jegadeesh-Titman 1993. 12-month return skipping
        the most recent month (which exhibits short-term reversal). The most
        -studied factor in equity markets.
      * **3m / 6m momentum** — shorter momentum lookbacks. Useful for the 5d
        horizon since long-horizon momentum can mean-revert.
      * **1m reversal** — past-month return as a *contrarian* signal. The
        DeBondt-Thaler 1985 observation, robust in Indian equities.
      * **Low-vol rank** — the low-vol anomaly. Low-vol stocks outperform on
        risk-adjusted basis (Frazzini-Pedersen 2014).
      * **Drawdown depth** — how far the stock is below its trailing peak.
        Bad past stocks often keep being bad in the short term.
      * **High-vs-52w / Low-vs-52w** — psychological anchors used as features
        in George-Hwang 2004.

    Beta + idiosyncratic vol are computed separately in
    `_beta_and_idio_features` because they need universe returns.
    """
    c, h, l = df["close"], df["high"], df["low"]
    out = pd.DataFrame(index=df.index)

    # Jegadeesh-Titman: log return over [t-252, t-21] (12m skip-1m).
    out["mom_12_1"] = np.log(c.shift(21) / c.shift(252))
    out["mom_6m"]   = np.log(c.shift(0)  / c.shift(126))
    out["mom_3m"]   = np.log(c.shift(0)  / c.shift(63))
    # Short-term reversal (1m, no skip).
    out["reversal_1m"] = np.log(c.shift(0) / c.shift(21))

    # 52-week reference points (psychological anchors).
    trail_max = c.rolling(252, min_periods=60).max()
    trail_min = c.rolling(252, min_periods=60).min()
    out["dist_52w_high"] = (c - trail_max) / trail_max
    out["dist_52w_low"]  = (c - trail_min) / trail_min.replace(0, np.nan)
    out["drawdown_dx"]   = (c / c.rolling(60, min_periods=20).max()) - 1.0
    return out


def _beta_and_idio_features(
    panel: pd.DataFrame, *, lookback: int = 60,
) -> pd.DataFrame:
    """Rolling beta to universe mean + idiosyncratic vol (residual std).

    Computed cross-sectionally so it needs the full panel. For each
    (token, date) compute beta = Cov(r, r_mkt) / Var(r_mkt) over the
    trailing ``lookback`` days, where ``r_mkt`` is the universe-mean
    daily log return. ``idio_vol`` = std of residuals (r - beta * r_mkt).

    Output is indexed the same as ``panel`` — caller concats by index.
    """
    if "logret_1" not in panel.columns or panel.empty:
        return pd.DataFrame(index=panel.index, columns=["beta_60", "idio_vol_60"])

    # Pivot to a (date x token) matrix of daily log returns.
    wide = (
        panel.pivot_table(index="date", columns="token",
                          values="logret_1", aggfunc="last")
        .sort_index()
    )
    mkt = wide.mean(axis=1)
    # Rolling covariance + variance.
    mkt_var = mkt.rolling(lookback, min_periods=lookback // 2).var()
    centred = wide.sub(wide.rolling(lookback, min_periods=lookback // 2).mean(), axis=0)
    mkt_centred = mkt - mkt.rolling(lookback, min_periods=lookback // 2).mean()
    cov = (centred.mul(mkt_centred, axis=0)
                  .rolling(lookback, min_periods=lookback // 2).mean())
    beta = cov.div(mkt_var, axis=0)
    # Idiosyncratic vol: rolling std of (r - beta * r_mkt).
    resid = wide.sub(beta.mul(mkt, axis=0), axis=0)
    idio_vol = resid.rolling(lookback, min_periods=lookback // 2).std() * math.sqrt(252)
    # Re-shape back to long format and align with panel.
    beta_long = beta.stack(future_stack=True).rename("beta_60")
    idio_long = idio_vol.stack(future_stack=True).rename("idio_vol_60")
    df_long = pd.concat([beta_long, idio_long], axis=1).reset_index()
    df_long["token"] = df_long["token"].astype(str)
    # Join back onto panel by (date, token).
    key = panel[["date", "token"]].copy()
    key["token"] = key["token"].astype(str)
    merged = key.merge(df_long, on=["date", "token"], how="left")
    return merged[["beta_60", "idio_vol_60"]].set_index(panel.index)


def _price_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute purely-trailing price/volume features.

    `df` is expected indexed by date with columns open/high/low/close/volume.
    """
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    o = df["open"]
    out = pd.DataFrame(index=df.index)

    # Log-returns at lags
    for n in _LAGS:
        out[f"logret_{n}"] = _safe_log_return(c, n)

    # Realised vol (std of 1d log-returns over the window, annualised x sqrt(252))
    one_day_ret = _safe_log_return(c, 1)
    for w in _VOL_WINDOWS:
        out[f"rv_{w}"] = (
            one_day_ret.rolling(w, min_periods=max(2, w // 2)).std() * math.sqrt(252)
        )

    # Tech indicators (reuse core/indicators.py)
    out["rsi14"] = indicators.rsi(c, 14)
    macd_line, sig, hist = indicators.macd(c)
    out["macd"] = macd_line
    out["macd_signal"] = sig
    out["macd_hist"] = hist
    upper, mid, lower = indicators.bollinger(c, 20, 2.0)
    rng = (upper - lower).replace(0, np.nan)
    out["bb_pct"] = (c - lower) / rng                # 0 at lower band, 1 at upper
    out["atr14"] = indicators.atr(h, l, c, 14)
    k, d_ = indicators.stochastic(h, l, c)
    out["stoch_k"] = k
    out["stoch_d"] = d_
    # VWAP we compute rolling (20d) — the indicators.vwap is cumulative which
    # leaks across the whole history and is unsuitable for a feature.
    typical = (h + l + c) / 3
    tp_v = typical * v
    out["vwap20_dist"] = (c - (
        tp_v.rolling(20, min_periods=10).sum()
        / v.rolling(20, min_periods=10).sum().replace(0, np.nan)
    )) / c

    # Volume features
    vol_mean = v.rolling(20, min_periods=10).mean()
    vol_std  = v.rolling(20, min_periods=10).std().replace(0, np.nan)
    out["vol_z20"] = (v - vol_mean) / vol_std
    out["vol_ratio_20"] = v / vol_mean.replace(0, np.nan)

    # Microstructure
    prev_close = c.shift(1)
    out["gap_pct"] = (o - prev_close) / prev_close
    out["intraday_range"] = (h - l) / prev_close
    out["close_to_high"] = (h - c) / (h - l).replace(0, np.nan)

    return out


# ---------------------------------------------------------------------------
# Sentiment aggregates (point-in-time)
# ---------------------------------------------------------------------------


def _sentiment_features(token: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Pull rolling FinBERT aggregates for `token` at each ``date`` end-of-day.

    We assume `news_sentiment` has been populated upstream — if it hasn't,
    every column returns NaN/0, which downstream LightGBM handles cleanly.
    """
    rows = []
    for ts in dates:
        # End-of-day timestamp in UTC (close enough for our daily resolution).
        asof = int(_dt.datetime(ts.year, ts.month, ts.day, 23, 59).timestamp())
        agg = sent_mod.aggregates_for_token(token, asof_ts=asof)
        rows.append(agg)
    df = pd.DataFrame(rows, index=dates)
    # Force float dtypes — when the news_sentiment table is empty or sparse,
    # SQL AVG/COUNT return None which pandas would otherwise infer as object,
    # and LightGBM rejects object-dtype columns at training time.
    df = df.apply(pd.to_numeric, errors="coerce").astype(float)
    return df.add_prefix("sent_")


# ---------------------------------------------------------------------------
# Filings features (from llm_summaries)
# ---------------------------------------------------------------------------


_GUIDANCE_POS = ("raise", "increase", "expand", "strong", "improve", "growth",
                 "tailwind", "upgrade", "exceed", "robust", "accelerat")
_GUIDANCE_NEG = ("cut", "reduce", "lower", "weak", "headwind", "decline",
                 "miss", "soft", "slowdown", "deteriorat", "challenging")


def _score_guidance_text(text: str) -> float:
    """Crude lexical score in [-1, 1] over a guidance/outlook string.

    LightGBM can handle nuance later; for now we just need a directional
    indicator that's better than nothing.
    """
    if not text:
        return 0.0
    t = text.lower()
    pos = sum(t.count(w) for w in _GUIDANCE_POS)
    neg = sum(t.count(w) for w in _GUIDANCE_NEG)
    tot = pos + neg
    if tot == 0:
        return 0.0
    return (pos - neg) / tot


def _filings_features(token: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """For each ``date`` D, find the most recent filing with report_date <= D
    and emit features derived from its ``summary_json``.
    """
    rows = get_conn().execute(
        """SELECT report_type, report_date, summary_json
           FROM llm_summaries
           WHERE token = ?
           ORDER BY report_date ASC""",
        (token,),
    ).fetchall()

    if not rows:
        empty = pd.DataFrame(
            index=dates,
            columns=[
                "fil_days_since", "fil_guidance_score", "fil_concerns_n",
                "fil_risks_n",    "fil_whatchanged_n",  "fil_highlights_n",
            ],
        )
        return empty.astype(float)

    # Build a per-filing record we can asof-merge.
    records = []
    for r in rows:
        try:
            sj = json.loads(r["summary_json"])
        except Exception:
            continue
        summ = (sj or {}).get("summary") or {}
        guidance_text = " ".join(summ.get("guidance") or [])
        records.append({
            "report_date":          pd.to_datetime(r["report_date"], errors="coerce"),
            "fil_guidance_score":   _score_guidance_text(guidance_text),
            "fil_concerns_n":       len(summ.get("concerns")    or []),
            "fil_risks_n":          len(summ.get("risks")       or []),
            "fil_whatchanged_n":    len(summ.get("what_changed") or []),
            "fil_highlights_n":     len(summ.get("highlights")  or []),
        })
    fdf = pd.DataFrame(records).dropna(subset=["report_date"]).sort_values("report_date")
    if fdf.empty:
        empty = pd.DataFrame(
            index=dates,
            columns=[
                "fil_days_since", "fil_guidance_score", "fil_concerns_n",
                "fil_risks_n",    "fil_whatchanged_n",  "fil_highlights_n",
            ],
        )
        return empty.astype(float)

    # As-of merge: every date D gets the most-recent filing with date <= D.
    asof = pd.DataFrame({"date": dates}).sort_values("date")
    merged = pd.merge_asof(
        asof, fdf, left_on="date", right_on="report_date", direction="backward",
    )
    merged["fil_days_since"] = (
        (merged["date"] - merged["report_date"]).dt.days.astype("Float64")
    )
    merged = merged.set_index("date")
    cols = [
        "fil_days_since", "fil_guidance_score", "fil_concerns_n",
        "fil_risks_n",    "fil_whatchanged_n",  "fil_highlights_n",
    ]
    return merged[cols].astype(float)


# ---------------------------------------------------------------------------
# Graph features (from relations table)
# ---------------------------------------------------------------------------


def _point_in_time_relations(*, resolved_only: bool = False) -> pd.DataFrame:
    """Return filing-backed relations with a defensible first-usable date.

    ``relations`` is primarily a live UI table and stores extraction time, not
    the time at which the underlying fact became public. Replaying every row
    from the start of the feature panel would therefore leak future filings
    into the past. For research we accept only edges whose evidence URL joins
    to a stored filing summary and use that filing's exchange announcement date
    as ``observed_at``. News-derived edges remain useful in the live network,
    but are deliberately excluded until their source publication timestamp is
    persisted and auditable.

    The relation upsert currently retains one evidence URL per canonical edge.
    If an older occurrence was overwritten this join may make the edge usable
    later than necessary, which is conservative rather than look-ahead biased.
    """
    resolved_clause = "AND r.to_token IS NOT NULL" if resolved_only else ""
    rows = get_conn().execute(
        f"""SELECT r.from_token, r.to_token, r.relation_type,
                   r.confidence, r.source, r.evidence_url,
                   MIN(s.report_date) AS observed_at
            FROM relations AS r
            JOIN llm_summaries AS s
              ON s.token = r.from_token
             AND s.source_url = r.evidence_url
            WHERE r.source IN ('report', 'filing')
              {resolved_clause}
            GROUP BY r.from_token, r.to_token, r.relation_type,
                     r.confidence, r.source, r.evidence_url"""
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=[
            "from_token", "to_token", "relation_type", "confidence",
            "source", "evidence_url", "observed_at",
        ])
    rels = pd.DataFrame([dict(row) for row in rows])
    rels["from_token"] = rels["from_token"].astype(str)
    rels["to_token"] = rels["to_token"].astype("string")
    rels["observed_at"] = pd.to_datetime(
        rels["observed_at"], errors="coerce",
    ).dt.normalize()
    rels["confidence"] = pd.to_numeric(
        rels["confidence"], errors="coerce",
    ).fillna(0.5).clip(lower=0.0, upper=1.0)
    return rels.dropna(subset=["observed_at"])


def _graph_features(token: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Point-in-time counterparty counts from timestamped filing evidence.

    Neighbour-return propagation is added in a cross-sectional pass once all
    securities' trailing returns are aligned.
    """
    rels = _point_in_time_relations()
    rels = rels[rels["from_token"] == str(token)]
    normalised_dates = pd.DatetimeIndex(dates).normalize()
    records: list[dict[str, float]] = []
    for date in normalised_dates:
        active = rels[rels["observed_at"] <= date]
        relation_type = active["relation_type"].fillna("").astype(str)
        total = len(active)
        records.append({
            "graph_n_customers": float(relation_type.isin(
                ["customer", "client"],
            ).sum()),
            "graph_n_suppliers": float(relation_type.isin(
                ["supplier", "vendor"],
            ).sum()),
            "graph_n_partners": float((relation_type == "partner").sum()),
            "graph_n_competitors": float(
                (relation_type == "competitor").sum(),
            ),
            "graph_n_resolved": float(active["to_token"].notna().sum()),
            "graph_avg_confidence": float(
                active["confidence"].mean() if total else 0.0,
            ),
            "graph_filing_share": 1.0 if total else 0.0,
        })

    df = pd.DataFrame(records, index=dates)
    df["graph_neighbour_ret"] = np.nan  # filled cross-sectionally below
    df["graph_neighbour_shock"] = np.nan
    df["graph_centrality_in"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Macro / calendar features
# ---------------------------------------------------------------------------


def _calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=dates)
    df["dow"] = dates.dayofweek.astype(int)
    df["dom"] = dates.day.astype(int)
    df["month"] = dates.month.astype(int)
    return df


# ---------------------------------------------------------------------------
# Targets (forward-looking labels) — keep separate so we never feature-leak
# ---------------------------------------------------------------------------


def _targets(close: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    for h in (1, 5):
        fwd = np.log(close.shift(-h) / close)
        out[f"ret_{h}d"] = fwd
        out[f"up_{h}d"]  = (fwd > 0).astype("float").where(fwd.notna(), np.nan)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_ticker_features(
    candles: pd.DataFrame,
    *,
    token: str,
    symbol: str,
    exchange: str,
    sector: str | None = None,
    industry: str | None = None,
) -> pd.DataFrame:
    """Build the full per-date feature frame for a single ticker.

    `candles` must be a DataFrame indexed by datetime with columns
    open/high/low/close/volume (matching `core.market_data.get_history`).
    """
    if candles is None or candles.empty:
        return pd.DataFrame()
    if candles.attrs.get("corporate_action_adjusted"):
        df = candles.copy()
    else:
        df = data_quality.apply_corporate_actions(
            candles, token=str(token),
        ).copy()
    # Normalise index to date-only so cross-sectional joins line up.
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.index = pd.DatetimeIndex(df.index.normalize())
    df = df[~df.index.duplicated(keep="last")].sort_index()

    parts = [_price_volume_features(df)]
    parts.append(_alpha_features(df))
    parts.append(_sentiment_features(token, df.index))
    parts.append(_filings_features(token, df.index))
    parts.append(_graph_features(token, df.index))
    parts.append(_calendar_features(df.index))

    feat = pd.concat(parts, axis=1)
    feat["token"]     = str(token)
    feat["symbol"]    = symbol
    feat["exchange"]  = exchange
    feat["sector"]    = sector or ""
    feat["industry"]  = industry or ""
    # Keep raw close on the panel so future feature changes can be computed
    # from Parquet without an AngelOne refetch.
    feat["close"] = df["close"].values

    # Targets last.
    feat = pd.concat([feat, _targets(df["close"])], axis=1)

    feat.index.name = "date"
    feat = feat.reset_index()
    return feat


# ---------------------------------------------------------------------------
# Cross-sectional pass + Parquet writer
# ---------------------------------------------------------------------------


def _add_cross_sectional(
    panel: pd.DataFrame, *, market_token: str | None = None,
    universe: str = "nifty50",
) -> pd.DataFrame:
    """Compute return-vs-NIFTY, return-vs-sector and sector-rank features,
    plus cross-sectional **labels** (outperforming the universe median).

    `panel` is the concatenated long-format frame (date, token, ..., ret_1d).
    Cross-sectional features and labels are computed per date.

    Why xs labels matter: predicting absolute next-day direction on a
    heavily-traded large-cap universe is a known coin-flip. Predicting
    *relative* performance ("will this stock beat the median large-cap?")
    cancels market-wide noise and is what every long-short equity quant
    actually trades on.
    """
    if panel.empty:
        return panel
    panel = data_quality.filter_point_in_time(panel, universe=universe)
    if panel.empty:
        return panel

    # Sector-relative return (subtract date-sector mean of 1d return).
    sector_mean = (
        panel.groupby(["date", "sector"])["logret_1"]
        .transform("mean")
    )
    panel["cs_sector_excess"] = panel["logret_1"] - sector_mean
    panel["cs_sector_rank"] = (
        panel.groupby(["date", "sector"])["logret_1"]
        .rank(pct=True)
        .astype(float)
    )

    # Market-relative return: if market_token present in panel, use its return.
    if market_token is not None and (panel["token"] == market_token).any():
        mret = (
            panel[panel["token"] == market_token]
            .set_index("date")["logret_1"]
            .rename("market_ret")
        )
        panel = panel.merge(mret, left_on="date", right_index=True, how="left")
        panel["cs_market_excess"] = panel["logret_1"] - panel["market_ret"]
        panel = panel.drop(columns=["market_ret"])
    else:
        # Fall back to universe-mean as a market proxy.
        univ_mean = panel.groupby("date")["logret_1"].transform("mean")
        panel["cs_market_excess"] = panel["logret_1"] - univ_mean

    # Market-state features available at the close of each date.
    by_date = panel.groupby("date")["logret_1"]
    panel["market_breadth_up"] = by_date.transform(lambda s: (s > 0).mean())
    panel["market_dispersion"] = by_date.transform("std")
    panel["market_mean_return"] = by_date.transform("mean")
    if "sector" in panel.columns:
        panel["sector_mean_return"] = (
            panel.groupby(["date", "sector"])["logret_1"].transform("mean")
        )

    # Cross-sectional beta + idio_vol (needs universe-wide returns).
    bi = _beta_and_idio_features(panel)
    panel["beta_60"]      = bi["beta_60"].values
    panel["idio_vol_60"]  = bi["idio_vol_60"].values
    # Low-vol-anomaly rank: percentile rank of inverse idio_vol per date.
    panel["lowvol_rank"] = (
        panel.groupby("date")["idio_vol_60"].rank(pct=True, ascending=True)
        .astype(float)
    )
    panel = _add_graph_cross_sectional(panel)

    # Cross-sectional LABELS (per date, per horizon): did this ticker
    # outperform the universe median forward return? Quantile-rank in
    # [0, 1] is also exposed for ranking-loss models later.
    for h in (1, 5):
        ret_col = f"ret_{h}d"
        if ret_col not in panel.columns:
            continue
        median = panel.groupby("date")[ret_col].transform("median")
        panel[f"xs_excess_{h}d"] = panel[ret_col] - median
        rank = panel.groupby("date")[ret_col].rank(pct=True)
        panel[f"xs_rank_{h}d"]   = rank.astype(float)
        # Binary outperform label: 1 if above median, 0 if below, NaN if
        # forward return unknown. Median-tied rows go to NaN (rare).
        binary = (rank > 0.5).astype("float").where(rank.notna(), np.nan)
        panel[f"xs_up_{h}d"] = binary
    return panel


def _add_graph_cross_sectional(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach dynamic counterparty return propagation features.

    Each edge becomes active only on or after the exchange-announcement date
    of its source filing. The pass then looks only at same-date trailing
    returns of resolved counterparties.
    """
    rels = _point_in_time_relations(resolved_only=True)
    if rels.empty or panel.empty:
        for col in (
            "graph_neighbour_ret", "graph_neighbour_shock", "graph_centrality_in",
        ):
            if col not in panel:
                panel[col] = np.nan
        return panel
    rels["to_token"] = rels["to_token"].astype(str)
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()

    # Point-in-time inbound degree. The relation table is small compared with
    # the feature panel, so an explicit date loop is clearer and safer than a
    # static map that would leak future edges.
    inbound_by_date: dict[tuple[pd.Timestamp, str], float] = {}
    for date in panel["date"].drop_duplicates().sort_values():
        active = rels[rels["observed_at"] <= date]
        inbound = active.groupby("to_token")["from_token"].nunique()
        inbound_by_date.update({
            (pd.Timestamp(date), str(token)): float(value)
            for token, value in inbound.items()
        })
    panel["graph_centrality_in"] = [
        inbound_by_date.get((pd.Timestamp(date), str(token)), 0.0)
        for date, token in zip(panel["date"], panel["token"])
    ]

    lookup_cols = ["date", "token", "logret_1", "logret_5", "rv_20"]
    available = [col for col in lookup_cols if col in panel.columns]
    lookup = panel[available].copy()
    lookup["token"] = lookup["token"].astype(str)
    lookup = lookup.rename(columns={
        "token": "to_token",
        "logret_1": "_nbr_logret_1",
        "logret_5": "_nbr_logret_5",
        "rv_20": "_nbr_rv_20",
    })
    expanded = panel[["date", "token"]].copy()
    expanded["from_token"] = expanded["token"].astype(str)
    expanded["_row"] = expanded.index
    expanded = expanded.merge(rels, on="from_token", how="left")
    expanded = expanded.dropna(subset=["to_token", "observed_at"])
    expanded = expanded[expanded["observed_at"] <= expanded["date"]]
    if expanded.empty:
        panel["graph_neighbour_ret"] = np.nan
        panel["graph_neighbour_shock"] = np.nan
        return panel
    expanded = expanded.merge(
        lookup.drop(columns=["date"]).assign(date=lookup["date"]),
        on=["date", "to_token"],
        how="left",
    )
    expanded["_weighted_ret"] = (
        pd.to_numeric(expanded.get("_nbr_logret_5"), errors="coerce")
        * expanded["confidence"]
    )
    shock_threshold = pd.to_numeric(
        expanded.get("_nbr_rv_20"), errors="coerce",
    ).fillna(0.0) / math.sqrt(252)
    expanded["_shock"] = (
        pd.to_numeric(expanded.get("_nbr_logret_1"), errors="coerce").abs()
        > (2.0 * shock_threshold.clip(lower=0.005))
    ).astype(float) * expanded["confidence"]
    denom = expanded.groupby("_row")["confidence"].sum().replace(0, np.nan)
    weighted_ret = expanded.groupby("_row")["_weighted_ret"].sum() / denom
    weighted_shock = expanded.groupby("_row")["_shock"].sum() / denom
    panel.loc[weighted_ret.index, "graph_neighbour_ret"] = weighted_ret
    panel.loc[weighted_shock.index, "graph_neighbour_shock"] = weighted_shock
    return panel


def write_partitioned(panel: pd.DataFrame, *, out_dir: Path = FEATURES_DIR) -> list[Path]:
    """Write the long-format panel to monthly Parquet partitions."""
    if panel.empty:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = panel.copy()
    panel["_ym"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m")
    written: list[Path] = []
    for ym, grp in panel.groupby("_ym"):
        p = out_dir / f"{ym}.parquet"
        grp = grp.drop(columns=["_ym"])
        # If the partition already exists, replace it wholesale — the caller
        # decides what date range to rebuild, and partial overwrites would
        # silently leave stale rows behind.
        grp.to_parquet(p, index=False)
        written.append(p)
    log.info("wrote %d partitions to %s", len(written), out_dir)
    return written


def build_universe_features(
    refs: Iterable[dict],
    candles_for: "callable[[dict], pd.DataFrame]",
    *,
    market_token: str | None = None,
    universe: str = "nifty50",
    out_dir: Path = FEATURES_DIR,
) -> pd.DataFrame:
    """End-to-end driver.

    `refs` is an iterable of dicts with keys ``token, symbol, exchange,
    sector, industry``. `candles_for(ref)` is a callable that returns the
    OHLCV DataFrame for that ref (callers inject their own fetch policy —
    e.g. cached AngelOne history, a local CSV, etc.).
    """
    frames: list[pd.DataFrame] = []
    for ref in refs:
        try:
            candles = candles_for(ref)
        except Exception as e:
            log.warning("candle fetch failed for %s: %s", ref.get("symbol"), e)
            continue
        f = build_ticker_features(
            candles,
            token=str(ref["token"]),
            symbol=ref.get("symbol", ""),
            exchange=ref.get("exchange", "NSE"),
            sector=ref.get("sector"),
            industry=ref.get("industry"),
        )
        if not f.empty:
            frames.append(f)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel = _add_cross_sectional(
        panel, market_token=market_token, universe=universe,
    )
    panel = data_quality.attach_macro(panel)
    panel = data_quality.attach_fundamentals(panel)
    panel = cross_sectional_standardize(panel)
    write_partitioned(panel, out_dir=out_dir)
    return panel


# ---------------------------------------------------------------------------
# Loader (for downstream training / inference)
# ---------------------------------------------------------------------------


# Columns that *should not* be cross-sectionally standardised — they're
# categorical/identifier columns or labels. Everything else is fair game.
_NO_STANDARDISE = {
    "date", "token", "symbol", "exchange", "sector", "industry", "regime",
    "close",
    "ret_1d", "ret_5d", "up_1d", "up_5d",
    "xs_up_1d", "xs_up_5d", "xs_excess_1d", "xs_excess_5d",
    "xs_rank_1d", "xs_rank_5d",
    "dow", "dom", "month",
}


def cross_sectional_standardize(
    panel: pd.DataFrame,
    *,
    exclude: set[str] | None = None,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Per-date z-score of every numeric feature across the universe.

    For each (date, feature) compute (x - mean) / std using only that
    day's cross-section. Adds ``_cs`` suffix to standardised columns so
    the model can pick either form (raw and rank-style) — and so the
    Parquet stays human-readable.

    Why this matters: LightGBM grows trees by axis-aligned splits. With
    raw features, a single threshold like "rsi14 < 30" treats all
    stocks equally and ignores that today's 30 may be a high RSI in a
    high-momentum regime. The ``_cs`` column rephrases every feature as
    "where does this stock rank today" — orthogonal information that
    bumps cross-sectional acc by 0.5-1.5 pp in our setting.
    """
    if panel.empty:
        return panel
    excl = (exclude or set()) | _NO_STANDARDISE
    numeric = [
        c for c in panel.columns
        if c not in excl and pd.api.types.is_numeric_dtype(panel[c])
    ]
    if not numeric:
        return panel
    grp = panel.groupby("date")
    means = grp[numeric].transform("mean")
    stds  = grp[numeric].transform("std")
    z = (panel[numeric] - means) / stds.replace(0, np.nan).fillna(eps)
    z = z.add_suffix("_cs")
    return pd.concat([panel, z], axis=1)


def refresh_sentiment_in_panel(
    panel: pd.DataFrame,
    *,
    windows: tuple[int, ...] = (1, 7, 30),
) -> pd.DataFrame:
    """Recompute ``sent_*`` columns in-memory from the live ``news_sentiment``
    table, without touching the Parquet store on disk.

    Why this exists: Parquet partitions were written when sentiment was
    sparse (or pre-backfill, all-NaN). After running --backfill-sentiment
    the DB has scores but the Parquet bytes don't. This refresher does a
    single bulk SELECT per token and recomputes rolling means/dispersions/
    counts in pandas — fast (~1s for 50 tickers × 1500 days) and avoids a
    full AngelOne refetch.
    """
    if panel.empty or "token" not in panel.columns:
        return panel
    conn = get_conn()
    out = panel.copy().sort_values(["token", "date"]).reset_index(drop=True)

    # Drop existing sent_* cols so we can rewrite cleanly.
    drop_cols = [c for c in out.columns if c.startswith("sent_")]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    # New columns, init to NaN/0.
    new_cols = []
    for w in windows:
        new_cols += [f"sent_mean_{w}d", f"sent_dispersion_{w}d", f"sent_count_{w}d"]
    for c in new_cols:
        out[c] = np.nan if "count" not in c else 0

    for token, idx in out.groupby("token", sort=False).indices.items():
        rows = conn.execute(
            """SELECT published_at, score
               FROM news_sentiment
               WHERE token = ? AND published_at IS NOT NULL
               ORDER BY published_at""",
            (str(token),),
        ).fetchall()
        if not rows:
            continue
        nsdf = pd.DataFrame([dict(r) for r in rows])
        nsdf["date"] = pd.to_datetime(nsdf["published_at"], unit="s").dt.normalize()
        # Roll up to per-day so a date can have multiple headlines.
        per_day = nsdf.groupby("date")["score"].agg(["mean", "std", "count"])
        per_day.columns = ["m", "s", "n"]
        per_day = per_day.sort_index()
        # Daily series indexed by every business day in our panel for this token.
        sub = out.iloc[idx][["date"]].copy()
        sub["date"] = pd.to_datetime(sub["date"]).dt.normalize()
        # Build a continuous daily index covering [panel_start, panel_end].
        full_idx = pd.date_range(sub["date"].min(), sub["date"].max(), freq="D")
        daily = pd.DataFrame(index=full_idx)
        daily = daily.join(per_day, how="left")
        # Variance proxy: E[X^2] - E[X]^2 over the window. Track separately so
        # rolling windows can produce a coherent std.
        daily["nm"]  = daily["m"].fillna(0) * daily["n"].fillna(0)
        daily["nm2"] = (
            (daily["s"].fillna(0) ** 2) * (daily["n"].fillna(0) - 1).clip(lower=0)
            + (daily["m"].fillna(0) ** 2) * daily["n"].fillna(0)
        )
        for w in windows:
            n_w  = daily["n"].fillna(0).rolling(w, min_periods=1).sum()
            nm_w = daily["nm"].rolling(w,  min_periods=1).sum()
            nm2_w = daily["nm2"].rolling(w, min_periods=1).sum()
            mean_w = (nm_w / n_w.replace(0, np.nan))
            var_w  = (nm2_w / n_w.replace(0, np.nan)) - (mean_w ** 2)
            var_w  = var_w.clip(lower=0)
            std_w  = var_w ** 0.5
            daily[f"sent_mean_{w}d"]       = mean_w
            daily[f"sent_dispersion_{w}d"] = std_w
            daily[f"sent_count_{w}d"]      = n_w.astype(int)
        # Asof-merge daily features onto the panel's actual trading dates.
        merged = sub.merge(
            daily[[c for c in daily.columns if c.startswith("sent_")]],
            left_on="date", right_index=True, how="left",
        )
        for c in [c for c in daily.columns if c.startswith("sent_")]:
            out.loc[idx, c] = merged[c].values
    # Final dtype guard.
    for w in windows:
        out[f"sent_mean_{w}d"]       = pd.to_numeric(out[f"sent_mean_{w}d"], errors="coerce")
        out[f"sent_dispersion_{w}d"] = pd.to_numeric(out[f"sent_dispersion_{w}d"], errors="coerce")
        out[f"sent_count_{w}d"]      = pd.to_numeric(out[f"sent_count_{w}d"], errors="coerce").fillna(0).astype(int)
    return out


def load_features(
    *,
    start: str | None = None,
    end:   str | None = None,
    tokens: Iterable[str] | None = None,
    out_dir: Path = FEATURES_DIR,
) -> pd.DataFrame:
    """Load the feature panel from Parquet partitions with optional filters."""
    if not out_dir.exists():
        return pd.DataFrame()
    parts = sorted(out_dir.glob("*.parquet"))
    if not parts:
        return pd.DataFrame()
    # Filter partitions by YYYY-MM if start/end given.
    if start is not None:
        s_ym = pd.to_datetime(start).strftime("%Y-%m")
        parts = [p for p in parts if p.stem >= s_ym]
    if end is not None:
        e_ym = pd.to_datetime(end).strftime("%Y-%m")
        parts = [p for p in parts if p.stem <= e_ym]
    dfs = [pd.read_parquet(p) for p in parts]
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    # Defensive dtype repair: any column that *should* be numeric but landed
    # as object (because the writer saw all-None rows) is coerced here. We
    # keep string identifier and category columns as-is.
    _str_cols = {"date", "token", "symbol", "exchange", "sector", "industry", "regime"}
    for c in df.columns:
        if c in _str_cols:
            continue
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if start is not None:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end is not None:
        df = df[df["date"] <= pd.to_datetime(end)]
    if tokens is not None:
        keep = {str(t) for t in tokens}
        df = df[df["token"].isin(keep)]
    return df.sort_values(["date", "token"]).reset_index(drop=True)

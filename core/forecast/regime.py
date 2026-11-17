"""Market-regime detector — `trend` / `meanrev` / `chop`.

Why a regime gate at all? The same model often loses money flipping
between trending and mean-reverting markets — momentum signals work in
one, contrarian signals work in the other, and both lose in chop. The
PRISM stacker uses the regime tag as a categorical feature so it can
re-weight base models per regime instead of one-model-fits-all.

Per-ticker, per-day classification using only data ≤ that day. Inputs:

  * ``rv_20``           — 20-day realised vol of 1d log-returns (annualised)
  * ``adx_proxy``       — abs(20d log-return) / (20d rolling std × √20),
                          i.e. the Sharpe-like trend-strength of the
                          recent move. Higher = stronger directional drift.
  * ``vol_ratio``       — ``rv_5 / rv_60``: short-vs-long-horizon vol.
                          > 1.5 means short vol is spiking — a chop /
                          volatility regime signal.

Rules (deliberately simple and defensible; we can swap in an HMM later
without touching the stacker contract):

  * ``chop``    if ``vol_ratio`` > 1.5     OR ``rv_20`` > 0.50
  * ``trend``   if ``adx_proxy`` > 1.0     AND NOT ``chop``
  * ``meanrev`` otherwise

These thresholds were picked from inspection of NIFTY's regime breakdown
2020-2024 (COVID = chop, 2021 = trend, 2022-23 = meanrev). They are
parameters — `classify()` accepts overrides for the backtest sweep.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from ..db import get_conn, lock

log = logging.getLogger("forecast.regime")

REGIMES = ("trend", "meanrev", "chop")


@dataclass(frozen=True)
class Thresholds:
    chop_volratio: float = 1.5
    chop_rv:       float = 0.50
    trend_adx:     float = 1.0


DEFAULT_THRESHOLDS = Thresholds()


def _safe_log_return(close: pd.Series, n: int) -> pd.Series:
    return np.log(close / close.shift(n))


def compute_signals(close: pd.Series) -> pd.DataFrame:
    """Return a DataFrame of the three regime inputs aligned to ``close``."""
    one_day = _safe_log_return(close, 1)
    rv_5  = one_day.rolling(5,  min_periods=3).std()  * math.sqrt(252)
    rv_20 = one_day.rolling(20, min_periods=10).std() * math.sqrt(252)
    rv_60 = one_day.rolling(60, min_periods=30).std() * math.sqrt(252)
    twenty_day_ret = _safe_log_return(close, 20)
    twenty_day_sd  = one_day.rolling(20, min_periods=10).std()
    adx_proxy = (twenty_day_ret.abs()
                 / (twenty_day_sd.replace(0, np.nan) * math.sqrt(20)))
    vol_ratio = rv_5 / rv_60.replace(0, np.nan)
    out = pd.DataFrame({
        "rv_5":      rv_5,
        "rv_20":     rv_20,
        "rv_60":     rv_60,
        "adx_proxy": adx_proxy,
        "vol_ratio": vol_ratio,
    })
    return out


def classify(
    signals: pd.DataFrame,
    *,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> pd.Series:
    """Map a `compute_signals()` frame to a regime label per row."""
    is_chop = (
        (signals["vol_ratio"] > thresholds.chop_volratio)
        | (signals["rv_20"]    > thresholds.chop_rv)
    )
    is_trend = (signals["adx_proxy"] > thresholds.trend_adx) & ~is_chop
    out = pd.Series(index=signals.index, dtype=object)
    out[:] = "meanrev"
    out[is_trend] = "trend"
    out[is_chop]  = "chop"
    # Rows where all inputs are NaN (start of series) should be unknown,
    # not 'meanrev' — let downstream decide how to handle.
    nan_mask = signals[["rv_20", "adx_proxy", "vol_ratio"]].isna().all(axis=1)
    out[nan_mask] = None
    return out


def classify_close(
    close: pd.Series,
    *,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Convenience: signals + label in one frame."""
    sig = compute_signals(close)
    sig["regime"] = classify(sig, thresholds=thresholds)
    return sig


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def persist(
    token: str,
    frame: pd.DataFrame,
) -> int:
    """Upsert one `classify_close` frame into `regime_state`.

    `frame` must be indexed by date and contain a `regime` column. Other
    columns (rv_20, adx_proxy, etc.) are stored as ``score_json`` so we
    can retune the classifier later without re-deriving inputs.
    """
    rows: list[tuple] = []
    for ts, row in frame.iterrows():
        regime = row.get("regime")
        if not regime:  # skip the NaN warm-up rows
            continue
        d = pd.Timestamp(ts).strftime("%Y-%m-%d")
        scores = {
            k: (None if pd.isna(v) else float(v))
            for k, v in row.items()
            if k != "regime"
        }
        rows.append((d, str(token), regime, json.dumps(scores)))
    if not rows:
        return 0
    with lock():
        get_conn().executemany(
            """INSERT INTO regime_state (date, token, regime, score_json)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(date, token) DO UPDATE SET
                  regime=excluded.regime, score_json=excluded.score_json""",
            rows,
        )
    return len(rows)


def load_for_token(token: str) -> pd.DataFrame:
    rows = get_conn().execute(
        "SELECT date, regime FROM regime_state WHERE token = ? ORDER BY date",
        (str(token),),
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["date", "regime"])
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Feature-panel integration
# ---------------------------------------------------------------------------


def attach_to_panel(
    panel: pd.DataFrame,
    *,
    persist_too: bool = False,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Add a `regime` column to a feature panel (long format).

    Reuses the panel's own `logret_1` and `rv_20` where available; if a
    panel was produced without those columns we recompute from the
    parquet store's closing prices via the per-ticker `logret_1` series.
    """
    if panel.empty or "logret_1" not in panel.columns:
        return panel.assign(regime=None)

    out = panel.copy()
    out["regime"] = None
    for token, grp in out.groupby("token", sort=False):
        grp = grp.sort_values("date")
        one_day = grp["logret_1"]
        # We don't have `close` in the panel but we have logret_1; rv_20 and
        # rv_5/60 are already in the panel. Rebuild signals using available
        # columns where possible, fall back to the helper otherwise.
        rv_5  = grp.get("rv_5")
        rv_20 = grp.get("rv_20")
        rv_60 = grp.get("rv_60")
        if rv_5 is None or rv_20 is None or rv_60 is None:
            continue
        twenty_day_ret = one_day.rolling(20, min_periods=10).sum()
        adx_proxy = (twenty_day_ret.abs()
                     / (one_day.rolling(20, min_periods=10).std().replace(0, np.nan)
                        * math.sqrt(20)))
        vol_ratio = rv_5 / rv_60.replace(0, np.nan)
        sig = pd.DataFrame({
            "rv_5": rv_5, "rv_20": rv_20, "rv_60": rv_60,
            "adx_proxy": adx_proxy, "vol_ratio": vol_ratio,
        }, index=grp.index)
        labels = classify(sig, thresholds=thresholds)
        out.loc[grp.index, "regime"] = labels.values
        if persist_too:
            d = grp[["date"]].copy()
            d["regime"] = labels.values
            d = d.dropna(subset=["regime"])
            if not d.empty:
                d = d.set_index("date")
                d["adx_proxy"] = adx_proxy.values
                d["vol_ratio"] = vol_ratio.values
                d["rv_20"] = rv_20.values
                persist(str(token), d)
    return out

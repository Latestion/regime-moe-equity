"""Yahoo Finance + AngelOne research-data bootstrap.

Yahoo supplies adjusted long-history EOD, actions, macro series and statement
data. AngelOne supplies the canonical NSE token/symbol map and recent candles
used by the feature builder for reconciliation.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from ..db import get_conn, lock
from . import data_quality

log = logging.getLogger("forecast.sources")

MACRO_TICKERS = {
    "india_vix": "^INDIAVIX",
    "usdinr": "INR=X",
    "brent": "BZ=F",
}


def _yf():
    import yfinance
    return yfinance


def _instruments():
    from .. import instruments
    return instruments


@dataclass(frozen=True)
class UniverseRef:
    token: str
    symbol: str
    exchange: str = "NSE"

    @property
    def yahoo_symbol(self) -> str:
        return f"{self.symbol.removesuffix('-EQ')}.NS"


def current_members(universe: str = "nifty200") -> list[UniverseRef]:
    today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    rows = get_conn().execute(
        """SELECT token,symbol FROM universe_membership
           WHERE universe=? AND effective_from<=?
             AND (effective_to IS NULL OR effective_to>=?)
           ORDER BY token,effective_from DESC""",
        (universe.lower(), today, today),
    ).fetchall()
    refs: list[UniverseRef] = []
    seen: set[str] = set()
    for row in rows:
        token = str(row["token"])
        if token in seen:
            continue
        refs.append(UniverseRef(token, str(row["symbol"])))
        seen.add(token)
    return sorted(refs, key=lambda ref: ref.symbol)


def historical_members(universe: str = "nifty200") -> list[UniverseRef]:
    """Return every canonical token ever present in a dated universe.

    Feature-panel construction must fetch exited and delisted members too;
    loading only today's constituents creates survivorship bias before the
    point-in-time membership filter even runs. The most recent recorded symbol
    is used as the display/Yahoo lookup symbol, while canonical EOD storage is
    keyed by token.
    """
    rows = get_conn().execute(
        """SELECT token,symbol,effective_from
           FROM universe_membership
           WHERE universe=?
           ORDER BY token,effective_from DESC""",
        (universe.lower(),),
    ).fetchall()
    refs: list[UniverseRef] = []
    seen: set[str] = set()
    for row in rows:
        token = str(row["token"])
        if token in seen:
            continue
        refs.append(UniverseRef(token, str(row["symbol"])))
        seen.add(token)
    return sorted(refs, key=lambda ref: ref.symbol)


def discover_yahoo_angel_proxy(size: int = 200) -> list[UniverseRef]:
    """Discover a current India large/liquid proxy, never historical NIFTY 200."""
    yf = _yf()
    EquityQuery = yf.EquityQuery

    query = EquityQuery("and", [
        EquityQuery("eq", ["region", "in"]),
        EquityQuery("gt", ["intradaymarketcap", 0]),
        EquityQuery("gt", ["avgdailyvol3m", 0]),
    ])
    quotes = []
    for offset in range(0, 1000, 250):
        response = yf.screen(
            query, offset=offset, size=250,
            sortField="intradaymarketcap", sortAsc=False,
        )
        page = response.get("quotes") or []
        quotes.extend(page)
        if len(page) < 250:
            break
    master = _instruments().load_instruments()
    equities = master[
        (master["exch_seg"] == "NSE")
        & master["symbol"].fillna("").str.endswith("-EQ")
    ].copy()
    by_base = {
        str(row["symbol"]).removesuffix("-EQ").upper(): UniverseRef(
            str(row["token"]), str(row["symbol"]),
        )
        for _, row in equities.iterrows()
    }
    result = []
    seen = set()
    for quote in quotes:
        yahoo_symbol = str(quote.get("symbol") or "")
        if not yahoo_symbol.endswith(".NS"):
            continue
        ref = by_base.get(yahoo_symbol.removesuffix(".NS").upper())
        if ref and ref.token not in seen:
            result.append(ref)
            seen.add(ref.token)
        if len(result) >= size:
            break
    return result


def store_current_proxy(
    refs: Iterable[UniverseRef],
    *,
    universe: str = "yahoo_angel200",
) -> int:
    today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    universe = universe.lower()
    with lock():
        get_conn().execute(
            """DELETE FROM universe_membership
               WHERE universe=? AND COALESCE(source,'') LIKE '%proxy%'""",
            (universe,),
        )
    return data_quality.upsert_membership([
        {
            "universe": universe,
            "token": ref.token,
            "symbol": ref.symbol,
            "effective_from": today,
            "effective_to": None,
            "source": "yahoo_screen_current_proxy",
        }
        for ref in refs
    ])


def sync_eod_and_actions(
    refs: Iterable[UniverseRef],
    *,
    start: str = "2015-01-01",
    pause_seconds: float = 0.15,
) -> dict:
    counts = {"symbols": 0, "eod": 0, "actions": 0, "failures": []}
    now = int(time.time())
    for ref in refs:
        try:
            coverage = get_conn().execute(
                """SELECT COUNT(*) AS n,MIN(trade_date) AS first_date,
                          MAX(trade_date) AS last_date
                   FROM eod_prices WHERE token=?""",
                (ref.token,),
            ).fetchone()
            requested_start = pd.to_datetime(start)
            if (
                coverage and int(coverage["n"] or 0) >= 500
                and pd.to_datetime(coverage["first_date"]) <= requested_start + pd.Timedelta(days=31)
                and pd.to_datetime(coverage["last_date"]) >= pd.Timestamp.today().normalize() - pd.Timedelta(days=10)
            ):
                counts["symbols"] += 1
                counts.setdefault("skipped_complete", 0)
                counts["skipped_complete"] += 1
                continue
            ticker = _yf().Ticker(ref.yahoo_symbol)
            try:
                history = ticker.history(
                    start=start, interval="1d", auto_adjust=True,
                    actions=True, repair=True, raise_errors=True,
                )
            except Exception:
                history = ticker.history(
                    start=start, interval="1d", auto_adjust=True,
                    actions=True, repair=False, raise_errors=True,
                )
            if history.empty:
                raise RuntimeError("empty Yahoo history")
            history.index = pd.to_datetime(history.index).tz_localize(None)
            eod_values = []
            action_values = []
            for date, row in history.iterrows():
                close = _number(row.get("Close"))
                if close is None:
                    continue
                eod_values.append((
                    ref.token, str(date.date()),
                    _number(row.get("Open")), _number(row.get("High")),
                    _number(row.get("Low")), close, _number(row.get("Volume")),
                    1, "yahoo_auto_adjusted", now,
                ))
                split = _number(row.get("Stock Splits")) or 0
                if split:
                    action_values.append((
                        ref.token, str(date.date()), "split",
                        1.0 / split, split,
                        f'{{"ratio":{split},"source":"yahoo"}}',
                    ))
                dividend = _number(row.get("Dividends")) or 0
                if dividend:
                    action_values.append((
                        ref.token, str(date.date()), "dividend",
                        1.0, 1.0,
                        f'{{"cash":{dividend},"source":"yahoo"}}',
                    ))
            with lock():
                get_conn().executemany(
                    """INSERT INTO eod_prices(
                         token,trade_date,open,high,low,close,volume,
                         adjusted,source,ingested_at
                       ) VALUES (?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(token,trade_date) DO UPDATE SET
                         open=excluded.open,high=excluded.high,low=excluded.low,
                         close=excluded.close,volume=excluded.volume,
                         adjusted=excluded.adjusted,source=excluded.source,
                         ingested_at=excluded.ingested_at""",
                    eod_values,
                )
                get_conn().executemany(
                    """INSERT INTO corporate_actions(
                         token,action_date,action_type,price_factor,
                         volume_factor,detail_json
                       ) VALUES (?,?,?,?,?,?)
                       ON CONFLICT(token,action_date,action_type) DO UPDATE SET
                         detail_json=excluded.detail_json""",
                    action_values,
                )
            counts["symbols"] += 1
            counts["eod"] += len(eod_values)
            counts["actions"] += len(action_values)
            log.info("Yahoo history %s: %d rows", ref.symbol, len(eod_values))
        except Exception as exc:
            log.warning("Yahoo history failed %s: %s", ref.symbol, exc)
            counts["failures"].append({"symbol": ref.symbol, "error": str(exc)})
        time.sleep(max(0.0, pause_seconds))
    return counts


def sync_macro(start: str = "2015-01-01") -> dict:
    counts = {}
    values = []
    for series, ticker in MACRO_TICKERS.items():
        try:
            history = _yf().Ticker(ticker).history(
                start=start, interval="1d", auto_adjust=True,
                repair=True, raise_errors=True,
            )
            history.index = pd.to_datetime(history.index).tz_localize(None)
            rows = 0
            for date, value in history["Close"].items():
                value = _number(value)
                if value is None:
                    continue
                trade_date = str(date.date())
                values.append((
                    series, trade_date, trade_date, value,
                    f"yahoo:{ticker}",
                ))
                rows += 1
            counts[series] = rows
        except Exception as exc:
            counts[series] = {"error": str(exc)}
    with lock():
        get_conn().executemany(
            """INSERT INTO macro_observations(
                 series,observation_date,available_at,value,source
               ) VALUES (?,?,?,?,?)
               ON CONFLICT(series,observation_date) DO UPDATE SET
                 available_at=excluded.available_at,value=excluded.value,
                 source=excluded.source""",
            values,
        )
    counts["total"] = len(values)
    return counts


def sync_quarterly_fundamentals(
    refs: Iterable[UniverseRef],
    *,
    pause_seconds: float = 0.25,
) -> dict:
    counts = {"symbols": 0, "rows": 0, "failures": []}
    metric_maps = {
        "revenue": ("income", "Total Revenue"),
        "net_income": ("income", "Net Income"),
        "ebitda": ("income", "EBITDA"),
        "operating_cashflow": ("cashflow", "Operating Cash Flow"),
        "capital_expenditure": ("cashflow", "Capital Expenditure"),
    }
    for ref in refs:
        try:
            existing = get_conn().execute(
                """SELECT COUNT(*) AS n FROM fundamental_observations
                   WHERE token=?""",
                (ref.token,),
            ).fetchone()["n"]
            if int(existing or 0) > 0:
                counts["symbols"] += 1
                counts["rows"] += int(existing)
                counts.setdefault("skipped_complete", 0)
                counts["skipped_complete"] += 1
                continue
            ticker = _yf().Ticker(ref.yahoo_symbol)
            income = ticker.get_income_stmt(freq="quarterly")
            cashflow = ticker.get_cashflow(freq="quarterly")
            try:
                earnings = ticker.get_earnings_dates(limit=40)
            except (KeyError, IndexError):
                earnings = None
            publication_dates = (
                sorted(pd.to_datetime(earnings.index).tz_localize(None))
                if earnings is not None and not earnings.empty else []
            )
            frames = {"income": income, "cashflow": cashflow}
            values = []
            for metric, (frame_name, yahoo_name) in metric_maps.items():
                frame = frames[frame_name]
                if frame is None or frame.empty or yahoo_name not in frame.index:
                    continue
                series = frame.loc[yahoo_name]
                for period_end, value in series.items():
                    value = _number(value)
                    if value is None:
                        continue
                    period_end = pd.Timestamp(period_end).tz_localize(None)
                    available = next(
                        (date for date in publication_dates
                         if period_end < date <= period_end + pd.Timedelta(days=140)),
                        None,
                    )
                    if available is None:
                        continue  # no defensible publication timestamp
                    values.append((
                        ref.token, metric, str(period_end.date()),
                        str(available.date()), value,
                        f"yahoo_statement:{ref.yahoo_symbol}",
                    ))
            with lock():
                get_conn().executemany(
                    """INSERT INTO fundamental_observations(
                         token,metric,observation_date,available_at,value,source
                       ) VALUES (?,?,?,?,?,?)
                       ON CONFLICT(token,metric,observation_date) DO UPDATE SET
                         available_at=excluded.available_at,value=excluded.value,
                         source=excluded.source""",
                    values,
                )
            counts["symbols"] += 1
            counts["rows"] += len(values)
        except Exception as exc:
            log.warning("Yahoo fundamentals failed %s: %s", ref.symbol, exc)
            counts["failures"].append({"symbol": ref.symbol, "error": str(exc)})
        time.sleep(max(0.0, pause_seconds))
    return counts


def bootstrap(
    *,
    universe: str = "nifty200",
    start: str = "2015-01-01",
    proxy_current: bool = False,
    limit: int | None = None,
) -> dict:
    # A historical research download must include every past member, not just
    # the names active today. Point-in-time filtering happens after their price
    # histories have been loaded.
    refs = historical_members(universe)
    membership_mode = "point_in_time_import"
    if refs:
        sources = get_conn().execute(
            "SELECT source FROM universe_membership WHERE universe=?",
            (universe.lower(),),
        ).fetchall()
        if any("proxy" in str(row["source"] or "").lower() for row in sources):
            membership_mode = "current_proxy_only"
    if not refs and proxy_current:
        refs = discover_yahoo_angel_proxy(size=limit or 200)
        universe = "yahoo_angel200"
        store_current_proxy(refs, universe=universe)
        membership_mode = "current_proxy_only"
    if not refs:
        raise RuntimeError(
            "No current universe membership is available. Yahoo Finance and "
            "AngelOne do not expose historical NIFTY 200 entry/exit dates. "
            "Import membership first, or use --proxy-current for a clearly "
            "labelled non-NIFTY current Yahoo/Angel large-cap proxy."
        )
    if limit:
        refs = refs[:max(1, limit)]
    return {
        "universe": universe,
        "membership_mode": membership_mode,
        "members": len(refs),
        "eod_actions": sync_eod_and_actions(refs, start=start),
        "macro": sync_macro(start=start),
        "fundamentals": sync_quarterly_fundamentals(refs),
    }


def _number(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None

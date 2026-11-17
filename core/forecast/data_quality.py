"""Point-in-time universe and corporate-action utilities.

The tables are deliberately source-agnostic: an index-constituent or corporate
action importer can populate SQLite without changing the model pipeline.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..db import get_conn, lock


def upsert_membership(rows: Iterable[dict]) -> int:
    values = [
        (
            row["universe"], str(row["token"]), row["symbol"],
            str(row["effective_from"]), row.get("effective_to"), row.get("source"),
        )
        for row in rows
    ]
    if not values:
        return 0
    with lock():
        get_conn().executemany(
            """INSERT INTO universe_membership(
                 universe,token,symbol,effective_from,effective_to,source
               ) VALUES (?,?,?,?,?,?)
               ON CONFLICT(universe,token,effective_from) DO UPDATE SET
                 symbol=excluded.symbol,effective_to=excluded.effective_to,
                 source=excluded.source""",
            values,
        )
    return len(values)


def import_membership_csv(
    path: str | Path, *, universe: str = "nifty200", source: str = "csv",
) -> int:
    """Import date-effective membership.

    Required columns: symbol, token, effective_from. Optional:
    effective_to, universe, source.
    """
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    normalised = []
    for row in rows:
        if not row.get("symbol") or not row.get("token") or not row.get("effective_from"):
            raise ValueError("membership CSV requires symbol, token, effective_from")
        normalised.append({
            "universe": (row.get("universe") or universe).lower(),
            "token": str(row["token"]),
            "symbol": row["symbol"],
            "effective_from": row["effective_from"],
            "effective_to": row.get("effective_to") or None,
            "source": row.get("source") or source,
        })
    return upsert_membership(normalised)


def import_symbol_history_csv(path: str | Path, *, source: str = "csv") -> int:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = []
    for row in rows:
        if not row.get("canonical_token") or not row.get("symbol") or not row.get("effective_from"):
            raise ValueError(
                "symbol-history CSV requires canonical_token, symbol, effective_from"
            )
        values.append((
            str(row["canonical_token"]), row["symbol"],
            row.get("exchange") or "NSE", row["effective_from"],
            row.get("effective_to") or None, row.get("source") or source,
        ))
    with lock():
        get_conn().executemany(
            """INSERT INTO symbol_history(
                 canonical_token,symbol,exchange,effective_from,effective_to,source
               ) VALUES (?,?,?,?,?,?)
               ON CONFLICT(canonical_token,symbol,effective_from) DO UPDATE SET
                 exchange=excluded.exchange,effective_to=excluded.effective_to,
                 source=excluded.source""",
            values,
        )
    return len(values)


def import_corporate_actions_csv(path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = []
    for row in rows:
        values.append((
            str(row["token"]), row["action_date"], row["action_type"],
            float(row.get("price_factor") or 1),
            float(row.get("volume_factor") or 1),
            json.dumps({
                k: v for k, v in row.items()
                if k not in {
                    "token", "action_date", "action_type",
                    "price_factor", "volume_factor",
                }
            }),
        ))
    with lock():
        get_conn().executemany(
            """INSERT INTO corporate_actions(
                 token,action_date,action_type,price_factor,volume_factor,detail_json
               ) VALUES (?,?,?,?,?,?)
               ON CONFLICT(token,action_date,action_type) DO UPDATE SET
                 price_factor=excluded.price_factor,
                 volume_factor=excluded.volume_factor,
                 detail_json=excluded.detail_json""",
            values,
        )
    return len(values)


def import_macro_csv(path: str | Path, *, source: str = "csv") -> int:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = []
    for row in rows:
        if not row.get("series") or not row.get("observation_date"):
            raise ValueError("macro CSV requires series, observation_date, value")
        values.append((
            row["series"].lower(), row["observation_date"],
            row.get("available_at") or row["observation_date"],
            float(row["value"]), row.get("source") or source,
        ))
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
    return len(values)


def import_fundamentals_csv(path: str | Path, *, source: str = "csv") -> int:
    """Import long-form point-in-time fundamentals.

    Required: token, metric, observation_date, value. ``available_at`` must
    be the publication/filing date, not the fiscal period end.
    """
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = []
    for row in rows:
        required = ("token", "metric", "observation_date", "available_at", "value")
        if any(not row.get(name) for name in required):
            raise ValueError(
                "fundamentals CSV requires token, metric, observation_date, "
                "available_at, value"
            )
        values.append((
            str(row["token"]), row["metric"].lower(), row["observation_date"],
            row["available_at"], float(row["value"]),
            row.get("source") or source,
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
    return len(values)


def import_eod_csv(path: str | Path, *, source: str = "csv") -> int:
    """Import canonical EOD rows: token,date,open,high,low,close,volume."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    now = int(time.time())
    values = []
    for row in rows:
        if not row.get("token") or not (row.get("date") or row.get("trade_date")):
            raise ValueError("EOD CSV requires token and date/trade_date")
        values.append((
            str(row["token"]), row.get("trade_date") or row["date"],
            _float_or_none(row.get("open")), _float_or_none(row.get("high")),
            _float_or_none(row.get("low")), float(row["close"]),
            _float_or_none(row.get("volume")),
            int(str(row.get("adjusted") or "0").lower() in {"1", "true", "yes"}),
            row.get("source") or source, now,
        ))
    with lock():
        get_conn().executemany(
            """INSERT INTO eod_prices(
                 token,trade_date,open,high,low,close,volume,adjusted,source,ingested_at
               ) VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(token,trade_date) DO UPDATE SET
                 open=excluded.open,high=excluded.high,low=excluded.low,
                 close=excluded.close,volume=excluded.volume,
                 adjusted=excluded.adjusted,source=excluded.source,
                 ingested_at=excluded.ingested_at""",
            values,
        )
    return len(values)


def load_eod(
    token: str, *, start: str | None = None, end: str | None = None,
) -> pd.DataFrame:
    clauses = ["token=?"]
    params: list[object] = [str(token)]
    if start:
        clauses.append("trade_date>=?")
        params.append(start)
    if end:
        clauses.append("trade_date<=?")
        params.append(end)
    rows = get_conn().execute(
        f"""SELECT trade_date,open,high,low,close,volume,adjusted
            FROM eod_prices WHERE {' AND '.join(clauses)}
            ORDER BY trade_date""",
        params,
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame([dict(row) for row in rows])
    adjusted = bool(frame["adjusted"].fillna(0).astype(int).all())
    dates = pd.to_datetime(frame.pop("trade_date"))
    frame = frame.drop(columns=["adjusted"])
    frame.index = dates
    frame.index.name = "date"
    if not adjusted:
        frame = apply_corporate_actions(frame, token=str(token))
    frame.attrs["corporate_action_adjusted"] = True
    return frame


def _float_or_none(value):
    return None if value in (None, "") else float(value)


def filter_point_in_time(
    panel: pd.DataFrame, *, universe: str = "nifty50",
) -> pd.DataFrame:
    """Keep only rows whose token was a member on that row's date.

    If no membership history has been imported, return the panel unchanged so
    existing installations remain usable while clearly reporting zero coverage.
    """
    rows = get_conn().execute(
        """SELECT token,effective_from,effective_to,source
           FROM universe_membership WHERE universe=?""",
        (universe,),
    ).fetchall()
    if not rows or panel.empty:
        return panel
    membership = pd.DataFrame([dict(row) for row in rows])
    if membership["source"].fillna("").str.contains("proxy").any():
        return panel
    membership["effective_from"] = pd.to_datetime(membership["effective_from"])
    membership["effective_to"] = pd.to_datetime(membership["effective_to"])
    source = panel.copy()
    source["date"] = pd.to_datetime(source["date"])
    source["token"] = source["token"].astype(str)
    chunks = []
    for token, group in source.groupby("token", sort=False):
        periods = membership[membership["token"].astype(str) == token]
        if periods.empty:
            continue
        active = pd.Series(False, index=group.index)
        for _, period in periods.iterrows():
            mask = group["date"] >= period["effective_from"]
            if pd.notna(period["effective_to"]):
                mask &= group["date"] <= period["effective_to"]
            active |= mask
        chunks.append(group[active])
    return pd.concat(chunks, ignore_index=True) if chunks else source.iloc[0:0]


def apply_corporate_actions(candles: pd.DataFrame, *, token: str) -> pd.DataFrame:
    """Back-adjust OHLCV using explicitly stored factors.

    For a 2-for-1 split, store price_factor=0.5 and volume_factor=2.0. Factors
    apply to observations strictly before the action date.
    """
    if candles is None or candles.empty:
        return candles
    rows = get_conn().execute(
        """SELECT action_date,price_factor,volume_factor
           FROM corporate_actions WHERE token=? ORDER BY action_date""",
        (str(token),),
    ).fetchall()
    if not rows:
        return candles
    adjusted = candles.copy()
    adjusted["volume"] = pd.to_numeric(
        adjusted["volume"], errors="coerce",
    ).astype(float)
    index_dates = pd.to_datetime(adjusted.index).tz_localize(None).normalize()
    for row in rows:
        action_date = pd.to_datetime(row["action_date"]).normalize()
        mask = index_dates < action_date
        adjusted.loc[mask, ["open", "high", "low", "close"]] *= float(
            row["price_factor"]
        )
        adjusted.loc[mask, "volume"] *= float(row["volume_factor"])
    return adjusted


def attach_macro(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach macro values only from their declared availability date onward."""
    if panel.empty:
        return panel
    rows = get_conn().execute(
        """SELECT series,available_at,value
           FROM macro_observations ORDER BY available_at"""
    ).fetchall()
    out = panel.copy()
    series_names = ("india_vix", "usdinr", "brent", "india_10y")
    if not rows:
        for name in series_names:
            out[f"macro_{name}"] = float("nan")
        return out
    macro = pd.DataFrame([dict(row) for row in rows])
    macro["available_at"] = pd.to_datetime(macro["available_at"])
    wide = (
        macro.pivot_table(
            index="available_at", columns="series", values="value", aggfunc="last",
        )
        .sort_index()
        .reset_index()
    )
    dates = pd.DataFrame({
        "date": pd.to_datetime(out["date"]).drop_duplicates().sort_values(),
    })
    dated = pd.merge_asof(
        dates, wide, left_on="date", right_on="available_at", direction="backward",
    ).drop(columns=["available_at"])
    rename = {name: f"macro_{name}" for name in series_names if name in dated}
    dated = dated.rename(columns=rename)
    out["date"] = pd.to_datetime(out["date"])
    out = out.merge(dated, on="date", how="left")
    for name in series_names:
        col = f"macro_{name}"
        if col not in out:
            out[col] = float("nan")
    return out


def attach_fundamentals(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel
    rows = get_conn().execute(
        """SELECT token,metric,available_at,value
           FROM fundamental_observations ORDER BY token,available_at"""
    ).fetchall()
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    if not rows:
        out["fund_days_since"] = float("nan")
        out["fund_is_stale"] = 1.0
        out["fund_has_recent"] = 0.0
        return out
    observations = pd.DataFrame([dict(row) for row in rows])
    observations["available_at"] = pd.to_datetime(observations["available_at"])
    chunks = []
    for token, group in out.groupby("token", sort=False):
        token_obs = observations[
            observations["token"].astype(str) == str(token)
        ]
        if token_obs.empty:
            empty = group.copy()
            empty["fund_days_since"] = float("nan")
            empty["fund_is_stale"] = 1.0
            empty["fund_has_recent"] = 0.0
            chunks.append(empty)
            continue
        wide = token_obs.pivot_table(
            index="available_at", columns="metric", values="value", aggfunc="last",
        ).sort_index().add_prefix("fund_").reset_index()
        wide = _derive_fundamental_features(wide)
        wide["_fund_available_at"] = wide["available_at"]
        chunks.append(pd.merge_asof(
            group.sort_values("date"), wide,
            left_on="date", right_on="available_at", direction="backward",
        ).drop(columns=["available_at"]))
    result = pd.concat(chunks, ignore_index=True)
    if "_fund_available_at" in result.columns:
        result["fund_days_since"] = (
            result["date"] - result["_fund_available_at"]
        ).dt.days.astype("float")
        result = result.drop(columns=["_fund_available_at"])
    else:
        result["fund_days_since"] = float("nan")
    fundamental_cols = [
        col for col in result.columns
        if col.startswith("fund_")
        and col not in {"fund_days_since", "fund_is_stale", "fund_has_recent"}
    ]
    has_any = result[fundamental_cols].notna().any(axis=1) if fundamental_cols else False
    result["fund_is_stale"] = (
        result["fund_days_since"].isna() | (result["fund_days_since"] > 220)
    ).astype(float)
    result["fund_has_recent"] = (
        has_any & (result["fund_is_stale"] == 0.0)
    ).astype(float)
    return result


def _derive_fundamental_features(wide: pd.DataFrame) -> pd.DataFrame:
    """Add point-in-time derived fundamentals before the as-of merge."""
    out = wide.copy().sort_values("available_at")
    if "fund_operating_cashflow" in out and "fund_capital_expenditure" in out:
        capex = pd.to_numeric(out["fund_capital_expenditure"], errors="coerce")
        ocf = pd.to_numeric(out["fund_operating_cashflow"], errors="coerce")
        out["fund_fcf"] = ocf + capex
    if "fund_fcf" in out and "fund_market_cap" in out:
        market_cap = pd.to_numeric(out["fund_market_cap"], errors="coerce")
        out["fund_fcf_yield"] = pd.to_numeric(
            out["fund_fcf"], errors="coerce",
        ) / market_cap.replace(0, pd.NA)
    growth_sources = {
        "fund_revenue": "fund_revenue_growth_qoq",
        "fund_eps": "fund_eps_growth_qoq",
        "fund_ebitda": "fund_ebitda_growth_qoq",
        "fund_net_income": "fund_net_income_growth_qoq",
        "fund_fcf": "fund_fcf_growth_qoq",
    }
    for source, target in growth_sources.items():
        if source in out:
            out[target] = pd.to_numeric(
                out[source], errors="coerce",
            ).pct_change(fill_method=None)
    return out


def quality_report(panel: pd.DataFrame | None = None) -> dict:
    membership = get_conn().execute(
        "SELECT COUNT(*) AS n FROM universe_membership"
    ).fetchone()["n"]
    actions = get_conn().execute(
        "SELECT COUNT(*) AS n FROM corporate_actions"
    ).fetchone()["n"]
    report = {
        "membership_rows": int(membership),
        "corporate_actions": int(actions),
        "symbol_history_rows": int(get_conn().execute(
            "SELECT COUNT(*) AS n FROM symbol_history"
        ).fetchone()["n"]),
        "eod_rows": int(get_conn().execute(
            "SELECT COUNT(*) AS n FROM eod_prices"
        ).fetchone()["n"]),
        "fundamental_rows": int(get_conn().execute(
            "SELECT COUNT(*) AS n FROM fundamental_observations"
        ).fetchone()["n"]),
        "macro_rows": int(get_conn().execute(
            "SELECT COUNT(*) AS n FROM macro_observations"
        ).fetchone()["n"]),
    }
    if panel is not None and not panel.empty:
        sector_coverage = 0.0
        if "sector" in panel.columns:
            sectors = panel["sector"].fillna("").astype(str).str.strip()
            sector_coverage = float(
                (~sectors.isin(["", "Unknown", "unknown"])).mean()
            )
        report.update({
            "rows": int(len(panel)),
            "tokens": int(panel["token"].nunique()),
            "start": str(pd.to_datetime(panel["date"]).min().date()),
            "end": str(pd.to_datetime(panel["date"]).max().date()),
            "missing_fraction": float(panel.isna().mean().mean()),
            "duplicate_token_dates": int(
                panel.duplicated(subset=["token", "date"]).sum()
            ),
            "sector_coverage": sector_coverage,
        })
    return report


def validate_research_data(
    panel: pd.DataFrame,
    *,
    universe: str = "nifty200",
    min_membership_rows: int = 200,
    max_missing_fraction: float = 0.35,
    allow_current_proxy: bool = False,
) -> dict:
    report = quality_report(panel)
    universe_rows = get_conn().execute(
        "SELECT COUNT(*) AS n FROM universe_membership WHERE universe=?",
        (universe.lower(),),
    ).fetchone()["n"]
    membership_meta = get_conn().execute(
        """SELECT MIN(effective_from) AS first_date,
                  MAX(CASE WHEN source LIKE '%proxy%' THEN 1 ELSE 0 END) AS proxy
           FROM universe_membership WHERE universe=?""",
        (universe.lower(),),
    ).fetchone()
    failures = []
    membership_is_proxy = bool(
        membership_meta and int(membership_meta["proxy"] or 0)
    )
    effective_min_membership_rows = (
        1 if allow_current_proxy and membership_is_proxy else min_membership_rows
    )
    if int(universe_rows) < effective_min_membership_rows:
        failures.append("insufficient_point_in_time_membership")
    panel_start = pd.to_datetime(panel["date"]).min() if not panel.empty else None
    first_membership = (
        pd.to_datetime(membership_meta["first_date"])
        if membership_meta and membership_meta["first_date"] else None
    )
    if (
        panel_start is not None
        and (first_membership is None or first_membership > panel_start)
    ):
        failures.append("membership_does_not_cover_panel_start")
    if membership_is_proxy:
        failures.append("current_proxy_membership_not_point_in_time")
    if allow_current_proxy:
        failures = [
            failure for failure in failures
            if failure not in {
                "membership_does_not_cover_panel_start",
                "current_proxy_membership_not_point_in_time",
            }
        ]
    if report.get("duplicate_token_dates", 0):
        failures.append("duplicate_token_dates")
    if report.get("missing_fraction", 1.0) > max_missing_fraction:
        failures.append("excessive_missing_features")
    if report.get("tokens", 0) < 100 and universe.lower() == "nifty200":
        failures.append("insufficient_universe_breadth")
    if universe.lower() == "nifty200" and report.get("sector_coverage", 0.0) < 0.80:
        failures.append("insufficient_sector_coverage")
    if universe.lower() == "nifty200" and report.get("eod_rows", 0) < 10_000:
        failures.append("insufficient_canonical_eod_history")
    if universe.lower() == "nifty200" and report.get("symbol_history_rows", 0) < 100:
        failures.append("insufficient_symbol_history")
    report.update({
        "universe": universe.lower(),
        "universe_membership_rows": int(universe_rows),
        "membership_first_date": (
            str(first_membership.date()) if first_membership is not None else None
        ),
        "membership_is_proxy": membership_is_proxy,
        "exploratory_proxy_override": bool(allow_current_proxy),
        "passed": not failures,
        "failures": failures,
    })
    return report

"""Delivery-percentage and insider-disclosure features, as fixed in `PREREGISTRATION_FLOW.md`.

Everything is point-in-time:
  * delivery for session D is published after the close, so it is attached from session D+1;
  * an insider row is stamped by its exchange broadcast date B and counts from the first
    session strictly after B, so a feature at session t uses broadcasts with
    t - 30d < B < t and nothing from t's own day.
A stock with no match stays missing (NaN); it is never 0-filled. The one exception is the
insider columns for a symbol whose disclosure history was fetched: no rows in the window is
an observed zero.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "research" / "raw" / "nse_flow"

FLOW_FEATURES = [
    "deliv_pct", "deliv_rel20", "deliv_chg5",
    "ins_net_30d", "ins_buys_30d", "ins_sells_30d",
]
INSIDER_WINDOW_DAYS = 30


def _base_symbol(symbol: str) -> str:
    return str(symbol).removesuffix("-EQ")


def load_delivery(raw_dir: Path = RAW) -> pd.DataFrame:
    """Series-EQ rows of every cached `MTO_DDMMYYYY.DAT`: columns symbol, date, deliv_pct."""
    rows = []
    for path in sorted((raw_dir / "mto").glob("MTO_*.DAT")):
        date = pd.to_datetime(path.stem.removeprefix("MTO_"), format="%d%m%Y")
        for line in path.read_text(encoding="latin-1").splitlines():
            parts = line.strip().split(",")
            if len(parts) == 7 and parts[0] == "20" and parts[3] == "EQ":
                name, pct = parts[2], parts[6]
            elif len(parts) == 6 and parts[0] == "20":  # old files carry no series column
                name, pct = parts[2], parts[5]
            else:
                continue
            try:
                rows.append((name, date, float(pct)))
            except ValueError:
                continue
    return pd.DataFrame(rows, columns=["symbol", "date", "deliv_pct"])


def _num(value) -> float:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0


def load_insider(raw_dir: Path = RAW) -> tuple[pd.DataFrame, set[str]]:
    """Buy/Sell disclosures stamped by broadcast date, plus the symbols whose history exists.

    Returns (rows, fetched): rows has symbol, bdate (broadcast date), buy_val, sell_val;
    `fetched` is every symbol that has a cached response, even an empty one.
    """
    rows, fetched, seen = [], set(), set()
    for path in sorted((raw_dir / "pit").glob("*.json")):
        fetched.add(path.name.split("_")[0])
        for raw in json.loads(path.read_text(encoding="utf-8")):
            # Overlapping fetch ranges (e.g. a sample year inside a full run) repeat rows.
            fingerprint = json.dumps(raw, sort_keys=True)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            kind = (raw.get("tdpTransactionType") or "").strip().lower()
            if kind not in ("buy", "sell"):
                continue
            stamp = pd.to_datetime(raw.get("date"), format="%d-%b-%Y %H:%M", errors="coerce")
            if pd.isna(stamp):
                continue
            value = _num(raw.get("buyValue" if kind == "buy" else "sellValue")) or _num(raw.get("secVal"))
            rows.append((raw["symbol"], stamp.normalize(),
                         value if kind == "buy" else 0.0, value if kind == "sell" else 0.0,
                         int(kind == "buy"), int(kind == "sell")))
    cols = ["symbol", "bdate", "buy_val", "sell_val", "n_buy", "n_sell"]
    return pd.DataFrame(rows, columns=cols), fetched


def _delivery_columns(panel: pd.DataFrame, delivery: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """deliv_pct / deliv_rel20 / deliv_chg5 per (symbol, session), lagged one session."""
    wide = delivery.pivot_table(index="date", columns="symbol", values="deliv_pct", aggfunc="last")
    # Lag one session on the panel's own calendar: value from D is first visible at D+1.
    wide = wide.reindex(calendar).shift(1)
    prior_mean = wide.shift(1).rolling(20, min_periods=10).mean()
    rel20 = wide / prior_mean.replace(0.0, np.nan)
    chg5 = wide - wide.shift(5)
    out = []
    for name, frame in (("deliv_pct", wide), ("deliv_rel20", rel20), ("deliv_chg5", chg5)):
        long = frame.stack(future_stack=True).rename(name).reset_index()
        long.columns = ["date", "symbol", name]
        out.append(long.set_index(["date", "symbol"])[name])
    return pd.concat(out, axis=1).reset_index()


def _insider_columns(panel_keys: pd.DataFrame, insider: pd.DataFrame, fetched: set[str]) -> pd.DataFrame:
    """30-day insider net value and counts: broadcasts with t - 30d < B < t, per symbol."""
    window = pd.Timedelta(days=INSIDER_WINDOW_DAYS)
    out = panel_keys[["date", "symbol"]].copy()
    for col in ("ins_net_30d", "ins_buys_30d", "ins_sells_30d"):
        out[col] = np.nan
    # Fully ordered, so same-day rows sum in the same order whatever files were loaded
    # (float addition is not associative; an arbitrary tie order flips ~1e-8 residues).
    order = ["bdate", "buy_val", "sell_val", "n_buy", "n_sell"]
    by_symbol = {s: g.sort_values(order, kind="mergesort") for s, g in insider.groupby("symbol")}
    for symbol, idx in out.groupby("symbol").groups.items():
        if symbol not in fetched:
            continue  # no response at all stays missing
        t = out.loc[idx, "date"].to_numpy(dtype="datetime64[ns]")
        g = by_symbol.get(symbol)
        if g is None or g.empty:
            out.loc[idx, ["ins_net_30d", "ins_buys_30d", "ins_sells_30d"]] = 0.0
            continue
        b = g["bdate"].to_numpy(dtype="datetime64[ns]")
        cum_net = np.concatenate([[0.0], np.cumsum((g["buy_val"] - g["sell_val"]).to_numpy())])
        cum_buy = np.concatenate([[0], np.cumsum(g["n_buy"].to_numpy())])
        cum_sell = np.concatenate([[0], np.cumsum(g["n_sell"].to_numpy())])
        hi = np.searchsorted(b, t, side="left")                      # B < t
        lo = np.searchsorted(b, t - window.to_timedelta64(), side="right")  # B > t - 30d
        lo = np.minimum(lo, hi)
        # Rounded to 1e-4 rupees so offsetting buys and sells give exactly 0, not +/-1e-8.
        net = np.round(cum_net[hi] - cum_net[lo], 4)
        out.loc[idx, "ins_net_30d"] = np.sign(net) * np.log1p(np.abs(net))
        out.loc[idx, "ins_buys_30d"] = (cum_buy[hi] - cum_buy[lo]).astype(float)
        out.loc[idx, "ins_sells_30d"] = (cum_sell[hi] - cum_sell[lo]).astype(float)
    return out


def attach_flow(panel: pd.DataFrame, raw_dir: Path = RAW) -> pd.DataFrame:
    """Return `panel` with the six FLOW_FEATURES columns added (rows and order unchanged)."""
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["_base"] = data["symbol"].map(_base_symbol)
    calendar = pd.DatetimeIndex(sorted(data["date"].unique()))

    delivery = load_delivery(raw_dir)
    keys = data[["date", "_base"]].rename(columns={"_base": "symbol"})
    if delivery.empty:
        for col in FLOW_FEATURES[:3]:
            data[col] = np.nan
    else:
        dcols = _delivery_columns(data, delivery, calendar)
        merged = keys.merge(dcols, on=["date", "symbol"], how="left")
        for col in FLOW_FEATURES[:3]:
            data[col] = merged[col].to_numpy()

    insider, fetched = load_insider(raw_dir)
    icols = _insider_columns(keys.drop_duplicates(), insider, fetched)
    merged = keys.merge(icols, on=["date", "symbol"], how="left")
    for col in FLOW_FEATURES[3:]:
        data[col] = merged[col].to_numpy()
    return data.drop(columns="_base")

"""Write the features actually used by each fold's models.

Reads `outputs/replay/checkpoints/manifest.json` (written by
`tools/run_with_checkpoints.py`) and the candidate list in `core/forecast/ranking.py`,
and writes `data/features_by_fold.csv` (one row per pooled model: fold, test window,
n_features, 0/1 flag per candidate) and `docs/FEATURES.md`.

Usage:
    python tools/export_feature_list.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.forecast import ranking  # noqa: E402

GROUPS = {
    "Returns": ["logret_1", "logret_5", "logret_20", "logret_60", "gap_pct",
                "intraday_range", "close_to_high"],
    "Momentum / reversal / 52-week position": [
        "mom_12_1", "mom_6m", "mom_3m", "reversal_1m", "dist_52w_high",
        "dist_52w_low", "drawdown_dx"],
    "Volatility / risk": ["rv_5", "rv_20", "rv_60", "atr14", "vol_ratio_20", "vol_z20",
                          "beta_60", "idio_vol_60", "lowvol_rank"],
    "Oscillators / trend indicators": ["rsi14", "macd", "macd_signal", "macd_hist",
                                       "bb_pct", "stoch_k", "stoch_d", "vwap20_dist"],
    "Cross-section / market context (computed from the same price panel)": [
        "cs_market_excess", "cs_sector_excess", "cs_sector_rank", "market_breadth_up",
        "market_dispersion", "market_mean_return", "sector_mean_return"],
    "Macro series": ["macro_india_vix", "macro_usdinr", "macro_brent", "macro_india_10y"],
    "Calendar": ["dow", "dom", "month"],
}


def main() -> int:
    manifest = json.loads(
        (ROOT / "outputs/replay/checkpoints/manifest.json").read_text(encoding="utf-8"),
    )
    pooled = [m for m in manifest["models"] if m["model"] == "pooled"]
    pooled.sort(key=lambda m: m["fold"])
    candidates = list(ranking.PRICE_RANK_FEATURES)

    ic_fold = pd.read_csv(ROOT / "outputs/replay/ic_by_fold.csv")
    windows = (ic_fold.groupby("fold")[["test_start", "test_end"]].first())

    rows = []
    counts: collections.Counter = collections.Counter()
    for m in pooled:
        used = set(m["feature_cols"])
        counts.update(used)
        row = {"fold": m["fold"],
               "test_start": windows.loc[m["fold"], "test_start"],
               "test_end": windows.loc[m["fold"], "test_end"],
               "train_rows": m["train_rows"], "n_features": m["n_features"]}
        row.update({c: int(c in used) for c in candidates})
        rows.append(row)
    table = pd.DataFrame(rows)
    (ROOT / "data").mkdir(exist_ok=True)
    table.to_csv(ROOT / "data/features_by_fold.csv", index=False)

    n_folds = len(pooled)
    in_all = [c for c in candidates if counts[c] == n_folds]
    never = [c for c in candidates if counts[c] == 0]
    lines = [
        "# Features used by the models",
        "",
        "The ranker is the `price_only` set in `core/forecast/ranking.py` "
        f"({len(candidates)} candidate columns). It uses price-derived and "
        "market-context columns only: no fundamentals, filings, sentiment or graph features.",
        "",
        "Within each fold, using only that fold's (purged) training rows, a candidate is "
        "dropped if it is non-missing on fewer than 60% of rows, is constant, or is flagged "
        "\"high\" severity by the drift monitor (`ranking.stable_feature_columns`, "
        "`monitor.drift_report`). The pooled model and the three regime experts of a fold "
        "share the resulting list.",
        "",
        f"* Features kept per fold: min {int(table['n_features'].min())}, "
        f"max {int(table['n_features'].max())}, median {int(table['n_features'].median())} "
        f"(`data/features_by_fold.csv`, one row per fold, 0/1 per candidate).",
        f"* Kept in all {n_folds} folds ({len(in_all)}): " + ", ".join(f"`{c}`" for c in in_all) + ".",
        f"* Never kept ({len(never)}): " + (", ".join(f"`{c}`" for c in never) or "none") + ".",
        "* Sector metadata is missing for this universe (a single sector value), so the "
        "`cs_sector_*` and `sector_mean_return` columns carry no sector information.",
        "",
        "## Candidates by group, with the number of folds (of "
        f"{n_folds}) in which each was kept",
        "",
    ]
    seen = set()
    for group, cols in GROUPS.items():
        present = [c for c in cols if c in candidates]
        if not present:
            continue
        seen.update(present)
        lines.append(f"**{group}**")
        lines.append("")
        lines.append("| feature | folds kept |")
        lines.append("|---|---|")
        for c in present:
            lines.append(f"| `{c}` | {counts[c]} |")
        lines.append("")
    leftover = [c for c in candidates if c not in seen]
    if leftover:
        lines.append("**Other**")
        lines.append("")
        lines.append("| feature | folds kept |")
        lines.append("|---|---|")
        for c in leftover:
            lines.append(f"| `{c}` | {counts[c]} |")
        lines.append("")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/FEATURES.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"candidates={len(candidates)} per-fold min/max="
          f"{table['n_features'].min()}/{table['n_features'].max()} "
          f"in_all={len(in_all)} never={len(never)}")
    print("never:", never)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

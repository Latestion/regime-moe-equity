"""Reproduce the feature-selection leakage that the paper discloses (Section 3.4).

The defect: `ranking.stable_feature_columns` (which drops "high drift" features, judged on the
last 60 days of whatever frame it receives) was computed once on the whole panel, so a fold's
feature set could depend on data after its test window. The fix computes it per fold from that
fold's purged training rows.

This script runs the pooled model through the same 25 folds twice:
  leaky  - feature list computed once on the entire panel (the defect)
  fixed  - feature list computed per fold from training rows (the pipeline used in the paper)
and reports, per regime, the mean daily rank IC and the independence-based t-statistic.
The `fixed` run must reproduce the pooled baseline in data/regime_mixture_result.json
(tolerance 1e-6) or the script exits non-zero.

Usage:
    python tools/run_leakage_check.py [--out outputs/leakage]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import regenerate_mixture_ic_series as rep  # noqa: E402
import robust_inference as ri  # noqa: E402
from core.forecast import features, ranking, regime  # noqa: E402


def pooled_ic_series(panel: pd.DataFrame, base_config: ranking.RankingConfig, *, leaky: bool,
                     train_years: int = 3, test_months: int = 6) -> pd.DataFrame:
    data = panel.copy()
    if "regime" not in data.columns:
        data = regime.attach_to_panel(data)
    data["date"] = pd.to_datetime(data["date"])
    start, end = data["date"].min(), data["date"].max()
    fold_start = start + pd.DateOffset(years=train_years)
    global_cols = (ranking.stable_feature_columns(data, ablation=base_config.ablation)
                   if leaky else None)
    rows, fold = [], 0
    while fold_start + pd.DateOffset(months=test_months) <= end:
        test_end = fold_start + pd.DateOffset(months=test_months)
        train = ranking._purged_before(data, fold_start)
        test = data[(data["date"] >= fold_start) & (data["date"] < test_end)]
        if train.empty or test.empty:
            fold_start = test_end
            continue
        fold += 1
        cols = global_cols if leaky else ranking.stable_feature_columns(
            train, ablation=base_config.ablation)
        art = ranking.train_ranker(train, base_config, feature_cols=cols)
        scored = ranking.score(art, test)
        for label in regime.REGIMES:
            ic = rep._ic_frame(scored, label)
            if len(ic):
                rows.append(pd.DataFrame({"date": pd.to_datetime(pd.Series(list(ic.index))),
                                          "regime": label, "fold": fold, "ic": ic.to_numpy()}))
        fold_start = test_end
    return pd.concat(rows, ignore_index=True)


def summarise(series: pd.DataFrame) -> dict:
    out = {}
    for label in ("trend", "chop", "meanrev"):
        x = series[series["regime"] == label].sort_values("date")["ic"].to_numpy(float)
        out[label] = {"n_dates": int(len(x)), "mean_ic": float(x.mean()),
                      "t_independence": ri.independence_t(x)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/leakage")
    args = ap.parse_args()
    published = json.loads(rep.PUBLISHED.read_text(encoding="utf-8"))
    base = ranking.RankingConfig(**published["config"]["base_config"])
    panel = features.load_features(start="2010-01-01")
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {}
    for mode in ("fixed", "leaky"):
        t0 = time.time()
        series = pooled_ic_series(panel, base, leaky=(mode == "leaky"))
        report[mode] = summarise(series)
        print(f"{mode}: " + ", ".join(
            f"{r}: IC={v['mean_ic']:+.4f} t={v['t_independence']:+.2f}"
            for r, v in report[mode].items()) + f"  ({time.time() - t0:.0f}s)", flush=True)

    bad = []
    for label, stats in published["pooled_baseline"]["regimes"].items():
        got = report["fixed"][label]
        if abs(got["mean_ic"] - stats["mean_ic"]) > 1e-6 or abs(got["t_independence"] - stats["t_stat"]) > 1e-6:
            bad.append(label)
    report["gate_fixed_reproduces_published_pooled"] = not bad
    (out_dir / "leakage_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if bad:
        print("GATE FAILED for regimes:", bad)
        return 2
    print("gate passed: the fixed run reproduces the published pooled baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

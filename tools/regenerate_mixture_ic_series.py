"""Replay the regime-mixture walk-forward and persist its per-date / per-fold IC.

`data/regime_mixture_result.json` keeps only per-regime summary statistics, so the
series behind the paper's Stage 1 result cannot be plotted from it. This script
re-runs the identical fold loop as
`regime_mixture.nested_walk_forward_regime_mixture` (same purge, same base config,
same seed) and additionally writes out the per-date and per-fold rank-IC series.

It refuses to emit anything unless the reproduced aggregates match the published
JSON, so no figure can ever disagree with the tables already in the paper.

Usage:
    python tools/regenerate_mixture_ic_series.py [--out data/paper_figures]
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.forecast import (  # noqa: E402
    features,
    ranking,
    regime,
    regime_mixture,
    research_statistics,
)

PUBLISHED = ROOT / "data" / "regime_mixture_result.json"
TOLERANCE = 1e-6


def _ic_frame(scored: pd.DataFrame, regime_label: str) -> pd.Series:
    """Per-date rank IC for one regime's rows, or an empty series."""
    group = scored[scored["regime"].astype(str) == regime_label].dropna(
        subset=["score", "xs_excess_5d"],
    )
    if group.empty:
        return pd.Series(dtype=float)
    try:
        return research_statistics.rank_ic_series(
            group["score"], group["xs_excess_5d"], group["date"],
        )
    except ValueError:
        return pd.Series(dtype=float)


def replay(panel: pd.DataFrame, *, config: regime_mixture.RegimeMixtureConfig,
           train_years: int = 3, test_months: int = 6) -> dict:
    """Mirror of `nested_walk_forward_regime_mixture`, keeping the scored frames."""
    base_config = ranking.RankingConfig(**config.base_config)
    data = panel.copy()
    if "regime" not in data.columns:
        data = regime.attach_to_panel(data)
    data["date"] = pd.to_datetime(data["date"])
    start, end = data["date"].min(), data["date"].max()
    fold_start = start + pd.DateOffset(years=train_years)

    folds: list[dict] = []
    pooled_frames: list[pd.DataFrame] = []
    mixture_frames: list[pd.DataFrame] = []
    per_date_rows: list[pd.DataFrame] = []
    per_fold_rows: list[dict] = []

    fold_index = 0
    while fold_start + pd.DateOffset(months=test_months) <= end:
        test_end = fold_start + pd.DateOffset(months=test_months)
        train = ranking._purged_before(data, fold_start)
        test = data[(data["date"] >= fold_start) & (data["date"] < test_end)]
        if train.empty or test.empty:
            fold_start = test_end
            continue

        fold_index += 1
        t0 = time.time()
        stable_cols = ranking.stable_feature_columns(train, ablation=base_config.ablation)
        pooled_artifact = ranking.train_ranker(train, base_config, feature_cols=stable_cols)
        pooled_scored = ranking.score(pooled_artifact, test)
        pooled_frames.append(pooled_scored)

        specialist_coverage: dict[str, dict] = {}
        specialists: dict[str, ranking.RankingArtifact] = {}
        for regime_label in regime.REGIMES:
            regime_train = train[train["regime"] == regime_label]
            specialist_coverage[regime_label] = {"train_rows": int(len(regime_train))}
            if len(regime_train) >= config.min_regime_train_rows:
                specialists[regime_label] = ranking.train_ranker(
                    regime_train, base_config, feature_cols=stable_cols,
                )
                specialist_coverage[regime_label]["used_specialist"] = True
            else:
                specialist_coverage[regime_label]["used_specialist"] = False

        mixture_scored = regime_mixture._score_regime_routed(
            pooled_artifact, specialists, test,
        )
        mixture_frames.append(mixture_scored)

        # Per-date and per-fold IC, paired on the dates both models scored.
        for regime_label in regime.REGIMES:
            pooled_ic = _ic_frame(pooled_scored, regime_label)
            mixture_ic = _ic_frame(mixture_scored, regime_label)
            common = pooled_ic.index.intersection(mixture_ic.index)
            if len(common) == 0:
                continue
            per_date_rows.append(pd.DataFrame({
                "date": pd.to_datetime(pd.Series(list(common))),
                "regime": regime_label,
                "fold": fold_index,
                "pooled_ic": pooled_ic.loc[common].to_numpy(),
                "mixture_ic": mixture_ic.loc[common].to_numpy(),
            }))
            per_fold_rows.append({
                "fold": fold_index,
                "regime": regime_label,
                "test_start": str(fold_start.date()),
                "test_end": str(test_end.date()),
                "n_dates": int(len(common)),
                "pooled_mean_ic": float(pooled_ic.loc[common].mean()),
                "mixture_mean_ic": float(mixture_ic.loc[common].mean()),
                "mean_ic_diff": float(
                    (mixture_ic.loc[common] - pooled_ic.loc[common]).mean()
                ),
                "train_rows": specialist_coverage[regime_label]["train_rows"],
                "used_specialist": specialist_coverage[regime_label]["used_specialist"],
            })

        folds.append({
            "train_end": str(pd.to_datetime(train["date"]).max().date()),
            "test_start": str(fold_start.date()),
            "test_end": str(test_end.date()),
            "specialist_coverage": specialist_coverage,
        })
        print(
            f"  fold {fold_index:2d}  {fold_start.date()} -> {test_end.date()}  "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        fold_start = test_end

    pooled_all = pd.concat(pooled_frames, ignore_index=True)
    mixture_all = pd.concat(mixture_frames, ignore_index=True)
    return {
        "folds": folds,
        "pooled_baseline": regime_mixture._regime_ic_report(pooled_all),
        "mixture": regime_mixture._regime_ic_report(mixture_all),
        "paired_comparison": regime_mixture._paired_ic_comparison(pooled_all, mixture_all),
        "per_date": pd.concat(per_date_rows, ignore_index=True),
        "per_fold": pd.DataFrame(per_fold_rows),
    }


def _compare(published: dict, reproduced: dict) -> tuple[list[dict], list[dict]]:
    """Field-by-field numeric comparison of the two result payloads."""
    checks: list[dict] = []
    failures: list[dict] = []

    def check(path: str, want, got) -> None:
        if isinstance(want, bool) or not isinstance(want, (int, float)):
            ok = want == got
            delta = None
        else:
            ok = got is not None and abs(float(want) - float(got)) <= TOLERANCE
            delta = None if got is None else abs(float(want) - float(got))
        record = {"field": path, "published": want, "reproduced": got, "match": ok}
        if delta is not None:
            record["abs_diff"] = delta
        checks.append(record)
        if not ok:
            failures.append(record)

    for block in ("pooled_baseline", "mixture"):
        for regime_label, stats in published[block]["regimes"].items():
            got_stats = (reproduced[block] or {}).get("regimes", {}).get(regime_label, {})
            for key in ("n_obs", "mean_ic", "ic_std", "t_stat", "p_value", "significant"):
                check(f"{block}.{regime_label}.{key}", stats[key], got_stats.get(key))

    for regime_label, stats in published["paired_comparison"].items():
        got_stats = reproduced["paired_comparison"].get(regime_label, {})
        for key in ("n_common_dates", "mean_ic_diff", "paired_t_stat",
                    "pooled_mean_ic", "mixture_mean_ic"):
            check(f"paired.{regime_label}.{key}", stats[key], got_stats.get(key))

    return checks, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/paper_figures")
    parser.add_argument("--start", default="2010-01-01")
    args = parser.parse_args()

    published = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    config = regime_mixture.RegimeMixtureConfig(
        base_config=dict(published["config"]["base_config"]),
        min_regime_train_rows=int(published["config"]["min_regime_train_rows"]),
    )

    print(f"loading feature panel from {args.start} ...", flush=True)
    panel = features.load_features(start=args.start)
    if panel.empty:
        raise RuntimeError("feature panel is empty")
    print(f"panel: {len(panel):,} rows, "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}", flush=True)

    print("replaying walk-forward folds ...", flush=True)
    result = replay(panel, config=config)

    checks, failures = _compare(published, result)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "published_source": str(PUBLISHED.relative_to(ROOT)).replace("\\", "/"),
        "tolerance": TOLERANCE,
        "n_checks": len(checks),
        "n_failures": len(failures),
        "reproduced": len(failures) == 0,
        "n_folds_published": len(published["folds"]),
        "n_folds_reproduced": len(result["folds"]),
        "checks": checks,
    }
    (out_dir / "reproduction_check.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )

    if failures or len(result["folds"]) != len(published["folds"]):
        print(f"\nREPRODUCTION FAILED: {len(failures)} mismatched field(s), "
              f"{len(result['folds'])} folds vs {len(published['folds'])} published.",
              flush=True)
        for record in failures[:12]:
            print(f"  {record['field']}: published={record['published']!r} "
                  f"reproduced={record['reproduced']!r}", flush=True)
        print("No series written. Figures must not disagree with the paper's tables.",
              flush=True)
        return 2

    result["per_date"].sort_values(["regime", "date"]).to_csv(
        out_dir / "ic_by_date.csv", index=False, date_format="%Y-%m-%d",
    )
    result["per_fold"].sort_values(["regime", "fold"]).to_csv(
        out_dir / "ic_by_fold.csv", index=False,
    )
    print(f"\nreproduced all {len(checks)} published statistics within {TOLERANCE:g}.")
    print(f"wrote {out_dir / 'ic_by_date.csv'} "
          f"({len(result['per_date']):,} rows)")
    print(f"wrote {out_dir / 'ic_by_fold.csv'} ({len(result['per_fold'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

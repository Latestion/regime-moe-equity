"""Regime-threshold sensitivity (paper Section 3.2 / Fig. 4).

The three regime constants (vol_ratio 1.5, rv_20 0.50, adx_proxy 1.0) are scaled
jointly by each factor in --scales, regimes are re-assigned from the panel with
`regime.attach_to_panel(thresholds=...)`, and the full 25-fold walk-forward replay
(`regenerate_mixture_ic_series.replay`, identical folds, features, learner, purge,
seed) is run again. For every scale and regime the paired mixture-minus-pooled
IC difference is summarised with `tools/robust_inference.py`
(Newey-West L=5, stationary bootstrap, seed 42).

The scale 1.0 run is a gate: it must reproduce every statistic in
`data/regime_mixture_result.json` within 1e-6 or the script exits non-zero.

Outputs under outputs/sensitivity/:
  scale_<s>_ic_by_date.csv        per-date IC series for that scale
  threshold_sensitivity.json      per-scale regime shares, expert coverage and paired inference

Usage:
    python tools/run_threshold_sensitivity.py [--scales 0.8 0.9 1.0 1.1 1.2]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import regenerate_mixture_ic_series as rep  # noqa: E402
import robust_inference as ri  # noqa: E402
from core.forecast import features, regime, regime_mixture  # noqa: E402

BASE = regime.DEFAULT_THRESHOLDS


def scaled(s: float) -> regime.Thresholds:
    return regime.Thresholds(
        chop_volratio=BASE.chop_volratio * s,
        chop_rv=BASE.chop_rv * s,
        trend_adx=BASE.trend_adx * s,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scales", nargs="*", type=float, default=[0.8, 0.9, 1.0, 1.1, 1.2])
    ap.add_argument("--out", default="outputs/sensitivity")
    ap.add_argument("--start", default="2010-01-01")
    args = ap.parse_args()

    published = json.loads(rep.PUBLISHED.read_text(encoding="utf-8"))
    config = regime_mixture.RegimeMixtureConfig(
        base_config=dict(published["config"]["base_config"]),
        min_regime_train_rows=int(published["config"]["min_regime_train_rows"]),
    )
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading feature panel ...", flush=True)
    panel = features.load_features(start=args.start)

    report: dict = {"scales": {}, "seed": 42, "gate": None}
    for s in args.scales:
        t0 = time.time()
        th = scaled(s)
        print(f"\n== scale {s}: {th}", flush=True)
        labelled = regime.attach_to_panel(panel, thresholds=th)
        shares = labelled["regime"].value_counts(normalize=True).round(4).to_dict()
        result = rep.replay(labelled, config=config)
        csv_path = out_dir / f"scale_{s:.1f}_ic_by_date.csv"
        result["per_date"].sort_values(["regime", "date"]).to_csv(
            csv_path, index=False, date_format="%Y-%m-%d")
        inference = ri.run(csv_path, lags=5, mean_block=10.0, n_boot=10_000, seed=42)
        coverage = [f["specialist_coverage"] for f in result["folds"]]
        report["scales"][f"{s:.1f}"] = {
            "thresholds": {"vol_ratio": th.chop_volratio, "rv_20": th.chop_rv, "adx": th.trend_adx},
            "regime_share_of_stock_days": shares,
            "folds_where_expert_not_used": {
                r: int(sum(1 for c in coverage if not c[r]["used_specialist"]))
                for r in regime.REGIMES},
            "paired_mixture_minus_pooled": {
                r: inference["regimes"][r]["paired_mixture_minus_pooled"]
                for r in ri.REGIMES},
            "pooled_mean_ic": {r: inference["regimes"][r]["pooled"]["mean_ic"] for r in ri.REGIMES},
            "mixture_mean_ic": {r: inference["regimes"][r]["mixture"]["mean_ic"] for r in ri.REGIMES},
            "folds_favouring_mixture": {
                r: inference["regimes"][r]["folds_favouring_mixture"] for r in ri.REGIMES},
        }
        if abs(s - 1.0) < 1e-12:
            _, failures = rep._compare(published, result)
            report["gate"] = {"scale": 1.0, "n_failures": len(failures), "passed": not failures}
            print(f"   gate (scale 1.0 vs published): "
                  f"{'PASSED' if not failures else 'FAILED'} ({len(failures)} mismatches)")
        print(f"   paired diff (mixture - pooled): " + ", ".join(
            f"{r}={report['scales'][f'{s:.1f}']['paired_mixture_minus_pooled'][r]['mean_ic']:+.4f}"
            for r in ri.REGIMES) + f"   ({time.time() - t0:.0f}s)", flush=True)
        (out_dir / "threshold_sensitivity.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")

    gate = report["gate"]
    if gate is not None and not gate["passed"]:
        print("GATE FAILED: scale 1.0 does not reproduce the published statistics.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

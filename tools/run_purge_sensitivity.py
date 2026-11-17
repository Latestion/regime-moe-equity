"""Purge-length sensitivity: repeat the 25-fold replay with a 10-session purge.

The main experiment purges the final 5 observed sessions before each test window (the
5-session label horizon, `ranking.RANK_LABEL_HORIZON`). This repeats the replay with a
10-session purge (labels unchanged) and reports the paired mixture-minus-pooled IC
difference with the same Newey-West / stationary-bootstrap inference.

Usage:
    python tools/run_purge_sensitivity.py [--purge 10] [--out outputs/purge10]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import regenerate_mixture_ic_series as rep  # noqa: E402
import robust_inference as ri  # noqa: E402
from core.forecast import features, ranking, regime_mixture  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--purge", type=int, default=10)
    ap.add_argument("--out", default="outputs/purge10")
    args = ap.parse_args()

    original = ranking._purged_before

    def purged(frame, boundary, *, horizon_sessions=args.purge):
        return original(frame, boundary, horizon_sessions=horizon_sessions)

    ranking._purged_before = purged

    published = json.loads(rep.PUBLISHED.read_text(encoding="utf-8"))
    config = regime_mixture.RegimeMixtureConfig(
        base_config=dict(published["config"]["base_config"]),
        min_regime_train_rows=int(published["config"]["min_regime_train_rows"]))
    panel = features.load_features(start="2010-01-01")
    result = rep.replay(panel, config=config)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ic_by_date.csv"
    result["per_date"].sort_values(["regime", "date"]).to_csv(csv_path, index=False, date_format="%Y-%m-%d")
    inference = ri.run(csv_path, lags=5, mean_block=10.0, n_boot=10_000, seed=42)
    (out_dir / "purge_sensitivity.json").write_text(
        json.dumps({"purge_sessions": args.purge, "label_horizon_sessions": ranking.RANK_LABEL_HORIZON,
                    "inference": inference}, indent=2) + "\n", encoding="utf-8")
    for r in ri.REGIMES:
        b = inference["regimes"][r]
        d = b["paired_mixture_minus_pooled"]
        print(f"{r:8s} pooled={b['pooled']['mean_ic']:+.4f} mixture={b['mixture']['mean_ic']:+.4f} "
              f"diff={d['mean_ic']:+.4f} t={d['t_newey_west']:+.2f} p={d['p_newey_west']:.4f} "
              f"CI=[{d['boot_ci95'][0]:+.4f}, {d['boot_ci95'][1]:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

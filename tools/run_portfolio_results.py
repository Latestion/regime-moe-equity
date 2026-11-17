"""Regenerate every portfolio result quoted in the paper.

Runs the walk-forward portfolio simulations from `core/forecast/regime_mixture.py`
and `core/forecast/selective.py` on the same 25 folds, base config and seed as the IC
replay, and writes one JSON per construction to `outputs/portfolio/`.

  portfolio            raw top-15 book, pooled vs mixture        -> regime_mixture_portfolio_result.json
  conditional          regime-conditional sizing (10 / 25 / 15)   -> regime_conditional_construction_result.json
  mixture_blend_w05    low-vol / mixture-forecast blend, weight .5 -> regime_mixture_lowvol_tilt_result.json
  mixture_blend_w03    same, weight .3                             -> regime_mixture_lowvol_tilt_w03_result.json
  pooled_blend_w05     low-vol / pooled-forecast blend, weight .5  (new)
  pooled_blend_w03     same, weight .3                             (new)
  lowvol_control       no-model low-volatility control (weight 0)  (new)

Where a committed JSON exists, the aggregate metrics are compared with it
(tolerance 1e-6); the new pooled-blend and control results are printed next to the
values quoted in `docs/FINDINGS_REGIME_MIXTURE_2026-09-04.md`.

Usage:
    python tools/run_portfolio_results.py [--only name ...] [--out outputs/portfolio]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.forecast import features, regime_mixture, selective  # noqa: E402

PUBLISHED = ROOT / "data" / "regime_mixture_result.json"
TOL = 1e-6
KEYS = ("n_days", "total_return", "cagr", "ann_vol", "sharpe", "sortino", "calmar",
        "max_drawdown", "win_rate", "hit_rate", "exposure")

# name -> (committed json to compare against or None, aggregate key path, quoted Sharpe or None)
CASES = {
    "portfolio": ("regime_mixture_portfolio_result.json", None, None),
    "conditional": ("regime_conditional_construction_result.json", "aggregate", 0.894),
    "mixture_blend_w05": ("regime_mixture_lowvol_tilt_result.json", "aggregate", 1.145),
    "mixture_blend_w03": ("regime_mixture_lowvol_tilt_w03_result.json", "aggregate", 1.471),
    "pooled_blend_w05": (None, "aggregate", 1.173),
    "pooled_blend_w03": (None, "aggregate", 1.504),
    "lowvol_control": (None, "aggregate", 1.515),
}


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
    except Exception:  # pragma: no cover
        pass
    return str(o)


def _compare(committed: dict, produced: dict, label: str) -> list[str]:
    bad = []
    for key in KEYS:
        if key in committed:
            a, b = committed[key], produced.get(key)
            if b is None or abs(float(a) - float(b)) > TOL:
                bad.append(f"{label}.{key}: committed={a} produced={b}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=list(CASES))
    ap.add_argument("--out", default="outputs/portfolio")
    ap.add_argument("--start", default="2010-01-01")
    args = ap.parse_args()

    published = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    base = dict(published["config"]["base_config"])
    mix_cfg = regime_mixture.RegimeMixtureConfig(
        base_config=base,
        min_regime_train_rows=int(published["config"]["min_regime_train_rows"]),
    )
    print(f"loading feature panel from {args.start} ...", flush=True)
    panel = features.load_features(start=args.start)
    print(f"panel: {len(panel):,} rows", flush=True)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    summary: dict[str, dict] = {}

    for name in args.only:
        if name not in CASES:
            raise SystemExit(f"unknown case {name!r}; choose from {list(CASES)}")
        t0 = time.time()
        print(f"\n== {name}", flush=True)
        if name == "portfolio":
            result = regime_mixture.portfolio_walk_forward_regime_mixture(panel, config=mix_cfg)
        elif name == "conditional":
            result = regime_mixture.nested_walk_forward_regime_conditional_construction(
                panel, config=mix_cfg)
        elif name.startswith("mixture_blend"):
            weight = 0.5 if name.endswith("w05") else 0.3
            result = regime_mixture.nested_walk_forward_lowvol_tilt_mixture(
                panel, config=mix_cfg, model_weight=weight, top_k=40)
        else:
            weight = {"pooled_blend_w05": 0.5, "pooled_blend_w03": 0.3, "lowvol_control": 0.0}[name]
            cfg = selective.LowVolTiltConfig(base_config=base, model_weight=weight, top_k=40)
            result = selective.nested_walk_forward_lowvol_tilt(panel, config=cfg)
        (out_dir / f"{name}.json").write_text(
            json.dumps(result, indent=2, default=_json_default) + "\n", encoding="utf-8")

        committed_name, agg_key, quoted = CASES[name]
        if name == "portfolio":
            produced = {"pooled": result["pooled_aggregate"], "mixture": result["mixture_aggregate"]}
            summary[name] = {k: v["sharpe"] for k, v in produced.items()}
        else:
            produced = {"aggregate": result[agg_key]}
            summary[name] = {"sharpe": result[agg_key]["sharpe"], "quoted_in_findings_note": quoted}
        if committed_name:
            committed = json.loads((ROOT / "data" / committed_name).read_text(encoding="utf-8"))
            if name == "portfolio":
                failures += _compare(committed["pooled_aggregate"], produced["pooled"], "pooled")
                failures += _compare(committed["mixture_aggregate"], produced["mixture"], "mixture")
            else:
                failures += _compare(committed[agg_key], produced["aggregate"], name)
        print(f"   {summary[name]}   ({time.time() - t0:.0f}s)", flush=True)

    (out_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "mismatches_vs_committed": failures,
                    "tolerance": TOL}, indent=2) + "\n", encoding="utf-8")
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    if failures:
        print("\nMISMATCHES against committed JSONs:")
        for f in failures:
            print("  ", f)
        return 2
    print("\nall committed portfolio results reproduced within 1e-6.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

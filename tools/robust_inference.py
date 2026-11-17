"""Serial-dependence-robust inference for the paper's Table 1.

Input: the per-date, per-regime daily rank-IC series written by
`tools/run_with_checkpoints.py` / `tools/regenerate_mixture_ic_series.py`
(columns: date, regime, fold, pooled_ic, mixture_ic).

For each regime and for each of {pooled, mixture, mixture - pooled}:
  * mean daily rank IC and number of dates T
  * independence t-statistic   mean / (sd / sqrt(T))
  * Newey-West t-statistic     mean / sqrt(V / T), with Bartlett weights
        V = g0 + 2 * sum_{l=1..L} (1 - l/(L+1)) * g_l,   g_l = lag-l autocovariance
        (divisor T for every autocovariance, no small-sample correction)
  * two-sided p-value from the standard normal, and the Bonferroni decision
    (alpha = 0.05 / 3 regimes)
  * 95% percentile interval from the stationary bootstrap (Politis-Romano):
        mean block length 10, 10,000 resamples, circular, fixed seed.
Every setting is recorded in the output so the table can be regenerated exactly.

Usage:
    python tools/robust_inference.py [--ic outputs/replay/ic_by_date.csv]
                                     [--out data/robustness] [--lags 5]
                                     [--block 10] [--boot 10000] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REGIMES = ("trend", "chop", "meanrev")
ALPHA = 0.05
BONFERRONI_ALPHA = ALPHA / 3.0


def independence_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / math.sqrt(n))) if sd > 0 else float("nan")


def newey_west_long_run_variance(x: np.ndarray, lags: int) -> float:
    """Bartlett-kernel long-run variance of the series (divisor T, no correction)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    e = x - x.mean()
    v = float(e @ e) / n
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        v += 2.0 * w * float(e[l:] @ e[:-l]) / n
    return v


def newey_west_t(x: np.ndarray, lags: int = 5) -> float:
    x = np.asarray(x, dtype=float)
    v = newey_west_long_run_variance(x, lags)
    return float(x.mean() / math.sqrt(v / len(x))) if v > 0 else float("nan")


def normal_two_sided_p(t: float) -> float:
    return float(2.0 * (1.0 - NormalDist().cdf(abs(t))))


def stationary_bootstrap_means(
    x: np.ndarray, *, mean_block: float = 10.0, n_boot: int = 10_000,
    seed: int = 42, chunk: int = 1000,
) -> np.ndarray:
    """Means of `n_boot` stationary-bootstrap resamples (circular, geometric blocks)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    p_new = 1.0 / float(mean_block)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = np.empty((b, n), dtype=np.int64)
        idx[:, 0] = rng.integers(0, n, size=b)
        restart = rng.random((b, n)) < p_new
        fresh = rng.integers(0, n, size=(b, n))
        for i in range(1, n):
            idx[:, i] = np.where(restart[:, i], fresh[:, i], (idx[:, i - 1] + 1) % n)
        means[done:done + b] = x[idx].mean(axis=1)
        done += b
    return means


def summarise(x: np.ndarray, *, lags: int, mean_block: float, n_boot: int,
              seed: int) -> dict:
    x = np.asarray(x, dtype=float)
    t_nw = newey_west_t(x, lags)
    p_nw = normal_two_sided_p(t_nw)
    t_ind = independence_t(x)
    boots = stationary_bootstrap_means(
        x, mean_block=mean_block, n_boot=n_boot, seed=seed,
    )
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n_dates": int(len(x)),
        "mean_ic": float(x.mean()),
        "t_independence": t_ind,
        "t_newey_west": t_nw,
        "p_newey_west": p_nw,
        "significant_bonferroni": bool(p_nw < BONFERRONI_ALPHA),
        "se_ratio_nw_over_independence": (
            float(t_ind / t_nw) if t_nw not in (0.0,) and not math.isnan(t_nw) else None
        ),
        "boot_ci95": [float(lo), float(hi)],
    }


def run(ic_path: Path, *, lags: int, mean_block: float, n_boot: int, seed: int) -> dict:
    df = pd.read_csv(ic_path)
    df["date"] = pd.to_datetime(df["date"])
    out: dict = {
        "method": {
            "long_run_variance": "Newey-West, Bartlett weights 1 - l/(L+1), divisor T, no small-sample correction",
            "lags_L": lags,
            "p_value": "two-sided, standard normal",
            "bonferroni_alpha": BONFERRONI_ALPHA,
            "bootstrap": "stationary (Politis-Romano), circular, percentile 95% interval",
            "bootstrap_mean_block": mean_block,
            "bootstrap_resamples": n_boot,
            "bootstrap_seed": seed,
            "series": "daily cross-sectional rank IC over the stocks in each regime, pooled over all folds, sorted by date",
            "input": str(ic_path.relative_to(ROOT)).replace("\\", "/") if ic_path.is_relative_to(ROOT) else str(ic_path),
        },
        "regimes": {},
    }
    for regime in REGIMES:
        g = df[df["regime"] == regime].sort_values("date")
        pooled = g["pooled_ic"].to_numpy(float)
        mixture = g["mixture_ic"].to_numpy(float)
        diff = mixture - pooled
        fold_diff = (g.assign(d=diff).groupby("fold")["d"].mean())
        kw = dict(lags=lags, mean_block=mean_block, n_boot=n_boot, seed=seed)
        out["regimes"][regime] = {
            "pooled": summarise(pooled, **kw),
            "mixture": summarise(mixture, **kw),
            "paired_mixture_minus_pooled": summarise(diff, **kw),
            "folds_favouring_mixture": int((fold_diff > 0).sum()),
            "n_folds": int(len(fold_diff)),
        }
        # Sensitivity of the Newey-West t to the lag choice (no bootstrap).
        out["regimes"][regime]["lag_sensitivity_t"] = {
            str(L): {
                "pooled": newey_west_t(pooled, L),
                "mixture": newey_west_t(mixture, L),
                "paired": newey_west_t(diff, L),
            } for L in (0, 5, 10, 20)
        }
    return out


def _flat_rows(result: dict) -> pd.DataFrame:
    rows = []
    for regime, blocks in result["regimes"].items():
        for name in ("pooled", "mixture", "paired_mixture_minus_pooled"):
            s = blocks[name]
            rows.append({
                "regime": regime, "series": name, "n_dates": s["n_dates"],
                "mean_ic": s["mean_ic"], "t_independence": s["t_independence"],
                "t_newey_west": s["t_newey_west"], "p_newey_west": s["p_newey_west"],
                "significant_bonferroni": s["significant_bonferroni"],
                "ci95_low": s["boot_ci95"][0], "ci95_high": s["boot_ci95"][1],
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ic", default="outputs/replay/ic_by_date.csv")
    ap.add_argument("--out", default="data/robustness")
    ap.add_argument("--name", default="table1_inference")
    ap.add_argument("--lags", type=int, default=5)
    ap.add_argument("--block", type=float, default=10.0)
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ic_path = (ROOT / args.ic) if not Path(args.ic).is_absolute() else Path(args.ic)
    result = run(ic_path, lags=args.lags, mean_block=args.block,
                 n_boot=args.boot, seed=args.seed)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.name}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    table = _flat_rows(result)
    table.to_csv(out_dir / f"{args.name}.csv", index=False, float_format="%.6g")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {out_dir / (args.name + '.json')} and .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

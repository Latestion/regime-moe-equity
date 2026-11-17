"""Fig. 4: mean IC difference (mixture - pooled) by regime under joint threshold scaling
and under the HMM gate. Bars are 95% stationary-bootstrap intervals.

Reads outputs/sensitivity/threshold_sensitivity.json and outputs/hmm/hmm_gate_result.json
(written by tools/run_threshold_sensitivity.py and tools/run_hmm_gate.py) and writes
figures/fig_threshold_sensitivity.pdf (+ .png).

Usage:
    python tools/make_fig_sensitivity.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REGIMES = (("trend", "Trend"), ("chop", "Choppy"), ("meanrev", "Mean-reversion"))
ORANGE, INK, GRID = "#eb6834", "#0b0b0b", "#e1e0d9"


def main() -> int:
    sens = json.loads((ROOT / "outputs/sensitivity/threshold_sensitivity.json").read_text(encoding="utf-8"))
    hmm_path = ROOT / "outputs/hmm/hmm_gate_result.json"
    hmm = json.loads(hmm_path.read_text(encoding="utf-8")) if hmm_path.exists() else None

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": "#c3c2b7"})
    fig, axes = plt.subplots(1, 3, figsize=(4.6, 2.3), sharey=True)
    scales = sorted(sens["scales"], key=float)
    labels = scales + (["HMM"] if hmm else [])
    for ax, (key, title) in zip(axes, REGIMES):
        for i, s in enumerate(scales):
            d = sens["scales"][s]["paired_mixture_minus_pooled"][key]
            lo, hi = d["boot_ci95"]
            ax.bar(i, d["mean_ic"], color=ORANGE, width=0.65, zorder=2)
            ax.plot([i, i], [lo, hi], color=INK, lw=1.0, zorder=3)
        if hmm:
            d = hmm["inference"]["regimes"][key]["paired_mixture_minus_pooled"]
            lo, hi = d["boot_ci95"]
            i = len(scales)
            ax.plot([i], [d["mean_ic"]], marker="*", color=INK, ms=7, zorder=4, ls="none")
            ax.plot([i, i], [lo, hi], color=INK, lw=1.0, zorder=3)
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_title(title, fontsize=8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        ax.set_xlabel("Threshold scale", fontsize=7)
    axes[0].set_ylabel("Mean IC, mixture − pooled", fontsize=7)
    fig.tight_layout()
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "fig_threshold_sensitivity.pdf")
    fig.savefig(out / "fig_threshold_sensitivity.png", dpi=200)
    print("wrote", out / "fig_threshold_sensitivity.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

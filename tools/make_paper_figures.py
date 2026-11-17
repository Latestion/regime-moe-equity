"""Render the Stage 1 forecasting figures for the icSoftComp2026 submission.

Reads the verified per-date / per-fold IC series written by
`tools/regenerate_mixture_ic_series.py` and the published summary statistics in
`data/regime_mixture_result.json`, and writes three vector PDFs (plus PNG proofs)
into `UPDATED_LIT_REVIEW/template/figures/`.

Every regime and every fold is plotted as computed - the figures make the paper's
positive Stage 1 claims, they do not filter data to do it.

Usage:
    python tools/make_paper_figures.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SERIES_DIR = ROOT / "data" / "paper_figures"
FIG_DIR = ROOT / "UPDATED_LIT_REVIEW" / "template" / "figures"
PUBLISHED = ROOT / "data" / "regime_mixture_result.json"
CHECK = SERIES_DIR / "reproduction_check.json"

# Regimes in the paper's own table order, with the paper's display labels.
REGIME_ORDER = ("trend", "chop", "meanrev")
REGIME_LABEL = {"trend": "Trend", "chop": "Choppy", "meanrev": "Mean-reversion"}

# Bonferroni-corrected threshold used throughout the paper (alpha 0.05 / 3 regimes).
CORRECTED_ALPHA = 0.05 / 3

# Paired two-sided p-values as printed in the paper's Table 1, used only to check
# that what this script computes from the series is what the paper reports.
PAPER_PAIRED_P = {"trend": 0.0094, "chop": 1.5e-6, "meanrev": 0.90}

# Two series only: the pooled baseline and the proposed mixture. Fixed hue order
# from the validated categorical palette (blue slot 1, orange slot 2; adjacent-pair
# CVD dE 24.7 protan / 32.7 tritan, normal-vision dE 33.6 - all six checks pass).
# Greyscale print safety does not rest on hue: bars carry a hatch, lines carry a
# dash pattern, and every series is direct-labelled or titled.
C_POOLED = "#2a78d6"
C_MIXTURE = "#eb6834"
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

TEXT_WIDTH_IN = 4.6  # the llncs text block is 4.7in; leave a hair of margin.


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.2,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def _recessive(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)


def _save(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(SERIES_DIR / f"{stem}.png", dpi=220)
    plt.close(fig)
    print(f"wrote {FIG_DIR / f'{stem}.pdf'}  (proof: {SERIES_DIR / f'{stem}.png'})")


def _paired_stats(by_date: pd.DataFrame, regime: str) -> tuple[float, float]:
    """Paired t and two-sided p on (mixture IC - pooled IC) for one regime."""
    frame = by_date[by_date["regime"] == regime]
    diff = (frame["mixture_ic"] - frame["pooled_ic"]).to_numpy()
    n = len(diff)
    t_stat = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(n)))
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))
    return t_stat, p_value


def _p_label(p_value: float) -> str:
    if p_value >= 0.01:
        return f"$p={p_value:.2f}$"
    if p_value >= 1e-4:
        return f"$p={p_value:.4f}$"
    exponent = int(np.floor(np.log10(p_value)))
    mantissa = p_value / 10 ** exponent
    if round(mantissa, 1) >= 10.0:  # 9.97e-8 must print as 1.0e-7, not 10.0e-8
        mantissa, exponent = mantissa / 10.0, exponent + 1
    return f"$p{{=}}{mantissa:.1f}{{\\times}}10^{{{exponent}}}$"


# --------------------------------------------------------------------------
# Figure 1 - mean out-of-sample rank IC by regime, pooled vs mixture
# --------------------------------------------------------------------------
def figure_regime_ic(by_date: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.5))
    width = 0.33
    positions = np.arange(len(REGIME_ORDER), dtype=float)

    series = [
        ("pooled_ic", C_POOLED, "", "Pooled model", -0.5),
        ("mixture_ic", C_MIXTURE, "////", "Regime-specialist mixture", 0.5),
    ]
    for column, colour, hatch, label, side in series:
        means, errors = [], []
        for regime in REGIME_ORDER:
            values = by_date.loc[by_date["regime"] == regime, column].to_numpy()
            means.append(float(values.mean()))
            # 95% CI on the mean of the per-date IC series - the same daily
            # observations the paper's own t-tests are computed over.
            errors.append(
                1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
            )
        centres = positions + side * (width + 0.02)
        ax.bar(
            centres, means, width, color=colour, edgecolor=SURFACE, linewidth=0.9,
            hatch=hatch, label=label, zorder=3,
        )
        ax.errorbar(
            centres, means, yerr=errors, fmt="none", ecolor=INK, elinewidth=0.8,
            capsize=2.2, capthick=0.8, zorder=4,
        )
        for centre, mean, error in zip(centres, means, errors):
            sign = 1.0 if mean >= 0 else -1.0
            ax.annotate(
                f"{mean:+.4f}",
                (centre, mean + sign * (error + 0.0009)),
                ha="center", va="bottom" if mean >= 0 else "top",
                fontsize=6.5, color=INK, zorder=5,
            )

    ax.axhline(0, color=INK, linewidth=0.9, zorder=2)
    ax.set_xticks(positions)
    ax.set_xticklabels([REGIME_LABEL[regime] for regime in REGIME_ORDER])
    ax.set_ylabel("Mean out-of-sample rank IC")
    _recessive(ax)

    lower, upper = ax.get_ylim()
    band = upper + 0.30 * (upper - lower)
    ax.set_ylim(lower, band)
    for index, regime in enumerate(REGIME_ORDER):
        t_stat, p_value = _paired_stats(by_date, regime)
        significant = p_value < CORRECTED_ALPHA
        note = _p_label(p_value) + ("" if significant else " (tie)")
        ax.annotate(
            f"paired $t={t_stat:.2f}$\n{note}",
            (positions[index], upper + 0.03 * (upper - lower)),
            ha="center", va="bottom", fontsize=6.5,
            color=INK if significant else INK_MUTED,
        )

    ax.legend(
        loc="lower left", bbox_to_anchor=(-0.01, 1.005), ncol=2, frameon=False,
        handlelength=1.5, borderaxespad=0.0, columnspacing=1.4,
    )
    _save(fig, "fig_regime_ic")


# --------------------------------------------------------------------------
# Figure 2 - cumulative daily rank IC over the walk-forward period
# --------------------------------------------------------------------------
def figure_cumulative_ic(by_date: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.0), sharey=True)
    for ax, regime in zip(axes, REGIME_ORDER):
        frame = by_date[by_date["regime"] == regime].sort_values("date")
        dates = frame["date"].to_numpy()
        ax.plot(dates, frame["pooled_ic"].cumsum().to_numpy(), color=C_POOLED,
                linestyle="-", linewidth=1.1)
        ax.plot(dates, frame["mixture_ic"].cumsum().to_numpy(), color=C_MIXTURE,
                linestyle="--", dashes=(3.4, 1.6), linewidth=1.1)
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.set_title(REGIME_LABEL[regime], pad=3)
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        _recessive(ax)
    axes[0].set_ylabel("Cumulative daily rank IC")
    handles = [
        Line2D([], [], color=C_POOLED, linestyle="-", linewidth=1.1,
               label="Pooled model"),
        Line2D([], [], color=C_MIXTURE, linestyle="--", dashes=(3.4, 1.6),
               linewidth=1.1, label="Regime-specialist mixture"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.09),
               ncol=2, frameon=False, handlelength=2.2, columnspacing=1.8)
    fig.subplots_adjust(wspace=0.10)
    _save(fig, "fig_cumulative_ic")


# --------------------------------------------------------------------------
# Figure 3 - per-fold mixture-minus-pooled IC across the walk-forward folds
# --------------------------------------------------------------------------
def figure_fold_consistency(by_fold: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.15), sharey=True)
    for ax, regime in zip(axes, REGIME_ORDER):
        frame = by_fold[by_fold["regime"] == regime].sort_values("fold")
        diffs = frame["mean_ic_diff"].to_numpy()
        ax.bar(
            frame["fold"].to_numpy(), diffs, width=0.74,
            color=[C_MIXTURE if value >= 0 else C_POOLED for value in diffs],
            edgecolor=SURFACE, linewidth=0.35, zorder=3,
        )
        ax.axhline(0, color=INK, linewidth=0.9, zorder=2)
        ax.axhline(float(diffs.mean()), color=INK, linestyle=":", linewidth=0.9,
                   zorder=4)
        wins = int((diffs > 0).sum())
        ax.set_title(REGIME_LABEL[regime], pad=3)
        ax.annotate(
            f"{wins}/{len(diffs)} folds\nfavour mixture",
            (0.03, 0.98), xycoords="axes fraction", ha="left", va="top",
            fontsize=6.5, color=INK,
        )
        ax.set_xticks([1, 5, 10, 15, 20, 25])
        _recessive(ax)
    axes[0].set_ylabel("Mean IC, mixture $-$ pooled")
    axes[1].set_xlabel("Walk-forward fold")
    handles = [
        Patch(facecolor=C_MIXTURE, edgecolor="none", label="Mixture better"),
        Patch(facecolor=C_POOLED, edgecolor="none", label="Pooled better"),
        Line2D([], [], color=INK, linestyle=":", linewidth=0.9, label="Regime mean"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.17),
               ncol=3, frameon=False, handlelength=1.6, columnspacing=1.5)
    fig.subplots_adjust(wspace=0.14)
    _save(fig, "fig_fold_consistency")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["ic", "cumulative", "folds"], default=None)
    args = parser.parse_args()

    check = json.loads(CHECK.read_text(encoding="utf-8"))
    if not check.get("reproduced"):
        raise SystemExit(
            "reproduction_check.json reports a mismatch against the published "
            "results; refusing to plot numbers the paper's tables do not state."
        )
    published = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    by_date = pd.read_csv(SERIES_DIR / "ic_by_date.csv", parse_dates=["date"])
    by_fold = pd.read_csv(SERIES_DIR / "ic_by_fold.csv")

    # The figures must state the same numbers as the paper's tables.
    for regime in REGIME_ORDER:
        t_stat, p_value = _paired_stats(by_date, regime)
        want_t = float(published["paired_comparison"][regime]["paired_t_stat"])
        want_p = PAPER_PAIRED_P[regime]
        if abs(t_stat - want_t) > 5e-3 or abs(p_value - want_p) > 0.05 * want_p:
            raise SystemExit(
                f"{regime}: figure would print t={t_stat:.4f}, p={p_value:.3g} "
                f"against the paper's t={want_t:.4f}, p={want_p:.3g}"
            )
        n_dates = int((by_date["regime"] == regime).sum())
        want_n = int(published["paired_comparison"][regime]["n_common_dates"])
        if n_dates != want_n:
            raise SystemExit(
                f"{regime}: {n_dates} dates in series vs {want_n} published"
            )

    _style()
    if args.only in (None, "ic"):
        figure_regime_ic(by_date)
    if args.only in (None, "cumulative"):
        figure_cumulative_ic(by_date)
    if args.only in (None, "folds"):
        figure_fold_consistency(by_fold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

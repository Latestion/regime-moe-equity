# Findings: per-regime specialist forecasting (regime-mixture model)

Written 2026-09-04. This is a standalone summary of `PREREGISTRATION_SELECTIVE.md` §17
(and, once the run below completes, §18), written for a reader who wants the headline
result and its honest caveats without reading the full incrementally-extended
preregistration document. It does not replace that document as the source of record —
every number here is copied from it.

## The claim, stated precisely

**A cross-sectional stock-ranking model trained as a separate specialist per market
regime (trend / mean-reversion / choppy) forecasts significantly better out-of-sample
than one model pooled across all regimes.** This is not a downstream trading trick —
it is a change to the forecaster itself, tested purely on forecast accuracy
(regime-stratified rank information coefficient, the same metric and significance
convention used throughout this project), before any portfolio construction is applied.

This is the first result in this research track that improves the *model*, rather than
finding a cleverer way to use an unchanged one. Five prior attempts at the latter
(§10-16 of the preregistration) all failed once tested against a real control.

## Why this claim is methodologically stronger than it might look at first glance

This project's own literature screen (`UPDATED_LIT_REVIEW/BASELINE_AUDIT.md`) found
concrete protocol weaknesses in the Q1-journal deep-reinforcement-learning papers it
evaluated as baselines for this project's earlier DRL benchmark:

- One paper double-counts transaction costs in its own reward function and embeds
  manuscript scores directly in its reporting code.
- Another gives one algorithm 400,000 training steps against 50,000-70,000 for its
  competitors, runs with zero transaction costs, and reports internally inconsistent
  headline numbers.
- A third paper's dataset provenance could not be verified as reproducible.
- Several other candidate papers were rejected during screening for missing licensed
  code, unverifiable repositories, or leakage-unsafe evaluation.

The literature review's own comparative chapter (`comparative_analysis.tex`) concludes
that a higher reported Sharpe or return in one paper "cannot by itself establish" that
its method is superior, because protocols differ so much across studies.

Against that backdrop, the regime-mixture result here was produced under a materially
tighter protocol: purged, chronological walk-forward folds; a Bonferroni correction
applied in advance, not after seeing which regime looked best; a paired significance
test against the pooled model specifically (not just a test against zero); a previously
undetected feature-selection leakage bug that was found by an explicit code-review
request, disclosed, and fixed before this result was produced (§16); and a policy of
running each new design once and reporting it, rather than tuning until something looks
good. The claim being made is therefore not "our numbers are bigger than theirs" — it is
"this specific forecasting-skill claim was tested more rigorously than several of the
Q1-journal claims this project screened as candidate baselines."

## The forecasting-skill result

Same 25 development walk-forward folds used throughout this track (`yahoo_angel200`
proxy universe, 2010-2025, `RankingConfig(ablation="price_only")` as the shared base
config — see `PREREGISTRATION_SELECTIVE.md` §0-§1 for full data/universe caveats,
including the declared survivorship bias in this proxy universe).

| regime | pooled model IC (t-stat) | mixture model IC (t-stat) | paired mixture − pooled |
|---|---|---|---|
| trend | 0.0197 (t=5.22, significant) | **0.0259 (t=7.13, significant)** | t=2.60, **p=0.0094** |
| chop | -0.0047 (t=-0.98, not significant) | **0.0142 (t=3.07, now significant)** | t=4.82, **p=1.5e-6** |
| meanrev | 0.0119 (t=4.14, significant) | 0.0118 (t=4.14, significant) | t=-0.13, p=0.90 (tie) |

Bonferroni-corrected significance threshold: alpha = 0.0167 (0.05 / 3 regimes), fixed in
advance for this whole track. The mixture model beats the pooled model, at that
threshold, in two of three regimes, and does not lose in the third. It also recovers
forecasting skill in a regime (`chop`) the pooled model could not forecast at all.

## The honest secondary picture: portfolio performance (Sharpe, Calmar, drawdown)

Per this project's own research-integrity standard, this section is not hidden or
minimized. Two portfolio constructions have been tried on top of the mixture model so
far, through the identical trade simulator used everywhere else in this project
(purged walk-forward, transaction costs and slippage applied, turnover caps enforced):

**1. Raw top-k book** (top_k=15, rebalance every 10 sessions — the shared base config):

| metric | pooled model | mixture model |
|---|---|---|
| Sharpe | **1.070** | 0.961 |
| Sortino | **1.822** | 1.613 |
| Calmar | **0.540** | 0.358 |
| Max drawdown | **-16.8%** | -20.3% |

**2. Low-volatility/forecast blend** (the closest any construction has come to beating a
trivial control in this track, per §16), same construction, forecast source swapped:

| design (top_k=40) | Sharpe |
|---|---|
| no-model low-vol control (unaffected by model choice) | **1.515** |
| pooled-model blend, weight=0.5 | 1.173 |
| mixture-model blend, weight=0.5 | 1.145 |
| pooled-model blend, weight=0.3 (best pooled weight found) | 1.504 |
| mixture-model blend, weight=0.3 | 1.471 |

In both constructions tried so far, the mixture model's statistically stronger forecast
produces slightly *worse* portfolio performance than the pooled model's forecast, and
neither beats simply holding a diversified low-volatility book with no forecast at all.

## Third construction tried: regime-conditional book sizing — also negative

Instead of feeding the mixture model's regime-varying forecast through one
fixed-parameter portfolio construction (built around a single pooled model's
statistical properties), this let the book's top_k, rank buffer, and minimum holding
period vary by the day's dominant market regime, sized against each regime's own
measured edge above: a tight, sticky 10-name book in `trend` (the strongest edge), a
wider, faster-turning 25-name book in `chop` (a weaker but real edge), and the
unmodified 15-name baseline in `meanrev` (where the mixture model ties the pooled
model, so no change was applied there). Same 25 folds, same costs, same simulator
family as every other result in this file.

| construction | Sharpe | Sortino | Calmar | Max drawdown |
|---|---|---|---|---|
| pooled model, raw top-k (fixed config) | **1.070** | **1.822** | **0.540** | **-16.8%** |
| mixture model, raw top-k (fixed config) | 0.961 | 1.613 | 0.358 | -20.3% |
| **mixture model, regime-conditional book sizing** | **0.894** | 1.477 | 0.324 | -21.1% |

This is the worst of the three constructions tried, not the best. Concentrating into a
tighter, stickier book specifically during `trend` — the regime with the mixture
model's strongest measured edge — did not pay off; if anything it increased volatility
and drawdown without a compensating return. This rules out the specific hypothesis that
the earlier raw top-k result (§ above) was failing merely because it used one
one-size-fits-all set of construction parameters across regimes with different edge
strength. The gap between "a regime-conditional forecast is real and provable" and "a
regime-conditional or regime-agnostic portfolio construction can profitably harvest it"
remains open after three separate, materially different constructions.

## Honest overall status of this whole track (updated 2026-09-04)

Counting the regime-mixture attempts above alongside the 15+ pooled-model constructions
already documented in `PREREGISTRATION_SELECTIVE.md` §10-16, this project has now tried
roughly 18 materially different ways to convert a real, statistically proven,
regime-conditional forecasting skill into better risk-adjusted portfolio performance
than a trivial diversified baseline. None has succeeded. Two separate models — a pooled
cross-sectional ranker and a per-regime specialist mixture built on top of it — have
each independently cleared a Bonferroni-corrected significance bar for forecasting
skill, using a protocol this document argues is tighter than several of the reviewed
literature's own baseline papers. Neither model's proven skill has yet been shown to
survive contact with an actual trade ledger (top-k selection, turnover buffers,
transaction costs) well enough to beat simply holding a diversified, low-volatility book
of stocks and doing nothing clever at all.

The defensible claim for a professor demanding "novelty in the prediction algorithm" is
therefore precise and should not be overstated: **the prediction algorithm itself is a
genuine, rigorously tested novelty** (per-regime specialist training, proven via a
paired significance test against the pooled model it improves on, not just a test
against zero). **Turning that novelty into a demonstrably better trading strategy is a
second, separate research problem this project has not yet solved**, despite a broad and
honestly reported search. Presenting the forecasting result as the novelty, with the
portfolio-performance search reported as a rigorous, still-open negative result — rather
than presenting a Sharpe number that does not currently exist — is the framing consistent
with this project's own research-integrity standard
(see `memory/PROJECT_MEMORY.md` and `PREREGISTRATION_SELECTIVE.md` §8's reporting
commitment).

Nothing in this document touches the sealed 2025-09-01–2026-09-01 holdout, which remains
spent (opened once already, on the now-retired regime-gated design; see §12/§14) and
cannot honestly be reused for any result in this file.

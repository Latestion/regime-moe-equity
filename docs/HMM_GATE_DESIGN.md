# HMM regime gate: design (fixed before the first run)

Purpose: replace the hand-set rule of `core/forecast/regime.py` with a statistical
regime model and repeat the pooled-versus-mixture comparison, as a check that the
choppy-regime result does not depend on the hand-set constants (paper Section 3.2).

Everything below is decided before any result from this gate exists. The script is
`tools/run_hmm_gate.py`. Nothing is tuned after seeing results; if the run fails or
behaves badly it is reported as is.

## Model
* One Gaussian hidden Markov model per stock, 3 states, 1-dimensional observation:
  the 1-day log return (`logret_1`, in percent), `covariance_type="diag"`,
  `n_iter=100`, `tol=1e-3`, `random_state=42`, a single fit per stock per fold
  (no restarts, no model selection).
* A stock with fewer than 500 observations before a fold start gets no regime label for
  that fold (its rows are excluded from regime-specific training and testing, in the
  pooled model's scoring they are still scored). Fits that raise an error are counted and
  handled the same way.

## Timing (no look-ahead)
* Expanding window: for the fold whose test window starts on date F, each stock's model
  is fitted on that stock's returns strictly before F (the same data the fold's training
  rows come from; the 5-session purge is applied to the training rows exactly as in the
  main experiment).
* Labels use the forward-filtered state probabilities only, never the smoothed
  (forward-backward) or Viterbi path: the label at day t depends on observations up to t
  and on parameters fitted before F. This applies to training rows and test rows alike.
* The state of a day is the argmax of its filtered probability.

## Mapping states to the three regimes
1. The state with the highest fitted variance = `chop`.
2. Of the other two, the state with the larger absolute fitted mean return = `trend`;
   the remaining state = `meanrev`.

## What stays identical to the main experiment
Data, features and per-fold feature filtering, LightGBM LambdaRank settings, seed 42,
25 folds and windows, purge, expert-training floor (5,000 rows), percentile scoring,
IC definition, paired Newey-West (L=5) and stationary bootstrap (block 10, 10,000, seed 42)
from `tools/robust_inference.py`.

## Reported
Per regime: pooled and mixture mean IC, paired difference, Newey-West t and p,
95% bootstrap interval, folds favouring the mixture, share of stock-days in each regime,
number of fits that failed or had too little history.

## Outputs
`outputs/hmm/hmm_gate_result.json`, `outputs/hmm/ic_by_date.csv`, run log in `outputs/hmm/logs/`.

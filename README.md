# Regime-Conditional Mixture-of-Experts for Cross-Sectional Equity Prediction

Code, data, checkpoints and logs for the ICTCS 2026 paper (Paper ID 889).

A causal, rule-based gate assigns each stock-day to a trend, mean-reversion or
choppy regime. One LightGBM expert per regime is trained only on that regime's
rows and compared with one pooled LightGBM model trained on all rows, under the
same folds, features, hyperparameters and seed.

## Contents

| Path | Contents |
|---|---|
| `core/forecast/regime.py` | Regime rule (`vol_ratio > 1.5` or `rv_20 > 0.50` -> chop; `adx_proxy > 1.0` -> trend; else mean-reversion); `attach_to_panel(thresholds=...)` |
| `core/forecast/regime_mixture.py` | Pooled vs mixture walk-forward, paired IC comparison, the portfolio constructions (raw top-k, regime-conditional, low-vol blend) |
| `core/forecast/ranking.py`, `selective.py` | Features, purged split, LightGBM ranker, portfolio simulator, pooled low-vol blend |
| `tools/run_with_checkpoints.py` | Replays all 25 folds, saves every model, checks 51 statistics against `data/regime_mixture_result.json` (tolerance 1e-6) |
| `tools/robust_inference.py` | Newey-West and stationary-bootstrap inference behind Table 1 |
| `tools/run_portfolio_results.py` | Regenerates every portfolio result in the paper, including the pooled blends and the no-model control |
| `tools/run_threshold_sensitivity.py`, `tools/run_hmm_gate.py` | Regime-threshold scaling (0.8-1.2) and the HMM gate (design: `docs/HMM_GATE_DESIGN.md`) |
| `tools/run_purge_sensitivity.py` | Replay with a 10-session purge |
| `tools/run_leakage_check.py` | Reproduces the disclosed feature-selection leakage, leaky vs fixed |
| `tools/export_feature_list.py`, `docs/FEATURES.md`, `data/features_by_fold.csv` | Features used in every fold |
| `tools/make_paper_figures.py`, `tools/make_fig_sensitivity.py` | Figures |
| `data/features/` | Monthly feature panel, 2010-01 to 2025-08 (190 stocks, 3,866 sessions) |
| `data/regime_mixture_result.json`, `data/regime_*_result.json` | Published IC statistics and portfolio results (committed originals) |
| `data/research/nifty_dataset_manifest.json` | Dataset description and sha256 of the close series |
| `outputs/replay/` | 100 checkpoints (`checkpoints/`), `manifest.json`, run log, `reproduction_check.json`, IC series |
| `outputs/portfolio/` | Regenerated portfolio JSONs and `summary.json` (reproduce the committed ones within 1e-6) |
| `outputs/sensitivity/`, `outputs/hmm/`, `outputs/purge10/`, `outputs/leakage/` | Results and logs of the robustness runs |
| `data/robustness/table1_inference.{json,csv}` | Table 1 inference output |

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q tests                      # unit tests
python tools/run_with_checkpoints.py           # IC replay + checkpoints + log
python tools/robust_inference.py               # Table 1 inference
python tools/run_portfolio_results.py          # portfolio results (about 35 min on the reference machine)
python tools/run_threshold_sensitivity.py      # five full replays (about 36 min)
python tools/run_hmm_gate.py                   # HMM gate (about 20 min; needs hmmlearn)
python tools/run_purge_sensitivity.py          # 10-session purge replay (about 8 min)
python tools/run_leakage_check.py              # leaky vs fixed pooled model (about 9 min)
```

The replay exits non-zero if any statistic differs from the published JSON by more than 1e-6.

### Hardware, time and memory

Reference machine: Windows 11, Intel CPU with 20 logical cores, 31.8 GB RAM, Python 3.11.9,
numpy 2.2.6, pandas 2.2.2, scipy 1.14.1, LightGBM 4.6.0 (all threads).

| Run | Wall time | Peak memory | Result |
|---|---|---|---|
| `run_with_checkpoints.py`, 20 threads | about 8 minutes | not measured | 51/51 reproduced |
| same, `OMP_NUM_THREADS=4` | about 13.5 minutes | 7.1 GB | 51/51 reproduced, maximum difference 0 |

The 4-thread run peaked at 7.1 GB (panel plus per-fold model fits), so a machine with
8 GB of RAM may run out of memory in the later folds (training sets grow with each fold). 16 GB or more
is recommended. Fewer threads only change the run time here; results were identical with 4 and 20 threads.

## Inference method (Table 1)

Newey-West long-run variance with Bartlett weights `1 - l/(L+1)`, `L = 5`, divisor T, no small-sample
correction; two-sided p-values from the standard normal; Bonferroni threshold 0.05/3. Confidence
intervals: stationary bootstrap (Politis-Romano), mean block length 10, 10,000 circular resamples,
percentile 95% interval, seed 42. The series is the daily cross-sectional rank IC of the stocks in a
regime, pooled over all 25 folds and sorted by date. All settings are stored in
`data/robustness/table1_inference.json`.

## Checkpoints

Each file is a LightGBM model in text form, gzip-compressed. `manifest.json` lists, for every model, the
fold, regime, number of training rows, training date range, feature columns, ranking configuration and sha256.

```python
import gzip, lightgbm as lgb
booster = lgb.Booster(model_str=gzip.open("outputs/replay/checkpoints/fold_01_pooled.txt.gz", "rt").read())
```

## Implementation details as run

- Model: LightGBM LambdaRank (`ranking.train_ranker`). Relevance grade = floor(9 x cross-sectional percentile
  rank of the 5-session forward excess return), clipped to 0-9. 500 rounds, learning rate 0.03, 31 leaves,
  `min_data_in_leaf` 100, `feature_fraction` 0.8, bagging fraction 0.8 every 5 rounds, `lambda_l2` 1, seed 42.
  Experts use the same settings as the pooled model. IC is measured against the raw 5-session excess return.
- Features: the `price_only` set in `ranking.py` (45 candidate columns), after dropping columns that are mostly
  missing, constant, or flagged by the drift monitor on the fold's own training data; 16 to 39 remain per fold
  (`docs/FEATURES.md`).
- Purge: the final 5 observed sessions (`RANK_LABEL_HORIZON = 5`) before each test window are excluded from
  training. A 10-session purge is in `outputs/purge10/` (the choppy gain is unchanged).
- Folds: expanding window, 3 years of initial history, 6-month test windows, 25 folds, 2013-01-04 to 2025-07-04.
- Portfolio costs: 10 bps per unit one-way turnover plus 5 bps slippage. Raw top-k: 15 names, rebalance every
  10 sessions. Regime-conditional: top-k / rank buffer / minimum holding days = 10/8/15 (trend), 25/3/5 (choppy),
  15/5/10 (mean-reversion) by the day's modal regime; see `DEFAULT_REGIME_CONFIG_OVERRIDES`.
- Universe: 200 NSE names screened at a single snapshot date, 190 with enough history. Not point-in-time;
  survivorship bias applies (see the manifest). Sector metadata are missing, so sector caps are inert.

## Not included

Raw price downloads and any credentials. Data from 2025-09 onward is not included.

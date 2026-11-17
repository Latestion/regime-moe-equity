# Features used by the models

The ranker is the `price_only` set in `core/forecast/ranking.py` (45 candidate columns). It uses price-derived and market-context columns only: no fundamentals, filings, sentiment or graph features.

Within each fold, using only that fold's (purged) training rows, a candidate is dropped if it is non-missing on fewer than 60% of rows, is constant, or is flagged "high" severity by the drift monitor (`ranking.stable_feature_columns`, `monitor.drift_report`). The pooled model and the three regime experts of a fold share the resulting list.

* Features kept per fold: min 16, max 39, median 33 (`data/features_by_fold.csv`, one row per fold, 0/1 per candidate).
* Kept in all 25 folds (10): `logret_1`, `vol_z20`, `vol_ratio_20`, `close_to_high`, `cs_sector_excess`, `cs_sector_rank`, `cs_market_excess`, `dow`, `dom`, `lowvol_rank`.
* Never kept (5): `macro_india_vix`, `macro_usdinr`, `macro_brent`, `macro_india_10y`, `month`.
* Sector metadata is missing for this universe (a single sector value), so the `cs_sector_*` and `sector_mean_return` columns carry no sector information.

## Candidates by group, with the number of folds (of 25) in which each was kept

**Returns**

| feature | folds kept |
|---|---|
| `logret_1` | 25 |
| `logret_5` | 24 |
| `logret_20` | 20 |
| `logret_60` | 19 |
| `gap_pct` | 23 |
| `intraday_range` | 23 |
| `close_to_high` | 25 |

**Momentum / reversal / 52-week position**

| feature | folds kept |
|---|---|
| `mom_12_1` | 10 |
| `mom_6m` | 13 |
| `mom_3m` | 17 |
| `reversal_1m` | 20 |
| `dist_52w_high` | 16 |
| `dist_52w_low` | 15 |
| `drawdown_dx` | 17 |

**Volatility / risk**

| feature | folds kept |
|---|---|
| `rv_5` | 24 |
| `rv_20` | 22 |
| `rv_60` | 19 |
| `atr14` | 15 |
| `vol_ratio_20` | 25 |
| `vol_z20` | 25 |
| `beta_60` | 23 |
| `idio_vol_60` | 19 |
| `lowvol_rank` | 25 |

**Oscillators / trend indicators**

| feature | folds kept |
|---|---|
| `rsi14` | 21 |
| `macd` | 13 |
| `macd_signal` | 13 |
| `macd_hist` | 18 |
| `bb_pct` | 24 |
| `stoch_k` | 24 |
| `stoch_d` | 24 |
| `vwap20_dist` | 22 |

**Cross-section / market context (computed from the same price panel)**

| feature | folds kept |
|---|---|
| `cs_market_excess` | 25 |
| `cs_sector_excess` | 25 |
| `cs_sector_rank` | 25 |
| `market_breadth_up` | 16 |
| `market_dispersion` | 7 |
| `market_mean_return` | 11 |
| `sector_mean_return` | 11 |

**Macro series**

| feature | folds kept |
|---|---|
| `macro_india_vix` | 0 |
| `macro_usdinr` | 0 |
| `macro_brent` | 0 |
| `macro_india_10y` | 0 |

**Calendar**

| feature | folds kept |
|---|---|
| `dow` | 25 |
| `dom` | 25 |
| `month` | 0 |

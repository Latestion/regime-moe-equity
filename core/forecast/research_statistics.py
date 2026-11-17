"""Multiple-testing-aware statistics for trading-strategy research.

The functions in this module operate on already causal, synchronous, net-of-
cost return streams. They do not repair look-ahead, survivorship, bad execution
assumptions, or an incomplete experiment history.

References
----------
Bailey & Lopez de Prado, "The Sharpe Ratio Efficient Frontier" (PSR),
"The Deflated Sharpe Ratio" (DSR), and Bailey et al., "The Probability of
Backtest Overfitting" (CSCV/PBO).
"""
from __future__ import annotations

import itertools
import math
from statistics import NormalDist
from typing import Iterable

import numpy as np
import pandas as pd


_NORMAL = NormalDist()
_EULER_MASCHERONI = 0.5772156649015329


def _clean_returns(values: Iterable[float]) -> np.ndarray:
    returns = np.asarray(list(values), dtype=float)
    if returns.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if len(returns) < 4:
        raise ValueError("at least four return observations are required")
    if not np.isfinite(returns).all():
        raise ValueError("returns contain NaN or infinity")
    std = float(np.std(returns, ddof=1))
    scale = max(1.0, float(np.max(np.abs(returns))))
    if std <= np.finfo(float).eps * scale:
        raise ValueError("returns have zero variance")
    return returns


def _shape_stats(returns: np.ndarray) -> tuple[float, float]:
    series = pd.Series(returns, dtype=float)
    skew = float(series.skew())
    pearson_kurtosis = float(series.kurt() + 3.0)
    if not np.isfinite([skew, pearson_kurtosis]).all():
        raise ValueError("skewness or kurtosis is undefined")
    return skew, pearson_kurtosis


def probabilistic_sharpe_ratio(
    returns: Iterable[float],
    *,
    benchmark_sharpe_annual: float = 0.0,
    periods_per_year: int = 252,
    confidence: float = 0.95,
) -> dict:
    """Probability that the true Sharpe exceeds a declared benchmark.

    The Bailey-Lopez de Prado formula uses per-period Sharpe and Pearson
    kurtosis. Returned ``psr`` is a probability, not a corrected Sharpe ratio.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")
    values = _clean_returns(returns)
    n = len(values)
    period_sr = float(values.mean() / values.std(ddof=1))
    annual_sr = period_sr * math.sqrt(periods_per_year)
    benchmark_period = float(benchmark_sharpe_annual) / math.sqrt(
        periods_per_year,
    )
    skew, pearson_kurtosis = _shape_stats(values)
    denominator_sq = (
        1.0 - skew * period_sr
        + ((pearson_kurtosis - 1.0) / 4.0) * period_sr**2
    )
    if denominator_sq <= 0 or not math.isfinite(denominator_sq):
        raise ValueError("PSR denominator is non-positive")
    z = (
        (period_sr - benchmark_period) * math.sqrt(n - 1)
        / math.sqrt(denominator_sq)
    )
    psr = float(_NORMAL.cdf(z))
    min_track_record = None
    if period_sr > benchmark_period:
        min_track_record = float(
            1.0
            + denominator_sq
            * (
                _NORMAL.inv_cdf(confidence)
                / (period_sr - benchmark_period)
            ) ** 2
        )
    return {
        "observations": n,
        "periods_per_year": int(periods_per_year),
        "sharpe_period": period_sr,
        "sharpe_annual": annual_sr,
        "benchmark_sharpe_period": benchmark_period,
        "benchmark_sharpe_annual": float(benchmark_sharpe_annual),
        "skew": skew,
        "pearson_kurtosis": pearson_kurtosis,
        "psr_z": float(z),
        "psr": psr,
        "min_track_record_observations": min_track_record,
    }


def deflated_sharpe_ratio(
    selected_returns: Iterable[float],
    trial_sharpes_annual: Iterable[float],
    *,
    raw_trial_count: int | None = None,
    benchmark_sharpe_annual: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """Deflate the selected strategy's Sharpe for the research search size.

    ``trial_sharpes_annual`` must contain comparable trials evaluated on the
    same calendar, costs, and benchmark. ``raw_trial_count`` may be larger than
    the surviving valid list, but never smaller. With one trial DSR is
    undefined; ordinary PSR is returned and ``dsr`` is ``None``.
    """
    trial_sharpes = np.asarray(list(trial_sharpes_annual), dtype=float)
    if not len(trial_sharpes) or not np.isfinite(trial_sharpes).all():
        raise ValueError("trial Sharpes must be a non-empty finite sequence")
    raw_count = int(raw_trial_count or len(trial_sharpes))
    if raw_count < len(trial_sharpes) or raw_count <= 0:
        raise ValueError("raw_trial_count cannot be smaller than supplied trials")
    ordinary = probabilistic_sharpe_ratio(
        selected_returns,
        benchmark_sharpe_annual=benchmark_sharpe_annual,
        periods_per_year=periods_per_year,
    )
    if raw_count == 1:
        return {
            **ordinary,
            "raw_trial_count": 1,
            "trial_sharpe_std_period": 0.0,
            "expected_max_sharpe_period": None,
            "expected_max_sharpe_annual": None,
            "dsr": None,
        }

    period_trials = trial_sharpes / math.sqrt(periods_per_year)
    trial_std = float(
        np.std(period_trials, ddof=1) if len(period_trials) > 1 else 0.0
    )
    n = float(raw_count)
    expected_max_multiplier = (
        (1.0 - _EULER_MASCHERONI) * _NORMAL.inv_cdf(1.0 - 1.0 / n)
        + _EULER_MASCHERONI
        * _NORMAL.inv_cdf(1.0 - 1.0 / (n * math.e))
    )
    benchmark_period = float(benchmark_sharpe_annual) / math.sqrt(
        periods_per_year,
    )
    expected_max_period = benchmark_period + trial_std * expected_max_multiplier
    deflated = probabilistic_sharpe_ratio(
        selected_returns,
        benchmark_sharpe_annual=(
            expected_max_period * math.sqrt(periods_per_year)
        ),
        periods_per_year=periods_per_year,
    )
    return {
        **ordinary,
        "raw_trial_count": raw_count,
        "trial_sharpe_std_period": trial_std,
        "expected_max_sharpe_period": float(expected_max_period),
        "expected_max_sharpe_annual": float(
            expected_max_period * math.sqrt(periods_per_year)
        ),
        "dsr_z": deflated["psr_z"],
        "dsr": deflated["psr"],
    }


def rank_ic_series(
    scores: Iterable[float],
    forward_returns: Iterable[float],
    dates: Iterable,
) -> pd.Series:
    """Per-date Spearman rank IC between cross-sectional scores and returns.

    ``scores``, ``forward_returns`` and ``dates`` must be row-aligned
    asset-date observations (one row per asset per date). A date is dropped,
    not zeroed, if it has fewer than three names or zero score/return
    variance, since Spearman correlation is undefined there.
    """
    frame = pd.DataFrame({
        "score": np.asarray(list(scores), dtype=float),
        "forward_return": np.asarray(list(forward_returns), dtype=float),
        "date": list(dates),
    })
    if frame.empty:
        raise ValueError("scores/forward_returns/dates must be non-empty")
    if not np.isfinite(frame[["score", "forward_return"]].to_numpy()).all():
        raise ValueError("scores and forward_returns must be finite")

    def _daily_ic(group: pd.DataFrame) -> float:
        if len(group) < 3:
            return float("nan")
        if group["score"].nunique() < 2 or group["forward_return"].nunique() < 2:
            return float("nan")
        return float(group["score"].corr(group["forward_return"], method="spearman"))

    daily = frame.groupby("date", sort=True).apply(_daily_ic, include_groups=False)
    daily = daily.dropna()
    if daily.empty:
        raise ValueError("no date had enough names to compute a rank IC")
    daily.name = "rank_ic"
    return daily


def ic_significance(
    ic_values: Iterable[float],
    *,
    periods_per_year: int = 252,
) -> dict:
    """Two-sided t-test that a stream of per-date rank ICs has non-zero mean.

    Mirrors the PSR/DSR treatment of a return stream: the ICs are the
    observations, ``ic_information_ratio`` is the IC analogue of an
    annualized Sharpe ratio (mean IC over IC volatility, annualized by
    periods_per_year).
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    values = np.asarray(list(ic_values), dtype=float)
    if values.ndim != 1:
        raise ValueError("ic_values must be one-dimensional")
    n = len(values)
    if n < 2:
        raise ValueError("at least two IC observations are required")
    if not np.isfinite(values).all():
        raise ValueError("ic_values contain NaN or infinity")
    mean_ic = float(values.mean())
    ic_std = float(values.std(ddof=1))
    if ic_std <= 0:
        raise ValueError("ic_values have zero variance")
    t_stat = mean_ic / (ic_std / math.sqrt(n))
    p_value = float(2.0 * (1.0 - _NORMAL.cdf(abs(t_stat))))
    icir = mean_ic / ic_std * math.sqrt(periods_per_year)
    return {
        "n_obs": n,
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "t_stat": float(t_stat),
        "p_value": p_value,
        "ic_information_ratio": float(icir),
    }


def ic_significance_multi_regime(
    ic_by_regime: dict[str, Iterable[float]],
    *,
    alpha: float = 0.05,
    method: str = "bonferroni",
    periods_per_year: int = 252,
) -> dict:
    """Per-regime IC significance with multiple-testing correction.

    Each regime is tested independently at the raw per-regime alpha, then
    flagged significant against a corrected threshold. Bonferroni is used
    because each day belongs to exactly one regime (trend/meanrev/chop are
    mutually exclusive by construction in ``core.forecast.regime``), so the
    raw number of regimes tested is the correct trial count — not an
    inflated effective-trial count meant for correlated, overlapping trials.
    """
    if method != "bonferroni":
        raise ValueError(f"unsupported correction method: {method!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if not ic_by_regime:
        raise ValueError("ic_by_regime must be non-empty")
    n_regimes = len(ic_by_regime)
    corrected_alpha = alpha / n_regimes
    regimes: dict[str, dict] = {}
    significant_regimes: list[str] = []
    for regime, ic_values in ic_by_regime.items():
        stats = ic_significance(ic_values, periods_per_year=periods_per_year)
        is_significant = stats["p_value"] < corrected_alpha
        regimes[regime] = {**stats, "significant": is_significant}
        if is_significant:
            significant_regimes.append(regime)
    return {
        "method": method,
        "alpha": alpha,
        "n_regimes_tested": n_regimes,
        "corrected_alpha": corrected_alpha,
        "regimes": regimes,
        "significant_regimes": significant_regimes,
    }


def effective_trial_count(correlation: pd.DataFrame | np.ndarray) -> dict:
    """Correlation-adjusted trial count proposed in the DSR paper."""
    matrix = np.asarray(correlation, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("correlation matrix must be square")
    m = matrix.shape[0]
    if m == 0 or not np.isfinite(matrix).all():
        raise ValueError("correlation matrix must be finite and non-empty")
    if m == 1:
        return {"raw_trial_count": 1, "average_correlation": 1.0, "n_eff": 1.0}
    off_diagonal = matrix[~np.eye(m, dtype=bool)]
    average = float(off_diagonal.mean())
    estimate = average + (1.0 - average) * m
    estimate = float(min(float(m), max(1.0, estimate)))
    return {
        "raw_trial_count": m,
        "average_correlation": average,
        "n_eff": estimate,
    }


def _annual_sharpes(matrix: np.ndarray, periods_per_year: int) -> np.ndarray:
    means = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = means / std * math.sqrt(periods_per_year)
    return np.where(np.isfinite(result), result, -np.inf)


def cscv_probability_of_backtest_overfitting(
    return_matrix: pd.DataFrame | np.ndarray,
    *,
    blocks: int = 16,
    periods_per_year: int = 252,
    include_logits: bool = False,
) -> dict:
    """Estimate PBO with combinatorially symmetric cross-validation.

    Columns are precomputed causal candidate return streams on one common
    calendar. CSCV selects among streams; it does not retrain ML models.
    """
    values = np.asarray(return_matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("return_matrix needs at least two candidate columns")
    if not np.isfinite(values).all():
        raise ValueError("return_matrix must be synchronous and contain no NaN")
    observations, candidates = values.shape
    if blocks < 2 or blocks % 2:
        raise ValueError("blocks must be a positive even number")
    if observations % blocks:
        raise ValueError("observations must divide exactly into CSCV blocks")
    block_size = observations // blocks
    if block_size < 2:
        raise ValueError("each CSCV block needs at least two observations")
    if np.any(values.std(axis=0, ddof=1) <= 0):
        raise ValueError("zero-variance candidates must be excluded before CSCV")

    block_indices = [
        np.arange(i * block_size, (i + 1) * block_size) for i in range(blocks)
    ]
    logits: list[float] = []
    selected_is: list[float] = []
    selected_oos: list[float] = []
    oos_losses = 0
    for chosen in itertools.combinations(range(blocks), blocks // 2):
        chosen_set = set(chosen)
        is_rows = np.concatenate([block_indices[i] for i in chosen])
        oos_rows = np.concatenate([
            block_indices[i] for i in range(blocks) if i not in chosen_set
        ])
        is_scores = _annual_sharpes(values[is_rows], periods_per_year)
        oos_scores = _annual_sharpes(values[oos_rows], periods_per_year)
        if not np.isfinite(is_scores).any():
            raise ValueError("a CSCV in-sample split has no valid candidate")
        winner = int(np.argmax(is_scores))  # stable column-order tie-break
        ranks = pd.Series(oos_scores).rank(method="average").to_numpy()
        omega = float(ranks[winner] / (candidates + 1.0))
        logit = float(math.log(omega / (1.0 - omega)))
        logits.append(logit)
        selected_is.append(float(is_scores[winner]))
        selected_oos.append(float(oos_scores[winner]))
        oos_losses += int(oos_scores[winner] < 0)

    logit_array = np.asarray(logits)
    is_array = np.asarray(selected_is)
    oos_array = np.asarray(selected_oos)
    degradation_alpha = None
    degradation_beta = None
    finite_degradation = np.isfinite(is_array) & np.isfinite(oos_array)
    is_fit = is_array[finite_degradation]
    oos_fit = oos_array[finite_degradation]
    if len(is_fit) > 1 and float(is_fit.std(ddof=0)) > 0:
        degradation_beta, degradation_alpha = np.polyfit(
            is_fit, oos_fit, deg=1,
        )
    result = {
        "observations": observations,
        "candidates": candidates,
        "blocks": blocks,
        "block_size": block_size,
        "combination_count": len(logits),
        "pbo": float((logit_array <= 0).mean()),
        "oos_probability_of_loss": float(oos_losses / len(logits)),
        "degradation_alpha": (
            float(degradation_alpha) if degradation_alpha is not None else None
        ),
        "degradation_beta": (
            float(degradation_beta) if degradation_beta is not None else None
        ),
    }
    if include_logits:
        result["logits"] = logits
    return result

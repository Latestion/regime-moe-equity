import importlib.util
import math
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "robust_inference", Path(__file__).resolve().parents[1] / "tools" / "robust_inference.py",
)
ri = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ri)


def _naive_nw_t(x, lags):
    x = [float(v) for v in x]
    n = len(x)
    m = sum(x) / n
    e = [v - m for v in x]
    gamma = [sum(e[i] * e[i - l] for i in range(l, n)) / n for l in range(lags + 1)]
    v = gamma[0] + 2 * sum((1 - l / (lags + 1)) * gamma[l] for l in range(1, lags + 1))
    return m / math.sqrt(v / n)


def test_newey_west_matches_explicit_formula():
    rng = np.random.default_rng(1)
    x = rng.normal(size=300) + 0.1
    for lags in (0, 1, 5, 10):
        assert abs(ri.newey_west_t(x, lags) - _naive_nw_t(x, lags)) < 1e-12


def test_lag_zero_is_population_variance_t():
    rng = np.random.default_rng(2)
    x = rng.normal(size=200)
    t0 = ri.newey_west_t(x, 0)
    t_ind = ri.independence_t(x)
    # lag 0 uses divisor T, the independence t uses T-1
    assert abs(t0 / t_ind - math.sqrt(len(x) / (len(x) - 1))) < 1e-9


def test_positive_autocorrelation_lowers_t_by_a_nonconstant_factor():
    rng = np.random.default_rng(3)
    e = rng.normal(size=4000)
    x = np.empty_like(e)
    x[0] = e[0]
    for i in range(1, len(e)):
        x[i] = 0.7 * x[i - 1] + e[i]
    x = x + 0.05
    assert abs(ri.newey_west_t(x, 5)) < abs(ri.independence_t(x))
    iid = rng.normal(size=4000) + 0.05
    ratio_iid = ri.independence_t(iid) / ri.newey_west_t(iid, 5)
    ratio_ar = ri.independence_t(x) / ri.newey_west_t(x, 5)
    assert ratio_ar > ratio_iid + 0.5


def test_bootstrap_is_reproducible_and_centred():
    rng = np.random.default_rng(4)
    x = rng.normal(size=400) + 0.2
    a = ri.stationary_bootstrap_means(x, n_boot=300, seed=7)
    b = ri.stationary_bootstrap_means(x, n_boot=300, seed=7)
    assert np.array_equal(a, b)
    assert abs(a.mean() - x.mean()) < 0.03
    lo, hi = np.percentile(a, [2.5, 97.5])
    assert lo < x.mean() < hi


def test_normal_p_value():
    assert abs(ri.normal_two_sided_p(1.959964) - 0.05) < 1e-5

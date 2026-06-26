import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.markets import _pearson, _ols, _bh_significant


def test_pearson_perfect_correlation():
    r, p = _pearson([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
    assert r > 0.999
    assert p < 0.01


def test_ols_recovers_slope():
    beta, alpha, r2, se = _ols([0, 1, 2, 3, 4], [1, 3, 5, 7, 9])  # y = 2x + 1
    assert abs(beta - 2.0) < 1e-6
    assert abs(alpha - 1.0) < 1e-6
    assert r2 > 0.999


def test_bh_selects_only_strong_pvalues():
    passed = _bh_significant([0.001, 0.4, 0.6, 0.8], 0.10)
    assert 0 in passed
    assert 1 not in passed


# ---- P5: correct the best-of-N-lags search (METHODOLOGY_REVIEW F10) ----

def test_sidak_inflates_pvalue_for_lag_search():
    from pipeline.markets import _sidak
    assert _sidak(0.04, 1) == 0.04                  # no search, no correction
    assert _sidak(0.04, 7) > 0.04                   # 7 lags searched -> inflated
    assert _sidak(0.04, 7) < 1.0
    assert _sidak(0.04, 7) > 0.05                   # a marginal lag is no longer "significant"

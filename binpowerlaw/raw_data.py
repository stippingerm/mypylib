#
# Author: Marcell Stippinger 2018
# License: BSD 3 clause
#

###   Notes:
# The implementation is mostly based on the pulication
#   Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). Power-law distributions in empirical data.
#   Society for Industrial and Applied Mathematics, 51(4), 661–703. https://doi.org/10.1137/070710111
# The functionality of this submodule can be found in the python package `powerlaw` too.

import numpy as np
from scipy.stats import pareto, multinomial
from .base import progress_bar, truncated_pareto, make_search_grid, \
    gen_surrogate_data as _surrogate, _work_axis


def _check_data(data, flatten=False, sort=False):
    """
    :param points: observed values, shape (n,)
    :param counts: number of occurrences for `points`, shape (n,)
    """
    data = np.atleast_1d(data)
    if flatten:
        data = np.ravel(data)
    if sort and not np.any(np.diff(data, axis=_work_axis) < 0):
        # sort if not increasing
        data = np.sort(data, axis=_work_axis)
    return data


def _trf_check_bounds(data, xmin, xmax):
    data = data.astype(float)
    use_data = np.logical_and(xmin <= data, data <= xmax)
    return data, use_data


def log_likelihood(data, alpha, xmin, xmax=np.inf):
    """
    Give the likelihood of data assuming the distribution parameters.
    :param data: observed samples, shape (m,)
    :param xmin: the lower cutoff of the power-law
    :param xmax: the upper cutoff of the power-law, currently ignored
    :param alpha: exponent, float
    """
    data = _check_data(data)
    if xmax == np.inf:
        ll = pareto.logpdf(data, alpha, scale=xmin)
    else:
        ll = truncated_pareto.logpdf(data, alpha - 1, float(xmax) / xmin, scale=xmin)
    return np.sum(ll)


def hill_estimator(data, xmin, xmax=np.inf):
    """
    Give the MLE for continuous power-law distribution exponent.
    :param bins: increasing bin boundaries, shape (n+1,)
    :param counts: counts in the bin, shape (n,)
    :param xmin: the cutoff of the power law
    """
    data = _check_data(data)
    data, use_data = _trf_check_bounds(data, xmin, xmax)
    powered = np.log(data) - np.log(xmin)
    alpha = 1 + np.sum(use_data) / np.sum(powered[use_data])
    return alpha


def KS_test(data, alpha, xmin, xmax=np.inf):
    """
    Give the Kolmogorov-Smirnov distance between the theoretic distribution and the data.
    :param data: data samples, increasingly ordered if possible, shape (n,)
    :param alpha: the exponent being tested, float
    :param xmin: the lower cutoff of the power-law, float
    :param xmax: the upper cutoff of the power-law, float
    """
    data = _check_data(data, sort=True)
    data, use_data = _trf_check_bounds(data, xmin, xmax)
    if xmax == np.inf:
        cdf = pareto.cdf(data[use_data], alpha - 1, scale=xmin)
    else:
        cdf = truncated_pareto.cdf(data[use_data], alpha - 1, float(xmax) / xmin, scale=xmin)
    n = np.sum(use_data)
    emp1 = np.arange(n) / float(n)
    emp2 = np.arange(1, n + 1) / float(n)
    ks = np.maximum(np.abs(emp1 - cdf), np.abs(emp2 - cdf))
    return np.max(ks) if len(ks) else np.inf


def find_xmin_xmax_ks(data, grid=None, scaling_range=10, max_range=np.inf, req_samples=100,
                      no_xmax=True, ranking=False):
    """
    Find the best scaling interval, exponent and the Kolmogorov-Smirnov distance which measures the fit quality.
    :param data: samples, if provided then used, shape (m,)
    :param grid: inspected boundary values, increasing, shape (m,)
    :param scaling_range, max_range: the minimal and maximal factor between `xmin` and `xmax`, float
    :param req_samples: the minimal number of samples in the chosen interval, int
    :param no_xmax: assume that xmax=np.inf, bool
    :return xmin, xmax, ahat, ks:
    """
    if grid is None:
        grid = np.unique(data)
    counts = np.histogram(data, grid)
    n_cum = np.concatenate(([0], np.cumsum(counts)))

    low, high, n_low, n_high = make_search_grid(grid, n_cum, no_xmax, scaling_range, max_range, req_samples)

    alpha_est = np.array([hill_estimator(data, xmin, xmax) for xmin, xmax in zip(low, high)])
    ks = np.array([KS_test(data, ahat, xmin, xmax) for ahat, xmin, xmax in zip(alpha_est, low, high)])
    which = np.argsort(ks) if ranking else np.nanargmin(ks)
    return alpha_est[which], low[which], high[which], ks[which]


def goodness_of_fit(data, alpha, xmin, xmax=np.inf, n_iter=1000, grid=None, debug=False, **kwargs):
    # bins is required to reduce number of guesses
    """
    Find the p-value of `data` coming from the pareto of given parameters.
    :param data: data samples, increasingly ordered if possible, shape (n,)
    :param xmin: the lower cutoff of the hypothesized power-law, float
    :param xmax: the upper cutoff of the hypothesized power-law, float
    :param alpha: the hypothesized exponent to be tested, float
    """

    def gen_surrogate_ks(n_point, p_cat, low, high, alpha, xmin, xmax, grid):
        sample = _surrogate(n_point, p_cat, low, high, alpha, xmin, xmax)
        counts, _ = np.histogram(sample, grid)
        _xmin, _xmax, _ahat, _ks = find_xmin_xmax_ks(grid, sample, no_xmax=(xmax == np.inf))
        return _ks

    n_point = len(data)
    low, mid, high = data[data < xmin], data[(xmin <= data) & (data < xmax)], data[xmax <= data]
    p_cat = np.array([len(low), len(mid), len(high)]) / float(n_point)
    if grid is None:
        grid = np.unique(data)

    ks_collection = [gen_surrogate_ks(n_point, p_cat, low, high, alpha, xmin, xmax, grid) for i in
                     progress_bar(range(n_iter))]
    ks_collection = np.sort(ks_collection)
    ks_data = KS_test(data, alpha, xmin, xmax)

    p = np.searchsorted(ks_collection, ks_data) / float(n_iter)
    if debug:
        print(ks_collection, ks_data)
    return p

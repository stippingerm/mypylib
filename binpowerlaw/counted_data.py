#
# Author: Marcell Stippinger 2018
# License: BSD 3 clause
#

###   Notes:
# The implementation is mostly based on the publication
#   Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). Power-law distributions in empirical data.
#   Society for Industrial and Applied Mathematics, 51(4), 661–703. https://doi.org/10.1137/070710111
# This submodule is intended to provide functionality and performance improvement over the python package
# `powerlaw` when the data is already sorted and converted to number of occurrences.

import numpy as np
from scipy.stats import pareto, multinomial
from .base import progress_bar, truncated_pareto, \
    edges_from_geometric_centers as make_grid, \
    aggregate_counts, make_search_grid, _adaptive_xmin_xmax_ks as adaptive_search, \
    gen_surrogate_counts as _surrogate, _work_axis


def _check_points_counts(points, counts, flatten=False, sort=False, min_range=None):
    """
    :param points: observed values, shape (n,)
    :param counts: number of occurrences for `points`, shape (n,)
    """
    points = np.atleast_1d(points)
    counts = np.atleast_1d(counts)
    if flatten:
        points = np.ravel(points)
        counts = np.ravel(counts)
    if len(counts) < 2:
        raise ValueError('at east two counts are requred for reasonable calculation, got'
                         'points="%s", counts="%s"'%(points,counts))
    if points.shape != counts.shape:
        raise ValueError('points and counts must have the same shape')
    input_ordering = np.unique(np.sign(np.diff(points, axis=_work_axis)))
    if 0 in input_ordering:
        raise ValueError('duplicate entry for points')
    if sort:
        # sort if not increasing
        if len(input_ordering) == 1:
            input_ordering = int(input_ordering)
            points = points[::input_ordering]
            counts = counts[::input_ordering]
        else:
            idx = np.argsort(points, axis=_work_axis)
            # TODO: in the 2d case, use advance indexing, (e.g., [ [[0],[1],[2], ...], idx ]) for sorting
            points = points[idx]
            counts = counts[idx]
    if min_range is not None:
        low, high = np.min(points), np.max(points)
        if low * min_range > high:
            raise ValueError('edges span smaller interval than the entire range')
    return points, counts


def _trf_check_bounds(points, counts, xmin, xmax):
    points = points.astype(float)
    use_data = np.logical_and(xmin <= points, points <= xmax)
    return points, use_data


def log_likelihood(points, counts, alpha, xmin, xmax=np.inf):
    """
    Give the likelihood of binned data assuming the distribution parameters.
    This is a naive estimate that assumes all data points are in the geometric center of the bin
    :param alpha: exponent, float
    :param points: observed values, shape (n,)
    :param counts: number of occurrences for `points`, shape (n,)
    :param xmin: the lower cutoff of the power-law
    :param xmax: the upper cutoff of the power-law
    """
    points, counts = _check_points_counts(points, counts)
    # Out of range must result in logpdf returning ll == -np.inf
    if xmax == np.inf:
        ll = pareto.logpdf(points, alpha - 1, scale=xmin)
    else:
        ll = truncated_pareto.logpdf(points, alpha - 1, float(xmax) / xmin, scale=xmin)
    return np.sum(ll * counts, axis=_work_axis)


def hill_estimator(points, counts, xmin, xmax=np.inf):
    """
    Give the MLE for continuous power-law distribution exponent.
    :param points: observed values, shape (n,)
    :param counts: number of occurrences for `points`, shape (n,)
    :param xmin: the cutoff of the power law
    :param xmax: the upper cutoff of the power-law
    """
    points, counts = _check_points_counts(points, counts)
    points, use_data = _trf_check_bounds(points, counts, xmin, xmax)
    powered = counts * (np.log(points) - np.log(xmin))
    alpha = 1 + np.sum(counts[use_data]) / np.sum(powered[use_data])
    return alpha


def KS_test(points, counts, alpha, xmin, xmax=np.inf):
    """
    Give the Kolmogorov-Smirnov distance between the theoretic distribution and the data.
    :param points: observed values, shape (n,)
    :param counts: number of occurrences for `points`, shape (n,)
    :param xmin: the lower cutoff of the power-law, float
    :param xmax: the upper cutoff of the power-law, float
    :param alpha: the exponent being tested, float
    """
    points, counts = _check_points_counts(points, counts, sort=True)
    points, use_data = _trf_check_bounds(points, counts, xmin, xmax)
    if np.isinf(xmax):
        cdf = pareto.cdf(points[use_data], alpha - 1, scale=xmin)
    else:
        cdf = truncated_pareto.cdf(points[use_data], alpha - 1, float(xmax) / xmin, scale=xmin)
    emp1 = np.cumsum(counts[use_data]) / float(np.sum(counts[use_data]))
    emp2 = np.concatenate(([0], emp1[:-1]))
    ks = np.maximum(np.abs(emp1 - cdf), np.abs(emp2 - cdf))
    return np.max(ks) if len(ks) else np.inf


def find_xmin_xmax_ks(points, counts, grid=None, scaling_range=10, max_range=np.inf,
                      clip_low=np.inf, clip_high=0, req_samples=100,
                      no_xmax=True, ranking=False):
    """
    Find the best scaling interval, exponent and the Kolmogorov-Smirnov distance which measures the fit quality.
    :param points: observed values, shape (n,)
    :param counts: number of occurrences for `points`, shape (n,)
    :param grid: inspected boundary values, increasing, shape (m,)
    :param scaling_range, max_range: the minimal and maximal factor between `xmin` and `xmax`, float
    :param req_samples: the minimal number of samples in the chosen interval, int
    :param no_xmax: assume that xmax=np.inf, bool
    :return xmin, xmax, ahat, ks:
    """
    points, counts = _check_points_counts(points, counts, min_range=scaling_range)
    if grid is None:
        grid, grid_counts = points, counts
        n_cum = np.cumsum(grid_counts)
    else:
        grid_counts, grid = aggregate_counts(points, counts, grid)
        n_cum = np.concatenate(([0], np.cumsum(grid_counts)))
    assert len(grid) == len(n_cum)

    low, high, n_low, n_high = make_search_grid(grid, n_cum, no_xmax, scaling_range, max_range,
                                                clip_low, clip_high, req_samples=req_samples)

    alpha_est = np.array([hill_estimator(points, counts, xmin, xmax) for xmin, xmax in zip(low, high)])
    ks = np.array([KS_test(points, counts, ahat, xmin, xmax) for ahat, xmin, xmax in zip(alpha_est, low, high)])
    which = np.argsort(ks) if ranking else np.nanargmin(ks)
    return alpha_est[which], low[which], high[which], ks[which]


def adaptive_xmin_xmax_ks(edges, counts, *args, **kwargs):
    edges, counts = _check_points_counts(edges, counts)
    # _adaptive_xmin_xmax_ks(fun, edges, *args, n_work, method='twopass', debug=False, **kwargs)
    return adaptive_search(find_xmin_xmax_ks, edges, counts, *args, **kwargs)


def point_based_goodness(points, counts, alpha, xmin, xmax=np.inf, n_iter=1000, grid=None, debug=False, **kwargs):
    # bins is required to reduce number of guesses
    """
    Find the p-value of `data` coming from the pareto of given parameters.
    :param data: data samples, increasingly ordered if possible, shape (n,)
    :param alpha: the hypothesized exponent to be tested, float
    :param xmin: the lower cutoff of the hypothesized power-law, float
    :param xmax: the upper cutoff of the hypothesized power-law, float
    """

    def gen_surrogate_ks(n_point, p_cat, p_low, p_high, alpha, xmin, xmax, grid):
        counts = _surrogate(n_point, p_cat, p_low, p_high, alpha, xmin, xmax, grid)
        _xmin, _xmax, _ahat, _ks = find_xmin_xmax_ks(grid, counts, no_xmax=(xmax == np.inf), **kwargs)
        return _ks

    points, counts = _check_points_counts(points, counts)
    n_point = np.sum(counts)
    c_low, c_mid, c_high = counts[points < xmin], counts[(xmin <= points) & (points < xmax)], counts[xmax <= points]
    p_cat = np.array([np.sum(c_low), np.sum(c_mid), np.sum(c_high)]) / float(n_point)
    p_low, p_high = c_low / np.sum(c_low), c_high / np.sum(c_high)
    if grid is None:
        grid = make_grid(points)

    ks_collection = [gen_surrogate_ks(n_point, p_cat, p_low, p_high, alpha, xmin, xmax, grid) for i in
                     progress_bar(range(n_iter))]
    ks_collection = np.sort(ks_collection)
    ks_data = KS_test(points, counts, alpha, xmin, xmax)

    p = np.searchsorted(ks_collection, ks_data) / float(n_iter)
    if debug:
        print(ks_collection, ks_data)
    return p

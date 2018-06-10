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
from .base import progress_bar, truncated_pareto, genzipf, truncated_zipf, \
    edges_from_geometric_centers as make_grid, \
    aggregate_counts, make_search_grid, _adaptive_xmin_xmax_ks as adaptive_search, \
    gen_surrogate_counts as _surrogate, _work_axis, check_random_state, \
    p_value_from_ks, uncertainty_of_alpha


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
                         'points="%s", counts="%s"' % (points, counts))
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


def _trf_check_bounds(points, counts, xmin, xmax, discrete, force_discard_end=False):
    if discrete:
        points = points.astype(int)
    else:
        points = points.astype(float)
    if discrete or force_discard_end:
        use_data = np.logical_and(xmin <= points, points < xmax)
    else:
        use_data = np.logical_and(xmin <= points, points <= xmax)
    return points, use_data


def _log_likelihood(points, counts, xmin, xmax, discrete, alpha):
    """
    Give the likelihood of binned data assuming the distribution parameters.
    This can be seen as a naive estimate for binned data that assumes
    all data points are in the center points of the bins.
    Note: nonzero counts for out of range points will set the return value -np.inf.
    :param alpha: exponent, float
    :param points: observed values, shape (n,)
    :param counts: number of occurrences for `points`, shape (n,)
    :param xmin: the lower cutoff of the power-law
    :param xmax: the upper cutoff of the power-law
    :param discrete: interpret as a discrete power-law (genrealized zipf) distribution
    """
    # TODO: use dispatch_pdf
    if discrete:
        if np.isinf(xmax):
            ll = genzipf.logpmf(points, alpha, xmin)
        else:
            ll = truncated_zipf.logpmf(points, alpha, xmin, xmax)
    else:
        if np.isinf(xmax):
            ll = pareto.logpdf(points, alpha - 1, scale=xmin)
        else:
            ll = truncated_pareto.logpdf(points, alpha - 1, float(xmax) / xmin, scale=xmin)
    return np.sum(ll * counts, axis=_work_axis)


def log_likelihood(points, counts, alpha, xmin, xmax=np.inf, discrete=False):
    """
    Give the likelihood of binned data assuming the distribution parameters.
    This can be seen as a naive estimate for binned data that assumes
    all data points are in the center points of the bins.
    Note: nonzero counts for out of range points will set the return value -np.inf.
    :param alpha: exponent, float
    :param points: observed values, shape (n,)
    :param counts: number of occurrences for `points`, shape (n,)
    :param xmin: the lower cutoff of the power-law
    :param xmax: the upper cutoff of the power-law
    :param discrete: interpret as a discrete power-law (genrealized zipf) distribution
    """
    points, counts = _check_points_counts(points, counts)
    points, use_data = _trf_check_bounds(points, counts, xmin, xmax, discrete)
    # Out of range must result in logpdf returning ll == -np.inf
    if np.any(counts[~use_data]):
        # TODO: broadcasting would be difficult
        ll = -np.inf
    else:
        ll = _log_likelihood(points[use_data], counts[use_data], xmin, xmax, discrete, alpha)
    return ll


def hill_estimator(points, counts, xmin, xmax=np.inf, discrete=False, **kwargs):
    """
    Give the MLE for continuous power-law distribution exponent.
    :param points: observed values, shape (n,)
    :param counts: number of occurrences for `points`, shape (n,)
    :param xmin: the cutoff of the power law
    :param xmax: the upper cutoff of the power-law
    :param discrete: interpret as a discrete power-law (genrealized zipf) distribution
    """
    points, counts = _check_points_counts(points, counts)
    points, use_data = _trf_check_bounds(points, counts, xmin, xmax, discrete)
    powered = counts * (np.log(points) - np.log(xmin))
    alpha = 1 + np.sum(counts[use_data]) / np.sum(powered[use_data])
    # The correction if xmax!=np.inf can be achieved by numerical approximation only:
    if discrete or not np.isinf(xmax):
        from scipy.optimize import minimize
        args = points[use_data], counts[use_data], xmin, xmax, discrete
        x0 = np.array([alpha])  # The zipf optimization task does not behave well.
        bounds = kwargs.pop('bounds', [(1.0001, None)])
        res = minimize(lambda a: -_log_likelihood(*args, a), x0, bounds=bounds, **kwargs)
        return np.asscalar(res.x)
    return alpha


def KS_test(points, counts, alpha, xmin, xmax=np.inf, discrete=False):
    """
    Give the Kolmogorov-Smirnov distance between the theoretic distribution and the data.
    :param points: observed values, shape (n,)
    :param counts: number of occurrences for `points`, shape (n,)
    :param xmin: the lower cutoff of the power-law, float
    :param xmax: the upper cutoff of the power-law, float
    :param alpha: the exponent being tested, float
    :param discrete: interpret as a discrete power-law (genrealized zipf) distribution
    """
    points, counts = _check_points_counts(points, counts, sort=True)
    points, use_data = _trf_check_bounds(points, counts, xmin, xmax, discrete, force_discard_end=True)
    # TODO: use dispatch_cdf
    if discrete:
        if np.isinf(xmax):
            cdf = genzipf.cdf(points[use_data], alpha, xmin)
        else:
            cdf = truncated_zipf.cdf(points[use_data], alpha, xmin, xmax)
    else:
        if np.isinf(xmax):
            cdf = pareto.cdf(points[use_data], alpha - 1, scale=xmin)
        else:
            cdf = truncated_pareto.cdf(points[use_data], alpha - 1, float(xmax) / xmin, scale=xmin)
    emp = np.cumsum(counts[use_data]) / float(np.sum(counts[use_data]))
    if not discrete:
        # This correction is needed because cdf_continuous[xmin] == 0 while emp[xmin] has an important weight
        emp = np.concatenate(([0], emp[:-1]))
    ks = np.abs(emp - cdf)
    return np.max(ks) if len(ks) else np.inf


def find_xmin_xmax_ks(points, counts, grid=None, scaling_range=10, max_range=np.inf,
                      clip_low=np.inf, clip_high=0, req_samples=100,
                      no_xmax=True, discrete=False, ranking=False, debug=False, **kwargs):
    """
    Find the best scaling interval, exponent and the Kolmogorov-Smirnov distance which measures the fit quality.
    :param points: observed values, shape (n,)
    :param counts: number of occurrences for `points`, shape (n,)
    :param grid: inspected boundary values, increasing, shape (m,)
    :param scaling_range, max_range: the minimal and maximal factor between `xmin` and `xmax`, float
    :param req_samples: the minimal number of samples in the chosen interval, int
    :param no_xmax: assume that xmax=np.inf, bool
    :param discrete: interpret as a discrete power-law (genrealized zipf) distribution
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
                                                clip_low, clip_high, req_samples=req_samples, debug=debug)

    alpha_est = np.array(
        [hill_estimator(points, counts, xmin, xmax, discrete, **kwargs) for xmin, xmax in zip(low, high)])
    ks = np.array(
        [KS_test(points, counts, ahat, xmin, xmax, discrete) for ahat, xmin, xmax in zip(alpha_est, low, high)])
    which = np.argsort(ks) if ranking else np.nanargmin(ks)
    return alpha_est[which], low[which], high[which], ks[which]


def adaptive_xmin_xmax_ks(edges, counts, *args, **kwargs):
    edges, counts = _check_points_counts(edges, counts)
    # _adaptive_xmin_xmax_ks(fun, edges, *args, n_work, method='twopass', debug=False, **kwargs)
    return adaptive_search(find_xmin_xmax_ks, edges, counts, *args, **kwargs)


def goodness_of_fit(points, counts, alpha, xmin, xmax=np.inf, discrete=False, n_iter=1000, grid=None, debug=False,
                    random_state=None, **kwargs):
    # bins is required to reduce number of guesses
    """
    Find the p-value of `data` coming from the pareto of given parameters.
    :param points: observed values, shape (n,)
    :param counts: number of occurrences for `points`, shape (n,)
    :param alpha: the hypothesized exponent to be tested, float
    :param xmin: the lower cutoff of the hypothesized power-law, float
    :param xmax: the upper cutoff of the hypothesized power-law, float
    :param discrete: interpret as a discrete power-law (genrealized zipf) distribution
    :param n_iter: the number of samples, int
    :param grid: inspected boundary values, increasing, shape (m,)
    :param debug: bool
    :param random_state:
    :param **kwargs:
    :return p: p-value
    """

    def gen_surrogate_ks(i):
        _counts = _surrogate(n_point, p_cat, p_low, p_high, alpha, xmin, xmax, bins=edges,
                             discrete=discrete, random_state=random_state)
        _ahat, _xmin, _xmax, _ks = find_xmin_xmax_ks(points, _counts, grid, no_xmax=no_xmax,
                                                     debug=debug and (i < 10), **kwargs)
        return _ahat, _ks

    points, counts = _check_points_counts(points, counts)
    random_state = check_random_state(random_state)
    alpha = float(alpha)
    no_xmax = np.isinf(xmax)

    if discrete:
        raise NotImplementedError
    n_point = np.sum(counts)
    c_low, c_mid, c_high = counts[points < xmin], counts[(xmin <= points) & (points < xmax)], counts[xmax <= points]
    p_cat = np.array([np.sum(c_low), np.sum(c_mid), np.sum(c_high)]) / float(n_point)
    p_low, p_high = c_low / np.sum(c_low), c_high / np.sum(c_high)
    # if grid is None:
    #    grid = make_grid(points)
    edges = make_grid(points)

    alpha_collection, ks_collection = zip(*[gen_surrogate_ks(i) for i in progress_bar(range(n_iter))])
    ks_data = KS_test(points, counts, alpha, xmin, xmax)

    return uncertainty_of_alpha(alpha_collection, alpha, debug), p_value_from_ks(ks_collection, ks_data, debug)

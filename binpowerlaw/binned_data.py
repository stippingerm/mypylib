#
# Author: Marcell Stippinger 2018
# License: BSD 3 clause
#

###   Notes:
# The implementation is mostly based on the publication
#   Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). Power-law distributions in empirical data.
#   Society for Industrial and Applied Mathematics, 51(4), 661–703. https://doi.org/10.1137/070710111
# This submodule is intended to provide additional functionality compared to the python package
# `powerlaw` when the data is already stored as a histogram (equidistant, log-scale or arbitrary).

import numpy as np
from scipy.stats import pareto, multinomial
from .base import progress_bar, truncated_pareto, \
    edges_from_arithmetic_centers, edges_from_geometric_centers, geometric_centers_from_edges as _get_centers, \
    aggregate_counts, make_search_grid, _adaptive_xmin_xmax_ks as adaptive_search, \
    gen_surrogate_counts as _surrogate, _work_axis, check_random_state, \
    p_value_from_ks, uncertainty_of_alpha
from .counted_data import hill_estimator as _naive_estimator


def _check_bins_counts(edges, counts, flatten=False, sort=False, min_range=None):
    edges = np.atleast_1d(edges)
    counts = np.atleast_1d(counts)
    if flatten:
        edges = np.ravel(edges)
        counts = np.ravel(counts)
    if len(counts) < 2:
        raise ValueError('at least two counts are requred for reasonable calculation, got'
                         'edges="%s", counts="%s"'%(edges,counts))
    if edges[1:].shape != counts.shape:
        raise ValueError('points and counts must have compatible shapes (n+1,) and (n,), '
                         'but they have shapes %s and %s'%(edges.shape, counts.shape))
    input_ordering = np.unique(np.sign(np.diff(edges, axis=_work_axis)))
    if 0 in input_ordering:
        raise ValueError('duplicate entry for points')
    if sort:
        # sort if not increasing
        if len(input_ordering) == 1:
            input_ordering = int(input_ordering)
            edges = edges[::input_ordering]
            counts = counts[::input_ordering]
        else:
            raise ValueError('edges must be increasing')
    if min_range is not None:
        low, high = np.min(edges), np.max(edges)
        if low * min_range > high:
            raise ValueError('edges span smaller interval than the entire range')
    return edges, counts


def _trf_check_bounds(edges, counts, xmin, xmax):
    edges = edges.astype(float)
    lefts = edges[:-1]
    rights = edges[1:]
    widths = rights / lefts
    use_data = np.logical_and(xmin <= lefts, rights <= xmax)
    use_edge = np.logical_and(xmin <= edges, edges <= xmax)
    return lefts, widths, use_data, use_edge


def _log_likelihood(lefts, widths, counts, xmin, xmax, alpha):
    """
    Give the likelihood of binned data assuming the distribution parameters.
    Note: the argument order supports easy optimization of alpha, see hill_estimator.
    :param lefts: bin boundaries, shape (n,)
    :param widths: bin lengths, shape (n,)
    :param counts: observed hits, shape (n,)
    :param xmin: the lower cutoff of the power-law
    :param xmax: the lower cutoff of the power-law
    :param alpha: exponent, float
    """
    # accounted_weight = pareto.cdf(xmax, scale=xmin) if xmax<np.inf else 1
    log_accounted_weight = np.log1p(-pareto.sf(xmax, alpha-1, scale=xmin))
    # I believe this analytically justified formulation using bin widths
    # is less prone to errors than calculating the same bin log likelihoods
    # as the difference of two (close) pareto.cdf values.
    ll = (- np.sum(counts) * log_accounted_weight
          + np.sum(counts * np.log1p(-np.power(widths, -(alpha - 1))))
          - (alpha - 1) * np.sum(counts * (np.log(lefts) - np.log(xmin))))
    # Maintenance info: zero counts out of bounds must not alter the result,
    # This requirement is set by log_likelihood.
    return ll


def log_likelihood(edges, counts, alpha, xmin, xmax=np.inf):
    """
    Give the likelihood of binned data assuming the distribution parameters.
    This is a correct estimate working for any sort of increasingly ordered bins.
    Note: nonzero counts for out of range bins will set the return value -np.inf.
    :param edges: bin boundaries, shape (n+1,)
    :param counts: observed hits, shape (n,)
    :param xmin: the lower cutoff of the power-law
    :param xmax: the lower cutoff of the power-law
    :param alpha: exponent, float
    """
    edges, counts = _check_bins_counts(edges, counts)
    lefts, widths, use_data, use_edge = _trf_check_bounds(edges, counts, xmin, xmax)
    if np.any(counts[~use_data]):
        # TODO: broadcasting would be difficult
        ll = -np.inf
    else:
        # Note: _log_likelihood is insensitive to zero counts.
        ll = _log_likelihood(lefts, widths, counts, xmin, xmax, alpha)
    return ll


def hill_estimator(edges, counts, xmin, xmax=np.inf, **kwargs):
    """
    Give the MLE for continuous power-law distribution exponent.
    :param edges: increasing bin boundaries, shape (n+1,)
    :param counts: counts in the bin, shape (n,)
    :param xmin: the cutoff of the power law
    """
    from scipy.optimize import minimize
    # from functools import partial
    edges, counts = _check_bins_counts(edges, counts)
    lefts, widths, use_data, use_edge = _trf_check_bounds(edges, counts, xmin, xmax)
    args = lefts[use_data], widths[use_data], counts[use_data], xmin, xmax
    x0 = np.array([_naive_estimator(_get_centers(edges), counts, xmin)])
    bounds = kwargs.pop('bounds', [(1.0001, None)])
    res = minimize(lambda a: -_log_likelihood(*args, a), x0, bounds=bounds, **kwargs)
    return np.asscalar(res.x)


def KS_test(edges, counts, alpha, xmin, xmax=np.inf):
    """
    Give the Kolmogorov-Smirnov distance between the theoretic distribution and the binned data.
    :param edges: increasing bin boundaries, shape (n+1,)
    :param counts: counts in the bin, shape (n,)
    :param alpha: the exponent being tested, float
    :param xmin: the lower cutoff of the power-law, float
    :param xmax: the upper cutoff of the power-law, float
    """
    edges, counts = _check_bins_counts(edges, counts, sort=True)
    lefts, widths, use_data, use_edge = _trf_check_bounds(edges, counts, xmin, xmax)
    if np.isinf(xmax):
        cdf = pareto.cdf(edges[1:][use_data], alpha - 1, scale=xmin)
    else:
        cdf = truncated_pareto.cdf(edges[1:][use_data], alpha - 1, float(xmax) / xmin, scale=xmin)
    emp = np.cumsum(counts[use_data]) / float(np.sum(counts[use_data]))
    ks = np.abs(emp - cdf)
    return np.max(ks) if len(ks) else np.inf


def find_xmin_xmax_ks(edges, counts, grid=None, scaling_range=10, max_range=np.inf,
                      clip_low=np.inf, clip_high=0, req_samples=100,
                      no_xmax=True, ranking=False, debug=False, **kwargs):
    """
    Find the best scaling interval, exponent and the Kolmogorov-Smirnov distance which measures the fit quality.
    :param edges: increasing bin boundaries, shape (n+1,)
    :param counts: counts in the bin, shape (n,)
    :param grid: a smaller grid to do the search, for best results use a subset of values given in `edges`, shape (m,)
    :param scaling_range, max_range: the minimal and maximal factor between `xmin` and `xmax`, float
    :param clip_low, clip_high: anchors for the maximal `xmin` and minimal `xmax`
    :param req_samples: the minimal number of samples in the chosen interval, int
    :param no_xmax: assume that xmax=np.inf, bool
    :param ranking: do not select best match but return all results ordered decreasingly
                    by the goodness of fit (KS-distance)
    :keyword ...: parameters to pass to minimize call
    :return ahat, xmin, xmax, ks:
    """
    edges, counts = _check_bins_counts(edges, counts, min_range=scaling_range)
    if grid is None:
        grid, grid_counts = edges, counts
    else:
        # NOTE: if grid is not a subset of edges the following might be a poor conversion
        grid_counts, grid = aggregate_counts(edges[:-1], counts, grid)
    n_cum = np.concatenate(([0], np.cumsum(grid_counts)))
    assert len(grid) == len(n_cum)

    low, high, n_low, n_high = make_search_grid(grid, n_cum, no_xmax, scaling_range, max_range,
                                                clip_low, clip_high, req_samples=req_samples, debug=debug)

    alpha_est = np.array([hill_estimator(edges, counts, xmin, xmax, **kwargs) for xmin, xmax in zip(low, high)])
    ks = np.array([KS_test(edges, counts, ahat, xmin, xmax) for ahat, xmin, xmax in zip(alpha_est, low, high)])
    which = np.argsort(ks) if ranking else np.nanargmin(ks)
    return alpha_est[which], low[which], high[which], ks[which]


def adaptive_xmin_xmax_ks(edges, counts, *args, **kwargs):
    edges, counts = _check_bins_counts(edges, counts)
    # _adaptive_xmin_xmax_ks(fun, edges, *args, n_work, method='twopass', debug=False, **kwargs)
    return adaptive_search(find_xmin_xmax_ks, edges, counts, *args, **kwargs)


def goodness_of_fit(edges, counts, alpha, xmin, xmax=np.inf, n_iter=1000, grid=None, debug=False,
                    random_state=None, **kwargs):
    # edges is required if xmax is not infty
    """
    Find the p-value of `data` coming from the pareto of given parameters.
    :param edges: increasing bin boundaries, shape (n+1,)
    :param counts: counts in the bin, shape (n,)
    :param alpha: the hypothesized exponent to be tested, float
    :param xmin: the lower cutoff of the hypothesized power-law, float
    :param xmax: the upper cutoff of the hypothesized power-law, float
    :param n_iter: the number of samples, int
    :param grid: inspected boundary values, increasing, shape (m,)
    :param debug: bool
    :param random_state:
    :param **kwargs:
    :return p: p-value
    """

    def gen_surrogate_data(i):
        _counts = _surrogate(n_point, p_cat, p_low, p_high, alpha, xmin, xmax, bins=edges,
                             discrete=False, random_state=random_state)
        _ahat, _xmin, _xmax, _ks = find_xmin_xmax_ks(edges, _counts, grid, no_xmax=no_xmax,
                                                     debug=debug and (i < 10), **kwargs)
        return _ahat, _ks

    edges, counts = _check_bins_counts(edges, counts)
    random_state = check_random_state(random_state)
    alpha = float(alpha)
    no_xmax = (xmax == np.inf)

    n_point = np.sum(counts)
    lefts, widths, use_data, use_edge = _trf_check_bounds(edges, counts, xmin, xmax)
    rights = edges[1:]
    if not ((xmin in lefts) and ((xmax == np.inf) or (xmax in rights))):
        raise ValueError('xmin and xmax should fall on bin boundaries.')
    c_low, c_mid, c_high = counts[rights <= xmin], counts[(xmin <= lefts) & (rights <= xmax)], counts[xmax <= lefts]
    p_cat = np.array([np.sum(c_low), np.sum(c_mid), np.sum(c_high)]) / float(n_point)
    p_low, p_high = c_low / np.sum(c_low), c_high / np.sum(c_high)
    #if grid is None:
    #    grid = edges

    alpha_collection, ks_collection = zip(*[gen_surrogate_data(i) for i in progress_bar(range(n_iter))])
    ks_data = KS_test(edges, counts, alpha, xmin, xmax)

    return uncertainty_of_alpha(alpha_collection, alpha, debug), p_value_from_ks(ks_collection, ks_data, debug)

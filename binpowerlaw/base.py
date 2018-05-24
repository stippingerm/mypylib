#
# Author: Marcell Stippinger 2018
# License: BSD 3 clause
#

import numpy as np
import pandas as pd
from scipy.stats import rv_continuous, multinomial, pareto

_work_axis = -1


def _identity(x):
    return x


try:
    from tqdm import tqdm as progress_bar
except ModuleNotFoundError:
    progress_bar = _identity


# In the scipy.pareto implementation the scale parameter plays the role of xmin while
# loc has no standard interpretation.

class truncated_pareto_gen(rv_continuous):
    """Truncated power-law distribution, see scipy.stats.pareto too"""

    def _cdf(self, x, b, m):
        return pareto.cdf(x, b) / pareto.cdf(m, b)

    def _logcdf(self, x, b, m):
        return pareto.logcdf(x, b) - pareto.logcdf(m, b)

    def _ppf(self, q, b, m):
        return np.power(1.0 - q * pareto.cdf(m, b), -1.0 / b)

    def _pdf(self, x, b, m):
        return np.select([x < m], [pareto.pdf(x, b) / pareto.cdf(m, b)], 0)

    def _logpdf(self, x, b, m):
        return np.select([x < m], [pareto.logpdf(x, b) - pareto.logcdf(m, b)], -np.inf)


truncated_pareto = truncated_pareto_gen(a=1.0, name="truncated_pareto")


def edges_from_geometric_centers(centers):
    data = np.atleast_1d(centers)
    if len(data) > 2:
        edge0 = np.power(data[0:1], 1.5) * np.power(data[1:2], -0.5)
        edgeE = np.power(data[-1:None], 1.5) * np.power(data[-2:-1], -0.5)
        edges = np.sqrt(data[1:] * data[:-1])
        return np.concatenate((edge0, edges, edgeE))
    else:
        raise ValueError('Too few bin centers.')


def geometric_centers_from_edges(edges):
    data = np.atleast_1d(edges)
    if len(data) > 1:
        return np.sqrt(edges[1:] * edges[:-1])
    else:
        raise ValueError('Too few bin edges.')


def edges_from_arithmetic_centers(centers):
    data = np.atleast_1d(centers)
    if len(data) > 2:
        edge0 = 1.5 * data[0:1] - 0.5 * data[1:2]
        edgeE = 1.5 * data[-1:None] - 0.5 * data[-2:-1]
        edges = np.sqrt(data[1:] * data[:-1])
        return np.concatenate((edge0, edges, edgeE))
    else:
        raise ValueError('Too few bin centers.')


def arithmetic_centers_from_edges(edges):
    data = np.atleast_1d(edges)
    if len(data) > 1:
        return 0.5 * (edges[1:] + edges[:-1])
    else:
        raise ValueError('Too few bin edges.')


def aggregate_counts(points, counts, bins, right=False):
    """
    Make a histogram of hit counts. This is a good way to dilute the bin boundaries of a histogram
    if the new boundaries do not coincide with the old ones.
    Note that a histogram has one more edges than the number of points listed in the hit counts.
    :param points: observed values, shape (n,)
    :param counts: number of occurrences for `points`, shape (n,)
    :param bins: bin edges, must be 1d and increasing
    :param right: whether to produce right semi-closed intervals, default: False
    :return counts, edges: tuple of data counts shape (m,) and histogram edges shape (m+1,)
    """
    points, counts, bins = np.array(points), np.array(counts), np.array(bins)
    if points.shape != counts.shape:
        raise ValueError('points and counts must have the same shape')
    if (len(bins.shape) != 1) or np.any(np.diff(bins) < 0):
        raise ValueError('bins must be 1d and increasing, got bins="%s"' % bins)
    n_bin = len(bins) - 1
    idx = np.digitize(points, bins, right)  # 0 and n_bin+1 mean out of bonds
    # Mimic np.histogram and make the last bin closed
    idx[points == bins[-1]] = n_bin
    idx[points == bins[0]] = 1
    df_counts = pd.DataFrame(list(zip(idx - 1, counts)), columns=['idx', 'counts'])
    agg_counts = df_counts.groupby('idx').sum().reindex(np.arange(n_bin), fill_value=0)
    return agg_counts['counts'].values, bins


def make_search_grid(edges, n_cum, no_xmax, scaling_range=1, max_range=np.inf,
                     clip_low=np.inf, clip_high=0, req_samples=0):
    if no_xmax:
        low, high = np.meshgrid(edges, np.inf)
        n_low, n_high = np.meshgrid(n_cum, n_cum[-1])
    else:
        low, high = np.meshgrid(edges, edges)
        n_low, n_high = np.meshgrid(n_cum, n_cum)
    low, high = low.ravel(), high.ravel()
    n_low, n_high = n_low.ravel(), n_high.ravel()

    acceptable = (low * scaling_range <= high) & (low * max_range >= high) & (
        low <= clip_low) & (high >= clip_high) & (req_samples < n_high - n_low)
    if np.sum(acceptable.astype(int)) == 0:
        raise ValueError('Empty search grid.')
    return low[acceptable], high[acceptable], n_low[acceptable], n_high[acceptable]


def _suggest_stepsizes(largest, method):
    """
    Suggest stepsizes for successive approximation of the result if exact solution is not feasible.
    :param largest: largest step to perform
    :param method: single for one-way no repetition, double for one-way but repeat each size, twopass for swipe
    :return: array of step sizes in the order they should be performed
    """
    log_largest = np.ceil(np.log2(largest)).astype(int)
    path = np.logspace(log_largest, 0, log_largest + 1, base=2.0).astype(int)
    relax = np.array([1, 1])
    if method == 'single':
        return np.concatenate((path, relax))
    if method == 'double':
        return np.concatenate((np.repeat(path, 2), relax))
    elif method == 'twopass':
        return np.concatenate((path, path[::-1], path, relax))
    else:
        raise ValueError('Unknown stepsize calculation method')


def _stable_unique(arr, limit=None):
    """
    Return unique values in their original order.
    :param arr:
    :param limit:
    :return:
    """
    arr = np.array(arr).ravel()
    idx = np.unique(arr, return_index=True)[1]
    return arr[np.sort(idx)[:limit]]


def _adaptive_xmin_xmax_ks(fun, edges, *args, n_work, method='twopass', debug=False, **kwargs):
    """
    Do a non-exhaustive grid search assuming that close values yield similar results.
    :param fun:
    :param edges:
    :param args:
    :param n_work:
    :param method:
    :param debug:
    :param kwargs:
    :return:
    """
    kwargs.pop('grid', None)
    ranking = kwargs.pop('ranking', False)
    n_edge = len(edges)
    n_work = int(n_work)
    largest_step, *intermediate_steps = _suggest_stepsizes(n_edge / float(n_work), method)
    df_edge_to_idx = pd.Series(np.arange(n_edge), index=edges, name='idx')
    work_edges = edges[::largest_step]
    if debug:
        print(largest_step, intermediate_steps)
        print('>>> %10s%10s%10s' % ('step', 'ks_dst', 'set_size'))
    for step in intermediate_steps:
        alpha, low, high, ks = fun(edges, *args, grid=work_edges, **kwargs, ranking=True)
        best_edges = _stable_unique(list(zip(low, high)), n_work)
        best_idx = df_edge_to_idx[best_edges].values
        if debug:
            if ks.size>0:
                print('+++ %10d%10f%10d' % (step, ks[:1], len(alpha)))
                print('l', df_edge_to_idx[low].values, 'h', df_edge_to_idx[high].values, 'b', best_idx,)
            else:
                print('--- %10d%10s%10d' % (step, 'empty', len(alpha)))
                raise ValueError('Empty result set, probably some of the grid conditions are not met '
                                 '(e.g. too many samples requested).')
        # * winning remains so scaling_range won't reduce the meshgrid to empty
        # * np.unique sorts
        extended_idx = np.unique(np.sum(np.ix_(best_idx, [-step, 0, step])).ravel())
        extended_idx = extended_idx[(0 <= extended_idx) & (extended_idx < n_edge)]
        work_edges = edges[extended_idx]
    return fun(edges, *args, grid=work_edges, **kwargs, ranking=ranking)


def gen_surrogate_data(n_point, p_cat, low, high, alpha, xmin, xmax):
    """
    Generate surrogate data points
    :param n_point: total number of data points
    :param p_cat: probability of `low`, `pareto` and `high` categories
    :param low, high: data to be subsampled (with replacement) for categories `low` and `high`
    :param alpha: exponent of the `pareto` regime
    :param xmin, xmax: boundaries of the `pareto` regime, so that all(low<xmin) and all (xmax<=high)
    :return: surrogate sample
    """
    s_low, s_mid, s_high = multinomial.rvs(n_point, p_cat)
    sample = np.zeros(n_point, dtype=float)
    if s_low:
        sample[0:s_low] = np.random.choice(low, s_low, replace=True)
    if s_high:
        sample[s_low + s_mid:n_point] = np.random.choice(high, s_high, replace=True)

    if xmax == np.inf:
        sample[s_low:s_low + s_mid] = pareto.rvs(alpha, scale=xmin, size=s_mid)
    else:
        sample[s_low:s_low + s_mid] = truncated_pareto.rvs(alpha, xmax / float(xmin), scale=xmin, size=s_mid)

    return sample


def gen_surrogate_counts(n_point, p_cat, p_low, p_high, alpha, xmin, xmax, bins):
    """
    Generate surrogate hit counts
    :param n_point: total number of data points
    :param p_cat: probability of `low`, `pareto` and `high` categories
    :param p_low, p_high: hit probabilities within categories `low` and `high`
    :param alpha: exponent of the `pareto` regime
    :param xmin, xmax: boundaries of the `pareto` regime, so that all(low<xmin) and all (xmax<=high)
    :return: surrogate hit counts
    """
    s_low, s_mid, s_high = multinomial.rvs(n_point, p_cat)
    if xmax == np.inf:
        sample = pareto.rvs(alpha, scale=xmin, size=s_mid)
    else:
        sample = truncated_pareto.rvs(alpha, xmax / float(xmin), scale=xmin, size=s_mid)

    counts, _ = np.histogram(sample, bins)
    if s_low:
        counts[0:len(p_low)] = multinomial.rvs(s_low, p_low)
    if s_high:
        counts[len(bins) - len(p_high):len(bins)] = multinomial.rvs(s_low, p_high)

    return counts

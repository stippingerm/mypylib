#
# Author: Marcell Stippinger 2018
# License: BSD 3 clause
#

import numpy as np
import pandas as pd
from scipy.stats import rv_continuous, rv_discrete, multinomial, pareto
from scipy.special import zeta
import numbers

_work_axis = -1


def _identity(x):
    return x


try:
    from tqdm import tqdm as progress_bar
except ModuleNotFoundError:
    progress_bar = _identity


# In the scipy.stats.pareto implementation the scale parameter plays the role of xmin while
# loc has no standard interpretation.

class truncated_pareto_gen(rv_continuous):
    """Truncated power-law distribution, see scipy.stats.pareto too"""

    def _argcheck(self, b, m):
        self.b = m
        return (0 < b) and (1 < m)

    def _cdf(self, x, b, m):
        return pareto.cdf(x, b) / pareto.cdf(m, b)

    def _logcdf(self, x, b, m):
        return pareto.logcdf(x, b) - pareto.logcdf(m, b)

    def _ppf(self, q, b, m):
        return np.power(1.0 - q * pareto.cdf(m, b), -1.0 / b)

    def _pdf(self, x, b, m):
        return pareto.pdf(x, b) / pareto.cdf(m, b)

    def _logpdf(self, x, b, m):
        return pareto.logpdf(x, b) - pareto.logcdf(m, b)


truncated_pareto = truncated_pareto_gen(a=1.0, name="truncated_pareto")


# In the scipy.stats.pareto implementation the scale parameter plays the role of xmin while
# loc has no standard interpretation.

class extended_pareto_gen(rv_continuous):
    """Truncated power-law distribution extended to work for all real exponents, see scipy.stats.pareto too"""

    @staticmethod
    def _np_integrate_1x(x, b):
        with np.errstate(invalid='ignore'):
            choices = [np.log(x), (1 - np.float_power(x, -b)) / b]
        return np.select([b == 0, b != 0], choices, np.nan)

    @staticmethod
    def _np_log_integrate_1x(x, b):
        with np.errstate(invalid='ignore'):
            choices = [np.log(np.log(x)), np.log1p(-np.float_power(x, -b)) - np.log(b),
                       np.log((1 - np.float_power(x, -b)) / b)]
        return np.select([b == 0, b > 0, b < 0], choices, np.nan)

    @staticmethod
    def _np_invert_fullq(q, b):
        with np.errstate(invalid='ignore'):
            choices = [np.exp(q), np.float_power(1 - q, -1.0 / b)]
        return np.select([b == 0, b != 0], choices, np.nan)

    @staticmethod
    def _integrate_1x(x, b):
        return np.log(x) if b == 0 else (1 - np.float_power(x, -b)) / b

    @staticmethod
    def _log_integrate_1x(x, b):
        return np.log(np.log(x)) if b == 0 else (
            np.log1p(-np.float_power(x, -b)) - np.log(b) if b > 0 else np.log((1 - np.float_power(x, -b)) / b))

    @staticmethod
    def _invert_fullq(q, b):
        return np.exp(q) if b == 0 else np.float_power(1 - b * q, -1.0 / b)

    @staticmethod
    def _vec_integrate_1x(x, b):
        fun = np.vectorize(extended_pareto_gen._integrate_1x)
        return fun(x, b)

    @staticmethod
    def _vec_log_integrate_1x(x, b):
        fun = np.vectorize(extended_pareto_gen._log_integrate_1x)
        return fun(x, b)

    @staticmethod
    def _vec_invert_fullq(q, b):
        fun = np.vectorize(extended_pareto_gen._invert_fullq)
        return fun(q, b)

    def _argcheck(self, b, m):
        self.b = m
        return np.all(1 < m)

    def _cdf(self, x, b, m):
        return self._vec_integrate_1x(x, b) / self._vec_integrate_1x(m, b)

    def _logcdf(self, x, b, m):
        return self._vec_log_integrate_1x(x, b) - self._vec_log_integrate_1x(m, b)

    def _ppf(self, q, b, m):
        fullq = q * self._vec_integrate_1x(m, b)
        return self._vec_invert_fullq(fullq, b)

    def _pdf(self, x, b, m):
        return np.float_power(x, -(b + 1)) / self._vec_integrate_1x(m, b)

    def _logpdf(self, x, b, m):
        return -(b + 1) * np.log(x) - self._vec_log_integrate_1x(m, b)


extended_pareto = extended_pareto_gen(a=1.0, name="extended_pareto")


# The scipy.stats.zipf distribution implements the discrete version for xmin=1.
# Note that the scale and loc parameters do not fit the purpose of xmin!=1.
# Here s is for lower (included) and m for upper (excluded) bound, i.e.,
# the support is k = xmin, ..., xmax + 1
# Generating samples for exponent b<2 is very tedious, advanced method needs to be implemented.
# For advanced rvs ideas see Section D and Ref. [46] within
#   Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). Power-law distributions in empirical data.
#   Society for Industrial and Applied Mathematics, 51(4), 661–703. https://doi.org/10.1137/070710111

class genzipf_gen(rv_discrete):
    """Generalized Zipf (discrete power-law) distribution, see scipy.stats.zipf too"""

    def _argcheck(self, b, s):
        self.a = int(s)
        return (0 < b) and (1 < s)  # and np.allclose(self.b, s)

    def _cdf(self, x, b, s):
        return 1.0 - zeta(b, x + 1) / zeta(b, s)

    def _logcdf(self, x, b, s):
        return np.log1p(-zeta(b, x + 1) / zeta(b, s))

    def _pmf(self, x, b, s):
        return np.power(x, np.asarray(-b, dtype=float)) / zeta(b, s)

    def _logpmf(self, x, b, s):
        return -b * np.log(x) - np.log(zeta(b, s))


genzipf = genzipf_gen(a=1.0, name="genzipf")


class truncated_zipf_gen(rv_discrete):
    """Truncated Zipf (discrete power-law) distribution, see scipy.stats.zipf too"""

    def _argcheck(self, b, s, m):
        self.a = int(s)
        self.b = int(m)
        return (0 < b) and (0 < s) and (s < m)  # and np.allclose([self.a, self.b], [s, m])

    def _cdf(self, x, b, s, m):
        return 1.0 - (zeta(b, x + 1) - zeta(b, m)) / (zeta(b, s) - zeta(b, m))

    def _logcdf(self, x, b, s, m):
        return np.log1p(-(zeta(b, x + 1) - zeta(b, m)) / (zeta(b, s) - zeta(b, m)))

    def _pmf(self, x, b, s, m):
        return np.power(x, np.asarray(-b, dtype=float)) / (zeta(b, s) - zeta(b, m))

    def _logpmf(self, x, b, s, m):
        return -b * np.log(x) - np.log(zeta(b, s) - zeta(b, m))


truncated_zipf = truncated_zipf_gen(name="truncated_zipf")


def dispatch_logpdf(data, alpha, xmin, xmax, discrete):
    if discrete:
        if np.isinf(xmax):
            ll = genzipf.logpmf(data, alpha, xmin)
        else:
            ll = truncated_zipf.logpmf(data, alpha, xmin, xmax)
    else:
        if np.isinf(xmax):
            ll = pareto.logpdf(data, alpha - 1, scale=xmin)
        else:
            ll = truncated_pareto.logpdf(data, alpha - 1, float(xmax) / xmin, scale=xmin)
    return ll


def dispatch_cdf(data, alpha, xmin, xmax, discrete):
    if discrete:
        if np.isinf(xmax):
            ll = genzipf.cdf(data, alpha, xmin)
        else:
            ll = truncated_zipf.cdf(data, alpha, xmin, xmax)
    else:
        if np.isinf(xmax):
            ll = pareto.cdf(data, alpha - 1, scale=xmin)
        else:
            ll = truncated_pareto.cdf(data, alpha - 1, float(xmax) / xmin, scale=xmin)
    return ll


def dispatch_rvs(alpha, xmin, xmax, discrete, size=1, random_state=None):
    if discrete:
        if np.isinf(xmax):
            ll = genzipf.rvs(alpha, xmin, size=size, random_state=random_state)
        else:
            ll = truncated_zipf.rvs(alpha, xmin, xmax, size=size, random_state=random_state)
    else:
        if np.isinf(xmax):
            ll = pareto.rvs(alpha - 1, scale=xmin, size=size, random_state=random_state)
        else:
            ll = truncated_pareto.rvs(alpha - 1, float(xmax) / xmin, scale=xmin, size=size, random_state=random_state)
    return ll


# copy-pasted from scikit-learn utils/validation.py
def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    If seed is None (or np.random), return the RandomState singleton used
    by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


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
                     clip_low=np.inf, clip_high=0, req_samples=0, debug=False):
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
    elif debug:
        print('Searching in %d combinations, low=%s, high=%s' % (
            int(np.sum(acceptable.astype(int))), low[acceptable][[0, -1]], high[acceptable][[0, -1]]
        ))
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
            if ks.size > 0:
                print('+++ %10d%10f%10d' % (step, ks[:1], len(alpha)))
                print('l', df_edge_to_idx[low].values, 'h', df_edge_to_idx[high].values, 'b', best_idx, )
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


def gen_surrogate_data(n_point, p_cat, low, high, alpha, xmin, xmax, discrete, random_state):
    """
    Generate surrogate data points
    :param n_point: total number of data points
    :param p_cat: probability of `low`, `pareto` and `high` categories
    :param low, high: data to be subsampled (with replacement) for categories `low` and `high`
    :param alpha: exponent of the `pareto` regime
    :param xmin, xmax: boundaries of the `pareto` regime, so that all(low<xmin) and all (xmax<=high)
    :param discrete: use zipf distribution instead of pareto, bool
    :param random_state:
    :return: surrogate sample
    """
    random_state = check_random_state(random_state)
    s_low, s_mid, s_high = multinomial.rvs(n_point, p_cat, random_state=random_state)
    sample = np.empty(n_point, dtype=float)
    if s_low:
        sample[0:s_low] = random_state.choice(low, s_low, replace=True)
    if s_high:
        sample[s_low + s_mid:n_point] = random_state.choice(high, s_high, replace=True)

    sample[s_low:s_low + s_mid] = dispatch_rvs(alpha, xmin, xmax, discrete, size=s_mid, random_state=random_state)

    random_state.shuffle(sample)
    return sample


def gen_surrogate_counts(n_point, p_cat, p_low, p_high, alpha, xmin, xmax, bins, discrete, random_state):
    """
    Generate surrogate hit counts
    :param n_point: total number of data points
    :param p_cat: probability of `low`, `pareto` and `high` categories
    :param p_low, p_high: hit probabilities within categories `low` and `high`
    :param alpha: exponent of the `pareto` regime
    :param xmin, xmax: boundaries of the `pareto` regime, so that all(low<xmin) and all(xmax<=high)
    :param bins: bin boundaries (used for calculating cdf and or binning samples)
    :param discrete: use zipf distribution instead of pareto, bool
    :param random_state:
    :return: surrogate hit counts
    """
    random_state = check_random_state(random_state)
    s_low, s_mid, s_high = multinomial.rvs(n_point, p_cat, random_state=random_state)
    # TODO: the same can be achieved by using the cdf and multinomial sampling, see whether it is stable enough.
    sample = dispatch_rvs(alpha, xmin, xmax, discrete, size=s_mid, random_state=random_state)

    counts, _ = np.histogram(sample, bins)
    if s_low:
        counts[0:len(p_low)] = multinomial.rvs(s_low, p_low, random_state=random_state)
    if s_high:
        counts[len(counts) - len(p_high):len(counts)] = multinomial.rvs(s_low, p_high, random_state=random_state)

    return counts


def p_value_from_ks(ks_collection, ks_data, debug=False):
    ks_collection = np.sort(ks_collection)
    p = 1 - np.searchsorted(ks_collection, ks_data) / float(len(ks_collection))
    if debug:
        print('#ks', ks_collection, ks_data)
    return p


def uncertainty_of_alpha(alpha_collection, alpha, debug=False):
    alpha_collection = np.asarray(alpha_collection)
    sigma = np.sqrt(np.mean((alpha_collection - alpha) ** 2))
    if debug:
        print('#alpha', alpha_collection, alpha)
    return sigma


# Make a section x0<x<x1 of cdf F having f0 = F(x0), f1 = F(x1)
# correspond to p0 = F~(x0), p1 = F~(x1) for x0<x<x1 by linear transformation.
# p0 = s * f0 + d
# p1 = s * f1 + d
# ----------------
# p0 - p1 = s (f0 - f1) --> s = (p0-p1)/(f0-f1)
# d = p0 - s * f0
def _solve_cdf_parts(p0, p1, f0, f1):
    s = float(p0 - p1) / (f0 - f1)
    d = p0 - s * f0
    return s, d


def exp_cutoff_sampler(exponent, xmin, xmax=np.inf, low_cut=0, discrete=False):
    # f1(x) = C1 * exp( -alpha * ( x/x_min - 1 ) ) = D1 * lambda_low * exp ( -lambda_low * (x-x1) )
    # f2(x) = C2 * ( x / x_min ) ^ (-alpha)
    # f3(x) = C3 * exp( -alpha * ( x/x_max - 1 ) ) = D3 * lambda_high * exp ( -lambda_high * (x-x3) )
    # f1(x_min)=f2(x_min)   --> C1 = C2,  D1 = C2 / lambda_low
    # f1'(x_min)=f2'(x_min) --> x1
    # f2(x_max)=f3(x_max)   --> C2 * (x_max/x_min) ^ (-alpha) = C3,  D3 = C2 * (x_max/x_min) ^ (-alpha) / lambda_high
    # f2'(x_min)=f3'(x_min) --> - alpha * C2 / x_min * (x_max/x_min) ^ (-alpha-1) = - C3 * alpha / x_max  -->  x3
    # lambda_low = alpha / x_min,  x1 = x_min / alpha
    # lambda_high = alpha / x_max,  x3 = x_max / alpha
    truncated = xmax != np.inf
    if discrete:
        raise NotImplementedError
        from scipy.stats import poisson
        base_class = rv_discrete
        low_regime = poisson(exponent)
        high_regime = poisson(exponent)
        if truncated:
            scaling_regime = genzipf(exponent, xmin)
            p_high = high_regime.pmf(xmax) / scaling_regime(xmax - 1)
        else:
            scaling_regime = truncated_zipf(exponent, xmin, xmax)
            p_high = 0
        p_low = low_regime.pmf(xmin) / scaling_regime(xmin)

    else:
        from scipy.stats import expon
        base_class = rv_continuous

        scale_low = float(xmin) / exponent  # 1 / lambda
        low_regime = expon(loc=low_cut, scale=scale_low)

        scaling_regime = pareto(exponent - 1, scale=xmin)

        scale_high = float(xmax) / exponent  # 1 / lambda
        high_regime = expon(loc=xmax, scale=scale_high)

        # The first term compensates for the rescaling of cdf
        p_low = low_regime.cdf(xmin) * scaling_regime.pdf(xmin) / low_regime.pdf(xmin)
        p_mid = scaling_regime.cdf(xmax)
        if truncated:
            # We get scaling_regime = pareto(exponent - 1, float(xmax) / xmin, scale=xmin) after rescaling
            p_high = high_regime.sf(xmax) * scaling_regime.pdf(xmax) / high_regime.pdf(xmax)
        else:
            p_high = 0.0

    weights = np.array([p_low, p_mid, p_high])
    p_low, p_mid, p_high = weights / np.sum(weights)

    s_low, d_low = _solve_cdf_parts(0, p_low, 0, low_regime.cdf(xmin))
    s_mid, d_mid = _solve_cdf_parts(p_low, p_low + p_mid, 0, scaling_regime.cdf(xmax))
    s_high, d_high = _solve_cdf_parts(p_low + p_mid, 1, high_regime.cdf(xmax), 1)

    class my_distribution(base_class):
        def _cdf(self, x):
            return np.select([x < xmin, x <= xmax, x > xmax],
                             [d_low + low_regime.cdf(x) * s_low, d_mid + scaling_regime.cdf(x) * s_mid,
                              d_high + high_regime.cdf(x) * s_high])

        def _ppf(self, q):
            return np.select([q < p_low, q <= p_low + p_mid, q > p_low + p_mid],
                             [low_regime.ppf((q - d_low) / s_low), scaling_regime.ppf((q - d_mid) / s_mid),
                              high_regime.ppf((q - d_high) / s_high)])

    my_rv = my_distribution(a=low_cut, name='synthetic')
    my_rv.weight_distribution = p_low, p_mid, p_high
    return my_rv

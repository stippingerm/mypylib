# Copyright (c) Gary Strangman.  All rights reserved
#
# Disclaimer
#
# This software is provided "as-is".  There are no expressed or implied
# warranties of any kind, including, but not limited to, the warranties
# of merchantability and fitness for a given application.  In no event
# shall Gary Strangman be liable for any direct, indirect, incidental,
# special, exemplary or consequential damages (including, but not limited
# to, loss of use, data or profits, or business interruption) however
# caused and on any theory of liability, whether in contract, strict
# liability or tort (including negligence or otherwise) arising in any way
# out of the use of this software, even if advised of the possibility of
# such damage.
#

#
# Heavily adapted for use by SciPy 2002 by Travis Oliphant
"""
A collection of basic statistical functions for python.  The function
names appear below.

 Some scalar functions defined here are also available in the scipy.special
 package where they work on arbitrary sized arrays.

Disclaimers:  The function list is obviously incomplete and, worse, the
functions are not optimized.  All functions have been tested (some more
so than others), but they are far from bulletproof.  Thus, as with any
free software, no warranty or guarantee is expressed or implied. :-)  A
few extra functions that don't appear in the list below can be found by
interested treasure-hunters.  These functions don't necessarily have
both list and array versions but were deemed useful.

Central Tendency
----------------
.. autosummary::
   :toctree: generated/

    gmean
    hmean
    mode

Moments
-------
.. autosummary::
   :toctree: generated/

    moment
    variation
    skew
    kurtosis
    normaltest

Moments Handling NaN:

.. autosummary::
   :toctree: generated/

    nanmean
    nanmedian
    nanstd

Altered Versions
----------------
.. autosummary::
   :toctree: generated/

    tmean
    tvar
    tstd
    tsem
    describe

Frequency Stats
---------------
.. autosummary::
   :toctree: generated/

    itemfreq
    scoreatpercentile
    percentileofscore
    histogram
    cumfreq
    relfreq

Variability
-----------
.. autosummary::
   :toctree: generated/

    obrientransform
    signaltonoise
    sem

Trimming Functions
------------------
.. autosummary::
   :toctree: generated/

   threshold
   trimboth
   trim1

Correlation Functions
---------------------
.. autosummary::
   :toctree: generated/

   pearsonr
   fisher_exact
   spearmanr
   pointbiserialr
   kendalltau
   linregress

Inferential Stats
-----------------
.. autosummary::
   :toctree: generated/

   ttest_1samp
   ttest_ind
   ttest_rel
   chisquare
   power_divergence
   ks_2samp
   mannwhitneyu
   ranksums
   wilcoxon
   kruskal
   friedmanchisquare

Probability Calculations
------------------------
.. autosummary::
   :toctree: generated/

   chisqprob
   zprob
   fprob
   betai

ANOVA Functions
---------------
.. autosummary::
   :toctree: generated/

   f_oneway
   f_value

Support Functions
-----------------
.. autosummary::
   :toctree: generated/

   ss
   square_of_sums
   rankdata

References
----------
.. [CRCProbStat2000] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
   Probability and Statistics Tables and Formulae. Chapman & Hall: New
   York. 2000.

"""

from __future__ import division, print_function, absolute_import

import warnings
import math

# from scipy.lib.six import xrange

# friedmanchisquare patch uses python sum
pysum = sum  # save it before it gets overwritten

# Scipy imports.
# from scipy.lib.six import callable, string_types
from numpy import array, asarray, ma, zeros, sum
import scipy.special as special
import scipy.linalg as linalg
import numpy as np

# from scipy.stats import futil
from scipy.stats import distributions

# from scipy.stats._rank import rankdata, tiecorrect

__all__ = ['ttest_1samp', 'ttest_ind', 'ttest_rel'
           ]


def _chk_asarray(a, axis):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis
    return a, outaxis


def _chk2_asarray(a, b, axis):
    if axis is None:
        a = np.ravel(a)
        b = np.ravel(b)
        outaxis = 0
    else:
        a = np.asarray(a)
        b = np.asarray(b)
        outaxis = axis
    return a, b, outaxis


def _chk_effective_sample_size(n, effective):
    if effective is None:
        out = n
    else:
        out = float(effective)
    return out


#####################################
#####  INFERENTIAL STATISTICS  #####
#####################################

def ttest_1samp(a, popmean, axis=0, n_effective=None):
    """
    Calculates the T-test for the mean of ONE group of scores.

    This is a two-sided test for the null hypothesis that the expected value
    (mean) of a sample of independent observations `a` is equal to the given
    population mean, `popmean`.

    Parameters
    ----------
    a : array_like
        sample observation
    popmean : float or array_like
        expected value in null hypothesis, if array_like than it must have the
        same shape as `a` excluding the axis dimension
    axis : int, optional, (default axis=0)
        Axis can equal None (ravel array first), or an integer (the axis
        over which to operate on a).

    Returns
    -------
    t : float or array
        t-statistic
    prob : float or array
        two-tailed p-value

    Examples
    --------
    >>> from scipy import stats

    >>> np.random.seed(7654567)  # fix seed to get the same result
    >>> rvs = stats.norm.rvs(loc=5, scale=10, size=(50,2))

    Test if mean of random sample is equal to true mean, and different mean.
    We reject the null hypothesis in the second case and don't reject it in
    the first case.

    >>> stats.ttest_1samp(rvs,5.0)
    (array([-0.68014479, -0.04323899]), array([ 0.49961383,  0.96568674]))
    >>> stats.ttest_1samp(rvs,0.0)
    (array([ 2.77025808,  4.11038784]), array([ 0.00789095,  0.00014999]))

    Examples using axis and non-scalar dimension for population mean.

    >>> stats.ttest_1samp(rvs,[5.0,0.0])
    (array([-0.68014479,  4.11038784]), array([  4.99613833e-01,   1.49986458e-04]))
    >>> stats.ttest_1samp(rvs.T,[5.0,0.0],axis=1)
    (array([-0.68014479,  4.11038784]), array([  4.99613833e-01,   1.49986458e-04]))
    >>> stats.ttest_1samp(rvs,[[5.0],[0.0]])
    (array([[-0.68014479, -0.04323899],
           [ 2.77025808,  4.11038784]]), array([[  4.99613833e-01,   9.65686743e-01],
           [  7.89094663e-03,   1.49986458e-04]]))

    """
    a, axis = _chk_asarray(a, axis)
    n = a.shape[axis]
    df = n - 1

    d = np.mean(a, axis) - popmean
    v = np.var(a, axis, ddof=1)
    n_effective = _chk_effective_sample_size(n, n_effective)
    denom = np.sqrt(v / float(n_effective))

    t = np.divide(d, denom)
    t, prob = _ttest_finish(df, t)

    return t, prob


def _ttest_finish(df,t):
    """Common code between all 3 t-test functions."""
    prob = distributions.t.sf(np.abs(t), df) * 2  # use np.abs to get upper tail
    if t.ndim == 0:
        t = t[()]

    return t, prob


def ttest_ind(a, b, axis=0, equal_var=True):
    """
    Calculates the T-test for the means of TWO INDEPENDENT samples of scores.

    This is a two-sided test for the null hypothesis that 2 independent samples
    have identical average (expected) values. This test assumes that the
    populations have identical variances.

    Parameters
    ----------
    a, b : array_like
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).
    axis : int, optional
        Axis can equal None (ravel array first), or an integer (the axis
        over which to operate on a and b).
    equal_var : bool, optional
        If True (default), perform a standard independent 2 sample test
        that assumes equal population variances [1]_.
        If False, perform Welch's t-test, which does not assume equal
        population variance [2]_.

        .. versionadded:: 0.11.0

    Returns
    -------
    t : float or array
        The calculated t-statistic.
    prob : float or array
        The two-tailed p-value.

    Notes
    -----
    We can use this test, if we observe two independent samples from
    the same or different population, e.g. exam scores of boys and
    girls or of two ethnic groups. The test measures whether the
    average (expected) value differs significantly across samples. If
    we observe a large p-value, for example larger than 0.05 or 0.1,
    then we cannot reject the null hypothesis of identical average scores.
    If the p-value is smaller than the threshold, e.g. 1%, 5% or 10%,
    then we reject the null hypothesis of equal averages.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/T-test#Independent_two-sample_t-test

    .. [2] http://en.wikipedia.org/wiki/Welch%27s_t_test

    Examples
    --------
    >>> from scipy import stats
    >>> np.random.seed(12345678)

    Test with sample with identical means:

    >>> rvs1 = stats.norm.rvs(loc=5,scale=10,size=500)
    >>> rvs2 = stats.norm.rvs(loc=5,scale=10,size=500)
    >>> stats.ttest_ind(rvs1,rvs2)
    (0.26833823296239279, 0.78849443369564776)
    >>> stats.ttest_ind(rvs1,rvs2, equal_var = False)
    (0.26833823296239279, 0.78849452749500748)

    `ttest_ind` underestimates p for unequal variances:

    >>> rvs3 = stats.norm.rvs(loc=5, scale=20, size=500)
    >>> stats.ttest_ind(rvs1, rvs3)
    (-0.46580283298287162, 0.64145827413436174)
    >>> stats.ttest_ind(rvs1, rvs3, equal_var = False)
    (-0.46580283298287162, 0.64149646246569292)

    When n1 != n2, the equal variance t-statistic is no longer equal to the
    unequal variance t-statistic:

    >>> rvs4 = stats.norm.rvs(loc=5, scale=20, size=100)
    >>> stats.ttest_ind(rvs1, rvs4)
    (-0.99882539442782481, 0.3182832709103896)
    >>> stats.ttest_ind(rvs1, rvs4, equal_var = False)
    (-0.69712570584654099, 0.48716927725402048)

    T-test with different means, variance, and n:

    >>> rvs5 = stats.norm.rvs(loc=8, scale=20, size=100)
    >>> stats.ttest_ind(rvs1, rvs5)
    (-1.4679669854490653, 0.14263895620529152)
    >>> stats.ttest_ind(rvs1, rvs5, equal_var = False)
    (-0.94365973617132992, 0.34744170334794122)

    """
    a, b, axis = _chk2_asarray(a, b, axis)
    if a.size == 0 or b.size == 0:
        return (np.nan, np.nan)

    v1 = np.var(a, axis, ddof=1)
    v2 = np.var(b, axis, ddof=1)
    n1 = a.shape[axis]
    n2 = b.shape[axis]

    if (equal_var):
        df = n1 + n2 - 2
        svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / float(df)
        denom = np.sqrt(svar * (1.0 / n1 + 1.0 / n2))
    else:
        vn1 = v1 / n1
        vn2 = v2 / n2
        df = ((vn1 + vn2)**2) / ((vn1**2) / (n1 - 1) + (vn2**2) / (n2 - 1))

        # If df is undefined, variances are zero (assumes n1 > 0 & n2 > 0).
        # Hence it doesn't matter what df is as long as it's not NaN.
        df = np.where(np.isnan(df), 1, df)
        denom = np.sqrt(vn1 + vn2)

    d = np.mean(a, axis) - np.mean(b, axis)
    t = np.divide(d, denom)
    t, prob = _ttest_finish(df, t)

    return t, prob


def ttest_rel(a, b, axis=0, n_effective=None):
    """
    Calculates the T-test on TWO RELATED samples of scores, a and b.

    This is a two-sided test for the null hypothesis that 2 related or
    repeated samples have identical average (expected) values.

    Parameters
    ----------
    a, b : array_like
        The arrays must have the same shape.
    axis : int, optional, (default axis=0)
        Axis can equal None (ravel array first), or an integer (the axis
        over which to operate on a and b).

    Returns
    -------
    t : float or array
        t-statistic
    prob : float or array
        two-tailed p-value

    Notes
    -----
    Examples for the use are scores of the same set of student in
    different exams, or repeated sampling from the same units. The
    test measures whether the average score differs significantly
    across samples (e.g. exams). If we observe a large p-value, for
    example greater than 0.05 or 0.1 then we cannot reject the null
    hypothesis of identical average scores. If the p-value is smaller
    than the threshold, e.g. 1%, 5% or 10%, then we reject the null
    hypothesis of equal averages. Small p-values are associated with
    large t-statistics.

    References
    ----------
    http://en.wikipedia.org/wiki/T-test#Dependent_t-test

    Examples
    --------
    >>> from scipy import stats
    >>> np.random.seed(12345678) # fix random seed to get same numbers

    >>> rvs1 = stats.norm.rvs(loc=5,scale=10,size=500)
    >>> rvs2 = (stats.norm.rvs(loc=5,scale=10,size=500) +
    ...         stats.norm.rvs(scale=0.2,size=500))
    >>> stats.ttest_rel(rvs1,rvs2)
    (0.24101764965300962, 0.80964043445811562)
    >>> rvs3 = (stats.norm.rvs(loc=8,scale=10,size=500) +
    ...         stats.norm.rvs(scale=0.2,size=500))
    >>> stats.ttest_rel(rvs1,rvs3)
    (-3.9995108708727933, 7.3082402191726459e-005)

    """
    a, b, axis = _chk2_asarray(a, b, axis)
    if a.shape[axis] != b.shape[axis]:
        raise ValueError('unequal length arrays')

    if a.size == 0 or b.size == 0:
        return (np.nan, np.nan)

    n = a.shape[axis]
    df = float(n - 1)

    d = (a - b).astype(np.float64)
    v = np.var(d, axis, ddof=1)
    n_effective = _chk_effective_sample_size(n, n_effective)
    dm = np.mean(d, axis)
    denom = np.sqrt(v / float(n_effective))

    t = np.divide(dm, denom)
    t, prob = _ttest_finish(df, t)

    return t, prob


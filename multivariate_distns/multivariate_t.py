#
# Author: Marcell Stippinger 2018
#
from __future__ import division, print_function, absolute_import

import math
import numpy as np
import scipy.linalg
from scipy.misc import doccer
from scipy.special import gammaln, psi, multigammaln, xlogy, entr, log1p
from scipy._lib._util import check_random_state
from scipy.linalg.blas import drot
from scipy.stats._multivariate import _PSD, _squeeze_output, _eigvalsh_to_eps, _pinv_1d

from scipy.stats._discrete_distns import binom
from scipy.stats import mvn

__all__ = ['multivariate_t',
           ]

_LOG_2PI = np.log(2 * np.pi)
_LOG_2 = np.log(2)
_LOG_PI = np.log(np.pi)


_doc_random_state = """\
random_state : None or int or np.random.RandomState instance, optional
    If int or RandomState, use it for drawing the random variates.
    If None (or np.random), the global np.random state is used.
    Default is None.
"""


class multi_rv_generic(object):
    """
    Class which encapsulates common functionality between all multivariate
    distributions.

    """
    def __init__(self, seed=None):
        super(multi_rv_generic, self).__init__()
        self._random_state = check_random_state(seed)

    @property
    def random_state(self):
        """ Get or set the RandomState object for generating random variates.

        This can be either None or an existing RandomState object.

        If None (or np.random), use the RandomState singleton used by np.random.
        If already a RandomState instance, use it.
        If an int, use a new RandomState instance seeded with seed.

        """
        return self._random_state

    @random_state.setter
    def random_state(self, seed):
        self._random_state = check_random_state(seed)

    def _get_random_state(self, random_state):
        if random_state is not None:
            return check_random_state(random_state)
        else:
            return self._random_state


class multi_rv_frozen(object):
    """
    Class which encapsulates common functionality between all frozen
    multivariate distributions.
    """
    @property
    def random_state(self):
        return self._dist._random_state

    @random_state.setter
    def random_state(self, seed):
        self._dist._random_state = check_random_state(seed)


_mvt_doc_default_callparams = """\
df : float
    Degrees of freedom of the distributon
mean : array_like, optional
    Mean (i.e., location) of the distribution (default zero)
cov : array_like, optional
    Shape matrix of the distribution (default one)
allow_singular : bool, optional
    Whether to allow a singular covariance matrix.  (Default: False)
"""

_mvt_doc_callparams_note = \
    """Setting the parameter `mean` to `None` is equivalent to having `mean`
    be the zero-vector. The parameter `cov` can be a scalar, in which case
    the covariance matrix is the identity times that value, a vector of
    diagonal entries for the covariance matrix, or a two-dimensional
    array_like.
    """

_mvt_doc_frozen_callparams = ""

_mvt_doc_frozen_callparams_note = \
    """See class definition for a detailed description of parameters."""

mvt_docdict_params = {
    '_mvt_doc_default_callparams': _mvt_doc_default_callparams,
    '_mvt_doc_callparams_note': _mvt_doc_callparams_note,
    '_doc_random_state': _doc_random_state
}

mvt_docdict_noparams = {
    '_mvt_doc_default_callparams': _mvt_doc_frozen_callparams,
    '_mvt_doc_callparams_note': _mvt_doc_frozen_callparams_note,
    '_doc_random_state': _doc_random_state
}

class multivariate_t_gen(multi_rv_generic):
    r"""
    A multivariate Student's T random variable.

    The `mean` keyword specifies the mean. The `cov` keyword specifies the
    covariance matrix.

    Methods
    -------
    ``pdf(x, df, mean=None, cov=1, allow_singular=False)``
        Probability density function.
    ``logpdf(x, df, mean=None, cov=1, allow_singular=False)``
        Log of the probability density function.
    ``cdf(x, df, mean=None, cov=1, allow_singular=False, maxpts=1000000*dim, abseps=1e-5, releps=1e-5)``
        Cumulative distribution function.
    ``logcdf(x, df, mean=None, cov=1, allow_singular=False, maxpts=1000000*dim, abseps=1e-5, releps=1e-5)``
        Log of the cumulative distribution function.
    ``rvs(df, mean=None, cov=1, size=1, random_state=None)``
        Draw random samples from a multivariate Student's T distribution.
    ``entropy()``
        Compute the differential entropy of the multivariate Student's T.

    Parameters
    ----------
    x : array_like
        Quantiles, with the last axis of `x` denoting the components.
    %(_mvt_doc_default_callparams)s
    %(_doc_random_state)s

    Alternatively, the object may be called (as a function) to fix the mean
    and covariance parameters, returning a "frozen" multivariate Student's T
    random variable:

    rv = multivariate_t(mean=None, cov=1, allow_singular=False)
        - Frozen object with the same methods but holding the given
          mean and covariance fixed.

    Notes
    -----
    %(_mvt_doc_callparams_note)s

    The covariance matrix `cov` must be a (symmetric) positive
    semi-definite matrix. The determinant and inverse of `cov` are computed
    as the pseudo-determinant and pseudo-inverse, respectively, so
    that `cov` does not need to have full rank.

    The probability density function for `multivariate_t` is

    .. math::

        f(x) = {\frac {\Gamma [(\nu +p)/2]}{\Gamma (\nu /2)\nu ^{p/2}\pi ^{p/2} |{\boldsymbol {\Sigma }}|^{1/2}}}
               \left[1+{\frac {1}{\nu }}({\mathbf {x} }-{\boldsymbol {\mu }})^{\rm {T}}
                    {\boldsymbol {\Sigma }}^{-1}({\mathbf {x} }-{\boldsymbol {\mu }})\right]^{-(\nu +p)/2},

    where :math:`\nu` is the degrees of freedom, :math:`\mu` is the mean, :math:`\Sigma` the covariance matrix,
    and :math:`k` is the dimension of the space where :math:`x` takes values.

    .. versionadded:: 1.0.0

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import multivariate_t

    >>> x = np.linspace(0, 5, 10, endpoint=False)
    >>> y = multivariate_t.pdf(x, df=2, mean=2.5, cov=0.5); y
    array([ 0.00108914,  0.01033349,  0.05946514,  0.20755375,  0.43939129,
            0.56418958,  0.43939129,  0.20755375,  0.05946514,  0.01033349])
    >>> fig1 = plt.figure()
    >>> ax = fig1.add_subplot(111)
    >>> ax.plot(x, y)

    The input quantiles can be any shape of array, as long as the last
    axis labels the components.  This allows us for instance to
    display the frozen pdf for a non-isotropic random variable in 2D as
    follows:

    >>> x, y = np.mgrid[-1:1:.01, -1:1:.01]
    >>> pos = np.dstack((x, y))
    >>> rv = multivariate_t(3, [0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
    >>> fig2 = plt.figure()
    >>> ax2 = fig2.add_subplot(111)
    >>> ax2.contourf(x, y, rv.pdf(pos))

    """

    def __init__(self, seed=None):
        super(multivariate_t_gen, self).__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__, mvt_docdict_params)

    def __call__(self, mean=None, cov=1, allow_singular=False, seed=None):
        """
        Create a frozen multivariate Student's T distribution.

        See `multivariate_t_frozen` for more information.

        """
        return multivariate_t_frozen(mean, cov,
                                          allow_singular=allow_singular,
                                          seed=seed)

    def _process_parameters(self, dim, df, mean, cov):
        """
        Infer dimensionality from mean or covariance matrix, ensure that
        mean and covariance are full vector resp. matrix.
        Ensure that ddof is positive
        """

        if not (np.isscalar(df) and (0 < df)):
            raise ValueError("Degrees of freedom must be a positive scalar.")
        df = np.float(df)

        # Try to infer dimensionality
        if dim is None:
            if mean is None:
                if cov is None:
                    dim = 1
                else:
                    cov = np.asarray(cov, dtype=float)
                    if cov.ndim < 2:
                        dim = 1
                    else:
                        dim = cov.shape[0]
            else:
                mean = np.asarray(mean, dtype=float)
                dim = mean.size
        else:
            if not np.isscalar(dim):
                raise ValueError("Dimension of random variable must be a scalar.")

        # Check input sizes and return full arrays for mean and cov if necessary
        if mean is None:
            mean = np.zeros(dim)
        mean = np.asarray(mean, dtype=float)

        if cov is None:
            cov = 1.0
        cov = np.asarray(cov, dtype=float)

        if dim == 1:
            mean.shape = (1,)
            cov.shape = (1, 1)

        if mean.ndim != 1 or mean.shape[0] != dim:
            raise ValueError("Array 'mean' must be a vector of length %d." % dim)
        if cov.ndim == 0:
            cov = cov * np.eye(dim)
        elif cov.ndim == 1:
            cov = np.diag(cov)
        elif cov.ndim == 2 and cov.shape != (dim, dim):
            rows, cols = cov.shape
            if rows != cols:
                msg = ("Array 'cov' must be square if it is two dimensional,"
                       " but cov.shape = %s." % str(cov.shape))
            else:
                msg = ("Dimension mismatch: array 'cov' is of shape %s,"
                       " but 'mean' is a vector of length %d.")
                msg = msg % (str(cov.shape), len(mean))
            raise ValueError(msg)
        elif cov.ndim > 2:
            raise ValueError("Array 'cov' must be at most two-dimensional,"
                             " but cov.ndim = %d" % cov.ndim)

        return dim, df, mean, cov

    def _process_quantiles(self, x, dim):
        """
        Adjust quantiles array so that last axis labels the components of
        each data point.

        """
        x = np.asarray(x, dtype=float)

        if x.ndim == 0:
            x = x[np.newaxis]
        elif x.ndim == 1:
            if dim == 1:
                x = x[:, np.newaxis]
            else:
                x = x[np.newaxis, :]

        return x

    def _logpdf(self, x, df, mean, prec_U, log_det_cov, rank):
        """
        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability
            density function
        df : float
            Degrees of freedom of the distribution
        mean : ndarray
            Mean of the distribution
        prec_U : ndarray
            A decomposition such that np.dot(prec_U, prec_U.T)
            is the precision matrix, i.e. inverse of the covariance matrix.
        log_det_cov : float
            Logarithm of the determinant of the covariance matrix
        rank : int
            Rank of the covariance matrix.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.
        :param df:

        """
        dev = x - mean
        maha = log1p(np.sum(np.square(np.dot(dev, prec_U)), axis=-1) / df)
        pref = gammaln(0.5 * (df+rank)) - gammaln(0.5 * df)
        return pref -0.5 * (rank * (df+_LOG_PI) + log_det_cov + (df+rank) * maha)

    def logpdf(self, x, df, mean=None, cov=1, allow_singular=False):
        """
        Log of the multivariate Student's T probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvt_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray or scalar
            Log of the probability density function evaluated at `x`

        Notes
        -----
        %(_mvt_doc_callparams_note)s

        """
        dim, df, mean, cov = self._process_parameters(None, df, mean, cov)
        x = self._process_quantiles(x, dim)
        psd = _PSD(cov, allow_singular=allow_singular)
        out = self._logpdf(x, df, mean, psd.U, psd.log_pdet, psd.rank)
        return _squeeze_output(out)

    def pdf(self, x, df, mean=None, cov=1, allow_singular=False):
        """
        Multivariate Student's T probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvt_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray or scalar
            Probability density function evaluated at `x`

        Notes
        -----
        %(_mvt_doc_callparams_note)s

        """
        dim, df, mean, cov = self._process_parameters(None, df, mean, cov)
        x = self._process_quantiles(x, dim)
        psd = _PSD(cov, allow_singular=allow_singular)
        out = np.exp(self._logpdf(x, df, mean, psd.U, psd.log_pdet, psd.rank))
        return _squeeze_output(out)

    def _cdf(self, x, df, mean, cov, maxpts, abseps, releps):
        """
        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the cumulative distribution function.
        df : float
            Degrees of freedom of the distribution
        mean : ndarray
            Mean of the distribution
        cov : array_like
            Covariance matrix of the distribution
        maxpts: integer
            The maximum number of points to use for integration
        abseps: float
            Absolute error tolerance
        releps: float
            Relative error tolerance

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'cdf' instead.

        .. versionadded:: 1.0.0

        """
        raise NotImplementedError
        lower = np.full(mean.shape, -np.inf)
        # mvnun expects 1-d arguments, so process points sequentially
        func1d = lambda x_slice: mvn.mvnun(lower, x_slice, mean, cov,
                                           maxpts, abseps, releps)[0]
        out = np.apply_along_axis(func1d, -1, x)
        return _squeeze_output(out)

    def logcdf(self, x, df, mean=None, cov=1, allow_singular=False, maxpts=None,
               abseps=1e-5, releps=1e-5):
        """
        Log of the multivariate Student's T cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvt_doc_default_callparams)s
        maxpts: integer, optional
            The maximum number of points to use for integration
            (default `1000000*dim`)
        abseps: float, optional
            Absolute error tolerance (default 1e-5)
        releps: float, optional
            Relative error tolerance (default 1e-5)

        Returns
        -------
        cdf : ndarray or scalar
            Log of the cumulative distribution function evaluated at `x`

        Notes
        -----
        %(_mvt_doc_callparams_note)s

        .. versionadded:: 1.0.0

        """
        dim, df, mean, cov = self._process_parameters(None, df, mean, cov)
        x = self._process_quantiles(x, dim)
        # Use _PSD to check covariance matrix
        _PSD(cov, allow_singular=allow_singular)
        if not maxpts:
            maxpts = 1000000 * dim
        out = np.log(self._cdf(x, df, mean, cov, maxpts, abseps, releps))
        return out

    def cdf(self, x, df, mean=None, cov=1, allow_singular=False, maxpts=None,
            abseps=1e-5, releps=1e-5):
        """
        Multivariate Student's T cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvt_doc_default_callparams)s
        maxpts: integer, optional
            The maximum number of points to use for integration
            (default `1000000*dim`)
        abseps: float, optional
            Absolute error tolerance (default 1e-5)
        releps: float, optional
            Relative error tolerance (default 1e-5)

        Returns
        -------
        cdf : ndarray or scalar
            Cumulative distribution function evaluated at `x`

        Notes
        -----
        %(_mvt_doc_callparams_note)s

        .. versionadded:: 1.0.0

        """
        dim, df, mean, cov = self._process_parameters(None, df, mean, cov)
        x = self._process_quantiles(x, dim)
        # Use _PSD to check covariance matrix
        _PSD(cov, allow_singular=allow_singular)
        if not maxpts:
            maxpts = 1000000 * dim
        out = self._cdf(x, df, mean, cov, maxpts, abseps, releps)
        return out

    def rvs(self, df, mean=None, cov=1, size=1, random_state=None):
        """
        Draw random samples from a multivariate Student's T distribution.

        Parameters
        ----------
        %(_mvt_doc_default_callparams)s
        size : integer, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the random variable.

        Notes
        -----
        %(_mvt_doc_callparams_note)s

        Sampling is based on the observation that
        .. math::

            X = \mu + \frac{Y}{\sqrt{\frac{U}{\nu}}} = \mu + Y\sqrt{\frac{\nu}{U}}

        has a math:`t_\nu({\boldsymbol {\mu}}, {\boldsymbol {\Sigma}})` distribution when
        math:`Y` has a math:`N(0,{\boldsymbol {\Sigma}})` distribution and independently
        math:`U` has a math:`\chi^2_{\boldsymbol {\nu}}` distribution.

        """
        dim, df, mean, cov = self._process_parameters(None, df, mean, cov)

        random_state = self._get_random_state(random_state)
        # Y, shape (*size, dim)
        norm_comp = random_state.multivariate_normal(np.zeros(dim), cov, size)
        # U, shape size
        chi2_comp = random_state.chisquare(df, size)

        out = mean + norm_comp * np.sqrt(df/chi2_comp[...,np.newaxis])
        return _squeeze_output(out)

    def entropy(self, df, mean=None, cov=1):
        """
        Compute the differential entropy of the multivariate Student's T.

        Parameters
        ----------
        %(_mvt_doc_default_callparams)s

        Returns
        -------
        h : scalar
            Entropy of the multivariate Student's T distribution

        Notes
        -----
        %(_mvt_doc_callparams_note)s

        """
        dim, df, mean, cov = self._process_parameters(None, df, mean, cov)
        raise NotImplementedError
        _, logdet = np.linalg.slogdet(2 * np.pi * np.e * cov)
        return 0.5 * logdet


multivariate_t = multivariate_t_gen()


class multivariate_t_frozen(multi_rv_frozen):
    def __init__(self, df, mean=None, cov=1, allow_singular=False, seed=None,
                 maxpts=None, abseps=1e-5, releps=1e-5):
        """
        Create a frozen multivariate Student's T distribution.

        Parameters
        ----------
        df : float
            Degrees of freedom of the distribution
        mean : array_like, optional
            Mean of the distribution (default zero)
        cov : array_like, optional
            Covariance matrix of the distribution (default one)
        allow_singular : bool, optional
            If this flag is True then tolerate a singular
            covariance matrix (default False).
        seed : None or int or np.random.RandomState instance, optional
            This parameter defines the RandomState object to use for drawing
            random variates.
            If None (or np.random), the global np.random state is used.
            If integer, it is used to seed the local RandomState instance
            Default is None.
        maxpts: integer, optional
            The maximum number of points to use for integration of the
            cumulative distribution function (default `1000000*dim`)
        abseps: float, optional
            Absolute error tolerance for the cumulative distribution function
            (default 1e-5)
        releps: float, optional
            Relative error tolerance for the cumulative distribution function
            (default 1e-5)

        Examples
        --------
        When called with the default parameters, this will create a 1D random
        variable with mean 0 and covariance 1:

        >>> from scipy.stats import multivariate_t
        >>> r = multivariate_t(3)
        >>> r.df
        3.0
        >>> r.mean
        array([ 0.])
        >>> r.cov
        array([[1.]])

        """
        self._dist = multivariate_t_gen(seed)
        self.dim, self.df, self.mean, self.cov = self._dist._process_parameters(None, df, mean, cov)
        self.cov_info = _PSD(self.cov, allow_singular=allow_singular)
        if not maxpts:
            maxpts = 1000000 * self.dim
        self.maxpts = maxpts
        self.abseps = abseps
        self.releps = releps

    def logpdf(self, x):
        x = self._dist._process_quantiles(x, self.dim)
        out = self._dist._logpdf(x, self.df, self.mean, self.cov_info.U, self.cov_info.log_pdet, self.cov_info.rank)
        return _squeeze_output(out)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def logcdf(self, x):
        return np.log(self.cdf(x))

    def cdf(self, x):
        x = self._dist._process_quantiles(x, self.dim)
        out = self._dist._cdf(x, self.df, self.mean, self.cov, self.maxpts, self.abseps,
                              self.releps)
        return _squeeze_output(out)

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(self.df, self.mean, self.cov, size, random_state)

    def entropy(self):
        """
        Computes the differential entropy of the multivariate Student's T.

        Returns
        -------
        h : scalar
            Entropy of the multivariate Student's T distribution

        """
        raise NotImplementedError
        log_pdet = self.cov_info.log_pdet
        rank = self.cov_info.rank
        return 0.5 * (rank * (_LOG_2PI + 1) + log_pdet)


# Set frozen generator docstrings from corresponding docstrings in
# multivariate_t_gen and fill in default strings in class docstrings
for name in ['logpdf', 'pdf', 'logcdf', 'cdf', 'rvs']:
    method = multivariate_t_gen.__dict__[name]
    method_frozen = multivariate_t_frozen.__dict__[name]
    method_frozen.__doc__ = doccer.docformat(method.__doc__, mvt_docdict_noparams)
    method.__doc__ = doccer.docformat(method.__doc__, mvt_docdict_params)


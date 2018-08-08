"""General copula."""

# Author: Marcell Stippinger
# Inspired by and based on sci-kit learn mixture models
# License: BSD 3 clause

# See: Berkes, Wood & Pillow (2009). Characterizing neural dependencies with copula models. NIPS 21, 129–136.
# Retrieved from http://papers.nips.cc/paper/3593-characterizing-neural-dependencies-with-copula-models.pdf
# See also: https://www.vosesoftware.com/riskwiki/Archimedeancopulas-theClaytonFrankandGumbel.php
# Two-variate implementation of Archimedean copulas is available in copulalib but it is far from complete.
# The continuation of copulalib is https://pypi.org/project/ambhas/ still bivariate copulas only.
# Sampling from cdf is difficult but feasible for Archimedean copulas: https://stackoverflow.com/a/50981679
# http://pluto.huji.ac.il/~galelidan/Copulas%20and%20Machine%20Learning.pdf

import numpy as np

from scipy import stats
from scipy.stats import rv_continuous, norm, gaussian_kde
from scipy.stats._multivariate import multivariate_normal_gen, multi_rv_generic
from scipy.stats._distn_infrastructure import argsreduce
from sklearn.base import TransformerMixin
from scipy.misc import derivative
from scipy.special import gamma, gammaln
from abc import abstractmethod


def _exist(*unused):
    """Check if the argument exists: in fact the check happens
       when calling this function and this function has nothing
       to do with it. It's merely a way to enforce the check.

       Parameters
       ----------
       unused : any_type
           list all variables as argument to be checked

       Notes
       -----
       To pass code inspection the function has to deal with its input
       therefore it deletes its own reference provided to the top level
       variables but this should not imply any side effects.
    """
    del unused
    pass


def _broadcast_shapes(a, b, align_as_numpy=True):
    a, b = np.atleast_1d(a, b)
    if a.ndim > 1 or b.ndim > 1:
        raise ValueError('broadcasting of single shapes is supported only')
    result_shape = np.maximum(len(a), len(b))
    padded_a = np.ones(result_shape, dtype=int)
    padded_b = np.ones(result_shape, dtype=int)
    if align_as_numpy:
        if len(a):
            padded_a[-len(a):] = a
        if len(b):
            padded_b[-len(b):] = b
    else:
        if len(a):
            padded_a[:len(a)] = a
        if len(b):
            padded_b[:len(b)] = b
    to_change_a = (padded_a != padded_b) & (padded_b != 1)
    if np.any(padded_a[to_change_a] != 1):
        raise ValueError('shapes %s and %s could not be broadcast together' % (a, b))
    padded_a[to_change_a] = padded_b[to_change_a]
    return padded_a


# ------------------------
#  Parameters of a copula
# ------------------------
#
# Training data alignment (like scipy.stats.multivariate_normal):
# n_observation x n_comp
# optionally n_obs_ax1 x ... x n_obs_ax_N x n_comp
#
# Marginals:
# n_comp x n_stat_param
# TODO: allow shape 1 x n_stat_param if iid_marginals
#
# Joint is expected in the following form:
# n_comp x n_joint_param_per_comp
# e.g. for a gaussian without compression: n_comp x (n_comp + 1)
# But we cannot rely on this therefore it is stored as a simple list.
#
# However parameters have a well-defined shape they are not suitable for
# scipy.stat.rv_continuous because it tries to broadcast parameters with
# data.
# To circumvent this there are two options:
# 1) Store parameters in a way less efficient than np.array,
#    e.g., hide details using a n_comp long list
# 2) Use multi_rv_generic which does not try to validate data.
#    This way one loses the automatic shortcuts that infer functions not
#    provided explicitly, e.g., rv_continuous.sf = 1 - rv_continuous.isf
#    To some point this is reasonable because the quadratures that are
#    designed for one dimension do not work in two or higher dimensions.
#
# ------------------
#  How copula works
# ------------------
#
# The copula is a multivariate distribution with support of $[0,1]^n$.
# To achieve this one uses an arbitrary joint distribution $F_n$ and its
# marginal version $F_1$ (they have the support $I^n$ and $I$ respectively)
# and combines them to obtain the distribution of the copula:
#
#  $$ C(u_1, ..., u_n) = F_nn(F_1^{-1}(u_1}, ..., F_n^{-1}(u_n}) $$
#
# here $F_i^{-1}$ denoting the inverse function of $F_i$ and $u_i \in [0,1]$.
# $\phi(u_i) = F_i^{-1}(u_i)$ is termed the generator of the copula. Its
# derivative is $d \phi(u_i) / d u_i = 1 / f_i(F_i^{-1}(u_i))$ therefore
# the pdf of the copula is as follows:
#
#  $$ c(u_1, ..., u_n) = f_nn(F_1^{-1}(u_1), ..., F_n^{-1}(u_n} / (
#                            f_1(F_1^{-1}(u_1)) ... f_n(F_n^{-1}(u_n}) ) $$
#
# with $f_nn$ denoting the function $F_nn$ derived once in each of its args.
#
# Since the definition was given through the cdf no normalization is needed.
# In our implementation, Archimedean copula generators are implemented as
# `ppf` and the inverse of the generator is a `cdf` satisfying usual domain
# of definition. Where possible the nth derivative of the cdf is given too.
#
# The copula distribution accounts for the correlations between variables
# but it has to be mapped to the desired marginals $G_1, ..., G_n$ by the
# histogram normalization method:
#
#  $$ u_i = G_i(y_i) $$
#
# Obviously this is a measure change $ d u_i / d y_i = g_i(y_i) $.
#  $$ \Prod g_i(y_i) d y_i = c(u_1, ..., u_n) \Prod d u_i $$
#

def _process_parameters(dim, marginal, joint, dtype_marginal=float, dtype_joint=float):
    """
    Infer dimensionality from marginal or covariance matrix, ensure that
    marginal and covariance are full vector resp. matrix.

    Parameters
    ----------
    dim : None, int
        number of dimensions of the multivariate distribution
    marginal : numpy.ndarray, shape (n_comp, ...)
    joint: numpy.ndarray, shape (n_comp, ...)
    dtype_marginal: Union[Type, numpy.dtype]
    dtype_joint: Union[Type, numpy.dtype]
    """
    # Adapted form scipy.stats._multivariate._process_parameters

    # Try to infer dimensionality
    if dim is None:
        if marginal is None:
            if joint is None:
                dim = 1
            else:
                joint = np.asarray(joint, dtype=dtype_joint)
                if joint.ndim < 2:
                    dim = 1
                else:
                    dim = joint.shape[0]
                import warnings
                warnings.warn("It is not safe to infer dim from 'joint'")
        else:
            marginal = np.asarray(marginal, dtype=dtype_marginal)
            dim = marginal.shape[0]
    else:
        if not np.isscalar(dim):
            raise ValueError("Dimension of random variable must be a scalar.")

    # Check input sizes and return full arrays for marginal and joint if necessary
    if marginal is None:
        marginal = np.zeros(dim)
    marginal = np.asarray(marginal, dtype=dtype_marginal)

    if joint is None:
        joint = 1.0
    if dtype_joint is not None:
        joint = np.asarray(joint, dtype=dtype_joint)

    if marginal.shape[0] != dim:
        raise ValueError("Array 'marginal' must be of length %d." % dim)
    # if joint.shape[0] != dim:
    #    raise ValueError("Array 'joint' must be of length %d." % dim)
    if marginal.ndim > 2:
        raise ValueError("Array 'marginal' must be at most two-dimensional,"
                         " but marginal.ndim = %d" % marginal.ndim)
    # if joint.ndim > 2:
    #    raise ValueError("Array 'joint' must be at most two-dimensional,"
    #                         " but joint.ndim = %d" % joint.ndim)

    return dim, marginal, joint


def _process_quantiles(x, dim):
    """
    Adjust quantiles array so that last axis labels the components of
    each data point.
    """
    # Adapted form scipy.stats._multivariate._process_quantiles

    x = np.asarray(x, dtype=float)

    if x.ndim == 0:
        x = x[np.newaxis]
    elif x.ndim == 1:
        if dim == 1:
            x = x[:, np.newaxis]
        else:
            x = x[np.newaxis, :]

    return x


def _align_vars(stat, data, params):
    params = np.asarray(params)
    data = np.asarray(data)
    stat = np.asarray(stat)
    n_sample = data.shape[-1]
    stat = np.broadcast_to(stat, _broadcast_shapes(stat.shape, (n_sample,)))
    params = np.broadcast_to(params.T, _broadcast_shapes(params.T.shape, (n_sample,))).T
    return stat, data, params


def _fit_individual_marginals(stat, data):
    """Estimate the parameters for marginal distributions.

    Parameters
    ----------
    stat : scipy.stats.rv_continuous or an array of them, shape (n_comp,)

    data : ndarray, shape (..., n_comp)
        The components are listed in rows while realisations in columns

    Returns
    -------
    params: list, shape (n_comp,) of tuples"""
    stat, data, _ = _align_vars(stat, data, 0)
    ret = [s.fit(comp) for s, comp in zip(stat, data.T)]
    return np.asarray(ret)


def _range_check(stat, data, params=None):
    """
    Check the range of input data against the support of the distribution

    Parameters
    ----------
    data : ndarray, shape (..., n_comp)
        Data samples, with the last axis of `data` denoting the components.
    """
    if stat is None:
        return True
    # FIXME: this is wrong because it does not account for loc and scale
    if params is None:
        lo, up = stat.a, stat.b
        ret = (np.all(lo <= comp) & np.all(comp <= up) for comp in data.T)
    else:
        try:
            limits = ((stat(p).a, stat(p).b) for p in params)
            ret = (np.all(lo <= comp) & np.all(comp <= up) for (comp, (lo, up)) in zip(data.T, limits))
        except AttributeError as e:
            raise ValueError("Could not verify bounds.") from e
    if not all(ret):
        raise ValueError("Data point out of the support of marginal distribution")


def _no_range_check(*args, **kwargs):
    del args, kwargs
    pass


def _unit_interval_check(stat, data, params=None):
    """
    Check the range of input data against the support of the distribution

    Parameters
    ----------
    data : ndarray, shape (..., n_comp)
        Data samples, with the last axis of `data` denoting the components.
    """
    del stat, params
    lower, upper = 0, 1
    ret = (np.all(lower <= comp) & np.all(comp <= upper) for comp in data.T)
    if not all(ret):
        raise ValueError("Data point out of the support [0,1] of marginal distribution")


def _repeat_params(params, dim):
    """Broadcast params to (dim, *params.shape)"""
    params = params[np.newaxis, ...]
    final_shape = _broadcast_shapes(params.shape, dim, align_as_numpy=False)
    repeated = np.broadcast_to(params, final_shape)
    return repeated


def _fit_common_marginal(stat, data, repeat=True):
    """Estimate the parameters for marginal distributions.

    Parameters
    ----------
    stat : scipy.stats.rv_continuous

    data : array, shape (..., n_comp)
        The components are listed in rows while realisations in columns

    Returns
    -------
    params: array, shape (n_comp, n_param_per_comp, ...)"""
    _, data, _ = _align_vars(0, data, 0)
    params = np.array(stat.fit(data.ravel()), ndmin=1)
    dim = data.shape[-1] if repeat else 1
    repeated = _repeat_params(params, dim)
    return repeated


def _transform_to_hypercube(stat, data, params):
    """Transform the observable domain to the [0,1]^n hypercube representation.

    Parameters
    ----------
    stat : scipy.stats.rv_continuous or an array of them, shape (n_comp,)

    data : ndarray, shape (..., n_comp)
        The components are listed in rows while realisations in columns.
        Values should conform the domain of definition of the `stat` class.

    params : ndarray, shape (n_comp, ...) or (1, ...)
    """
    stat, data, params = _align_vars(stat, data, params)
    ret = [s.cdf(comp, *p) for (s, comp, p) in zip(stat, data.T, params)]
    return np.asarray(ret).T  # revert transpose of data if it's multi-dim.


def _transform_to_domain_of_def(stat, data, params):
    """Transform the [0,1]^n hypercube representation to the observable domain.

    Parameters
    ----------
    stat : scipy.stats.rv_continuous or an array of them, shape (n_comp,)

    data : ndarray, shape (..., n_comp)
        The components are listed in rows while realisations in columns.
        Values must be in the [0,1] range.

    params: ndarray, shape(n_comp, ...) or (1, ...)
    """
    stat, data, params = _align_vars(stat, data, params)
    ret = [s.ppf(comp, *p) for (s, comp, p) in zip(stat, data.T, params)]
    return np.asarray(ret).T  # revert transpose of data if it's multi-dim.


def _jacobian(stat, data, params):
    """Calculate the product of univariate probability distribution functions.

    Parameters
    ----------
    stat : scipy.stats.rv_continuous or an array of them, shape (n_comp,)

    data : ndarray, shape (..., n_comp)
        The components are listed in rows while realisations in columns
        Elements in support of stat

    params : ndarray, shape (n_comp, ...) or (1, ...)

    Notes
    -----
    This function is suitable for calculating the density transformation
    of copula (data = F_i^{-1}(u_i)) and marginalizer (data = x_i) too.
    """
    stat, data, params = _align_vars(stat, data, params)
    ret = [s.pdf(comp, *p) for (s, comp, p) in zip(stat, data.T, params)]
    return np.prod(ret, axis=0).T  # revert transpose of data if it's multi-dim.


def _log_jacobian(stat, data, params):
    """Calculate the sum of log univariate probability distribution functions.

    Parameters
    ----------
    stat : scipy.stats.rv_continuous or an array of them, shape (n_comp,)

    data : ndarray, shape (..., n_comp)
        The components are listed in rows while realisations in columns

    params : ndarray, shape (n_comp, ...) or (1, ...)

    Notes
    -----
    This function is suitable for calculating the density transformation
    of copula (data = F_i^{-1}(u_i)) and marginalizer (data = x_i) too.
    """
    stat, data, params = _align_vars(stat, data, params)
    ret = [s.logpdf(comp, *p) for (s, comp, p) in zip(stat, data.T, params)]
    return np.sum(ret, axis=0).T  # revert transpose of data if it's multi-dim.


def nan_to_neg_inf(x):
    y = x.copy()
    y[np.isnan(y)] = -np.inf
    return y


_mj_doc_default_callparams = """\
joint : tuple
    Parameters of the joint distribution
marginal : array_like, shape (n_comp, ...)
    Parameters of the marginal distributions
iid_marginals : bool
    Whether to use a single marginal for all components
"""

_mj_doc_callparams_note = \
    """No special notes.
    """


class multivariate_transform_gen(multi_rv_generic):
    def __init__(self, marginal_gen, joint_gen, iid_marginals, *args,
                 fit_method='exact', fit_init=None, fit_bounds=None,
                 dtype_marginal=float, dtype_joint=float, name=None, **kwargs):
        """
        From a common parameter set provide the separate parameters for marginal and joint distributions

        Parameters
        ----------
        iid_marginals: use the same parametrization of the marginal on all axis
        marginal_gen: subclass of scipy._rv_continuous
        joint_gen: subclass of scipy.stats._multivariate.multi_rv_generic
        iid_marginals: bool
        *args: arguments for multi_rv_generic
        dtype_marginal: Union[Type, np.dtype]
        dtype_joint: Union[Type, np.dtype]
        **kwargs: keyword arguments for  multi_rv_generic
        """
        self.iid_marginals = bool(iid_marginals)
        self.marginal_gen = marginal_gen
        self.joint_gen = joint_gen
        self._dtype_marginal = None if dtype_marginal is None else np.dtype(dtype_marginal)
        self._dtype_joint = None if dtype_joint is None else np.dtype(dtype_joint)
        self.name = name
        self._range_check = _no_range_check
        if fit_method == 'exact':
            _exist(joint_gen.fit, marginal_gen.fit)
        elif fit_method == 'optimize':
            if fit_init is None:
                raise ValueError('Initial values to fit must be provided')
            self._fit_init = fit_init
            self._fit_bounds = fit_bounds
        else:
            raise ValueError('fit_method must be one of the following: exact, optimize')
        self._fit_method = fit_method
        super(multivariate_transform_gen, self).__init__(*args, **kwargs)

    @abstractmethod
    def fit(self, data, loc=0, scale=1):
        """
        Fit both the parameters of the marginals and the joint distribution

        Parameters
        ----------
        data : ndarray, shape (..., n_comp)
            Data samples, with the last axis of `data` denoting the components.

        Notes
        -----
        One needs to specify at construction time how to perform fit.
        """
        pass

    @abstractmethod
    def _logpdf(self, x, marginal, joint):
        """
        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability
            density function
        marginal : array_like, shape (n_comp, n_param) or (1, n_param)
            Parameters of the marginal distribution(s)
        joint : tuple, array_like
            Parameters of the joint distribution
            
        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.
        """
        pass

    def logpdf(self, x, marginal, joint):
        """
        Log of the multivariate probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mj_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray or scalar
            Log of the probability density function evaluated at `x`

        Notes
        -----
        %(_mj_doc_callparams_note)s
        """
        dim, marginal, joint = _process_parameters(None, marginal, joint, dtype_marginal=self._dtype_marginal,
                                                   dtype_joint=self._dtype_joint)
        x = _process_quantiles(x, dim)
        self._range_check(self.marginal_gen, x, marginal)

        ret = self._logpdf(x, marginal, joint)
        return ret

    def pdf(self, x, marginal, joint):
        """
        Multivariate probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mj_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray or scalar
            Log of the probability density function evaluated at `x`

        Notes
        -----
        %(_mj_doc_callparams_note)s
        """
        dim, marginal, joint = _process_parameters(None, marginal, joint, dtype_marginal=self._dtype_marginal,
                                                   dtype_joint=self._dtype_joint)
        x = _process_quantiles(x, dim)
        self._range_check(self.marginal_gen, x, marginal)

        ret = np.exp(self._logpdf(x, marginal, joint))
        return ret

    @abstractmethod
    def _cdf(self, x, marginal, joint):
        """
        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the cumulative distribution function.
        marginal : array_like, shape (n_comp, n_param) or (1, n_param)
            Parameters of the marginal distribution(s)
        joint : tuple, array_like
            Parameters of the joint distribution

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'cdf' instead.
        """
        pass

    def logcdf(self, x, marginal, joint):
        """
        Log of the cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mj_doc_default_callparams)s

        Returns
        -------
        cdf : ndarray or scalar
            Log of the cumulative distribution function evaluated at `x`

        Notes
        -----
        %(_mj_doc_callparams_note)s
        """
        dim, marginal, joint = _process_parameters(None, marginal, joint, dtype_marginal=self._dtype_marginal,
                                                   dtype_joint=self._dtype_joint)
        x = _process_quantiles(x, dim)
        self._range_check(self.marginal_gen, x, marginal)

        ret = np.log(self._cdf(x, marginal, joint))
        return ret

    def cdf(self, x, marginal, joint):
        """
        The cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mj_doc_default_callparams)s

        Returns
        -------
        cdf : ndarray or scalar
            Log of the cumulative distribution function evaluated at `x`

        Notes
        -----
        %(_mj_doc_callparams_note)s
        """
        dim, marginal, joint = _process_parameters(None, marginal, joint, dtype_marginal=self._dtype_marginal,
                                                   dtype_joint=self._dtype_joint)
        x = _process_quantiles(x, dim)
        self._range_check(self.marginal_gen, x, marginal)

        ret = self._cdf(x, marginal, joint)
        return ret

    @abstractmethod
    def _rvs(self, marginal, joint, size, random_state):
        """
        Parameters
        ----------
        marginal : array_like, shape (n_comp, n_param) or (n_param,)
            Parameters of the marginal distribution(s)
        joint : tuple
            Parameters of the joint distribution
        size : int or tuple
            Number of `N` dimensional random variates to generate.
        random_state : numpy.random_state
            Random state instance

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the random variable.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'rvs' instead.
        """
        pass

    def rvs(self, marginal, joint, size=1, random_state=None):
        """
        Draw random samples from an `N`-dimensional copula distribution.
        
        Parameters
        ----------
        %(_mj_doc_default_callparams)s
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
        %(_mj_doc_callparams_note)s
        """
        dim, marginal, joint = _process_parameters(None, marginal, joint, dtype_marginal=self._dtype_marginal,
                                                   dtype_joint=self._dtype_joint)
        random_state = self._get_random_state(random_state)

        ret = self._rvs(marginal, joint, size, random_state)

        return ret


class histogram_normalization_gen(multivariate_transform_gen):
    def __init__(self, marginal_gen, joint_gen, iid_marginals, *args,
                 fit_method='exact', fit_init=None, fit_bounds=None,
                 dtype_marginal=float, dtype_joint=float, name=None, **kwargs):
        """
        From a common parameter set provide the separate parameters for marginal and joint distributions

        Parameters
        ----------
        marginal_gen: subclass of scipy._rv_continuous
        joint_gen: subclass of scipy.stats._multivariate.multi_rv_generic
        iid_marginals: bool, use the same parametrization of the marginal on all axis
        *args: arguments for multi_rv_generic
        dtype_marginal: Union[Type, np.dtype]
        dtype_joint: Union[Type, np.dtype]
        **kwargs: keyword arguments for  multi_rv_generic
        """
        self._range_check = _no_range_check
        if fit_method == 'exact':
            _exist(joint_gen.fit, marginal_gen.fit)
        elif fit_method == 'optimize':
            if fit_init is None:
                raise ValueError('Initial values to fit must be provided')
            self._fit_init = fit_init
            self._fit_bounds = fit_bounds
        else:
            raise ValueError('fit_method must be one of the following: exact, optimize')
        self._fit_method = fit_method
        super(histogram_normalization_gen, self).__init__(marginal_gen, joint_gen, iid_marginals, *args,
                                                          dtype_marginal=dtype_marginal, dtype_joint=dtype_joint,
                                                          name=name, **kwargs)

    def fit(self, data, loc=0, scale=1):
        """
        Fit both the parameters of the marginals and the joint distribution one after another

        Parameters
        ----------
        data : ndarray, shape (..., n_comp)
            Data samples, with the last axis of `data` denoting the components.

        Notes
        -----
        This function uses the fit capability of underlying distributions.
        TODO: create a wrapper class for distributions that do not have fit.
        """
        if self.iid_marginals:
            marginal = _fit_common_marginal(self.marginal_gen, data)
        else:
            marginal = _fit_individual_marginals(self.marginal_gen, data)
        uniformized = _transform_to_hypercube(stat=self.marginal_gen, data=data, params=marginal)

        joint = self.joint_gen.fit(uniformized)
        return marginal, joint

    def _logpdf(self, x, marginal, joint):
        """
        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability
            density function
        marginal : array_like, shape (n_comp, n_param) or (n_param,)
            Parameters of the marginal distribution(s)
        joint : tuple, array_like
            Parameters of the joint distribution

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.
        """
        uniformized = _transform_to_hypercube(stat=self.marginal_gen, data=x, params=marginal)
        marginal_logpdf = _log_jacobian(stat=self.marginal_gen, data=x, params=marginal)
        joint_logpdf = self.joint_gen.logpdf(uniformized, *joint)

        # TODO: make this sanitization optional (it was introduced due to values much off the centre of mv normal)
        return np.nan_to_num(nan_to_neg_inf(joint_logpdf + marginal_logpdf))

    def _cdf(self, x, marginal, joint):
        """
        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the cumulative distribution function.
        marginal : array_like, shape (n_comp, n_param) or (n_param,)
            Parameters of the marginal distribution(s)
        joint : tuple, array_like
            Parameters of the joint distribution

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'cdf' instead.
        """
        uniformized = _transform_to_hypercube(stat=self.marginal_gen, data=x, params=marginal)
        joint_cdf = self.joint_gen.cdf(uniformized, *joint)
        return joint_cdf

    def _rvs(self, marginal, joint, size, random_state):
        """
        Parameters
        ----------
        marginal : array_like, shape (n_comp, n_param) or (n_param,)
            Parameters of the marginal distribution(s)
        joint : tuple
            Parameters of the joint distribution
        size : int or tuple
            Number of `N` dimensional random variates to generate.
        random_state : numpy.random_state
            Random state instance

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the random variable.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'rvs' instead.
        """
        uniformized = self.joint_gen.rvs(*joint, size=size, random_state=random_state)
        ret = _transform_to_domain_of_def(stat=self.marginal_gen, data=uniformized, params=marginal)
        return ret


class copula_base_gen(multivariate_transform_gen):
    def __init__(self, marginal_gen, joint_gen, iid_marginals, *args,
                 fit_method='exact', fit_init=None, fit_bounds=None,
                 dtype_marginal=float, dtype_joint=float, name=None, **kwargs):
        """
        From a common parameter set provide the separate parameters for marginal and joint distributions

        Parameters
        ----------
        marginal_gen: subclass of scipy._rv_continuous
        joint_gen: subclass of scipy.stats._multivariate.multi_rv_generic
        *args: arguments for multi_rv_generic
        dtype_marginal: Union[Type, np.dtype]
        dtype_joint: Union[Type, np.dtype]
        **kwargs: keyword arguments for  multi_rv_generic
        """
        self._range_check = _unit_interval_check
        if fit_method == 'exact':
            _exist(joint_gen.fit, marginal_gen.fit)
        elif fit_method == 'optimize':
            if fit_init is None:
                raise ValueError('Initial values to fit must be provided')
            self._fit_init = fit_init
            self._fit_bounds = fit_bounds
        else:
            raise ValueError('fit_method must be one of the following: exact, optimize')
        self._fit_method = fit_method
        super(copula_base_gen, self).__init__(marginal_gen, joint_gen, iid_marginals, *args,
                                              dtype_marginal=dtype_marginal, dtype_joint=dtype_joint,
                                              name=name, **kwargs)

    def _logpdf(self, x, marginal, joint):
        """
        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability
            density function
        marginal : array_like, shape (n_comp, n_param) or (n_param,)
            Parameters of the marginal distribution(s)
        joint : tuple, array_like
            Parameters of the joint distribution

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.
        """
        internal = _transform_to_domain_of_def(stat=self.marginal_gen, data=x, params=marginal)
        marginal_logpdf = -_log_jacobian(stat=self.marginal_gen, data=internal, params=marginal)
        joint_logpdf = self.joint_gen.logpdf(internal, *joint)

        # TODO: make this sanitization optional (it was introduced due to values much off the centre of mv normal)
        return np.nan_to_num(nan_to_neg_inf(joint_logpdf + marginal_logpdf))

    def _cdf(self, x, marginal, joint):
        """
        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the cumulative distribution function.
        marginal : array_like, shape (n_comp, n_param) or (n_param,)
            Parameters of the marginal distribution(s)
        joint : tuple, array_like
            Parameters of the joint distribution

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'cdf' instead.
        """
        internal = _transform_to_domain_of_def(stat=self.marginal_gen, data=x, params=marginal)
        joint_cdf = self.joint_gen.cdf(internal, *joint)
        return joint_cdf

    def _rvs(self, marginal, joint, size, random_state):
        """
        Parameters
        ----------
        marginal : array_like, shape (n_comp, n_param) or (n_param,)
            Parameters of the marginal distribution(s)
        joint : tuple
            Parameters of the joint distribution
        size : int or tuple
            Number of `N` dimensional random variates to generate.
        random_state : numpy.random_state
            Random state instance

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the random variable.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'rvs' instead.
        """
        internal = self.joint_gen.rvs(*joint, size=size, random_state=random_state)
        ret = _transform_to_hypercube(stat=self.marginal_gen, data=internal, params=marginal)
        return ret


class archimedean_copula_gen(copula_base_gen):
    def __init__(self, marginal_gen, joint_gen, *args,
                 fit_init=None, fit_bounds=None, dtype_marginal=float, dtype_joint=float, name=None, **kwargs):
        """
        From a common parameter set provide the separate parameters for marginal and joint distributions

        Parameters
        ----------
        marginal_gen: subclass of scipy._rv_continuous
        joint_gen: subclass of scipy.stats._multivariate.multi_rv_generic
        iid_marginals: bool
        *args: arguments for multi_rv_generic
        fit_method: string['exact', 'optimize']
        fit_init: ndarray
        fit_bounds: None, ndarray
        dtype_marginal: Union[Type, np.dtype]
        dtype_joint: Union[Type, np.dtype]
        **kwargs: keyword arguments for  multi_rv_generic
        """
        super(archimedean_copula_gen, self).__init__(marginal_gen, joint_gen, True, *args,
                                                     fit_method='optimize', fit_init=fit_init, fit_bounds=fit_bounds,
                                                     dtype_marginal=dtype_marginal, dtype_joint=dtype_joint,
                                                     name=name, **kwargs)
        self._range_check = _unit_interval_check

    def _logpdf(self, x, marginal, joint):
        """
        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability
            density function
        marginal : array_like, shape (n_comp, n_param) or (n_param,)
            Parameters of the marginal distribution(s)
        joint : tuple, array_like
            Parameters of the joint distribution

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.
        """
        # Note: if dealing with generators logpdf does not necessarily exist
        internal = _transform_to_domain_of_def(stat=self.marginal_gen, data=x, params=marginal)
        marginal_logpdf = -_log_jacobian(stat=self.marginal_gen, data=internal, params=marginal)
        # marginal_logpdf = -np.log(_jacobian(stat=self.marginal_gen, data=internal, params=marginal))

        dim = internal.shape[-1]
        # joint_pdf = derivative(lambda x0: self.joint_gen.cdf(x0, *joint), np.sum(internal, axis=-1), dx=1e-6, n=dim)
        joint_pdf = np.abs(self.joint_gen.cdfd(np.sum(internal, axis=-1), *joint, n=dim))

        return np.log(joint_pdf) + marginal_logpdf

    def _cdf(self, x, marginal, joint):
        """
        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the cumulative distribution function.
        marginal : array_like, shape (n_comp, n_param) or (n_param,)
            Parameters of the marginal distribution(s)
        joint : tuple, array_like
            Parameters of the joint distribution

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'cdf' instead.
        """
        internal = _transform_to_domain_of_def(stat=self.marginal_gen, data=x, params=marginal)
        joint_cdf = self.joint_gen.cdf(np.sum(internal, axis=-1), *joint)
        return joint_cdf

    def _rvs(self, marginal, joint, size, random_state):
        """
        Parameters
        ----------
        marginal : array_like, shape (n_comp, n_param) or (n_param,)
            Parameters of the marginal distribution(s)
        joint : tuple, array_like
            Parameters of the joint distribution
        size : int or tuple
            Number of `N` dimensional random variates to generate.
        random_state : numpy.random_state
            Random state instance

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the random variable.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'rvs' instead.
        """
        raise NotImplementedError


class passthru_norm_gen(type(norm)):
    """Example extension to the multivariate normal distribution without
       ML fit capability"""

    def fit(self, X, **kwargs):
        """Return default distribution parameters.

        Parameters
        ----------
        X : integer, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s

        Returns
        -------
        mn : 0, scalar
            Mean vector of the distribution

        va : 1, scalar
            Variance parameter of the distribution

        """
        return 0, 1


class multivariate_cov_only_normal_gen(multivariate_normal_gen):
    """Example extension to the multivariate normal distribution with
       ML fit capability"""

    def __init__(self, *args, fit_mean=False, **kwargs):
        self._fit_mean = fit_mean
        super(multivariate_normal_gen, self).__init__(*args, **kwargs)

    # def logpdf(self, x, mean=None, cov=1, allow_singular=False):
    #     y = super(my_multivariate_normal_gen, self).logpdf(x, mean, cov, allow_singular)
    #     return nan_to_neg_inf(y)

    def fit(self, X):
        """Provide analytic solution to ML fit.

        Parameters
        ----------
        X : integer, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s

        Returns
        -------
        mn : ndarray or scalar
            Mean vector of the distribution

        co : ndarray or scalar
            Covariance parameter of the distribution

        """
        X = np.array(X, ndmin=1)
        mn = np.mean(X, axis=0)
        if not self._fit_mean:
            mn = np.zeros_like(mn)
        co = np.cov(X, rowvar=False)
        return mn, co


class my_gaussian_kde_gen(rv_continuous):
    """Example extension to the multivariate normal distribution with
       ML fit capability

       Notes
       -----
       This class overrides methods like pdf, logpdf and cdf directly because
       the scipy.stats implementation enforces too strict parameter validation,
       i.e., all parameters must be scalar floats while our intent is to pass
       along the point cloud, which was obtained in fit, for further processing.
       Due to this override the parameter validation on the side of the `x`
       quantiles is weaker too.
       """

    def _check_params_kde(self, *args, **kwargs):
        if len(args) != 2:
            raise TypeError('The function should be called with a two arguments')
        x = args[0]
        try:
            x = np.asarray(x, dtype=float)
        except ValueError:
            raise ValueError('The first argument should be an array of floats')
        inst = args[1]
        if not isinstance(inst, gaussian_kde):
            raise TypeError('The second argument should be a child of gaussian_kde')
        return x, inst

    def fit(self, X, *args, **kwds):
        """Provide analytic solution to ML fit.

        Parameters
        ----------
        X : integer, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s

        Returns
        -------
        inst: scipy.stats.gaussian_kde
            Covariance parameter of the distribution

        """
        try:
            inst = gaussian_kde(X, bw_method=None)
        except ValueError:
            import warnings
            warnings.warn("Could not fit KDE")
            inst = gaussian_kde([-1, 0, 1], bw_method=None)
        return np.array([inst])

    def _pdf(self, x, inst):
        """
        Parameters
        ----------
        inst: scipy.stats.gaussian_kde
            Covariance parameter of the distribution

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'pdf' instead.
        """
        return inst.pdf(x)

    def pdf(self, x, inst):
        """
        Parameters
        ----------
        inst: scipy.stats.gaussian_kde
            Covariance parameter of the distribution

        """
        self._check_params_kde(x, inst)
        return self._pdf(x, inst)

    def _logpdf(self, x, inst):
        """
        Parameters
        ----------
        inst: scipy.stats.gaussian_kde
            Covariance parameter of the distribution

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'pdf' instead.
        """
        return inst.logpdf(x)

    def logpdf(self, x, inst):
        """
        Parameters
        ----------
        inst: scipy.stats.gaussian_kde
            Covariance parameter of the distribution

        """
        self._check_params_kde(x, inst)
        return self._logpdf(x, inst)

    def _cdf(self, x, inst):
        """
        Parameters
        ----------
        inst: scipy.stats.gaussian_kde
            Covariance parameter of the distribution

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'pdf' instead.

        """
        func = np.vectorize(lambda v: inst.integrate_box_1d(-10000, v))
        return func(x)

    def cdf(self, x, inst):
        """
        Parameters
        ----------
        inst: scipy.stats.gaussian_kde
            Covariance parameter of the distribution

        """
        self._check_params_kde(x, inst)
        return self._cdf(x, inst)

    # def _argcheck(self, *args):
    #    return 1

    def __call__(self, inst, *args, **kwargs):
        return inst


#
# class invert_cdf(rv_continuous):
#     def __init__(self, underlying, *args, **kwargs):
#         self._cdf = underlying._ppf
#         self._ppf = underlying._cdf
#         super(invert_cdf, self).__init__(*args, **kwargs)

def invert_cdf(stat):
    stat._cdf, stat._ppf = stat._ppf, stat._cdf
    stat._pdf = super(type(stat), stat)._pdf
    return stat


class archimedean_generator_gen(rv_continuous):
    """Base class for generators of Archimedean copulas that provides
    numerical differentiation of the inverse of the generator.
    Note that these are not valid probability distributions because the
    cdf is monotoically decreasing and therefore the pdf is negative."""
    # The "generator of the copula" phi: [0,1] --> [0,inf)
    # This is a strictly decreasing function such that phi(1)=0 and
    # it is applied to the marginal distribution.
    # def _ppf(self, q, *args):
    # archimedean_generator_gen._ppf.__doc__ += \
    #     "This is the generator of the Archimedean copula"

    # The inverse of the generator phi^{-1}: [0,inf) --> [0,1]
    # This applied to the sum of the generators valid arguments fall into
    # [0, phi(0)]. Outside this range the pseudoinverse is defined as zero.
    # def _cdf(self, x, theta):
    # archimedean_generator_gen._cdf.__doc__ += \
    #     "This is the inverse of the generator of the Archimedean copula"

    # Make sure logpdf can be calculated (positive argument to log)
    def _logpdf(self, x, *args):
        return np.log(-self._pdf(x, *args))

    # The nth derivative of the inverse of the generator
    def _cdfd(self, x, *args, n=1):
        # min order = n - (n % 2) + 3
        order = 2 * n + 1
        return derivative(lambda x0: self._cdf(x0, *args), x, dx=1e-2, n=n, order=order)

    # def _logcdfd(self, x, *args, n=1):
    #    return np.log(self._cdfd(x, *args, n=n))

    def cdfd(self, x, *args, n=1, **kwds):
        """
        nth derivative of the cumulative distribution function of the given RV.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        cdf : ndarray
            nth derivative of the cumulative distribution function evaluated at `x`

        """
        # Based on scipy.stats._distn_infrastructure.rv_generic.cdf
        args, loc, scale = self._parse_args(*args, **kwds)
        n = int(n)
        if n < 1:
            raise ValueError('Only positive integer is accepted as order of derivative.')
        x, loc, scale = map(np.asarray, (x, loc, scale))
        args = tuple(map(np.asarray, args))
        dtyp = np.find_common_type([x.dtype, np.float64], [])
        x = np.asarray((x - loc) / scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x) & (scale > 0)
        cond2 = (x >= self.b) & cond0
        cond = cond0 & cond1
        output = np.zeros(np.shape(cond), dtyp)
        np.place(output, (1 - cond0) + np.isnan(x), self.badvalue)
        np.place(output, cond2, 1.0)
        if np.any(cond):  # call only if at least 1 entry
            goodargs = argsreduce(cond, *((x,) + args))
            vec_cdfd = np.vectorize(self._cdfd)
            np.place(output, cond, vec_cdfd(*goodargs, n=n))
        if output.ndim == 0:
            return output[()]
        return output


def _test_cdfd(stat, x, *args, n=1):
    if stat._cdfd.__code__ is archimedean_generator_gen._cdfd.__code__:
        return ValueError('Numerically differentiating method is not overridden with analytic one.')
    x, *args, n = np.atleast_1d(x, *args, n)
    ana = stat._cdfd(x, *args, n=n)
    vecdiff = np.vectorize(archimedean_generator_gen._cdfd)
    num = vecdiff(stat, x, *args, n=n)
    good = np.allclose(ana, num)
    if not good:
        print('Ana: %s\nNum: %s\n' % (ana, num))
    return good


class independent_generator_gen(archimedean_generator_gen):
    def _argcheck(self, *args):
        """Default check for correct values on args and keywords.

        Returns condition array of 1's where arguments are correct and
         0's where they are not.

        """
        cond = 1
        for arg in args:
            cond = np.logical_and(cond, (~np.isnan(arg)))
        return cond

    def _ppf(self, q):
        ret = -np.log(q)
        return ret

    def _cdf(self, x):
        ret = np.exp(-x)
        return ret

    def _pdf(self, x):
        ret = -np.exp(-x)
        return ret

    def _logpdf(self, x):
        ret = -x
        return ret

    def _cdfd(self, x, n):
        ret = np.power(-1, n % 2) * np.exp(-x)
        return ret

    # def _logcdfd(self, x, n):
    #    ret = np.power(-1, n), np.log(-x)
    #    return ret


independent_generator = independent_generator_gen(name='indep')
# _test_cdfd(independent_generator, [0.1, 5, 3], n=[5, 2, 1])


def _log_prod_arithmetic_progression(a, d, n):
    """The product of the members of a finite arithmetic progression
    with an initial element a1, common differences d, and n elements
    in total is determined in a closed expression"""
    # https://en.wikipedia.org/wiki/Arithmetic_progression#Product
    frac = np.true_divide(a, d)
    return n * np.log(d) + gammaln(frac + n) - gammaln(frac)


class clayton_generator_gen(archimedean_generator_gen):
    def _ppf(self, q, theta):
        ret = 1.0 / theta * (np.power(q, -theta) - 1)
        return ret

    def _cdf(self, x, theta):
        ret = np.power(theta * x + 1, -1.0 / theta)
        return ret

    def _pdf(self, x, theta):
        ret = -np.power(theta * x + 1, -1.0 / theta - 1)
        return ret

    def _logpdf(self, x, theta):
        ret = (-1.0 / theta - 1) * np.log(theta * x + 1)
        return ret

    def _cdfd(self, x, theta, n=1):
        fact = np.exp(_log_prod_arithmetic_progression(1, theta, n))
        ret = fact * np.power(1 + theta * x, -n - 1. / theta) * np.power(-1, n % 2)
        return ret


clayton_generator = clayton_generator_gen(name='clayton')
# _test_cdfd(clayton_generator, [0.1, 5, 3], 2.2, n=[5, 2, 1])
# _test_cdfd(clayton_generator, [0.1, 5, 3], 1.5, n=[5, 2, 1])


class gumbel_generator_gen(archimedean_generator_gen):
    def _ppf(self, q, theta):
        ret = np.power(-np.log(q), -theta)
        return ret

    def _cdf(self, x, theta):
        ret = np.exp(-np.power(x, 1. / theta))
        return ret

    def _pdf(self, x, theta):
        ret = -np.power(x, 1./theta-1) * np.exp(-np.power(x, 1. / theta))
        return ret

    def _logpdf(self, x, theta):
        ret = (1./theta-1) * np.log(x) - np.power(x, 1. / theta)
        return ret


gumbel_generator = gumbel_generator_gen(name='gumbel')


class frank_generator_gen(archimedean_generator_gen):
    def _ppf(self, q, theta):
        ret = -np.log(np.expm1(-theta * q) / np.expm1(-theta))
        return ret

    def _cdf(self, x, theta):
        ret = -1.0 / theta * np.log1p(np.exp(-x) * np.expm1(-theta))
        return ret


frank_generator = frank_generator_gen(name='frank')


passthru_norm = passthru_norm_gen()
# _exist(passthru_norm.fit)

gaussian_generator = invert_cdf(passthru_norm_gen(a=0.0, b=1.0, name='invgauss'))

multivariate_cov_only_normal = multivariate_cov_only_normal_gen(fit_mean=False)
# _exist(multivariate_cov_only_normal.fit)

my_gaussian_kde = my_gaussian_kde_gen(0, name="gaussian_kde")

gaussian_copula = copula_base_gen(marginal_gen=passthru_norm, joint_gen=multivariate_cov_only_normal,
                                  iid_marginals=True,
                                  fit_method='exact', dtype_joint=None, name="Gaussian")

iid_gaussian_copula = copula_base_gen(marginal_gen=norm,
                                      joint_gen=gaussian_copula,
                                      iid_marginals=True, dtype_joint=None)

independent_copula = archimedean_copula_gen(marginal_gen=independent_generator, joint_gen=independent_generator,
                                            fit_init=0, fit_bounds=[0, 1], name="Independent")
clayton_copula = archimedean_copula_gen(marginal_gen=clayton_generator, joint_gen=clayton_generator,
                                        fit_init=1, fit_bounds=[0, np.inf], name="Clayton")
frank_copula = archimedean_copula_gen(marginal_gen=frank_generator, joint_gen=frank_generator,
                                      fit_init=1,
                                      fit_bounds=[-np.inf, np.inf], name="Frank")


# sep_gaussian_copula = copula_gen(marginal_gen=norm, joint_gen=my_multivariate_normal,
#                                  iid_marginals=False)
#
# ker_gaussian_copula = copula_gen(marginal_gen=my_gaussian_kde, joint_gen=my_multivariate_normal,
#                                  iid_marginals=False, tuning={'dtype_marginal': object})
#
# kir_gaussian_copula = copula_gen(marginal_gen=my_gaussian_kde, joint_gen=my_multivariate_normal,
#                                  iid_marginals=True, tuning={'dtype_marginal': object})

def make_copula(marginal, joint, iid_marginals=True):
    """
    Set up a histogram normalization scheme for the provided marginal and
    use one of the copulae to relate the axes.

    Parameters
    ----------
    marginal: Union[string['kde'], subclass[scipy.stats.rv_continuous]]
        The assumed shape of the marginals

    joint: string['gaussian', 'clayton', 'independent']
        Covariance parameter of the distribution

    iid_marginals: bool
        Use the same distribution parameters for all marginals, default=True

    Notes
    -----
    As this function does no argument checking, it should not be
    called directly; use 'pdf' instead.

    """
    # allowed_marginals = {'kde': my_gaussian_kde, 'norm': norm}
    allowed_joint = {'gaussian': gaussian_copula, 'clayton': clayton_copula, 'independent': independent_copula}
    tuning = {}
    if marginal == 'kde':
        tuning['dtype_marginal'] = object
        marginal_gen = my_gaussian_kde
    else:
        _exist(marginal.fit)
        marginal_gen = marginal
    if joint == 'gaussian':
        tuning['dtype_joint'] = None
    joint_gen = allowed_joint[joint]

    iid_marginals = bool(iid_marginals)
    return multivariate_transform_gen(marginal_gen=marginal_gen,
                                      joint_gen=joint_gen,
                                      iid_marginals=iid_marginals,
                                      **tuning)

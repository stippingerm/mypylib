"""Arbitrary Mixture Model."""

#
# Author: Marcell Stippinger
# License: BSD 3 clause
#

# Notes:
# This submodule was inspired by and based on scikit-learn mixture models

import numpy as np
import warnings

# sklearn.mixture.gaussian_mixture
from sklearn.mixture.base import BaseMixture
from sklearn.mixture.gaussian_mixture import _check_weights
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
# sklearn.mixture.base
from sklearn.utils import check_array, check_random_state
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture.base import _check_X
# classifiers
from sklearn.utils.validation import check_X_y
from .base import MixtureClassifierMixin


def _infer_stat_type(stat, mv_stat=None):
    """Infer whether stat is a mulvtivariate statistics"""
    if mv_stat is None:
        from scipy import stats
        if isinstance(stat, (stats.rv_continuous, stats.rv_discrete)):
            mv_stat = False
        elif isinstance(stat, (stats._multivariate.multi_rv_generic)):
            mv_stat = True
        else:
            import warnings
            warnings.warn("Could not infer whether 'stat' is multivariate.")
            mv_stat = False
    return mv_stat


def _estimate_1d_stat_parameters(stat, X, resp, resolution=None):
    # TODO the scipy interface does not support weights, discretize
    """Estimate the arbitrary 1d distribution parameters.

    Parameters
    ----------
    stat : scipy.stats.rv_continuous
           A scipy.stats class implementing the distribution which is
           characterized by n_shape_params floats

    X : array-like, shape (n_samples, n_features)
        The input data array.

    resp : array-like, shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    Returns
    -------
    nk : array-like, shape (n_components,)
        The numbers of data samples in the current components.

    params : array-like, shape (n_components, n_features, n_shape_params)
        The centers of the current components.
    """

    def arr_fit(x):
        """Force stat.fit to return an array"""
        return np.array(stat.fit(x), ndmin=1)

    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    n_components = resp.shape[1]
    if resolution is None:
        repeats = (resp > 0).astype(int)
    else:
        raise NotImplementedError
    params = np.empty(n_components, dtype=object)

    # vec_fit = np.vectorize(arr_fit, signature='(n)->(p)')
    for i in range(0, n_components):
        approximated = np.repeat(X, repeats[:, i], axis=0)
        component_params = np.apply_along_axis(arr_fit, 1, approximated.T)
        # component_params : array-like, shape (n_features, n_shape_params)
        params[i] = component_params
    params = np.stack(params, axis=0)
    return nk, params


def _estimate_log_1d_stat_prob(stat, X, params):
    """Estimate the log 1d distribution probability.

    Parameters
    ----------
    stat : scipy.stats.rv_continuous
           A scipy.stats class implementing the distribution which is
           characterized by n_shape_params floats

    X : array-like, shape (n_samples, n_features)

    params : array-like, shape (n_components, n_features, n_shape_params)

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """

    def logpdf(x, shape_par):
        """Decompress parameter list for stat.logpdf"""
        return stat.logpdf(x, *shape_par)

    n_samples, n_features = X.shape
    n_components = params.shape[0]

    log_prob = np.empty((n_samples, n_components))
    vec_logpdf = np.vectorize(logpdf, signature='(n),(p)->(n)')
    for k, component_params in enumerate(params):
        # component_params: array-like, shape (n_features, n_shape_params)
        component_logpdf = vec_logpdf(X.T, component_params)
        # component_logpdf: array-like, shape (n_features, n_samples)
        log_prob[:, k] = np.sum(component_logpdf, 0)
    return log_prob


def _estimate_mv_stat_parameters(stat, X, resp, resolution=None):
    # TODO the scipy interface does not support weights, discretize
    """Estimate the arbitrary multivariate distribution parameters.

    Parameters
    ----------
    stat : scipy.stats.rv_continuous
           A scipy.stats class implementing the distribution which is
           characterized by n_shape_params floats

    X : array-like, shape (n_samples, n_features)
        The input data array.

    resp : array-like, shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    Returns
    -------
    nk : array-like, shape (n_components,)
        The numbers of data samples in the current components.

    params : tuple
        The centers of the current components.
    """

    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    n_components = resp.shape[1]
    if resolution is None:
        repeats = (resp > 0).astype(int)
    else:
        raise NotImplementedError
    params = np.empty(n_components, dtype=object)

    for i in range(0, n_components):
        approximated = np.repeat(X, repeats[:, i], axis=0)
        component_params = stat.fit(approximated)
        # component_params : tuple
        params[i] = component_params
    return nk, params


def _estimate_log_mv_stat_prob(stat, X, params):
    """Estimate the log multivariate distribution probability.

    Parameters
    ----------
    stat : scipy.stats.rv_continuous
           A scipy.stats class implementing the distribution which is
           characterized by n_shape_params floats

    X : array-like, shape (n_samples, n_features)

    params : array-like, shape (n_components, n_features, n_shape_params)

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """

    n_samples, n_features = X.shape
    n_components = params.shape[0]

    log_prob = np.empty((n_samples, n_components))
    for k, component_params in enumerate(params):
        # component_params: tuple
        component_logpdf = stat.logpdf(X, *component_params)
        # component_logpdf: array-like, shape (n_samples, )
        log_prob[:, k] = component_logpdf
    return log_prob


class PluginClassifier(MixtureClassifierMixin, BaseMixture):
    """Arbitrary Mixture.

    Representation of any mixture model probability distribution.
    This class allows classification based on any mixture components
    conforming the scipy.stats API. The main difference to mixture
    distributions is that responsibilities are set based on class label.
    This introduces some computational overhead but opens the door
    towards classification between mixture distributions. However,
    this will need to set the number of mixture components for each class.

    Read more in the :ref:`User Guide <gmm>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    stat : subclass of Union[scipy.stats.rv_contiuous, scipy.stats._multivariate.multi_rv_generic] : {fit,log_pdf,rvs}
        The scipy statistics to plug in to the classifier as components
        Requirements: support methods fit, log_pdf and rvs

    n_components : int or array-like shape (n_components,), defaults to 1.
        The number of mixture components per class. NOT IMPLEMENTED YET.

    tol : float, defaults to 1e-3.
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.

    max_iter : int, defaults to 100.
        The number of EM iterations to perform.
        If n_components is 1 only one iteration is performed because the
        results converge without EM. TODO

    n_init : int, defaults to 1.
        The number of initializations to perform. The best results are kept. TODO

    init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
        The method used to initialize the weights, the means and the
        precisions.
        Must be one of::

            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.

    classes_init : array-like, shape (n_components, ), optional
        The user-provided component to class assignments if n_components is
        not 1. TODO: TO BE IMPLEMENTED

    weights_init : array-like, shape (n_components, ), optional
        The user-provided initial weights, defaults to None.
        If it None, weights are initialized using the `init_params` method.

    use_weights: bool, optional
        If set to false, do not use weights for prediction (useful if classes
        have different weights in the training and test set)

    params_init : array-like, shape (n_components, n_features, n_shape_params), optional
        The user-provided initial means, defaults to None,
        If it None, shape parameters are initialized using the `init_params` method.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    warm_start : bool, default to False.
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several time on similar problems.

    verbose : int, default to 0.
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default to 10.
        Number of iteration done before the next print.

    Attributes
    ----------
    weights_ : array-like, shape (n_components,)
        The weights of each mixture components.

    params_ : array-like, shape (n_components, n_features, n_shape_params)
        The mean of each mixture component.

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.
        If n_components is 1, no EM must be performed, in this case, set to 1.

    lower_bound_ : float
        Log-likelihood of the best fit of EM.

    See Also
    --------
    BayesianGaussianMixture : Gaussian mixture model fit with a variational
        inference.
    """

    # TODO: when sampling assign learned class instead of ordinals
    # (this can be done by an overload that hooks to the super class)

    def __init__(self, stat, n_components=1, tol=1e-3, max_iter=100, n_init=1, init_params='kmeans', classes_init=None,
                 weights_init=None, use_weights=True, params_init=None, random_state=None, warm_start=False, verbose=0,
                 verbose_interval=10, mv_stat=None):
        BaseMixture.__init__(self,
                             n_components=n_components, tol=tol, reg_covar=0, 
                             max_iter=max_iter, n_init=n_init, init_params=init_params,
                             random_state=random_state, warm_start=warm_start,
                             verbose=verbose, verbose_interval=verbose_interval)
        MixtureClassifierMixin.__init__(self)
        self.stat = stat
        self.mv_stat = _infer_stat_type(stat, mv_stat)
        self.n_components_per_class = 1
        self.params_init = params_init
        self.use_weights = use_weights
        self.weights_init = weights_init
        self.classes_init = classes_init

    def decision_function(self, X):
        """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        self._check_is_fitted()
        # X = _check_X(X, n_features=self.means_.shape[1]) TODO
        X = _check_X(X)
        if self.use_weights:
            ret = self._estimate_weighted_log_prob(X)
        else:
            ret = self._estimate_log_prob(X)
        return ret

    def _get_parameters(self):
        return (self.weights_, self.params_)

    def _set_parameters(self, params):
        (self.weights_, self.params_) = params

    def _check_parameters(self, X):
        """Check the mixture parameters are well defined."""
        _, n_features = X.shape

        if self.weights_init is not None:
            self.weights_init = _check_weights(self.weights_init,
                                               self.n_components)

    def _initialize(self, X, resp, *arg, **kwarg):
        """Initialization of the Gaussian mixture parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        resp : array-like, shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        if self.mv_stat:
            weights, params = _estimate_mv_stat_parameters(
                self.stat, X, resp)  # self.reg_covar
        else:
            weights, params = _estimate_1d_stat_parameters(
                self.stat, X, resp)  # self.reg_covar
        weights /= n_samples

        self.weights_ = (weights if self.weights_init is None
        else self.weights_init)
        self.params_ = params if self.params_init is None else self.params_init

    # TODO: def _e_step(self, X):
    # override gaussian

    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape

        if self.mv_stat:
            self.weights_, self.params_ = (
                _estimate_mv_stat_parameters(self.stat, X, np.exp(log_resp)))
        else:
            self.weights_, self.params_ = (
                _estimate_1d_stat_parameters(self.stat, X, np.exp(log_resp)))
        self.weights_ /= n_samples
        # self.precisions_cholesky_ = _compute_precision_cholesky(
        #    self.covariances_, self.covariance_type)

    def _estimate_log_prob(self, X):
        if self.mv_stat:
            return _estimate_log_mv_stat_prob(self.stat, X, self.params_)
        else:
            return _estimate_log_1d_stat_prob(self.stat, X, self.params_)

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _check_is_fitted(self):
        check_is_fitted(self, ['params_'])

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        raise NotImplementedError

    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        bic : float
            The lower the better.
        """
        raise NotImplementedError

    def aic(self, X):
        """Akaike information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        aic : float
            The lower the better.
        """
        raise NotImplementedError

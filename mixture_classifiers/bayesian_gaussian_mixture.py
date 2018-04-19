"""Fully Bayesian Gaussian Mixture Model."""

#
# Author: Marcell Stippinger
# License: BSD 3 clause
#

# Notes:
# This submodule was inspired by and based on scikit-learn mixture models

import numpy as np

# sklearn.mixture.gaussian_mixture
from sklearn.mixture.gaussian_mixture import GaussianMixture, _compute_precision_cholesky
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
# sklearn.mixture.base
from sklearn.mixture.base import _check_X
# classifiers

from .base import MixtureClassifierMixin


def _no_progress_bar(x, *args, **kwargs):
    return x


###############################################################################
# Gaussian mixture parameters sampling (used by the decision function)

def _sample_gaussian_parameters_full(count_mean, hyper_mean, count_covar, hyper_covar, random_state):
    """Estimate the full covariance matrices.

    Parameters
    ----------
    count_mean : array-like, shape (n_components,)

    hyper_mean : array-like, shape (n_components, n_features)

    count_covar : array-like, shape (n_components,)

    hyper_covar : array-like, shape (n_components, n_features, n_features)

    random_state : int, RandomState

    Returns
    -------
    means : array, shape (n_components, n_features,)
        The mean vector of the current components.

    covariances : array, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.
    """
    from scipy.stats import invwishart, wishart, multivariate_normal as mnorm
    # precisions = [wishart.rvs(n, pre, random_state=random_state) for pre, n in zip(hyper_precis, count_precis)]
    covariances = [invwishart.rvs(n, cov, random_state=random_state) for cov, n in zip(hyper_covar, count_covar)]
    means = [mnorm.rvs(m, cov / n, random_state=random_state) for m, cov, n in zip(hyper_mean, covariances, count_mean)]
    return means, covariances


def _sample_gaussian_parameters_tied(count_mean, hyper_mean, count_covar, hyper_covar, random_state):
    """Estimate the full covariance matrices.

    Parameters
    ----------
    count_mean : array-like, shape (n_components,)

    hyper_mean : array-like, shape (n_components, n_features)

    count_covar : array-like, shape (1,)

    hyper_covar : array-like, shape (n_features, n_features)

    random_state : int, RandomState

    Returns
    -------
    means : array, shape (n_components, n_features,)
        The mean vector of the current components.

    covariances : array, shape (n_features, n_features)
        The covariance matrix of the current components.
    """
    from scipy.stats import invwishart, wishart, multivariate_normal as mnorm
    # precisions = wishart.rvs(np.sum(count_precis), hyper_precis, random_state=random_state)
    covariances = invwishart.rvs(np.sum(count_covar), hyper_covar, random_state=random_state)
    means = [mnorm.rvs(m, covariances / n, random_state=random_state) for m, n in zip(hyper_mean, count_mean)]
    return means, covariances


def _sample_gaussian_parameters_diag(count_mean, hyper_mean, count_covar, hyper_covar, random_state):
    """Estimate the full covariance matrices.

    Parameters
    ----------
    count_mean : array-like, shape (n_components,)

    hyper_mean : array-like, shape (n_components, n_features)

    count_covar : array-like, shape (1,)

    hyper_covar : array-like, shape (n_components, n_features)

    random_state : int, RandomState

    Returns
    -------
    means : array, shape (n_components, n_features,)
        The mean vector of the current components.

    covariances : array, shape (n_components, n_features)
        The covariance matrix of the current components.
    """
    from scipy.stats import invwishart, wishart, norm
    # precisions = [np.diag(wishart.rvs(n, np.diag(pre), random_state=random_state))
    #               for pre, n in zip(hyper_precis, count_precis)]
    covariances = [np.diag(invwishart.rvs(n, np.diag(cov), random_state=random_state))
                   for cov, n in zip(hyper_covar, count_covar)]
    means = [norm.rvs(m, cov / n, random_state=random_state) for m, cov, n in zip(hyper_mean, covariances, count_mean)]
    return means, covariances


def _sample_gaussian_parameters_spherical(count_mean, hyper_mean, count_covar, hyper_covar, random_state):
    """Estimate the full covariance matrices.

    Parameters
    ----------
    count_mean : array-like, shape (n_components,)

    hyper_mean : array-like, shape (n_components, n_features)

    count_covar : array-like, shape (1,)

    hyper_covar : array-like, shape (n_components,)

    random_state : int, RandomState

    Returns
    -------
    means : array, shape (n_components, n_features,)
        The mean vector of the current components.

    covariances : array, shape (n_components,)
        The covariance matrix of the current components.
    """
    from scipy.stats import invwishart, wishart, norm
    # precisions = [wishart.rvs(n, pre, random_state=random_state) for pre, n in zip(hyper_precis, count_precis)]
    covariances = [invwishart.rvs(n, cov, random_state=random_state) for cov, n in zip(hyper_covar, count_covar)]
    means = [norm.rvs(m, cov / n, random_state=random_state) for m, cov, n in zip(hyper_mean, covariances, count_mean)]
    return means, covariances


def _sample_gaussian_parameters(count_mean, hyper_mean, count_covar, hyper_covar, covariance_type, random_state):
    """Estimate the Gaussian distribution parameters.

    Parameters
    ----------
    count_mean : array-like, shape (n_components,)
        The number of observations used to establish the hyperparameter mean.

    hyper_mean : array-like, shape (n_components, n_features)
        The hyperparameter mean.

    count_covar : array-like
        The number of observations used to establish the hyperparameter covariances.

    hyper_covar : array-like
        The hyperparameter covariances.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    means : array-like, shape (n_components, n_features)
        The centers of the current components.

    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    """
    means, covariances = {"full": _sample_gaussian_parameters_full,
                          "tied": _sample_gaussian_parameters_tied,
                          "diag": _sample_gaussian_parameters_diag,
                          "spherical": _sample_gaussian_parameters_spherical
                          }[covariance_type](count_mean, hyper_mean, count_covar, hyper_covar, random_state)
    return np.array(means), np.array(covariances)


class GaussianClassifier(MixtureClassifierMixin, GaussianMixture):
    """Gaussian Mixture.

    Representation of a Gaussian mixture model probability distribution.
    This class allows classification based on a Gaussian mixture
    components. The main difference to GaussianMixture is that
    responsibilities are set based on class label. This introduces some
    computational overhead but opens the door towards classification
    between Gaussian mixture distributions. However, this will need to
    set the number of mixture components for each class.

    Read more in the :ref:`User Guide <gmm>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_components : int or array-like shape (n_components,), defaults to 1.
        The number of mixture components per class. TODO: IMPLEMENT OTHER THAN 1.

    covariance_type : {'full', 'tied', 'diag', 'spherical'},
            defaults to 'full'.
        String describing the type of covariance parameters to use.
        Must be one of::

            'full' (each component has its own general covariance matrix),
            'tied' (all components share the same general covariance matrix),
            'diag' (each component has its own diagonal covariance matrix),
            'spherical' (each component has its own single variance).

    tol : float, defaults to 1e-3.
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.

    reg_covar : float, defaults to 1e-6.
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

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
        not 1. TO BE IMPLEMENTED

    weights_init : array-like, shape (n_components, ), optional
        The user-provided initial weights, defaults to None.
        If it None, weights are initialized using the `init_params` method.

    use_weights: bool, optional
        If set to false, do not use weights for prediction (useful if classes
        have different weights in the training and test set)

    means_init : array-like, shape (n_components, n_features), optional
        The user-provided initial means, defaults to None,
        If it None, means are initialized using the `init_params` method.

    precisions_init : array-like, optional.
        The user-provided initial precisions (inverse of the covariance
        matrices), defaults to None.
        If it None, precisions are initialized using the 'init_params' method.
        The shape depends on 'covariance_type'::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

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

    means_ : array-like, shape (n_components, n_features)
        The mean of each mixture component.

    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_ : array-like
        The precision matrices for each component in the mixture. A precision
        matrix is the inverse of a covariance matrix. A covariance matrix is
        symmetric positive definite so the mixture of Gaussian can be
        equivalently parameterized by the precision matrices. Storing the
        precision matrices instead of the covariance matrices makes it more
        efficient to compute the log-likelihood of new samples at test time.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_cholesky_ : array-like
        The cholesky decomposition of the precision matrices of each mixture
        component. A precision matrix is the inverse of a covariance matrix.
        A covariance matrix is symmetric positive definite so the mixture of
        Gaussian can be equivalently parameterized by the precision matrices.
        Storing the precision matrices instead of the covariance matrices makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

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

    def __init__(self, n_components=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans', n_integral_points=100,
                 classes_init=None, counts_init=None,
                 weights_init=None, use_weights=True, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 progress_bar=None, verbose=0, verbose_interval=10):
        GaussianMixture.__init__(self,
                                 n_components=n_components, covariance_type=covariance_type, tol=tol,
                                 reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params,
                                 weights_init=weights_init, means_init=means_init, precisions_init=precisions_init,
                                 random_state=random_state, warm_start=warm_start, verbose=verbose,
                                 verbose_interval=verbose_interval)
        MixtureClassifierMixin.__init__(self)
        self.n_integral_points = n_integral_points
        self.use_weights = use_weights
        self.classes_init = classes_init
        self.counts_init = counts_init
        self.progress_bar = _no_progress_bar if progress_bar is None else progress_bar

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
        X = _check_X(X, n_features=self.means_.shape[1])
        n_integral_points = self.n_integral_points
        saved_params = self._get_parameters()
        ret = np.array([[0]], dtype=float)

        for base_params in self.progress_bar(
                self._sample_base_params(saved_params, n_integral_points)):
            super(GaussianClassifier, self)._set_parameters(base_params)
            if self.use_weights:
                ret = self._estimate_weighted_log_prob(X) + ret
            else:
                ret = self._estimate_log_prob(X) + ret
        self._set_parameters(saved_params)
        return ret / n_integral_points

    def _sample_base_params(self, hyper_params, n_integral_points):
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
        (classes_, counts_, weights_, means_, covariances_,
         precisions_cholesky_) = hyper_params

        for i in range(n_integral_points):
            means, covariances = _sample_gaussian_parameters(
                counts_, means_, counts_, covariances_, self.covariance_type, self.random_state)
            precisions_cholesky = _compute_precision_cholesky(
                covariances, self.covariance_type)
            yield weights_, means, covariances, precisions_cholesky

    def fit(self, X, y):
        classes_, self.counts_ = np.unique(y, return_counts=True)

        # Delegate most of parameter checks
        super(GaussianClassifier, self).fit(X, y)

        if not np.all(self.classes_ == classes_):
            raise ValueError('Implementation inconsistent, classes returned in different order')

    # def predict(self, X):
    #    return MixtureClassifierMixin.predict(self, X)

    def _get_parameters(self):
        return (self.classes_, self.counts_, self.weights_, self.means_, self.covariances_,
                self.precisions_cholesky_)

    def _set_parameters(self, params):
        (self.classes_, self.counts_, *base_params) = params
        super(GaussianClassifier, self)._set_parameters(base_params)


def _exampleNormal(n_features, beta_param=1, random_state=None):
    from scipy.stats import beta, expon
    # partial correlations between [-1,1]
    P = beta.rvs(beta_param, beta_param, size=(n_features, n_features),
                 random_state=random_state) * 2 - 1
    amp = np.diag(np.sqrt(expon.rvs(size=(n_features,), random_state=random_state)))
    return np.dot(np.dot(amp, vineCorr(P)), amp)


def _fullCorr(n_components, n_features, beta_param=1, random_state=None):
    corr = [_exampleNormal(n_features, beta_param, random_state) for _ in range(0, n_components)]
    inv = [np.linalg.inv(a) for a in corr]
    return np.array(corr), np.array(inv)


def _tiedCorr(n_components, n_features, beta_param=1, random_state=None):
    del n_components
    corr = _exampleNormal(n_features, beta_param, random_state)
    inv = np.linalg.inv(corr)
    return corr, inv


def _diagCorr(n_components, n_features, expon_param=1, random_state=None):
    from scipy.stats import expon
    corr = expon.rvs(0, expon_param, size=(n_components, n_features), random_state=random_state)
    inv = 1.0 / corr
    return corr, inv


def _sphericalCorr(n_components, n_features, expon_param=1, random_state=None):
    from scipy.stats import expon
    del n_features
    corr = expon.rvs(0, expon_param, size=(n_components,), random_state=random_state)
    inv = 1.0 / corr
    return corr, inv


def exampleClassifier(n_components, n_features, covariance_type='full', random_state=None):
    from scipy.stats import norm
    weights = np.full((n_components,), 1.0 / n_components)
    means = norm.rvs(0, 1, size=(n_components, n_features), random_state=random_state)
    covariances, precisions = {'full': _fullCorr, 'tied': _tiedCorr, 'diag': _diagCorr,
                               'spherical': _sphericalCorr}[covariance_type](n_components, n_features,
                                                                             random_state=random_state)

    clf = GaussianClassifier(n_components=n_components, covariance_type=covariance_type, random_state=random_state)
    clf._set_parameters((weights, means, covariances, precisions))
    return clf

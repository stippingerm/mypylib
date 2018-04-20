"""Fully Bayesian Gaussian Mixture Model."""

#
# Author: Marcell Stippinger
# License: BSD 3 clause
#

# Notes:
# This submodule was inspired by and based on scikit-learn mixture models

import numpy as np

from scipy.special import gammaln, log1p
# sklearn.mixture.gaussian_mixture
from sklearn.mixture.gaussian_mixture import GaussianMixture, _compute_precision_cholesky, _compute_log_det_cholesky
from .simple_gaussian_mixture import _fullCorr, _tiedCorr, _diagCorr, _sphericalCorr

from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
# sklearn.mixture.base
from sklearn.mixture.base import _check_X
from sklearn.utils.extmath import row_norms
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
        The hyperparameter covariances. The hyperparameter in the Normal-(inverse-)Wishart
        is the sum of pairwise deviations, i.e., {count_covar[i] * hyper_covar[i] for all i}

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


def _estimate_log_gaussian_prob(X, count_mean, means, count_precis, precisions_chol, covariance_type):
    """Estimate the full Bayesian log Gaussian probability.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    count_mean : array-like, shape (n_components,)

    means : array-like, shape (n_components, n_features)

    count_precis : array-like, shape (n_components,)

    precisions_chol : array-like,
        Cholesky decompositions of the precision matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    Returns
    -------
    mx_product : array, shape (n_samples, n_components)

    Notes
    -----
    This is the posterior predictive of the Normal-(inverse-)Wishart distribution.
    It is a multivariate Student's t distribution with
    * df = count_covar - rank + 1
    * mean = mean
    * scale = [ (count_mean + 1) / ( count_mean * (count_covar-rank+1) ) ] * ( count_covar * covar )
      where the last term is the sum pairwise deviation products which is the parameter of
      the Normal-(inverse-)Wishart distribution instead of the covariance)
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    df = count_precis - n_features + 1.0

    if covariance_type == 'tied':
        scale_pref = (np.sum(count_mean) + 1.0) * np.sum(count_precis) / (
                (np.sum(count_precis) - n_features + 1.0) * np.sum(count_mean))
        scale_mx = precisions_chol / np.sqrt(scale_pref)
    else:
        scale_pref = (count_mean + 1.0) * count_precis / ((count_precis - n_features + 1.0) * count_mean)
        scale_mx = np.array([p/np.sqrt(s) for p,s in zip(precisions_chol,scale_pref)])
    # det(precision_chol) is half of det(precision)
    log_det = _compute_log_det_cholesky(
        scale_mx, covariance_type, n_features)

    if covariance_type == 'full':
        mx_product = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, scale_mx)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            mx_product[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'tied':
        mx_product = np.empty((n_samples, n_components))
        for k, mu in enumerate(means):
            y = np.dot(X, scale_mx) - np.dot(mu, scale_mx)
            mx_product[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'diag':
        precisions = scale_mx ** 2
        mx_product = (np.sum((means ** 2 * precisions), 1) -
                      2. * np.dot(X, (means * precisions).T) +
                      np.dot(X ** 2, precisions.T))

    elif covariance_type == 'spherical':
        precisions = scale_mx ** 2
        mx_product = (np.sum(means ** 2, 1) * precisions -
                      2 * np.dot(X, means.T * precisions) +
                      np.outer(row_norms(X, squared=True), precisions))

    gams = gammaln(0.5 * (df + n_features)) - gammaln(0.5 * df)
    return gams - 0.5 * (n_features * np.log(df * np.pi) +
                         (df + n_features) * log1p(mx_product / df)) + log_det

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
        ret : array, shape (n_samples, n_classes)
            Component likelihoods.
        """
        self._check_is_fitted()
        X = _check_X(X, n_features=self.means_.shape[1])
        n_integral_points = self.n_integral_points
        if n_integral_points > 0:
            saved_params = self._get_parameters()
            ret = self._sample_decision_function(X, saved_params, n_integral_points)
        else:
            if self.use_weights:
                # ret = self._estimate_weighted_log_prob(X)
                ret = self._estimate_log_bayesian_prob(X) + self._estimate_log_weights()
            else:
                ret = self._estimate_log_bayesian_prob(X)
        return ret

    def _sample_decision_function(self, X, hyper_params, n_integral_points):
        """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        ret : array, shape (n_samples, n_classes)
            Component likelihoods.
        """
        self._check_is_fitted()
        X = _check_X(X, n_features=self.means_.shape[1])
        ret = np.array([[0]], dtype=float)

        from .simple_gaussian_mixture import GaussianClassifier as SGM
        sgm = SGM(covariance_type=self.covariance_type, use_weights=self.use_weights)
        for base_params in self.progress_bar(
                self._sample_base_params(hyper_params, n_integral_points)):
            sgm._set_parameters(base_params)
            if self.use_weights:
                ret = ret + sgm._estimate_weighted_log_prob(X)
            else:
                ret = ret + sgm._estimate_log_prob(X)
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
            yield classes_, weights_, means, covariances, precisions_cholesky

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

    def _check_log_prob(self, X):
        # for debug purposes only
        from multivariate_distns import multivariate_t as MVT
        e1 = _estimate_log_gaussian_prob(
            X, self.counts_, self.means_, self.counts_, self.precisions_cholesky_, self.covariance_type)
        e2 = np.empty_like(e1)
        d = len(self.means_[0])
        if self.covariance_type == "full":
            covariances_ = self.covariances_
        if self.covariance_type == "tied":
            covariances_ = [self.covariances_ for k in self.counts_]
        if self.covariance_type == "diag":
            covariances_ = [np.diag(c) for c in self.covariances_]
        if self.covariance_type == "spherical":
            covariances_ = [np.diag(c*np.ones(d)) for c in self.covariances_]
        for i, (k, m, c) in enumerate(zip(self.counts_, self.means_, covariances_)):
            d = len(m)
            df = k - d + 1
            cov = (k+1)/(k*(k-d+1)) * k*c
            mvt = MVT.multivariate_t(df, m, cov)
            e2[:, i] = mvt.logpdf(X)
        pass

    def _estimate_log_bayesian_prob(self, X):
        #self._check_log_prob(X)
        return _estimate_log_gaussian_prob(
            X, self.counts_, self.means_, self.counts_, self.precisions_cholesky_, self.covariance_type)


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

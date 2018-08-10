"""Gaussian Mixture Model."""

#
# Author: Marcell Stippinger
# License: BSD 3 clause
#

# Notes:
# This submodule was inspired by and based on scikit-learn mixture models

import numpy as np

# sklearn.mixture.gaussian_mixture
from sklearn.mixture.gaussian_mixture import GaussianMixture, _estimate_gaussian_covariances_tied, \
    _compute_precision_cholesky
from sklearn.utils import check_random_state
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
# sklearn.mixture.base
from sklearn.mixture.base import _check_X
# classifiers

from sklearn.model_selection import StratifiedKFold
from .base import MixtureClassifierMixin


def _isSquareArray(P):
    P = np.asarray(P)
    return (P.ndim == 2) and np.all(np.array(P.shape) == P.shape[0])


def vineCorr(P):
    '''Calculate correlation matrix from partial correlations
    Note: to get a covariance matrix you must multiply this by
    the variance of each component
    :param P: array-like, shape (d,d)
              symmetric 2d array of partial correlations
    :return: array-like, shape (d,d)
             correlation matrix
    '''
    # For the construction see https://stats.stackexchange.com/a/125020
    # and https://stats.stackexchange.com/a/125017
    # Partial correlations and the recursive formula is explained in
    # https://en.wikipedia.org/wiki/Partial_correlation#Using_recursive_formula
    # all we need is the partial correlations distributed between [-1,1]

    P = np.atleast_2d(P)
    if not _isSquareArray(P):
        raise AttributeError('P must be a square matrix')

    d = P.shape[0]
    S = np.eye(d)

    for k in range(0, d - 1):
        for i in range(k + 1, d):
            p = P[k, i]
            for l in range((k - 1), -1, -1):  # converting partial correlation to raw correlation
                p *= np.sqrt((1 - P[l, i] ^ 2) * (1 - P[l, k] ^ 2)) + P[l, i] * P[l, k]
            S[k, i] = p
            S[i, k] = p

    # permuting the variables to make the distribution permutation-invariant
    permutation = np.random.permutation(d)
    S = S[permutation, :][:, permutation]
    return S


def _estimate_gaussian_covariances_tidi(resp, X, nk, means, reg_covar):
    """Estimate the tied covariance matrix.

    Parameters
    ----------
    resp : array-like, shape (n_samples, n_components)

    X : array-like, shape (n_samples, n_features)

    nk : array-like, shape (n_components,)

    means : array-like, shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariance : array, shape (1, n_features)
        The tied diagonal covariance matrix of the components.
        Note: shaoe conforms diagonal calculations.
    """
    avg_X2 = np.sum(X * X, axis=0)
    avg_means2 = np.dot(nk, means*means)
    covariance = avg_X2 - avg_means2
    covariance /= nk.sum()
    covariance += reg_covar
    return covariance[np.newaxis, :]


def _estimate_tidi_gaussian_parameters(X, resp, reg_covar, covariance_type):
    """Estimate the Gaussian distribution parameters.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input data array.

    resp : array-like, shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    reg_covar : float
        The regularization added to the diagonal of the covariance matrices.

    covariance_type : {'diag'}
        The type of precision matrices.

    Returns
    -------
    nk : array-like, shape (n_components,)
        The numbers of data samples in the current components.

    means : array-like, shape (n_components, n_features)
        The centers of the current components.

    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    """
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    # TODO: when megring code to sklearn, change this dict key
    covariances = {"diag": _estimate_gaussian_covariances_tidi,
                   }[covariance_type](resp, X, nk, means, reg_covar)
    return nk, means, covariances


def _estimate_fair_gaussian_parameters(X, resp, reg_covar, covariance_type, random_state):
    """Estimate the Gaussian distribution parameters.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input data array.

    resp : array-like, shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    reg_covar : float
        The regularization added to the diagonal of the covariance matrices.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    random_state : RandomState instance
        the random number generator;

    Returns
    -------
    nk : array-like, shape (n_components,)
        The numbers of data samples in the current components for mean estimation.

    nk_fair : array-like, shape (n_components,)
        The numbers of data samples in the current components for covariance estimation.

    means : array-like, shape (n_components, n_features)
        The centers of the current components.

    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    """
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    # Subsample data to be comparable with "full" that has fewer data per component
    n_samples, n_components = resp.shape
    uni = np.unique(resp)
    if uni.shape == (2,) and np.all(uni == [0, 1]):
        labels_ = [hash(tuple(x)) for x in resp]
        skf = StratifiedKFold(n_splits=n_components, shuffle=True, random_state=random_state)
        _, select = next(skf.split(X, labels_))
    else:
        select = random_state.choice(n_samples, int(n_samples / n_components), replace=False)
    X_fair = X[select, :]
    resp_fair = resp[select, :]
    nk_fair = resp_fair.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    # NOTE: it would be appealing to use the more precise means here too
    # but it may result in an invalid covariance matrix
    means_fair = np.dot(resp_fair.T, X_fair) / nk_fair[:, np.newaxis]
    covariances = {"tied": _estimate_gaussian_covariances_tied,
                   }[covariance_type](resp_fair, X_fair, nk_fair, means_fair, reg_covar)
    return nk, nk_fair, means, covariances


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
    n_components_per_class : int or array-like shape (n_components,), defaults to 1.
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

    # NOTE: sample() is in sklearn.mixture.gaussian_mixture.BaseMixture but uses properties
    # specific to sklearn.mixture.gaussian_mixture.GaussianMixture

    # TODO: when sampling assign learned class instead of ordinals
    # (this can be done by an overload that hooks to the super class)

    def __init__(self, n_components_per_class=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans', use_weights=True,
                 classes_init=None, weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10):
        GaussianMixture.__init__(self,
                                 n_components=n_components_per_class, covariance_type=covariance_type, tol=tol,
                                 reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params,
                                 weights_init=weights_init, means_init=means_init, precisions_init=precisions_init,
                                 random_state=random_state, warm_start=warm_start, verbose=verbose,
                                 verbose_interval=verbose_interval)
        MixtureClassifierMixin.__init__(self)
        self.use_weights = use_weights
        self.classes_init = classes_init
        self.n_components_per_class = n_components_per_class

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
        if self.use_weights:
            ret = self._estimate_weighted_log_prob(X)
        else:
            ret = self._estimate_log_prob(X)
        return ret

    def sample(self, n_samples=1):
        """Generate random samples from the fitted Gaussian distribution.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample

        y : array, shape (nsamples,)
            Component labels

        """
        X, y = super(GaussianClassifier, self).sample(n_samples)
        return self._relabel_samples(X, y)

    def _get_parameters(self):
        base_params = super(GaussianClassifier, self)._get_parameters()
        return (self.classes_, *base_params)

    def _set_parameters(self, params):
        """
        Parameters
        ----------
        classes_
        weight_concentration_
        means_,
        covariances_,
        precisions_cholesky_
        """
        (self.classes_, *base_params) = params
        self.classes_ = np.array(self.classes_)
        super(GaussianClassifier, self)._set_parameters(base_params)
        self.n_components = len(self.means_)


class FairTiedClassifier(GaussianClassifier):
    def __init__(self, n_components=1, covariance_type='tied', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 classes_init=None, weights_init=None, use_weights=True, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10):
        if covariance_type != 'tied':
            raise ValueError('Fair covariance estimation may be requested for tied covariances only.')
        GaussianClassifier.__init__(self,
                                    n_components_per_class=n_components, covariance_type=covariance_type, tol=tol,
                                    reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params,
                                    classes_init=classes_init, weights_init=weights_init, use_weights=use_weights,
                                    means_init=means_init, precisions_init=precisions_init,
                                    random_state=random_state, warm_start=warm_start, verbose=verbose,
                                    verbose_interval=verbose_interval)

    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        # TODO: provide reliable class info to stratified subsampling if multiple components per class used.
        n_samples, _ = X.shape
        random_state = check_random_state(self.random_state)
        self.weights_, _, self.means_, self.covariances_ = (
            _estimate_fair_gaussian_parameters(X, np.exp(log_resp), self.reg_covar,
                                               self.covariance_type, random_state))
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type)


class DiagTiedClassifier(GaussianClassifier):
    def __init__(self, n_components=1, covariance_type='tied', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 classes_init=None, weights_init=None, use_weights=True, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10):
        if covariance_type != 'diag':
            raise ValueError('Diag tied estimation may be requested for "diag" covariances only.')
        GaussianClassifier.__init__(self,
                                    n_components_per_class=n_components, covariance_type=covariance_type, tol=tol,
                                    reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params,
                                    classes_init=classes_init, weights_init=weights_init, use_weights=use_weights,
                                    means_init=means_init, precisions_init=precisions_init,
                                    random_state=random_state, warm_start=warm_start, verbose=verbose,
                                    verbose_interval=verbose_interval)

    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        # TODO: provide reliable class info to stratified subsampling if multiple components per class used.
        n_samples, _ = X.shape
        random_state = check_random_state(self.random_state)
        self.weights_, self.means_, self.covariances_ = (
            _estimate_tidi_gaussian_parameters(X, np.exp(log_resp), self.reg_covar, self.covariance_type))
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type)


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
    classes = np.arange(n_components)
    clf = GaussianClassifier(n_components_per_class=1, covariance_type=covariance_type, random_state=random_state)
    clf._set_parameters((classes, weights, means, covariances, precisions))
    return clf

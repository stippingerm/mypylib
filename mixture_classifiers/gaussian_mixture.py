"""Gaussian Mixture Model."""

#
# Author: Marcell Stippinger
# License: BSD 3 clause
#

# Notes:
# This submodule was inspired by and based on scikit-learn mixture models

import numpy as np

# sklearn.mixture.gaussian_mixture
from sklearn.mixture.gaussian_mixture import GaussianMixture as _BaseGaussianMixture, \
    _estimate_gaussian_covariances_full, _estimate_gaussian_covariances_tied, \
    _estimate_gaussian_covariances_diag, _estimate_gaussian_covariances_spherical, \
    _compute_precision_cholesky
from sklearn.utils import check_random_state
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
# sklearn.mixture.base
from sklearn.mixture.base import _check_X
# classifiers

from sklearn.model_selection import StratifiedKFold
from .base import MixtureClassifierMixin

from collections import namedtuple

_feature_mapper = namedtuple('_feature_mapper', ['shape', 'tied', 'fair'])

# Effective covariance --> Cov shape in sklearn API; same among components; reduce data amount to 1/n_components
# (listed here in order of increasing complexity and precision)
_feature_mapping = {
    'fair spherical': _feature_mapper('spherical', True, True),
    'tied spherical': _feature_mapper('spherical', True, False),
    'spherical': _feature_mapper('spherical', False, False),
    'fair diag': _feature_mapper('diag', True, True),
    'tied diag': _feature_mapper('diag', True, False),
    'diag': _feature_mapper('diag', False, False),
    'fair': _feature_mapper('tied', True, True),  # covariance values are shared
    'tied': _feature_mapper('tied', True, False),
    'fair corr': _feature_mapper('full', True, True),  # correlation coefficients are shared
    'tied corr': _feature_mapper('full', True, False),
    'full': _feature_mapper('full', False, False),  # nothing shared
}


# Note: although some ties would allow simpler shapes (i.e., for spherical and diag) but rather
# we use fallback to a broader shape to spare the effort of implementing the corresponding calculations.


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


def _estimate_gaussian_covariances_tied_diag(resp, X, nk, means, reg_covar):
    """Estimate the diagonal-only tied covariance matrix.

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
        Note: shape conforms diagonal calculations.
    """
    del resp
    avg_X2 = np.sum(X * X, axis=0)
    avg_means2 = np.dot(nk, means * means)
    covariance = avg_X2 - avg_means2
    covariance /= nk.sum()
    covariance += reg_covar
    return covariance[np.newaxis, :]


def _estimate_gaussian_covariances_tied_spherical(resp, X, nk, means, reg_covar):
    """Estimate the spherical variance values.

    Parameters
    ----------
    responsibilities : array-like, shape (n_samples, n_components)

    X : array-like, shape (n_samples, n_features)

    nk : array-like, shape (n_components,)

    means : array-like, shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    variances : array, shape (1,)
        The variance values of each components.
    """
    return _estimate_gaussian_covariances_tied_diag(resp, X, nk,
                                                    means, reg_covar).mean(1)


def _estimate_gaussian_correlations_tied(resp, X, nk, means, reg_covar,
                                         resp_fair=None, X_fair=None, nk_fair=None, means_fair=None):
    """Estimate the tied correlation matrix. Then obtain covariance matrix by scaling
    it using component-wise variances.

    Parameters
    ----------
    resp : array-like, shape (n_samples, n_components)

    X : array-like, shape (n_samples, n_features)

    nk : array-like, shape (n_components,)

    means : array-like, shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariance : array, shape (n_components, n_features, n_features)
        The correlation-tied covariance matrix of the components.
    """
    # Note: maybe we should require to either provide all or none of the fair parameters
    if resp_fair is None:
        resp_fair = resp
    if X_fair is None:
        X_fair = X
    if nk_fair is None:
        nk_fair = nk
    if means_fair is None:
        means_fair = means
    tied_covariance = _estimate_gaussian_covariances_tied(resp_fair, X_fair, nk_fair, means_fair, reg_covar)
    tied_inv_scaler = 1.0 / np.sqrt(np.diag(tied_covariance))
    tied_correlation = np.outer(tied_inv_scaler, tied_inv_scaler) * tied_covariance
    comp_variances = _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar)
    comp_scaler = [np.outer(c, c) for c in np.sqrt(comp_variances)]
    comp_covariance = [tied_correlation * s for s in comp_scaler]
    return np.array(comp_covariance)


def _subsampled_statistics(X, resp, random_state):
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
    return X_fair, resp_fair, nk_fair, means_fair


def _estimate_gaussian_parameters(X, resp, reg_covar, advanced_covariance_type, random_state):
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

    covariance_type, tied, fair = _feature_mapping[advanced_covariance_type]

    if fair:
        X_fair, resp_fair, nk_fair, means_fair = _subsampled_statistics(X, resp, random_state)
        if covariance_type == 'full':
            covariances = _estimate_gaussian_correlations_tied(
                resp, X, nk, means, reg_covar, resp_fair, X_fair, nk_fair, means_fair
            )
        else:
            covariances = {"tied": _estimate_gaussian_covariances_tied,
                           "diag": _estimate_gaussian_covariances_tied_diag,
                           "spherical": _estimate_gaussian_covariances_tied_spherical
                           }[covariance_type](resp_fair, X_fair, nk_fair, means_fair, reg_covar)
    elif tied:
        covariances = {"full": _estimate_gaussian_correlations_tied,
                       "tied": _estimate_gaussian_covariances_tied,
                       "diag": _estimate_gaussian_covariances_tied_diag,
                       "spherical": _estimate_gaussian_covariances_tied_spherical
                       }[covariance_type](resp, X, nk, means, reg_covar)
        nk_fair = nk
    else:
        covariances = {"full": _estimate_gaussian_covariances_full,
                       "diag": _estimate_gaussian_covariances_diag,
                       "spherical": _estimate_gaussian_covariances_spherical
                       }[covariance_type](resp, X, nk, means, reg_covar)
        nk_fair = nk
    return nk, nk_fair, means, covariances


class GaussianMixture(_BaseGaussianMixture):
    """Gaussian Mixture.

    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution.

    Read more in the :ref:`User Guide <gmm>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_components : int, defaults to 1.
        The number of mixture components.

    covariance_type : {'full' (default), 'tied', 'diag', 'spherical'}
        String describing the type of covariance parameters to use.
        Must be one of:

        'full'
            each component has its own general covariance matrix
        'fair corr', 'tied corr'
            all components share the same general correlation matrix
            while each component scales it to match its own variances
        'fair', 'tied'
            all components share the same general covariance matrix
        'diag'
            each component has its own diagonal covariance matrix
        'fair diag', 'tied diag'
            all components share the same diagonal covariance matrix
        'spherical'
            each component has its own single variance
        'fair spehrical', 'tied spherical'
            all components share the same single variance

        In the above, 'fair' indicates that covariances are estimated
        from data subsampled to the observations contained in an average
        component.

    tol : float, defaults to 1e-3.
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.

    reg_covar : float, defaults to 1e-6.
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    max_iter : int, defaults to 100.
        The number of EM iterations to perform.

    n_init : int, defaults to 1.
        The number of initializations to perform. The best results are kept.

    init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
        The method used to initialize the weights, the means and the
        precisions.
        Must be one of::

            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.

    weights_init : array-like, shape (n_components, ), optional
        The user-provided initial weights, defaults to None.
        If it None, weights are initialized using the `init_params` method.

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
        convergence when fit is called several times on similar problems.
        In that case, 'n_init' is ignored and only a single initialization
        occurs upon the first call.
        See :term:`the Glossary <warm_start>`.

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

    lower_bound_ : float
        Lower bound value on the log-likelihood (of the training data with
        respect to the model) of the best fit of EM.

    See Also
    --------
    BayesianGaussianMixture : Gaussian mixture model fit with a variational
        inference.
    """

    def __init__(self, n_components=1, covariance_type='full', advanced_covariance_type=None, tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10):
        if advanced_covariance_type is None:
            advanced_covariance_type = covariance_type
        else:
            covariance_type = _feature_mapping[advanced_covariance_type].shape
        _BaseGaussianMixture.__init__(self,
                                      n_components=n_components, covariance_type=covariance_type, tol=tol,
                                      reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params,
                                      weights_init=weights_init, means_init=means_init, precisions_init=precisions_init,
                                      random_state=random_state, warm_start=warm_start, verbose=verbose,
                                      verbose_interval=verbose_interval)
        self.advanced_covariance_type = advanced_covariance_type

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
            _estimate_gaussian_parameters(X, np.exp(log_resp), self.reg_covar,
                                          self.advanced_covariance_type, random_state))
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type)


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

    def __init__(self, n_components_per_class=1, covariance_type='full', advanced_covariance_type=None, tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans', use_weights=True,
                 classes_init=None, weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10):
        GaussianMixture.__init__(self,
                                 n_components=n_components_per_class, covariance_type=covariance_type,
                                 advanced_covariance_type=advanced_covariance_type, tol=tol,
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


def FairTiedClassifier(*args, covariance_type='tied', **kwargs):
    if covariance_type != 'tied':
        raise ValueError('Fair covariance estimation may be requested for tied covariances only.')
    return GaussianClassifier(*args, advanced_covariance_type='fair', **kwargs)


def DiagTiedClassifier(*args, covariance_type='tied', **kwargs):
    if covariance_type != 'diag':
        raise ValueError('Diag tied estimation may be requested for "diag" covariances only.')
    return GaussianClassifier(*args, advanced_covariance_type='tied diag', **kwargs)


def CorrTiedClassifier(*args, covariance_type='tied', **kwargs):
    if covariance_type != 'full':
        raise ValueError('Correlation tied estimation may be requested for "full" covariances only.')
    return GaussianClassifier(*args, advanced_covariance_type='tied corr', **kwargs)


def _exampleNormal(n_features, beta_param=1, random_state=None):
    from scipy.stats import beta, expon
    # partial correlations between [-1,1]
    P = beta.rvs(beta_param, beta_param, size=(n_features, n_features),
                 random_state=random_state) * 2 - 1
    amp = np.diag(np.sqrt(expon.rvs(size=(n_features,), random_state=random_state)))
    return np.dot(np.dot(amp, vineCorr(P)), amp)


def _full_cov(n_components, n_features, beta_param=1, random_state=None):
    cov = [_exampleNormal(n_features, beta_param, random_state) for _ in range(0, n_components)]
    inv = [np.linalg.inv(a) for a in cov]
    return np.array(cov), np.array(inv)


def _tied_cov(n_components, n_features, beta_param=1, random_state=None):
    del n_components
    cov = _exampleNormal(n_features, beta_param, random_state)
    inv = np.linalg.inv(cov)
    return cov, inv


def _tied_corr(n_components, n_features, beta_param=1, expon_param=1, random_state=None):
    from scipy.stats import expon
    tied_cov = _exampleNormal(n_features, beta_param, random_state)
    tied_inv_scaler = np.sqrt(np.diag(tied_cov))
    tied_corr = np.outer(tied_inv_scaler, tied_inv_scaler) * tied_cov
    comp_var = expon.rvs(0, expon_param, size=(n_components, n_features), random_state=random_state)
    comp_scaler = [np.outer(c, c) for c in np.sqrt(comp_var)]
    comp_cov = [tied_corr * s for s in comp_scaler]
    cov = np.array(comp_cov)
    inv = np.linalg.inv(cov)
    return cov, inv


def _tidi_cov(n_components, n_features, expon_param=1, random_state=None):
    from scipy.stats import expon
    cov = expon.rvs(0, expon_param, size=(1, n_features), random_state=random_state)
    cov = np.repeat(cov, n_components, axis=0)
    inv = 1.0 / cov
    return cov, inv


def _diag_cov(n_components, n_features, expon_param=1, random_state=None):
    from scipy.stats import expon
    cov = expon.rvs(0, expon_param, size=(n_components, n_features), random_state=random_state)
    inv = 1.0 / cov
    return cov, inv


def _spherical_cov(n_components, n_features, expon_param=1, random_state=None):
    from scipy.stats import expon
    del n_features
    cov = expon.rvs(0, expon_param, size=(n_components,), random_state=random_state)
    inv = 1.0 / cov
    return cov, inv


def exampleClassifier(n_components, n_features, covariance_type='full', random_state=None):
    from scipy.stats import norm
    weights = np.full((n_components,), 1.0 / n_components)
    means = norm.rvs(0, 1, size=(n_components, n_features), random_state=random_state)
    covariances, precisions = {'full': _full_cov, 'tied': _tied_cov, 'diag': _diag_cov,
                               'tidi': _tidi_cov, 'tied_corr': _tied_corr,
                               'spherical': _spherical_cov}[covariance_type](n_components, n_features,
                                                                             random_state=random_state)
    classes = np.arange(n_components)
    clf = GaussianClassifier(n_components_per_class=1, covariance_type=covariance_type, random_state=random_state)
    clf._set_parameters((classes, weights, means, covariances, precisions))
    return clf

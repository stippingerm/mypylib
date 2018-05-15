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
from sklearn.utils.fixes import logsumexp
from sklearn.utils import check_random_state
from sklearn.mixture.gaussian_mixture import GaussianMixture, _compute_precision_cholesky, _compute_log_det_cholesky
from sklearn.mixture.bayesian_mixture import BayesianGaussianMixture as _BaseBayesianGaussianMixture
from .gaussian_mixture import _fullCorr, _tiedCorr, _diagCorr, _sphericalCorr, _estimate_fair_gaussian_parameters

from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
# sklearn.mixture.base
from sklearn.mixture.base import _check_X, _check_shape
from sklearn.utils.extmath import row_norms
# classifiers

from .base import MixtureClassifierMixin


def _no_progress_bar(x, *args, **kwargs):
    return x

def _log_prob_to_resp(weighted_log_prob):
    """From log probabilities and infer responsibilities for each sample.

    From the log probabilities, compute weighted log probabilities per
    component and responsibilities for each sample in X with respect to
    the current state of the model.

    Parameters
    ----------
    weighted_log_prob : array-like, shape (n_samples, n_classes)

    Returns
    -------
    log_prob_norm : array, shape (n_samples,)
        log p(X)

    log_responsibilities : array, shape (n_samples, n_components)
        logarithm of the responsibilities
    """
    log_prob_norm = logsumexp(weighted_log_prob, axis=1)
    with np.errstate(under='ignore'):
        # ignore underflow
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
    return log_prob_norm, log_resp


###############################################################################
# Gaussian mixture parameters sampling (used by the decision function)

def _sample_gaussian_parameters_full(counts_means, hyper_means, counts_covars, hyper_covars, random_state):
    """Estimate the full covariance matrices.

    Parameters
    ----------
    counts_means : array-like, shape (n_components,)

    hyper_means : array-like, shape (n_components, n_features)

    counts_covars : array-like, shape (n_components,)

    hyper_covars : array-like, shape (n_components, n_features, n_features)

    random_state : int, RandomState

    Returns
    -------
    means : array, shape (n_components, n_features,)
        The mean vector of the current components.

    covariances : array, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.
    """
    from scipy.stats import invwishart, wishart, multivariate_normal as mnorm
    # precisions = [wishart.rvs(n, pre / nu, random_state=random_state)
    #               for pre, nu in zip(hyper_precis, count_precis)]
    covariances = [invwishart.rvs(nu, cov * nu, random_state=random_state)
                   for cov, nu in zip(hyper_covars, counts_covars)]
    means = [mnorm.rvs(m, cov / k, random_state=random_state)
             for m, cov, k in zip(hyper_means, covariances, counts_means)]
    return means, covariances


def _sample_gaussian_parameters_tied(counts_means, hyper_means, counts_covars, hyper_covars, random_state):
    """Estimate the full covariance matrices.

    Parameters
    ----------
    counts_means : array-like, shape (n_components,)

    hyper_means : array-like, shape (n_components, n_features)

    counts_covars : array-like, shape (1,)

    hyper_covars : array-like, shape (n_features, n_features)

    random_state : int, RandomState

    Returns
    -------
    means : array, shape (n_components, n_features,)
        The mean vector of the current components.

    covariances : array, shape (n_features, n_features)
        The covariance matrix of the current components.
    """
    from scipy.stats import invwishart, wishart, multivariate_normal as mnorm
    # precisions = wishart.rvs(np.sum(count_precis),
    #                          hyper_precis / np.sum(count_precis), random_state=random_state)
    covariances = invwishart.rvs(np.sum(counts_covars),
                                 hyper_covars * np.sum(counts_covars), random_state=random_state)
    means = [mnorm.rvs(m, covariances / k, random_state=random_state)
             for m, k in zip(hyper_means, counts_means)]
    return means, covariances


def _sample_gaussian_parameters_diag(counts_means, hyper_means, counts_covars, hyper_covars, random_state):
    """Estimate the full covariance matrices.

    Parameters
    ----------
    counts_means : array-like, shape (n_components,)

    hyper_means : array-like, shape (n_components, n_features)

    counts_covars : array-like, shape (1,)

    hyper_covars : array-like, shape (n_components, n_features)

    random_state : int, RandomState

    Returns
    -------
    means : array, shape (n_components, n_features,)
        The mean vector of the current components.

    covariances : array, shape (n_components, n_features)
        The covariance matrix of the current components.
    """
    from scipy.stats import invwishart, wishart, norm
    # precisions = [np.diag(wishart.rvs(nu, np.diag(pre * nu), random_state=random_state))
    #               for pre, nu in zip(hyper_precis, count_precis)]
    covariances = [np.diag(invwishart.rvs(nu, np.diag(cov * nu), random_state=random_state))
                   for cov, nu in zip(hyper_covars, counts_covars)]
    means = [norm.rvs(m, cov / k, random_state=random_state)
             for m, cov, k in zip(hyper_means, covariances, counts_means)]
    return means, covariances


def _sample_gaussian_parameters_spherical(counts_means, hyper_means, counts_covars, hyper_covars, random_state):
    """Estimate the full covariance matrices.

    Parameters
    ----------
    counts_means : array-like, shape (n_components,)

    hyper_means : array-like, shape (n_components, n_features)

    counts_covars : array-like, shape (1,)

    hyper_covars : array-like, shape (n_components,)

    random_state : int, RandomState

    Returns
    -------
    means : array, shape (n_components, n_features,)
        The mean vector of the current components.

    covariances : array, shape (n_components,)
        The covariance matrix of the current components.
    """
    from scipy.stats import invwishart, wishart, norm
    # precisions = [wishart.rvs(nu, pre / nu, random_state=random_state)
    #               for pre, nu in zip(hyper_precis, count_precis)]
    covariances = [invwishart.rvs(nu, cov * nu, random_state=random_state)
                   for cov, nu in zip(hyper_covars, counts_covars)]
    means = [norm.rvs(m, cov / k, random_state=random_state)
             for m, cov, k in zip(hyper_means, covariances, counts_means)]
    return means, covariances


def _sample_gaussian_parameters(counts_means, hyper_means, counts_covars, hyper_covars, covariance_type, random_state):
    """Estimate the Gaussian distribution parameters.

    Parameters
    ----------
    counts_means : array-like, shape (n_components,)
        The number of observations used to establish the hyperparameter mean.

    hyper_means : array-like, shape (n_components, n_features)
        The hyperparameter mean.

    counts_covars : array-like
        The number of observations used to establish the hyperparameter covariances.

    hyper_covars : array-like
        The hyperparameter covariances. The hyperparameter in the Normal-(inverse-)Wishart
        is the sum of pairwise deviations, i.e., {counts_covars[i] * hyper_covars[i] for all i}

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
                          }[covariance_type](counts_means, hyper_means, counts_covars, hyper_covars, random_state)
    return np.array(means), np.array(covariances)


def _estimate_log_t_prob(X, counts_means, means, count_precis, precisions_chol, covariance_type):
    """Estimate the full Bayesian log Gaussian probability.
    This is the posterior predictive of each class, i.e., a multivariate t probability.
    Note: this does not take into account the probability of belonging to that class.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    counts_means : array-like, shape (n_components,)

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
    * df = counts_covars - rank + 1
    * mean = mean
    * scale = [ (counts_means + 1) / ( counts_means * (counts_covars-rank+1) ) ] * ( counts_covars * covar )
      where the last term is the sum pairwise deviation products which is the parameter of
      the Normal-(inverse-)Wishart distribution instead of the covariance)
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape

    if covariance_type == 'tied':
        count_precis = np.sum(count_precis)

    # Contrary to the original bishop book, the shape matrix is normalized,
    # that is, it represents the maximum a posteriori covariances. I believe,
    # the reason for this is to calculate maximum a posteriori probabilities
    # the same way as in sklearn.mixture_models.GaussianMixture.
    df = count_precis - n_features + 1.0
    scale_pref = (counts_means + 1.0) * count_precis / (df * counts_means)
    # det(precision_chol) is half of det(precision)
    log_det = _compute_log_det_cholesky(
        precisions_chol, covariance_type, n_features) - 0.5 * n_components * np.log(scale_pref)

    if covariance_type == 'full':
        mx_product = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            mx_product[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'tied':
        mx_product = np.empty((n_samples, n_components))
        for k, mu in enumerate(means):
            y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
            mx_product[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'diag':
        precisions = precisions_chol ** 2
        mx_product = (np.sum((means ** 2 * precisions), 1) -
                      2. * np.dot(X, (means * precisions).T) +
                      np.dot(X ** 2, precisions.T))

    elif covariance_type == 'spherical':
        precisions = precisions_chol ** 2
        mx_product = (np.sum(means ** 2, 1) * precisions -
                      2 * np.dot(X, means.T * precisions) +
                      np.outer(row_norms(X, squared=True), precisions))

    gams = gammaln(0.5 * (df + n_features)) - gammaln(0.5 * df)

    return gams - 0.5 * (n_features * np.log(df * np.pi) +
                         (df + n_features) * log1p(mx_product / (scale_pref * df))) + log_det


class UninformedPriorChecks(object):
    """Uninformed priors for the Variational Bayesian estimation of a Gaussian mixture.

    This class allows to uninformed priors that do not bias the results in any way. In
    this setting the maximum a posteriori estimates implemented in
    `sklearn.mixture.bayesian_mixture.BayesianGaussianMixture` becomes the same as the
    maximum likelihood estimation of `sklearn.mixture.gaussian_mixture.GaussianMixture`.
    Sampling from the uninformed prior with no data is unadvised / may raise an error.
    Yet, the rationale behind using uninformed prior is to still have access to the
    posterior predictive distribution `_estimate_log_posterior_predictive_prob`.
    """

    def _check_means_parameters(self, X):
        """Check the parameters of the Gaussian distribution.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        _, n_features = X.shape

        if self.mean_precision_prior is None:
            self.mean_precision_prior_ = 0.  # uninformed
        elif self.mean_precision_prior >= 0.:
            self.mean_precision_prior_ = self.mean_precision_prior
        else:
            raise ValueError("The parameter 'mean_precision_prior' should be "
                             "greater than 0., but got %.3f."
                             % self.mean_precision_prior)

        if self.mean_prior is None:
            self.mean_prior_ = X.mean(axis=0)
        else:
            self.mean_prior_ = check_array(self.mean_prior,
                                           dtype=[np.float64, np.float32],
                                           ensure_2d=False)
            _check_shape(self.mean_prior_, (n_features, ), 'means')

    def _check_precision_parameters(self, X):
        """Check the prior parameters of the precision distribution.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        n_samples, n_features = X.shape

        # NOTE: as the data is used to estimate n_classes different components
        # the original least degrees_of_freedom_prior being n_features was not
        # a valid prior either. Since responsibilities were < 1.
        if self.degrees_of_freedom_prior is None:
            self.degrees_of_freedom_prior_ = np.maximum(0., n_features-n_samples)
        elif self.degrees_of_freedom_prior >= n_features - n_samples:
            self.degrees_of_freedom_prior_ = self.degrees_of_freedom_prior
        else:
            raise ValueError("The parameter 'degrees_of_freedom_prior' "
                             "should be greater than %d, but got %.3f."
                             % (n_features - n_samples, self.degrees_of_freedom_prior))


class BayesianGaussianMixture(_BaseBayesianGaussianMixture):
    """Variational Bayesian estimation of a Gaussian mixture.

    This class allows to infer an approximate posterior distribution over the
    parameters of a Gaussian mixture distribution. The effective number of
    components can be inferred from the data.

    This class implements two types of prior for the weights distribution: a
    finite mixture model with Dirichlet distribution and an infinite mixture
    model with the Dirichlet Process. In practice Dirichlet Process inference
    algorithm is approximated and uses a truncated distribution with a fixed
    maximum number of components (called the Stick-breaking representation).
    The number of components actually used almost always depends on the data.

    .. versionadded:: 0.18

    Read more in the :ref:`User Guide <bgmm>`.

    Parameters
    ----------
    n_components : int, defaults to 1.
        The number of mixture components. Depending on the data and the value
        of the `weight_concentration_prior` the model can decide to not use
        all the components by setting some component `weights_` to values very
        close to zero. The number of effective components is therefore smaller
        than n_components.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}, defaults to 'full'
        String describing the type of covariance parameters to use.
        Must be one of::

            'full' (each component has its own general covariance matrix),
            'tied' (all components share the same general covariance matrix),
            'diag' (each component has its own diagonal covariance matrix),
            'spherical' (each component has its own single variance).

    tol : float, defaults to 1e-3.
        The convergence threshold. EM iterations will stop when the
        lower bound average gain on the likelihood (of the training data with
        respect to the model) is below this threshold.

    reg_covar : float, defaults to 1e-6.
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    max_iter : int, defaults to 100.
        The number of EM iterations to perform.

    n_init : int, defaults to 1.
        The number of initializations to perform. The result with the highest
        lower bound value on the likelihood is kept.

    init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
        The method used to initialize the weights, the means and the
        covariances.
        Must be one of::

            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.

    use_weights: bool, optional
        If set to false, do not use weights for prediction (useful if classes
        have different weights in the training and test set)

    weight_concentration_prior_type : str, defaults to 'dirichlet_process'.
        String describing the type of the weight concentration prior.
        Must be one of::

            'dirichlet_process' (using the Stick-breaking representation),
            'dirichlet_distribution' (can favor more uniform weights).

    weight_concentration_prior : float | None, optional.
        The dirichlet concentration of each component on the weight
        distribution (Dirichlet). This is commonly called gamma in the
        literature. The higher concentration puts more mass in
        the center and will lead to more components being active, while a lower
        concentration parameter will lead to more mass at the edge of the
        mixture weights simplex. The value of the parameter must be greater
        than 0. If it is None, it's set to ``1. / n_components``.

    mean_precision_prior : float | None, optional.
        The precision prior on the mean distribution (Gaussian).
        Controls the extend to where means can be placed. Smaller
        values concentrate the means of each clusters around `mean_prior`.
        The value of the parameter must be greater than 0.
        If it is None, it's set to 1.

    mean_prior : array-like, shape (n_features,), optional
        The prior on the mean distribution (Gaussian).
        If it is None, it's set to the mean of X.

    degrees_of_freedom_prior : float | None, optional.
        The prior of the number of degrees of freedom on the covariance
        distributions (Wishart). If it is None, it's set to `n_features`.

    covariance_prior : float or array-like, optional
        The prior on the covariance distribution (Wishart).
        If it is None, the emiprical covariance prior is initialized using the
        covariance of X. The shape depends on `covariance_type`::

                (n_features, n_features) if 'full',
                (n_features, n_features) if 'tied',
                (n_features)             if 'diag',
                float                    if 'spherical'

    n_integral_points : int
        Number of sampled Gaussian Mixtures used for prediction.
        If set to 0 the analytic posterior predictive is used.

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

    progress_bar: callable
        Hook to report progress. The callable should pipe the iterable provided
        as its first argument. It can be used to measure the consumption of the
        iterable.

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
        The shape depends on ``covariance_type``::

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
        time. The shape depends on ``covariance_type``::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    n_iter_ : int
        Number of step used by the best fit of inference to reach the
        convergence.

    lower_bound_ : float
        Lower bound value on the likelihood (of the training data with
        respect to the model) of the best fit of inference.

    weight_concentration_prior_ : tuple or float
        The dirichlet concentration of each component on the weight
        distribution (Dirichlet). The type depends on
        ``weight_concentration_prior_type``::

            (float, float) if 'dirichlet_process' (Beta parameters),
            float          if 'dirichlet_distribution' (Dirichlet parameters).

        The higher concentration puts more mass in
        the center and will lead to more components being active, while a lower
        concentration parameter will lead to more mass at the edge of the
        simplex.

    weight_concentration_ : array-like, shape (n_components,)
        The dirichlet concentration of each component on the weight
        distribution (Dirichlet).

    mean_precision_prior : float
        The precision prior on the mean distribution (Gaussian).
        Controls the extend to where means can be placed.
        Smaller values concentrate the means of each clusters around
        `mean_prior`.

    mean_precision_ : array-like, shape (n_components,)
        The precision of each components on the mean distribution (Gaussian).

    means_prior_ : array-like, shape (n_features,)
        The prior on the mean distribution (Gaussian).

    degrees_of_freedom_prior_ : float
        The prior of the number of degrees of freedom on the covariance
        distributions (Wishart).

    degrees_of_freedom_ : array-like, shape (n_components,)
        The number of degrees of freedom of each components in the model.

    covariance_prior_ : float or array-like
        The prior on the covariance distribution (Wishart).
        The shape depends on `covariance_type`::

            (n_features, n_features) if 'full',
            (n_features, n_features) if 'tied',
            (n_features)             if 'diag',
            float                    if 'spherical'

    use_uninformed_prior : bool
        Use uniformed (completely agnostic) prior that does not introduce
        any bias. When set to True training requires sufficient samples.

    See Also
    --------
    GaussianMixture : Finite Gaussian mixture fit with EM.

    References
    ----------

    .. [1] `Bishop, Christopher M. (2006). "Pattern recognition and machine
       learning". Vol. 4 No. 4. New York: Springer.
       <http://www.springer.com/kr/book/9780387310732>`_

    .. [2] `Hagai Attias. (2000). "A Variational Bayesian Framework for
       Graphical Models". In Advances in Neural Information Processing
       Systems 12.
       <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.36.2841&rep=rep1&type=pdf>`_

    .. [3] `Blei, David M. and Michael I. Jordan. (2006). "Variational
       inference for Dirichlet process mixtures". Bayesian analysis 1.1
       <http://www.cs.princeton.edu/courses/archive/fall11/cos597C/reading/BleiJordan2005.pdf>`_
    """

    # TODO: in _check_precision_parameters the prior ddf is required to be >= n_features
    # but formulas can already be evaluated if that is true for the posterior

    def __init__(self, n_components=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans', use_weights=True,
                 weight_concentration_prior_type='dirichlet_process',
                 weight_concentration_prior=None,
                 mean_precision_prior=None, mean_prior=None,
                 degrees_of_freedom_prior=None, covariance_prior=None,
                 use_uninformed_prior=False,
                 n_integral_points=100, random_state=None, warm_start=False, verbose=0,
                 verbose_interval=10, progress_bar=None):
        super(BayesianGaussianMixture, self).__init__(
            n_components=n_components, covariance_type=covariance_type, tol=tol,
            reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params,
            weight_concentration_prior_type=weight_concentration_prior_type,
            weight_concentration_prior=weight_concentration_prior,
            mean_precision_prior=mean_precision_prior, mean_prior=mean_prior,
            degrees_of_freedom_prior=degrees_of_freedom_prior, covariance_prior=covariance_prior,
            random_state=random_state, warm_start=warm_start, verbose=verbose,
            verbose_interval=verbose_interval)

        self.n_integral_points = n_integral_points
        self.use_weights = use_weights
        self.progress_bar = _no_progress_bar if progress_bar is None else progress_bar
        self.use_uninformed_prior = use_uninformed_prior

    def _check_means_parameters(self, X):
        if self.use_uninformed_prior:
            UninformedPriorChecks._check_means_parameters(self, X)
        else:
            super(BayesianGaussianMixture, self)._check_means_parameters(X)

    def _check_precision_parameters(self, X):
        if self.use_uninformed_prior:
            UninformedPriorChecks._check_precision_parameters(self, X)
        else:
            super(BayesianGaussianMixture, self)._check_precision_parameters(X)

    def _dispatch_weighted_log_prob(self, X):
        """Predict posterior probability of samples just like `_estimate_weighted_log_prob`
        but dispatch to sampling and equalizing class probabilities.

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
        n_integral_points = self.n_integral_points
        if n_integral_points > 0:
            proba = self._MC_log_posterior_predictive(X, n_integral_points)
        else:
            proba = self._estimate_log_posterior_predictive_prob(X)

        if self.use_weights:
            proba += self._estimate_log_weights()
        else:
            proba -= np.log(self.n_components)
        return proba

    def uninform_prior(self, what='all'):
        # TODO: allow completely uninformed prior
        # (officially not supported by bayesian_mixture, but it shall work)
        raise NotImplementedError

    def score_samples(self, X):
        """Compute the weighted log probabilities for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log probabilities of each data point in X.
        """
        self._check_is_fitted()
        X = _check_X(X, None, self.means_.shape[1])

        return logsumexp(self._dispatch_weighted_log_prob(X), axis=1)

    def score(self, X, y=None):
        """Compute the per-sample average log-likelihood of the given data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_dimensions)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        log_likelihood : float
            Log likelihood of the Gaussian mixture given X.
        """
        return self.score_samples(X).mean()

    def predict(self, X):
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
        X = _check_X(X, None, self.means_.shape[1])
        return self._dispatch_weighted_log_prob(X).argmax(axis=1)

    def predict_proba(self, X):
        """Predict posterior probability of each component given the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        resp : array, shape (n_samples, n_components)
            Returns the probability each Gaussian (state) in
            the model given each sample.
        """
        self._check_is_fitted()
        X = _check_X(X, None, self.means_.shape[1])
        _, log_resp = _log_prob_to_resp(self._dispatch_weighted_log_prob(X))
        return np.exp(log_resp)


    def sample(self, n_samples=1, monte_carlo=False):
        """Generate random samples from the fitted Gaussian distribution.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        monte_carlo : bool, optinal
            Use the less efficient Monte Carlo sampling method. Default: False.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample

        y : array, shape (nsamples,)
            Component labels

        """
        if monte_carlo:
            return self._MC_sample_points(n_samples=n_samples)
        else:
            return self._sample_posterior_points(n_samples=n_samples)

    def _sample_posterior_points(self, n_samples):
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
        from multivariate_distns import multivariate_t as MVT
        self._check_is_fitted()

        if n_samples < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at "
                "least one sample." % (self.n_components))

        d, n_features = self.means_.shape
        rng = check_random_state(self.random_state)
        if self.use_weights:
            weights_ = self.weights_
        else:
            weights_ = 1.0/self.n_components
        n_samples_comp = rng.multinomial(n_samples, weights_)

        if self.covariance_type == 'full':
            X = np.vstack([
                MVT.multivariate_t.rvs(nu-d+1, mean,
                                       (k + 1) / (k * (nu-d+1)) * nu * covariance, size=int(sample))
                for (k, mean, nu, covariance, sample) in zip(
                    self.mean_precision_, self.means_,
                    self.degrees_of_freedom_, self.covariances_, n_samples_comp)])
        elif self.covariance_type == "tied":
            nu = np.sum(self.degrees_of_freedom_)
            X = np.vstack([
                MVT.multivariate_t.rvs(nu - d + 1, mean,
                                       (k + 1) / (k * (nu - d + 1)) * nu * self.covariances_, size=int(sample))
                for (k, mean, sample) in zip(
                self.mean_precision_, self.means_, n_samples_comp)])
        elif self.covariance_type == "diag":
            X = np.vstack([
                #  mean + rng.randn(sample, n_features) * np.sqrt(covariance)
                MVT.multivariate_t.rvs(nu - d + 1, mean,
                                       np.diag((k + 1) / (k * (nu - d + 1)) * nu * covariance), size=int(sample))
                for (k, mean, nu, covariance, sample) in zip(
                    self.mean_precision_, self.means_,
                    self.degrees_of_freedom_, self.covariances_, n_samples_comp)])
        elif self.covariance_type == "spherical":
            X = np.vstack([
                #  mean + rng.randn(sample, n_features) * np.sqrt(covariance)
                MVT.multivariate_t.rvs(nu - d + 1, mean,
                                       np.diag((k + 1) / (k * (nu - d + 1)) * nu * covariance * np.ones(d)),
                                       size=int(sample))
                for (k, mean, nu, covariance, sample) in zip(
                    self.mean_precision_, self.means_,
                    self.degrees_of_freedom_, self.covariances_, n_samples_comp)])

        y = np.concatenate([j * np.ones(sample, dtype=int)
                           for j, sample in enumerate(n_samples_comp)])

        return (X, y)

    def _check_log_prob(self, X):
        # for debug purposes only
        from multivariate_distns import multivariate_t as MVT
        e1 = _estimate_log_t_prob(
            X, self.mean_precision_, self.means_, self.degrees_of_freedom_, self.precisions_cholesky_, self.covariance_type)
        e2 = np.empty_like(e1)
        d = len(self.means_[0])
        if self.covariance_type == "full":
            covariances_ = self.covariances_
        if self.covariance_type == "tied":
            covariances_ = [self.covariances_ for k in self.mean_precision_]
        if self.covariance_type == "diag":
            covariances_ = [np.diag(c) for c in self.covariances_]
        if self.covariance_type == "spherical":
            covariances_ = [np.diag(c * np.ones(d)) for c in self.covariances_]
        for i, (k, m, c) in enumerate(zip(self.mean_precision_, self.means_, covariances_)):
            d = len(m)
            if self.covariance_type == "tied":
                n = np.sum(self.degrees_of_freedom_)
            else:
                n = k
            df = n - d + 1
            cov = (k + 1) / (k * df) * n * c
            mvt = MVT.multivariate_t(df, m, cov)
            e2[:, i] = mvt.logpdf(X)
        pass

    def _MC_log_posterior_predictive(self, X, n_integral_points):
        """Predict the labels for the data samples in X using
        Monte Carlo sampled models.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        prob : array, shape (n_samples, n_classes)
            Component likelihoods.
        """
        self._check_is_fitted()
        X = _check_X(X, n_features=self.means_.shape[1])
        prob = np.array([[0]], dtype=float)

        from sklearn.mixture.gaussian_mixture import GaussianMixture as SGM
        sgm = SGM(covariance_type=self.covariance_type)
        hyper_params = (self.mean_precision_, self.degrees_of_freedom_, self.weights_,
                        self.means_, self.covariances_, self.precisions_cholesky_)
        for base_params in self.progress_bar(
                self._sample_base_params(hyper_params, n_integral_points)):
            sgm._set_parameters(base_params)
            prob = prob + sgm._estimate_log_prob(X)
        return prob / n_integral_points

    def _MC_sample_points(self, hyper_params, n_samples):
        """Predict the labels for the data samples in X using
        Monte Carlo sampled models.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        prob : array, shape (n_samples, n_classes)
            Component likelihoods.
        """
        self._check_is_fitted()
        n_features = self.means_.shape[1]
        X = np.empty((n_samples, n_features))
        y = np.empty(n_samples)

        from sklearn.mixture.gaussian_mixture import GaussianMixture as SGM
        sgm = SGM(covariance_type=self.covariance_type)
        for i, base_params in self.progress_bar(
                enumerate(self._sample_base_params(hyper_params, n_samples))):
            sgm._set_parameters(base_params)
            X[i, :], y[i] = sgm.sample(1)
        return X

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
        (mean_precision_, degrees_of_freedom_, weights_, means_, covariances_,
         precisions_cholesky_) = hyper_params

        for i in range(n_integral_points):
            means, covariances = _sample_gaussian_parameters(
                mean_precision_, means_, degrees_of_freedom_, covariances_, self.covariance_type, self.random_state)
            precisions_cholesky = _compute_precision_cholesky(
                covariances, self.covariance_type)
            yield weights_, means, covariances, precisions_cholesky

    def _check_log_prob(self, X):
        # for debug purposes only
        from multivariate_distns import multivariate_t as MVT
        e1 = _estimate_log_t_prob(
            X, self.mean_precision_, self.means_, self.degrees_of_freedom_, self.precisions_cholesky_, self.covariance_type)
        e2 = np.empty_like(e1)
        d = len(self.means_[0])
        if self.covariance_type == "full":
            covariances_ = self.covariances_
        if self.covariance_type == "tied":
            covariances_ = [self.covariances_ for k in self.mean_precision_]
        if self.covariance_type == "diag":
            covariances_ = [np.diag(c) for c in self.covariances_]
        if self.covariance_type == "spherical":
            covariances_ = [np.diag(c * np.ones(d)) for c in self.covariances_]
        for i, (k, m, c) in enumerate(zip(self.mean_precision_, self.means_, covariances_)):
            d = len(m)
            if self.covariance_type == "tied":
                n = np.sum(self.degrees_of_freedom_)
            else:
                n = k
            df = n - d + 1
            cov = (k + 1) / (k * df) * n * c
            mvt = MVT.multivariate_t(df, m, cov)
            e2[:, i] = mvt.logpdf(X)
        pass

    def _estimate_log_posterior_predictive_prob(self, X):
        # self._check_log_prob(X)
        return _estimate_log_t_prob(
            X, self.mean_precision_, self.means_,
            self.degrees_of_freedom_, self.precisions_cholesky_,
            self.covariance_type)


class BayesianGaussianClassifier(MixtureClassifierMixin, BayesianGaussianMixture):
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

    def __init__(self, n_components_per_class=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans', use_weights=True,
                 classes_init=None, weight_concentration_prior_type='dirichlet_process',
                 weight_concentration_prior=None,
                 mean_precision_prior=None, mean_prior=None,
                 degrees_of_freedom_prior=None, covariance_prior=None,
                 use_uninformed_prior=False,
                 n_integral_points=100, random_state=None, warm_start=False, verbose=0,
                 verbose_interval=10, progress_bar=None):
        BayesianGaussianMixture.__init__(self,
                                         n_components=n_components_per_class, covariance_type=covariance_type, tol=tol,
                                         reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params,
                                         use_weights=use_weights,
                                         weight_concentration_prior_type=weight_concentration_prior_type,
                                         weight_concentration_prior=weight_concentration_prior,
                                         mean_precision_prior=mean_precision_prior, mean_prior=mean_prior,
                                         degrees_of_freedom_prior=degrees_of_freedom_prior,
                                         covariance_prior=covariance_prior,
                                         use_uninformed_prior=use_uninformed_prior,
                                         n_integral_points=n_integral_points, random_state=random_state,
                                         warm_start=warm_start, verbose=verbose,
                                         verbose_interval=verbose_interval, progress_bar=progress_bar)
        MixtureClassifierMixin.__init__(self)
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
        ret : array, shape (n_samples, n_classes)
            Component likelihoods.
        """
        self._check_is_fitted()
        X = _check_X(X, n_features=self.means_.shape[1])
        return self._dispatch_weighted_log_prob(X)

    def _get_parameters(self):
        base_params = super(BayesianGaussianClassifier, self)._get_parameters()
        return (self.classes_, *base_params)

    def _set_parameters(self, params):
        """
        Parameters
        ----------
        classes_
        weight_concentration_
        mean_precision_
        means_
        degrees_of_freedom_
        covariances_,
        precisions_cholesky_
        """
        (self.classes_, *base_params) = params
        self.classes_ = np.array(self.classes_)
        super(BayesianGaussianClassifier, self)._set_parameters(base_params)
        self.n_components = len(self.means_)


class BayesianFairTiedClassifier(BayesianGaussianClassifier):
    def __init__(self, n_components_per_class=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans', use_weights=True,
                 classes_init=None, weight_concentration_prior_type='dirichlet_process',
                 weight_concentration_prior=None,
                 mean_precision_prior=None, mean_prior=None,
                 degrees_of_freedom_prior=None, covariance_prior=None,
                 use_uninformed_prior=False,
                 n_integral_points=100, random_state=None, warm_start=False, verbose=0,
                 verbose_interval=10, progress_bar=None):
        BayesianGaussianClassifier.__init__(self,
                                            n_components_per_class=n_components_per_class, covariance_type=covariance_type, tol=tol,
                                            reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params,
                                            use_weights=use_weights, classes_init=classes_init,
                                            weight_concentration_prior_type=weight_concentration_prior_type,
                                            weight_concentration_prior=weight_concentration_prior,
                                            mean_precision_prior=mean_precision_prior, mean_prior=mean_prior,
                                            degrees_of_freedom_prior=degrees_of_freedom_prior,
                                            covariance_prior=covariance_prior,
                                            use_uninformed_prior=use_uninformed_prior,
                                            n_integral_points=n_integral_points, random_state=random_state,
                                            warm_start=warm_start, verbose=verbose,
                                            verbose_interval=verbose_interval, progress_bar=progress_bar)
        if covariance_type != 'tied':
            raise ValueError('Fair covariance estimation is needed only in the tied case.')

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
        random_state = check_random_state(self.random_state)

        nk, fk, xk, sk = _estimate_fair_gaussian_parameters(
            X, np.exp(log_resp), self.reg_covar, self.covariance_type, random_state)
        self._estimate_weights(nk)
        self._estimate_means(nk, xk)
        self._estimate_precisions(fk, xk, sk)


def exampleClassifier(n_components, n_features, covariance_type='full', random_state=None):
    from scipy.stats import norm
    weights = np.full((n_components,), 1.0 / n_components)
    means = norm.rvs(0, 1, size=(n_components, n_features), random_state=random_state)
    covariances, precisions = {'full': _fullCorr, 'tied': _tiedCorr, 'diag': _diagCorr,
                               'spherical': _sphericalCorr}[covariance_type](n_components, n_features,
                                                                             random_state=random_state)
    classes = np.arange(n_components)
    df = np.full(n_components, 42)
    clf = BayesianGaussianClassifier(n_components_per_class=1, covariance_type=covariance_type, random_state=random_state)
    clf._set_parameters((classes, weights, df, means, df, covariances, precisions))
    return clf

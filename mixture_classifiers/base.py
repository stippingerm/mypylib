"""
Generalized Mixture Classifiers.
"""

#
# Author: Marcell Stippinger
# License: BSD 3 clause
#

# Notes:
# This submodule was inspired by and based on scikit-learn mixture models
# https://github.com/scikit-learn/scikit-learn/sklearn/linear_model/base.py

# Important:
# sklearn.mixture.base is a good base class for fitting mixture distributions, except:
# * sampling implemented there assumes mixture of Gaussians,
#   therefore it should be moved to sklearn.mixture.gaussian_mixture
# We extend it by mixture classification, following the objectives:
# * use the arbitrary class labels provided as input
#   (not simple ordinals as in a mixture distribution)
# * respect these labels when generating samples
# * restrict the responsibilities of each component to their respective class

from __future__ import division

import numpy as np
import warnings

from abc import ABCMeta, abstractmethod
from sklearn.base import ClassifierMixin
from sklearn.utils.fixes import logsumexp
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_X_y
from sklearn.utils import check_array, check_random_state
from sklearn.exceptions import ConvergenceWarning


def _normalize_log_prob(weighted_log_prob):
    """Estimate log probabilities and responsibilities for each sample.

    Compute the log probabilities, weighted log probabilities per
    component and responsibilities for each sample in X with respect to
    the current state of the model.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

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


def _make_multi_responsibility_mask(classes, n_components, y):
    """Make log responsibilities based on fixed classes

    Parameters
    ----------
    classes : array-like, shape (n_classes, )
        The allowed class labels.

    n_components : int or array-like, shape (n_classes, )
        The allowed class labels.

    y : array-like, shape (n_samples, )
        The actual class labels.

    Returns
    -------
    resp : array-like, shape (n_samples, sum(n_components))
        The assumed responsibilities for each data sample in X."""
    n_classes = len(classes)
    n_components = np.broadcast_to(n_components, n_classes)
    class_start_indices = np.cumsum([0, *n_components])
    column_select = np.squeeze(np.searchsorted(classes, y, side='left'))
    n_samples = len(y)
    resp = np.zeros((n_samples, class_start_indices[-1]))
    for (i_sample, i_class) in zip(range(0, n_samples), column_select):
        component_range = class_start_indices[[i_class, i_class+1]]
        resp[i_sample, slice(*component_range)] = 1.0
    return resp


def _make_single_responsibility_mask(classes, y):
    """Make log responsibilities based on fixed classes

    Parameters
    ----------
    classes : array-like, shape (n_components, )
        The allowed class labels.

    y : array-like, shape (n_samples, )
        The actual class labels.

    Returns
    -------
    resp : array-like, shape (n_samples, n_components)
        The assumed responsibilities for each data sample in X."""
    n_components = len(classes)
    n_samples = len(y)
    column_select = np.squeeze(np.searchsorted(classes, y, side='left'))
    resp = np.zeros((n_samples, n_components))
    resp[np.arange(0, n_samples), column_select] = 1.0  # np :: adv. indexing
    return resp


class MixtureClassifierMixin(ClassifierMixin):
    """Mixin for multiclass classifiers.
    Handles prediction for sparse and dense X.
    """

    def __init__(self):
        self.classes_ = np.array([])
        self._refuse_inf_resp = True

    def _make_responsibilities(self, y):
        if np.any(self.n_components_per_class != 1):
            mask = _make_multi_responsibility_mask(self.classes_, self.n_components_per_class, y)
            resp = mask / np.sum(mask, axis=1)
        else:
            resp = _make_single_responsibility_mask(self.classes_, y)
        return resp

    def _limit_log_resp(self, log_resp, y):
        with np.errstate(divide='ignore'):
            # ignore log(zero)
            log_mask = np.log(self._make_responsibilities(y))
        if np.any(self.n_components_per_class != 1):
            limited_log_resp = _normalize_log_prob(log_resp + log_mask)
        else:
            limited_log_resp = log_mask
        if self._refuse_inf_resp:
            limited_log_resp = np.maximum(limited_log_resp, np.finfo(float).min)
        return limited_log_resp

    def _relabel_samples(self, X, y):
        n_classes = len(self.classes_)
        n_components = np.broadcast_to(self.n_components_per_class, n_classes)
        ids = np.digitize(y, np.cumsum([0, *n_components], dtype=int))-1
        return X, self.classes_[list(ids)]

    @abstractmethod
    def decision_function(self, X):
        """Predict probabilities for samples in X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.
        Returns
        -------
        C : array, shape = [n_samples]
            Predicted class label per sample.
        """
        pass

    def predict(self, X):
        """Predict class labels for samples in X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.
        Returns
        -------
        C : array, shape = [n_samples]
            Predicted class label per sample.
        """
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(np.int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]

    def fit(self, X, y):
        """Estimate model parameters with the EM algorithm.

        The method fit the model `n_init` times and set the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a `ConvergenceWarning` is raised.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : array-like, shape (n_samples,)
            List of classes assigned to data points. Each row
            corresponds to a single data point.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)
        self._check_initial_parameters(X)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self.n_components = np.sum(np.broadcast_to(self.n_components_per_class, n_classes))

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, 'converged_'))
        n_init = self.n_init if do_init else 1

        # Addendum N°1: if analytically solvable, do not iterate
        analytic_solution = np.all(self.n_components_per_class == 1)
        max_iter = 1 if analytic_solution else self.max_iter

        max_lower_bound = -np.infty
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, random_state)
                self.lower_bound_ = -np.infty

            for n_iter in range(max_iter):
                prev_lower_bound = self.lower_bound_

                log_prob_norm, log_resp = self._e_step(X)
                # Addendum N°2: limit responsibilities to classes
                log_resp = self._limit_log_resp(log_resp, y)
                self._m_step(X, log_resp)
                self.lower_bound_ = self._compute_lower_bound(
                    log_resp, log_prob_norm)

                change = self.lower_bound_ - prev_lower_bound
                self._print_verbose_msg_iter_end(n_iter, change)

                if abs(change) < self.tol:
                    self.converged_ = True
                    break

            self._print_verbose_msg_init_end(self.lower_bound_)

            if self.lower_bound_ > max_lower_bound:
                max_lower_bound = self.lower_bound_
                best_params = self._get_parameters()
                best_n_iter = n_iter

        if analytic_solution:
            self.converged_ = True

        if not self.converged_:
            warnings.warn('Initialization %d did not converge. '
                          'Try different init parameters, '
                          'or increase max_iter, tol '
                          'or check for degenerate data.'
                          % (init + 1), ConvergenceWarning)

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter

        return self

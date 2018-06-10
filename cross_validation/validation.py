"""
The :mod:`sklearn.model_selection._validation` module includes classes and
functions to validate the model.
"""

#
# Authors of sklearn.model_selection._validation:
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Raghav RV <rvraghav93@gmail.com>
# Author: Marcell Stippinger 2018
# License: BSD 3 clause
#

# Notes:
# This module was based on sklearn.model_selection._validation
# We extend its functionality by allowing different (but identically shaped)
# data to be used for testing. It is useful when training and test data have
# related samples, e.g. when testing morning vs afternoon sessions over the
# same period of the year.
# We found it useful to move the parallel loop out of cross_validate because
# it allows more efficient parallelization if there are only a few cv steps
# but several estimators, data sets or feature selections.
# We also implemented nested cross-validation similar to the one found in
# LogisticRegressionCV. This allows to test models which have many hyper-
# parameters and the user needs independent validation and test results.
# Validation in this case is transparent and requires post-processing, i.e.,
# all results for all possible hyperparameters are reported, the best is not
# selected here.

# Ref for nested cross-validation:
# https://stats.stackexchange.com/questions/65128/nested-cross-validation-for-model-selection
# https://sites.google.com/site/dtclabdcv/
# https://www.researchgate.net/post/What_is_double_cross_validation
# https://www.r-project.org/conferences/useR-2006/Abstracts/Francois+Langrognet.pdf

from __future__ import print_function
from __future__ import division

import warnings
import numbers
import time

import numpy as np
import scipy.sparse as sp

from sklearn.base import is_classifier, clone
from sklearn.utils import indexable, check_random_state, safe_indexing
from sklearn.utils.deprecation import DeprecationDict
from sklearn.utils.validation import _is_arraylike, _num_samples
from sklearn.utils.metaestimators import _safe_split
from sklearn.externals.joblib import Parallel, delayed, logger
from sklearn.externals.six.moves import zip
from sklearn.metrics.scorer import check_scoring, _check_multimetric_scoring
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection._split import check_cv
from sklearn.preprocessing import LabelEncoder
# own
from sklearn.utils.validation import check_consistent_length
from itertools import product as iter_product
from sklearn.exceptions import DataConversionWarning

__all__ = ['cross_validate', 'cross_val_score', 'cross_val_predict', 'nested_cross_validate',
           'concatenate_score_dicts']


def cross_validate(estimator, X, y=None, groups=None, scoring=None, cv=None,
                   X_for_test=None, n_jobs=1, verbose=0, fit_params=None,
                   pre_dispatch='2*n_jobs', return_train_score="warn"):
    """Evaluate metric(s) by cross-validation and also record fit/score times.

    Read more in the :ref:`User Guide <multimetric_cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like
        The data to fit. Can be for example a list, or an array.

    X_for_test : array-like
        The data to test. When omitted, X is used. It is assumed that
        the samples in X are related pairwisely to these samples.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set.

    scoring : string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

        See :ref:`multimetric_grid_search` for an example.

        If None, the estimator's default scorer (if available) is used.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    verbose : integer, optional
        The verbosity level.

    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    return_train_score : boolean, optional
        Whether to include train scores.

        Current default is ``'warn'``, which behaves as ``True`` in addition
        to raising a warning when a training score is looked up.
        That default will be changed to ``False`` in 0.21.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    Returns
    -------
    scores : dict of float arrays of shape=(n_splits,)
        Array of scores of the estimator for each run of the cross validation.

        A dict of arrays containing the score/time arrays for each scorer is
        returned. The possible keys for this ``dict`` are:

            ``test_score``
                The score array for test scores on each cv split.
            ``train_score``
                The score array for train scores on each cv split.
                This is available only if ``return_train_score`` parameter
                is ``True``.
            ``fit_time``
                The time for fitting the estimator on the train
                set for each cv split.
            ``score_time``
                The time for scoring the estimator on the test set for each
                cv split. (Note time for scoring on the train set is not
                included even if ``return_train_score`` is set to ``True``

    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.model_selection import cross_validate
    >>> from sklearn.metrics.scorer import make_scorer
    >>> from sklearn.metrics import confusion_matrix
    >>> from sklearn.svm import LinearSVC
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()

    Single metric evaluation using ``cross_validate``

    >>> cv_results = cross_validate(lasso, X, y, return_train_score=False)
    >>> sorted(cv_results.keys())                         # doctest: +ELLIPSIS
    ['fit_time', 'score_time', 'test_score']
    >>> cv_results['test_score']    # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    array([ 0.33sklearn..,  0.08sklearn..,  0.03sklearn..])

    Multiple metric evaluation using ``cross_validate``
    (please refer the ``scoring`` parameter doc for more information)

    >>> scores = cross_validate(lasso, X, y,
    sklearn..                         scoring=('r2', 'neg_mean_squared_error'))
    >>> print(scores['test_neg_mean_squared_error'])      # doctest: +ELLIPSIS
    [-3635.5sklearn.. -3573.3sklearn.. -6114.7sklearn..]
    >>> print(scores['train_r2'])                         # doctest: +ELLIPSIS
    [ 0.28sklearn..  0.39sklearn..  0.22sklearn..]

    See Also
    ---------
    :func:`sklearn.model_selection.cross_val_score`:
        Run cross-validation for single metric evaluation.

    :func:`sklearn.metrics.make_scorer`:
        Make a scorer from a performance metric or loss function.

    """
    X, y, groups = indexable(X, y, groups)
    if X_for_test is None:
        X_for_test = X
    else:
        X_for_test, y = indexable(X_for_test, y)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorers, _ = _check_multimetric_scoring(estimator, scoring=scoring)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    scores = parallel(
        delayed(_fit_and_score)(
            clone(estimator), X, X_for_test, y, scorers, train, test, verbose, None,
            fit_params, return_train_score=return_train_score,
            return_times=True)
        for train, test in cv.split(X, y, groups))

    if return_train_score:
        train_scores, test_scores, fit_times, score_times = zip(*scores)
        train_scores = _aggregate_score_dicts(train_scores)
    else:
        test_scores, fit_times, score_times = zip(*scores)
    test_scores = _aggregate_score_dicts(test_scores)

    # TODO: replace by a dict in 0.21
    ret = DeprecationDict() if return_train_score == 'warn' else {}
    ret['fit_time'] = np.array(fit_times)
    ret['score_time'] = np.array(score_times)

    for name in scorers:
        ret['test_%s' % name] = np.array(test_scores[name])
        if return_train_score:
            key = 'train_%s' % name
            ret[key] = np.array(train_scores[name])
            if return_train_score == 'warn':
                message = (
                    'You are accessing a training score ({!r}), '
                    'which will not be available by default '
                    'any more in 0.21. If you need training scores, '
                    'please set return_train_score=True').format(key)
                # warn on key access
                ret.add_warning(key, message, FutureWarning)

    return ret


def nested_cross_validate(estimator, X, y=None, groups=None, scoring=None, cv=None,
                          X_for_test=None, n_jobs=1, verbose=0, fit_params=None,
                          pre_dispatch='2*n_jobs', return_train_score=True):
    """Evaluate metric(s) by cross-validation and also record fit/score times.

    Read more in the :ref:`User Guide <multimetric_cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like
        The data to fit. Can be for example a list, or an array.

    X_for_test : array-like
        The data to test. When omitted, X is used. It is assumed that
        the samples in X are related pairwisely to these samples.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set.

    scoring : string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

        See :ref:`multimetric_grid_search` for an example.

        If None, the estimator's default scorer (if available) is used.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    verbose : integer, optional
        The verbosity level.

    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'


    Returns
    -------
    scores : dict of float arrays of shape=(n_splits,)
        Array of scores of the estimator for each run of the cross validation.

        A dict of arrays containing the score/time arrays for each scorer is
        returned. The possible keys for this ``dict`` are:

            ``test_score``
                The score array for test scores on each cv split.
            ``validation_score``
                The score array for train scores on each cv split.
            ``calibration_score``
                The score array for train scores on each cv split.
            ``fit_time``
                The time for fitting the estimator on the train
                set for each cv split.
            ``score_time``
                The time for scoring the estimator on the test set for each
                cv split. (Note time for scoring on the train set is not
                included even if ``return_train_score`` is set to ``True``


    See Also
    ---------
    :func:`sklearn.model_selection.cross_val_score`:
        Run cross-validation for single metric evaluation.

    :func:`sklearn.metrics.make_scorer`:
        Make a scorer from a performance metric or loss function.

    """
    X, y, groups = indexable(X, y, groups)
    if X_for_test is None:
        X_for_test = X
    else:
        X_for_test, y = indexable(X_for_test, y)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorers, _ = _check_multimetric_scoring(estimator, scoring=scoring)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    scores = parallel(
        delayed(_nested_fit_and_score)(
            clone(estimator), X, X_for_test, y, groups, scorers, calibrate_validate, test, cv, verbose, None,
            fit_params, return_train_score=True, return_times=True)
        for calibrate_validate, test in cv.split(X, y, groups))

    calibration_scores, validation_scores, test_scores, fit_times, score_times = zip(*scores)
    validation_scores = _aggregate_score_dicts(validation_scores)
    calibration_scores = _aggregate_score_dicts(calibration_scores)
    test_scores = _aggregate_score_dicts(test_scores)

    # TODO: replace by a dict in 0.21
    ret = DeprecationDict() if return_train_score == 'warn' else {}
    ret['fit_time'] = np.array(fit_times)
    ret['score_time'] = np.array(score_times)

    for name in scorers:
        ret['test_%s' % name] = np.array(test_scores[name])
        ret['validation_%s' % name] = np.array(validation_scores[name])
        ret['calibration_%s' % name] = np.array(calibration_scores[name])

    return ret


def cross_val_score(estimator, X, y=None, groups=None, scoring=None, cv=None,
                    X_for_test=None, n_jobs=1, verbose=0, fit_params=None,
                    pre_dispatch='2*n_jobs'):
    """Evaluate a score by cross-validation

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like
        The data to fit. Can be for example a list, or an array.

    X_for_test : array-like
        The data to test. When omitted, X is used. It is assumed that
        the samples in X are related pairwisely to these samples.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - An object to be used as a cross-validation generator.
        - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    verbose : integer, optional
        The verbosity level.

    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    Returns
    -------
    scores : array of float, shape=(len(list(cv)),)
        Array of scores of the estimator for each run of the cross validation.

    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.model_selection import cross_val_score
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()
    >>> print(cross_val_score(lasso, X, y))  # doctest: +ELLIPSIS
    [ 0.33150734  0.08022311  0.03531764]

    See Also
    ---------
    :func:`sklearn.model_selection.cross_validate`:
        To run cross-validation on multiple metrics and also to return
        train scores, fit times and score times.

    :func:`sklearn.metrics.make_scorer`:
        Make a scorer from a performance metric or loss function.

    """
    # To ensure multimetric format is not supported
    scorer = check_scoring(estimator, scoring=scoring)

    cv_results = cross_validate(estimator=estimator, X=X, y=y, groups=groups,
                                scoring={'score': scorer}, cv=cv,
                                return_train_score=False,
                                n_jobs=n_jobs, verbose=verbose,
                                fit_params=fit_params,
                                pre_dispatch=pre_dispatch)
    return cv_results['test_score']


def _fit_and_score(estimator, X, X_for_test, y, scorer, train, test, verbose,
                   parameters, fit_params, return_train_score=False,
                   return_parameters=False, return_n_test_samples=False,
                   return_times=False, error_score='raise'):
    """Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape at least 2D
        The data to fit.

    X_for_test : array-like
        The data to test. When omitted, X is used. It is assumed that
        the samples in X are related pairwisely to these samples.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.

        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.

        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.

    train : array-like, shape (n_train_samples,)
        Indices of training samples.

    test : array-like, shape (n_test_samples,)
        Indices of test samples.

    verbose : integer
        The verbosity level.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    parameters : dict or None
        Parameters to be set on the estimator.

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    return_train_score : boolean, optional, default: False
        Compute and return score on training set.

    return_parameters : boolean, optional, default: False
        Return parameters that has been used for the estimator.

    return_n_test_samples : boolean, optional, default: False
        Whether to return the ``n_test_samples``

    return_times : boolean, optional, default: False
        Whether to return the fit/score times.

    Returns
    -------
    train_scores : dict of scorer name -> float, optional
        Score on training set (for all the scorers),
        returned only if `return_train_score` is `True`.

    test_scores : dict of scorer name -> float, optional
        Score on testing set (for all the scorers).

    n_test_samples : int
        Number of test samples.

    fit_time : float
        Time spent for fitting in seconds.

    score_time : float
        Time spent for scoring in seconds.

    parameters : dict or None, optional
        The parameters that have been evaluated.
    """
    if verbose > 1:
        if parameters is None:
            msg = ''
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                                    for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, _index_param_value(X, v, train))
                       for k, v in fit_params.items()])

    test_scores = {}
    train_scores = {}
    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X_for_test, y, test, train)

    is_multimetric = not callable(scorer)
    n_scorers = len(scorer.keys()) if is_multimetric else 1

    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            if is_multimetric:
                test_scores = dict(zip(scorer.keys(),
                                       [error_score, ] * n_scorers))
                if return_train_score:
                    train_scores = dict(zip(scorer.keys(),
                                            [error_score, ] * n_scorers))
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
            warnings.warn("Classifier fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%r" % (error_score, e), FitFailedWarning)
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)")

    else:
        fit_time = time.time() - start_time
        # _score will return dict if is_multimetric is True
        test_scores = _score(estimator, X_test, y_test, scorer, is_multimetric)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = _score(estimator, X_train, y_train, scorer,
                                  is_multimetric)

    if verbose > 2:
        if is_multimetric:
            for scorer_name, score in test_scores.items():
                msg += ", %s=%s" % (scorer_name, score)
        else:
            msg += ", score=%s" % test_scores
    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = "%s, total=%s" % (msg, logger.short_format_time(total_time))
        print("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

    ret = [train_scores, test_scores] if return_train_score else [test_scores]

    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    return ret


def _nested_fit_and_score(estimator, X, X_for_test, y, groups, scorer, calibrate_validate, test, cv, verbose,
                          parameters, fit_params, return_train_score=True,
                          return_parameters=False, return_n_test_samples=False,
                          return_times=False, error_score='raise'):
    """Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape at least 2D
        The data to fit.

    X_for_test : array-like
        The data to test. When omitted, X is used. It is assumed that
        the samples in X are related pairwisely to these samples.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.

        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.

        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.

    calibrate_validate : array-like, shape (n_train_samples,)
        Indices of training samples.

    test : array-like, shape (n_test_samples,)
        Indices of test samples.

    verbose : integer
        The verbosity level.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    parameters : dict or None
        Parameters to be set on the estimator.

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    return_train_score : boolean, optional, default: False
        Compute and return score on training set.

    return_parameters : boolean, optional, default: False
        Return parameters that has been used for the estimator.

    return_n_test_samples : boolean, optional, default: False
        Whether to return the ``n_test_samples``

    return_times : boolean, optional, default: False
        Whether to return the fit/score times.

    Returns
    -------
    calibration_scores : dict of scorer name -> float, optional
        Score on training set (for all the scorers).

    validation_scores : dict of scorer name -> float, optional
        Score on training set (for all the scorers).

    test_scores : dict of scorer name -> float, optional
        Score on testing set (for all the scorers).

    n_test_samples : int
        Number of test samples.

    fit_time : float
        Time spent for fitting in seconds.

    score_time : float
        Time spent for scoring in seconds.

    parameters : dict or None, optional
        The parameters that have been evaluated.
    """

    nested_estimator = clone(estimator)
    results = _fit_and_score(estimator, X, X_for_test, y, scorer, calibrate_validate, test, verbose,
                             parameters, fit_params, return_train_score=False,
                             return_parameters=return_parameters, return_n_test_samples=return_n_test_samples,
                             return_times=return_times, error_score=error_score)

    X_ = safe_indexing(X, calibrate_validate)
    y_ = safe_indexing(y, calibrate_validate)
    g_ = groups if groups is None else safe_indexing(groups, calibrate_validate)
    Xt_ = X_for_test if X_for_test is None else safe_indexing(X_for_test, calibrate_validate)
    # TODO: make sense for X_for_test here
    nested_scores = cross_validate(nested_estimator, X_, y=y_, groups=g_, scoring=scorer, cv=cv,
                                   X_for_test=Xt_, n_jobs=1, verbose=0, fit_params=fit_params,
                                   return_train_score=True)
    calibration_scores = {k[6:]: np.nanmean(v) for k, v in nested_scores.items() if k.startswith('train_')}
    validation_scores = {k[5:]: np.nanmean(v) for k, v in nested_scores.items() if k.startswith('test_')}

    return [calibration_scores, validation_scores, *results]


def _score(estimator, X_test, y_test, scorer, is_multimetric=False):
    """Compute the score(s) of an estimator on a given test set.

    Will return a single float if is_multimetric is False and a dict of floats,
    if is_multimetric is True
    """
    if is_multimetric:
        return _multimetric_score(estimator, X_test, y_test, scorer)
    else:
        if y_test is None:
            score = scorer(estimator, X_test)
        else:
            score = scorer(estimator, X_test, y_test)

        if hasattr(score, 'item'):
            try:
                # e.g. unwrap memmapped scalars
                score = score.item()
            except ValueError:
                # non-scalar?
                pass

        if not isinstance(score, numbers.Number):
            raise ValueError("scoring must return a number, got %s (%s) "
                             "instead. (scorer=%r)"
                             % (str(score), type(score), scorer))
    return score


def _multimetric_score(estimator, X_test, y_test, scorers):
    """Return a dict of score for multimetric scoring"""
    scores = {}

    for name, scorer in scorers.items():
        if y_test is None:
            score = scorer(estimator, X_test)
        else:
            score = scorer(estimator, X_test, y_test)

        if hasattr(score, 'item'):
            try:
                # e.g. unwrap memmapped scalars
                score = score.item()
            except ValueError:
                # non-scalar?
                pass
        scores[name] = score

        if not isinstance(score, numbers.Number):
            raise ValueError("scoring must return a number, got %s (%s) "
                             "instead. (scorer=%s)"
                             % (str(score), type(score), name))
    return scores


def cross_val_predict(estimator, X, y=None, groups=None, cv=None,
                      X_for_test=None, n_jobs=1,
                      verbose=0, fit_params=None, pre_dispatch='2*n_jobs',
                      method='predict'):
    """Generate cross-validated estimates for each input data point

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.

    X : array-like
        The data to fit. Can be, for example a list, or an array at least 2d.

    X_for_test : array-like
        The data to test. When omitted, X is used. It is assumed that
        the samples in X are related pairwisely to these samples.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - An object to be used as a cross-validation generator.
        - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    verbose : integer, optional
        The verbosity level.

    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    method : string, optional, default: 'predict'
        Invokes the passed method name of the passed estimator. For
        method='predict_proba', the columns correspond to the classes
        in sorted order.

    Returns
    -------
    predictions : ndarray
        This is the result of calling ``method``

    Notes
    -----
    In the case that one or more classes are absent in a training portion, a
    default score needs to be assigned to all instances for that class if
    ``method`` produces columns per class, as in {'decision_function',
    'predict_proba', 'predict_log_proba'}.  For ``predict_proba`` this value is
    0.  In order to ensure finite output, we approximate negative infinity by
    the minimum finite float value for the dtype in other cases.

    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.model_selection import cross_val_predict
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()
    >>> y_pred = cross_val_predict(lasso, X, y)
    """
    X, y, groups = indexable(X, y, groups)
    if X_for_test is None:
        X_for_test = X
    else:
        X_for_test, y = indexable(X_for_test, y)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
        le = LabelEncoder()
        y = le.fit_transform(y)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(delayed(_fit_and_predict)(
        clone(estimator), X, X_for_test, y, train, test, verbose, fit_params, method)
                                 for train, test in cv.split(X, X_for_test, y, groups))

    # Concatenate the predictions
    predictions = [pred_block_i for pred_block_i, _ in prediction_blocks]
    test_indices = np.concatenate([indices_i
                                   for _, indices_i in prediction_blocks])

    if not _check_is_permutation(test_indices, _num_samples(X)):
        raise ValueError('cross_val_predict only works for partitions')

    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))

    # Check for sparse predictions
    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    else:
        predictions = np.concatenate(predictions)
    return predictions[inv_test_indices]


def _fit_and_predict(estimator, X, X_for_test, y, train, test, verbose, fit_params,
                     method):
    """Fit estimator and predict values for a given dataset split.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.

    X : array-like of shape at least 2D
        The data to fit.

    X_for_test : array-like
        The data to test. When omitted, X is used. It is assumed that
        the samples in X are related pairwisely to these samples.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    train : array-like, shape (n_train_samples,)
        Indices of training samples.

    test : array-like, shape (n_test_samples,)
        Indices of test samples.

    verbose : integer
        The verbosity level.

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    method : string
        Invokes the passed method name of the passed estimator.

    Returns
    -------
    predictions : sequence
        Result of calling 'estimator.method'

    test : array-like
        This is the value of the test parameter
    """
    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, _index_param_value(X, v, train))
                       for k, v in fit_params.items()])

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, _ = _safe_split(estimator, X_for_test, y, test, train)

    if y_train is None:
        estimator.fit(X_train, **fit_params)
    else:
        estimator.fit(X_train, y_train, **fit_params)
    func = getattr(estimator, method)
    predictions = func(X_test)
    if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
        n_classes = len(set(y))
        if n_classes != len(estimator.classes_):
            recommendation = (
                'To fix this, use a cross-validation '
                'technique resulting in properly '
                'stratified folds')
            warnings.warn('Number of classes in training fold ({}) does '
                          'not match total number of classes ({}). '
                          'Results may not be appropriate for your use case. '
                          '{}'.format(len(estimator.classes_),
                                      n_classes, recommendation),
                          RuntimeWarning)
            if method == 'decision_function':
                if (predictions.ndim == 2 and
                        predictions.shape[1] != len(estimator.classes_)):
                    # This handles the case when the shape of predictions
                    # does not match the number of classes used to train
                    # it with. This case is found when sklearn.svm.SVC is
                    # set to `decision_function_shape='ovo'`.
                    raise ValueError('Output shape {} of {} does not match '
                                     'number of classes ({}) in fold. '
                                     'Irregular decision_function outputs '
                                     'are not currently supported by '
                                     'cross_val_predict'.format(
                        predictions.shape, method,
                        len(estimator.classes_),
                        recommendation))
                if len(estimator.classes_) <= 2:
                    # In this special case, `predictions` contains a 1D array.
                    raise ValueError('Only {} class/es in training fold, this '
                                     'is not supported for decision_function '
                                     'with imbalanced folds. {}'.format(
                        len(estimator.classes_),
                        recommendation))

            float_min = np.finfo(predictions.dtype).min
            default_values = {'decision_function': float_min,
                              'predict_log_proba': float_min,
                              'predict_proba': 0}
            predictions_for_all_classes = np.full((_num_samples(predictions),
                                                   n_classes),
                                                  default_values[method])
            predictions_for_all_classes[:, estimator.classes_] = predictions
            predictions = predictions_for_all_classes
    return predictions, test


def _check_is_permutation(indices, n_samples):
    """Check whether indices is a reordering of the array np.arange(n_samples)

    Parameters
    ----------
    indices : ndarray
        integer array to test
    n_samples : int
        number of expected elements

    Returns
    -------
    is_partition : bool
        True iff sorted(indices) is np.arange(n)
    """
    if len(indices) != n_samples:
        return False
    hit = np.zeros(n_samples, dtype=bool)
    hit[indices] = True
    if not np.all(hit):
        return False
    return True


def _index_param_value(X, v, indices):
    """Private helper function for parameter value indexing."""
    if not _is_arraylike(v) or _num_samples(v) != _num_samples(X):
        # pass through: skip indexing
        return v
    if sp.issparse(v):
        v = v.tocsr()
    return safe_indexing(v, indices)


def _aggregate_score_dicts(scores):
    """Aggregate the list of dict to dict of np ndarray

    The aggregated output of _fit_and_score will be a list of dict
    of form [{'prec': 0.1, 'acc':1.0}, {'prec': 0.1, 'acc':1.0}, sklearn..]
    Convert it to a dict of array {'prec': np.array([0.1 sklearn..]), sklearn..}

    Parameters
    ----------

    scores : list of dict
        List of dicts of the scores for all scorers. This is a flat list,
        assumed originally to be of row major order.

    Example
    -------

    >>> scores = [{'a': 1, 'b':10}, {'a': 2, 'b':2}, {'a': 3, 'b':3},
    sklearn..           {'a': 10, 'b': 10}]                         # doctest: +SKIP
    >>> _aggregate_score_dicts(scores)                        # doctest: +SKIP
    {'a': array([1, 2, 3, 10]),
     'b': array([10, 2, 3, 10])}
    """
    out = {}
    for key in scores[0]:
        out[key] = np.asarray([score[key] for score in scores])
    return out


def concatenate_score_dicts(scores):
    """Concatenate the list of dict to dict of np ndarray

    The aggregated output of _fit_and_score will be a list of dict
    of form [{'prec': 0.1, 'acc':1.0}, {'prec': 0.1, 'acc':1.0}, sklearn..]
    Convert it to a dict of array {'prec': np.array([0.1 sklearn..]), sklearn..}

    Parameters
    ----------

    scores : list of dict
        List of dicts of the scores for all scorers. This is a flat list,
        assumed originally to be of row major order.

    Example
    -------

    >>> scores = [{'a': [1, 11], 'b':10}, {'a': [2, 22], 'b':2}, {'a': [3, 33], 'b':3},
    sklearn..     {'a': [10, 100], 'b': 10}]                              # doctest: +SKIP
    >>> concatenate_score_dicts(scores)                                   # doctest: +SKIP
    {'a': array([  1,  11,   2,  22,   3,  33,  10, 100]),
     'b': array([10,  2,  3, 10])}
    """
    out = {}
    for key in scores[0]:
        out[key] = np.concatenate([np.atleast_1d(score[key]) for score in scores])
    return out


def to_indexable(X):
    """Make indexable for cross-validation.

    Checks consistent length, passes through None, and ensures that everything
    can be indexed by converting sparse matrices to csr and converting
    non-interable objects to arrays.

    Parameters
    ----------
    X : list, dataframe, array, sparse matrix
    """
    if sp.issparse(X):
        return X.tocsr()
    elif hasattr(X, "__getitem__") or hasattr(X, "iloc"):
        return X
    elif X is None:
        return X
    else:
        return np.array(X, ndmin=1)


def to_dict(X):
    """Make dict-like.

    Parameters
    ----------
    X : dict
    """
    if hasattr(X, "items"):
        return X
    elif X is None:
        return {None: None}
    else:
        return {None: X}


def dict_indexable(*iterables):
    """Make arrays indexable for cross-validation.

    Checks consistent length, passes through None, and ensures that everything
    can be indexed by converting sparse matrices to csr and converting
    non-interable objects to arrays.

    Parameters
    ----------
    *iterables : lists, dataframes, arrays, sparse matrices
        List of objects to ensure sliceability.
    """
    result = []
    for X in iterables:
        if hasattr(X, "items"):
            result.append(X)
        elif X is None:
            result.append(X)
        else:
            result.append(np.array(X))
    check_consistent_length([v for d in result for k, v in d.items()])
    return result


def safe_features(X, indices):
    # Like safe_indexing in sklearn.utils
    """Return items or rows from X using indices.

    Allows simple indexing of lists or arrays.

    Parameters
    ----------
    X : array-like, sparse-matrix, list, pandas.DataFrame, pandas.Series.
        Data from which to sample rows or items.
    indices : array-like of int
        Indices according to which X will be subsampled.

    Returns
    -------
    subset
        Subset of X on first axis

    Notes
    -----
    CSR, CSC, and LIL sparse matrices are supported. COO sparse matrices are
    not supported.
    Plain python list of indices won't work with pandas.DataFrame as it does
    not have flags, convert to numpy.ndarray instead.
    """
    if hasattr(X, "iloc"):
        # Work-around for indexing with read-only indices in pandas
        indices = indices if indices.flags.writeable else indices.copy()
        # Pandas Dataframes and Series
        try:
            return X.iloc[:, indices]
        except ValueError:
            # Cython typed memoryviews internally used in pandas do not support
            # readonly buffers.
            warnings.warn("Copying input dataframe for slicing.",
                          DataConversionWarning)
            return X.copy().iloc[:, indices]
    elif hasattr(X, "shape"):
        if hasattr(X, 'take') and (hasattr(indices, 'dtype') and
                                   indices.dtype.kind == 'i'):
            # This is often substantially faster than X[indices]
            return X.take(indices, axis=1)
        else:
            return X[:, indices]
    else:
        return [[row[idx] for idx in indices] for row in X]


def _cross_validate_inner(est_key, train_key, test_key, feat, est, train, y, groups, scoring, cv, test, cv_params,
                          nested):
    # delay feature selection and cloning as it may require a lot of memory
    train, test = safe_features(train, feat), (test if test is None else safe_features(test, feat))
    if nested:
        result = nested_cross_validate(clone(est), train, y, groups, scoring, cv, test, **cv_params)
    else:
        result = cross_validate(clone(est), train, y, groups, scoring, cv, test, **cv_params)
    return est_key, train_key, test_key, len(feat), feat, result


def _no_progress_bar(x, *args, **kwargs):
    return x


def _default_feature_selection(n_feature):
    return [np.arange(n_feature)]


def _first_of_dict(d):
    return d[next(iter(d.keys()))]


class random_feature_selector():
    def __init__(self, n_total, random_state):
        self.random_state = check_random_state(random_state)
        self.n_total = int(n_total)

    def __call__(self, n_select):
        return self.random_state.choice(self.n_total, 1, replace=False)


def cross_validate_iterator(estimator, X, y=None, groups=None, scoring=None, cv=None, X_for_test=None,
                            feature_selection_generator=None, n_feature=0, how='product',
                            progress_bar=None, progress_params=None, n_jobs=1, verbose=0, fit_params=None,
                            pre_dispatch='2*n_jobs', return_train_score="warn", nested=False):
    """
    Iterate over inputs to cross-validation to keep the parallel execution queue filled.

    Parameters
    ----------
    estimator: dict of any to estimator
    X: dict of any to array-like
    y: array-like
    groups: array-like
    scoring: scorer
    cv: splitter
    X_for_test: dict of any to array-like
    feature_selection_generator: callable
    n_feature: int, array-like
    how: str('product')|str('zip')
    progress_bar: callable
    progress_params: dict
    n_jobs: int
    verbose: bool
    fit_params: dict
    pre_dispatch: str
    return_train_score: bool
    nested: bool

    Returns
    -------
    results: dict
    
    """
    # Limitation:
    # * since the Cartesian product of X and X_for_test is taken (if provided) the number of samples in all
    #   datasets must be the same, and only one set of y labels can be provided
    # *
    # This is not the place for generating synthetic data, because:
    # * the iteration structure requires a lot of extra memory (instead of just duplicating X)
    # * not clear what is going to be the train and the test window
    estimator, X, X_for_test = to_dict(estimator), to_dict(X), to_dict(X_for_test)
    n_feature = to_indexable(n_feature)

    if type(y) is dict:
        same_y = False
        cv = check_cv(cv, _first_of_dict(y), classifier=is_classifier(estimator))
    else:
        same_y = True
        cv = check_cv(cv, y, classifier=is_classifier(estimator))
    # scorers, _ = _check_multimetric_scoring(estimator, scoring=scoring)

    if how=='product':
        conditions = (estimator.items(), X.items(), X_for_test.items(), n_feature)
        conditions_iterator = iter_product(*conditions)
        total = np.product([len(x) for x in conditions])
    else:
        conditions = (estimator.items(), X.items(), n_feature)
        conditions_iterator = ((est, (train_key, train), (train_key, X_for_test[train_key]), n_feat) for
                               est, (train_key, train), n_feat in iter_product(*conditions))
        total = np.product([len(x) for x in conditions])

    if progress_bar is None:
        progress_bar = _no_progress_bar

    if feature_selection_generator is None:
        feature_selection_generator = _default_feature_selection
        inner_progress_bar = _no_progress_bar
    else:
        def inner_progress_bar(arr, *args, **kwargs):
            if len(arr) > 1:
                return progress_bar(arr, *args, **kwargs)
            else:
                return arr

    progress_params = {**({} if progress_params is None else progress_params), 'total': total}
    cv_params = {'verbose': verbose, 'fit_params': fit_params, 'return_train_score': return_train_score}

    # We need to clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)

    results = parallel(
        (delayed(_cross_validate_inner)(
            est_key, train_key, test_key, feat, est, train, (y if same_y else y[train_key]),
            groups, scoring, cv, test, cv_params, nested))
        for (est_key, est), (train_key, train), (test_key, test), n_feat in
        progress_bar(conditions_iterator, **progress_params)
        # no harm to convert to list, this is is going to be stored in memory  anyway
        for feat in inner_progress_bar(list(feature_selection_generator(train.shape[1] if n_feat == 0 else n_feat)),
                                       desc='Feat', leave=False))

    keys = ['estimator', 'train', 'test', 'n_feature', 'feature', 'scores']
    results = {k: np.asarray(v) for k, v in zip(keys, zip(*results))}
    scores = results.pop('scores')
    results.update(_aggregate_score_dicts(scores))
    return results

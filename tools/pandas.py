#
# Author: Marcell Stippinger
# License: BSD 3 clause
#

import numpy as np
import pandas as pd
from itertools import product as iter_product


def _no_progress_bar(x, *args, **kwargs):
    return x


def _fill_index_names(idx, single_default='index'):
    """Return names of index levels and substitute None values for default names.

    Parameters
    ----------
    idx: subclass of pandas.Index

    single_default: str, optional

    Returns
    -------
    names: list of str
    """
    data_type = type(idx)
    if issubclass(data_type, pd.MultiIndex):
        result = [('level_%d' % i if s is None else s) for i, s in enumerate(idx.names)]
    else:
        result = [single_default] if idx.name is None else idx.names
    return result


def _get_index_names(data, use_default=False):
    """Return names of index levels and substitute None values for default names.

    Parameters
    ----------
    data: pandas.Series, pandas.DataFrame

    use_default: bool, optional

    Returns
    -------
    names: list of str
    """
    if use_default:
        try:
            single_default = 'level_0' if 'index' in data.columns else 'index'
        except AttributeError:
            single_default = 'index'
        return _fill_index_names(data.index, single_default)
    else:
        return data.index.names


def _get_empty_df(data, column_names=None):
    """Return DataFrame with the same index as `data` and no columns.
    If column_names is provided use them as column levels otherwise preserve
    column levels of `data`.

    Parameters
    ----------
    data: pandas.Series, pandas.DataFrame, subclass of pandas.Index

    column_names: list of str, optional

    Returns
    -------
    df: pandas.DataFrame
    """
    data_type = type(data)
    # NOTE: the following does not capture the column levels
    # return pd.DataFrame(index=data.index, columns=[])
    if issubclass(data_type, pd.Series):
        result = data.to_frame()[[]]
    elif issubclass(data_type, pd.DataFrame):
        result = data[[]]
    elif issubclass(data_type, pd.Index):
        result = pd.DataFrame(index=data, columns=[])
    else:
        raise NotImplementedError('Operation not implemented for type "%s"' % data_type)
    if column_names is not None:
        result.columns = pd.MultiIndex.from_tuples([], names=column_names)
    return result


def pd_reindex(data, index):
    """Reindex with automatic level selection

    Parameters
    ----------
    data: pandas.Series, pandas.DataFrame
    index: subclass of pandas.Index

    Returns
    -------
    reindexed: same type as `data`
    """
    try:
        return data.reindex(index, level=index.names)
    except TypeError:
        return data.reindex(index)


def to_frame(data):
    """Return DataFrame from `data`.

    Parameters
    ----------
    data: pandas.Series, pandas.DataFrame

    Returns
    -------
    df: pandas.DataFrame
    """
    data_type = type(data)
    if issubclass(data_type, pd.Series):
        return data.to_frame()
    elif issubclass(data_type, pd.DataFrame):
        return data
    else:
        raise NotImplementedError('Operation not implemented for type "%s"' % data_type)


def fill_axis_names(data, inplace=False):
    """Fill in names of missing (i.e., None) index and column levels.

    Parameters
    ----------
    data: pandas.Series, pandas.DataFrame

    inplace: bool, optional

    Returns
    -------
    data: same as input
    """
    data_type = type(data)
    if not inplace:
        data = data.copy()
    data.index.names = _get_index_names(data, True)
    if issubclass(data_type, pd.DataFrame):
        data.columns.names = _fill_index_names(data.columns, 'columns')
    return data


def MultiIndex_from_product_indices(indices):
    """Create MultiIndex from cartesian product of (Multi)Indices

    Parameters
    ----------
    indices: subclass of pandas.Index

    Returns
    -------
    idx: pandas.MultiIndex
    """
    names = np.concatenate(idx.names for idx in indices)
    values = iter_product(indices)
    return pd.MultiIndex.from_tuples(values, names=names)


def join_multi(df1, df2, left_on=None, right_on=None, on=None, **kwarg):
    """Join DataFrames on MultiIndex. Just like join, this is a shortcut for the more general merge.

    Parameters
    ----------
    df1, df2: pandas.DataFrame

    left_on, right_on: int, str, list of thereof
        levels of MultiIndex to be used selectively

    on: int, str, list of thereof
        levels of MultiIndex to be used, if it differs for df1 and df2 use `left_on` and `right_on`

    Returns
    -------
    result: pandas.DataFrame
    """
    if on is None and left_on is None and right_on is None:
        left_keys = _get_index_names(df1, True)
        right_keys = _get_index_names(df2, True)
        on = list(set(list(left_keys)) & set(list(right_keys)))
    res_idx = _get_index_names(df1, True)
    result = df1.reset_index().merge(df2.reset_index(), left_on=left_on, right_on=right_on, on=on, **kwarg)
    # raises ValueError if columns is not unique after reset_index
    return result.set_index(res_idx)


def named_loc(df, *args, **kwargs):
    """named_loc(df, name1, crit1, ..., name=crit, ...)

    Take a subset of rows where the query for conditions evaluates True."""
    keep = np.full(len(df.index), True)
    matlab_style_params = np.array(args, dtype=object).reshape((-1,2), order='C')
    python_style_params = np.array(list(kwargs.items()), dtype=object).reshape((-1,2))
    for name, crit in np.concatenate((matlab_style_params, python_style_params)):
        try:
            # assume 'name' is a column
            values = df.loc[:, name].values
        except (KeyError, pd.core.indexing.IndexingError):
            # fall back to index (work either on DataFrame or Series)
            values = df.index.get_level_values(name)
        try:
            # assume 'crit' is callable
            passed = list(map(crit, values))
        except TypeError:
            # fall back to comparison
            passed = np.isin(values, crit)
        keep = passed & keep
    return df.loc[keep]


def outer(left, right, fun=None, **kwargs):
    """Bring two special DataFrames to the same form (same shape and index/columns), then
    analogously to outer product, perform custom function over selected index levels.
    Other index levels and columns are kept.

    Parameters
    ----------
    left, right: pandas.DataFrame
        `left` is allowed to have extra index level while `right` may have an extra column level

    fun: callable, optional
        The function must take two array-like arguments

    **kwargs: optional arguments to be passed to function

    Returns
    -------
    result: if `fun` is provided return outer product,
        otherwise return the two DataFrame objects that would be passed to `fun`
    """
    # TODO: add examples
    left_frame = to_frame(left)
    right_frame = to_frame(right)
    left_enlarged = join_multi(_get_empty_df(right_frame.T, left_frame.index.names), left_frame.T).T
    right_enlarged = join_multi(_get_empty_df(left, right_frame.columns.names), right)
    if fun is None:
        return left_enlarged, right_enlarged
    else:
        return fun(left_enlarged, right_enlarged, **kwargs)


def outer_single(data, levels=-1, fill_value=None, fun=None, **kwargs):
    """Analogously to outer product, perform custom function over selected index levels.
    Other index levels and columns are kept.

    Parameters
    ----------
    level : int, string, or list of these, default last leve
        Level(s) to unstack, can pass level name

    fill_value :
        Replace NaN with this value if the unstack produces missing values
    fun: callable, optional

    **kwargs: optional arguments to be passed to function

    Returns
    -------
    result: if `fun` is provided return outer product,
        otherwise return the two DataFrame objects that would be passed to `fun`
    """
    data = fill_axis_names(data)
    return outer(data, data.unstack(levels, fill_value=fill_value), fun=fun, **kwargs)


def wide_groupby(fun, df, columns=None, fun_args=tuple(),
                 n_jobs=1, verbose=0, pre_dispatch='2 * n_jobs',
                 progress_bar=None, **kwargs):
    """Group DataFrame data based on `columns` if provided else by unique index values and feed chunks to function.
    In contrast to DataFrame.groupby aggregation which works on Series, here the chunks are DataFrames with all columns.
    df: DataFrame
    fun: function
        Operation to be performed on the chunks
        Note: if using parallelism (n_jobs!=0) this function must be defined
        in a module, notebook cell definitions do not work.
    columns: list of str
        coluns to be used for grouping
    fun_args: tuple
        positional function arguments after data chunk
    n_jobs: int, default: 1
        The maximum number of concurrently running jobs, such as the number
        of Python worker processes when backend="multiprocessing"
        or the size of the thread-pool when backend="threading".
        If -1 all CPUs are used. If 1 is given, no parallel computing code
        is used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all
        CPUs but one are used.
    verbose: int, optional
        The verbosity level: if non zero, progress messages are
        printed. Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.
    pre_dispatch: {'all', integer, or expression, as in '3*n_jobs'}
        The number of batches (of tasks) to be pre-dispatched.
        Default is '2*n_jobs'. When batch_size="auto" this is reasonable
        default and the multiprocessing workers should never starve.
    progress_bar: tqdm, optional
        A progress bar that measures the consumption of an iterable.
    **kwargs: dict
        keyword arguments to be passed to function
    """
    from sklearn.externals.joblib import Parallel, delayed
    if progress_bar is None:
        progress_bar = _no_progress_bar
    if columns is not None:
        df = df.reset_index().set_index(columns)
    record_idx = df.index.unique()
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    values = parallel(delayed(fun)(df.loc[[idx], :], *fun_args, **kwargs) for idx in progress_bar(record_idx))
    if issubclass(type(record_idx), pd.Index):
        # This is a workaround of index.unique not keeping names
        record_idx.names = df.index.names
    else:
        # Index.unique may not return an index
        record_idx = pd.MultiIndex.from_tuples(record_idx, names=df.index.names)
    try:
        ret_df = pd.DataFrame(data=values, index=record_idx)
    except:
        ret_df = pd.DataFrame.from_dict({idx: val for idx, val in zip(record_idx, values)}, orient='index')
    return ret_df


def transition_graph(seq):
    """Transition graph of a chain with frequencies"""
    # TODO: same can be done with np.unique
    seq = np.asarray(seq)
    df = pd.DataFrame()
    df['from'] = seq[:-1]
    df['to'] = seq[1:]
    df['count'] = 1
    return df.groupby(['from', 'to']).sum().reset_index()
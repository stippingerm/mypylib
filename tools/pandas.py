import numpy as np
import pandas as pd
from itertools import product as iter_product


def _fill_index_names(idx, single_default='index'):
    data_type = type(idx)
    if issubclass(data_type, pd.MultiIndex):
        result = [('level_%d'%i if s is None else s) for i, s in enumerate(idx.names)]
    else:
        result = [single_default] if idx.name is None else idx.names
    return result


def _get_index_names(data, use_default=False):
    if use_default:
        try:
            single_default = 'level_0' if 'index' in data.columns else 'index'
        except AttributeError:
            single_default = 'index'
        return _fill_index_names(data.index, single_default)
    else:
        return data.index.names


def _get_empty_df(data, column_names=None):
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


def to_frame(data):
    data_type = type(data)
    if issubclass(data_type, pd.Series):
        return data.to_frame()
    elif issubclass(data_type, pd.DataFrame):
        return data
    else:
        raise NotImplementedError('Operation not implemented for type "%s"'%data_type)


def fill_axis_names(data, inplace=False):
    if not inplace:
        data = data.copy()
    data.index.names = _get_index_names(data, True)
    data.columns.names = _fill_index_names(data.columns, 'columns')
    return data


def MultiIndex_from_product_indices(indices):
    names = np.concatenate(idx.names for idx in indices)
    values = iter_product(indices)
    return pd.MultiIndex.from_tuples(values, names=names)


def join_multi(df1, df2, left_on=None, right_on=None, on=None, **kwarg):
    if left_on is None:
        left_on = _get_index_names(df1, True)
    if right_on is None:
        right_on = _get_index_names(df2, True)
    if on is None:
        on = list(set(list(left_on)) & set(list(right_on)))
    res_idx = _get_index_names(df1, True)
    result = df1.reset_index().merge(df2.reset_index(), on=on, **kwarg)
    # raises ValueError if columns is not unique after reset_index
    return result.set_index(res_idx)


def outer(fun, left, right, **kwargs):
    # left is allowed to have extra index level while right may have an extra column level
    # they are then brought to the same shape and index/columns
    left_frame = to_frame(left)
    right_frame = to_frame(right)
    left_enlarged = join_multi(_get_empty_df(right_frame.T, left_frame.index.names), left_frame.T).T
    right_enlarged = join_multi(_get_empty_df(left, right_frame.columns.names), right)
    if fun is None:
        return left_enlarged, right_enlarged
    else:
        return fun(left_enlarged, right_enlarged, **kwargs)


def outer_single(fun, data, levels, **kwargs):
    data = fill_axis_names(data)
    return outer(fun, data, data.unstack(levels), **kwargs)

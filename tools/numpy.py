# import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

python3 = sys.version_info > (3, 0)

# Alias for len
length = len


def mat_var(data, struc, field):
    """Get the 'field' from the 'struc' in the 'data' imported from Matlab"""
    return data[struc][field].item()


def compress_seq(seq, add_start=True, add_end=False):
    """Find changes between consecutive elements of `seq` (i.e., `seq[i-1]!=seq[i]`)
       and report their place along with the new value (place=i, new_value=seq[i]).
       Note: to reconstruct the whole sequence, both `add_start` and `add_end` are required."""
    seq = np.ravel(seq)
    start = [True] if add_start else []
    end = [True] if add_end else []
    change = np.concatenate((start, seq[1:] != seq[:-1], end)).astype(bool)
    where = np.where(change)[0]
    if add_end:
        what = np.concatenate((seq[where[:-1]], np.array([np.nan]).astype(seq.dtype)))
    else:
        what = seq[where]
    return where, what


def adjust_series(seq, axis=None):
    """Normalize the series into [0,1] via a shift and a multiplication"""
    if axis is None:
        smin = np.nanmin(seq)
        smax = np.nanmax(seq)
    else:
        smin = np.nanmin(seq, axis, keepdims=True)
        smax = np.nanmax(seq, axis, keepdims=True)
    ret = (seq - smin) * (1.0 / (smax - smin))
    return ret


def shrink_series(seq, axis=None):
    """Normalize the [0,any] series into [0,1] via a shift and a multiplication"""
    if axis is None:
        smax = np.nanmax(seq)
    else:
        smax = np.nanmax(seq, axis, keepdims=True)
    ret = seq * (1.0 / smax)
    return ret


def normalize_series(seq, axis=None):
    """Normalize the series E=0, D2=1 via a shift and a multiplication"""
    if axis is None:
        smean = np.nanmean(seq)
        sdev = np.nanstd(seq)
    else:
        smean = np.nanmean(seq, axis, keepdims=True)
        sdev = np.nanstd(seq, axis, keepdims=True)
    ret = (seq - smean) * (1.0 / (sdev))
    return ret


def square_angle(x, y):
    """return square-shaped angle between (0,8)"""
    # reg = (3 if y<0 else 2) if x<0 else (4 if y<0 else 1)
    reg = np.ones((len(x),))
    reg[x * y < 0] += 1
    reg[y < 0] += 2
    sub = np.sign(x) * y - np.sign(y) * x
    return reg * 2 + sub - 1


def bool_to_ord(arr):
    """Get ordinal numbers from a boolean array"""
    seq = np.arange(len(arr))
    idx = np.array(arr, dtype=bool)
    return seq[idx]


def ord_to_bool(seq, maximum=None):
    """Boolean selector from ordinal numbers"""
    if maximum is None:
        maximum = max(seq)
    arr = np.full(maximum + 1, False)
    arr[seq] = True
    return arr


def fill(seq, nanvalue=None):
    reg = np.array(seq)
    reg[reg is nanvalue] = np.nan
    reg = pd.Series(reg).fillna(method='pad').values
    return reg


def unfold(seq, nanvalue=None):
    """Unfold data represented on a [0, 2pi) torus, calculating the number of spins"""
    # NOTE: fill was removed
    seq = np.asarray(seq)
    diff = np.hstack(([0], seq[1:] - seq[:-1]))
    jump = np.abs(diff) > np.pi
    cw = np.sign(diff)
    cw[~jump] = 0
    spin = np.cumsum(-cw * 2 * np.pi)
    return seq + spin


def transition_graph(seq):
    """Transition graph of a chain with frequencies"""
    # TODO: same can be done with np.unique
    seq = np.asarray(seq)
    df = pd.DataFrame()
    df['from'] = seq[:-1]
    df['to'] = seq[1:]
    df['count'] = 1
    return df.groupby(['from', 'to']).sum().reset_index()


def py_strictly_increasing(L):
    """test series"""
    return all(x < y for x, y in zip(L[:-1], L[1:]))


def py_strictly_decreasing(L):
    """test series"""
    return all(x > y for x, y in zip(L[:-1], L[1:]))


def py_non_increasing(L):
    """test series"""
    return all(x >= y for x, y in zip(L[:-1], L[1:]))


def py_non_decreasing(L):
    """test series"""
    return all(x <= y for x, y in zip(L[:-1], L[1:]))


def hls_matrix(h, l, s):
    """map hls to rgb"""
    from colorsys import hls_to_rgb

    c = np.vectorize(hls_to_rgb)(h, l, s)  # --> tuple (r,g,b)
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0, 2)

    return c


def colorize_cplx(z):
    """nice colors for complex numbers"""
    r = np.abs(z)
    arg = np.angle(z)

    h = (arg + np.pi) / (2 * np.pi) + 0.5
    l = 1.0 - 1.0 / (1.0 + r ** 0.3)
    s = 0.8

    return hls_matrix(h, l, s)

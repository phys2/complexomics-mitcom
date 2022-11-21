"""
Distance calculation for profile components.
"""
import math

import numpy as np
from numba import njit


def pearson_1d(a: np.ndarray, b: np.ndarray, nan: float = 0.0) -> np.ndarray:
    """
    Return 1d-array Pearson's product-moment correlation coefficients.

    Correlate array `a` with all arrays from `b`.

    :param a: 1d-array
    :param b: 1d- or 2d-array.
    :param nan: If set, replace any NaN with with value
    :return: Pearson product-moment correlation coefficients as 1d-array.
    """
    r = np.corrcoef(a, b)[0, 1:]
    if nan is not None:
        r = np.nan_to_num(r, nan=nan, copy=False)
    return r


def pearson(a: np.ndarray, b: np.ndarray, nan: float = 0.0) -> np.ndarray:
    """
    Return 2d-array of Pearson's product-moment correlation coefficients.

    Correlate all rows array `a` with all arrays from `b`.

    :param a: 2d-array
    :param b: 2d-array
    :param nan: If set, replace any NaN with with value
    :return: Pearson product-moment correlation coefficients as 2d-array.
    """
    r = np.corrcoef(a, b)
    if nan is not None:
        r = np.nan_to_num(r, nan=nan, copy=False)
    return r


@njit(cache=True)
def custom_dist(a, b):
    """
    Note:
      a and b have following layout:
      [x0, x1, abund, r, <empty>] + [profile_data]
    """
    # number of features/dimensions
    n = 5

    # feature vector of n dimensions
    v = a[:n] - b[:n]

    # custom tranformation for slice number distances
    for i in [0, 1]:
        d = abs(v[i])
        v[i] = 0.5 * math.sqrt(d - 1) if d > 1 else 0

    # shift profile indices by length of leading metadata array
    ax0 = a[0] + n
    ax1 = a[1] + n
    bx0 = b[0] + n
    bx1 = b[1] + n

    x0 = max(ax0, bx0)
    x1 = min(ax1, bx1)

    if x0 > x1 - 4:
        v[-1] = 2
    else:
        x0 = min(ax0, bx0)
        x1 = max(ax1, bx1)
        p = np.corrcoef(a[x0 : x1 + 1], b[x0 : x1 + 1])[0, 1]
        if np.isnan(p):
            # print('p is NaN')
            v[-1] = 2
        else:
            v[-1] = 1 - p

    # euclidean distance in feature space
    res = np.linalg.norm(v)

    if res < 1e-10:
        res = 0.0

    return res

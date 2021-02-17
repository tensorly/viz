from warnings import warn

import numpy as np

from ._utils import is_iterable


def _leverage(factor_matrix):
    A = factor_matrix
    leverage = A * np.linalg.solve(A.T @ A, A.T)
    return leverage


def _slicewise_sse(estimated, true, axis=None):
    if not is_iterable(axis):
        axis = {axis}
    axis = set(axis)

    reduction_axis = tuple(i for i in range(true.ndim) if i not in axis)
    return ((estimated - true)**2).sum(axis=reduction_axis)




from warnings import warn

import numpy as np
import pandas as pd

from ._utils import is_iterable
from .factor_tools import construct_cp_tensor


_LEVERAGE_NAME = "Leverage score"
_SLABWISE_SSE_NAME = "Slabwise SSE"


def _compute_leverage(factor_matrix):
    A = factor_matrix
    leverage = A @ np.linalg.solve(A.T @ A, A.T)
    return np.diag(leverage)


def _compute_slabwise_sse(estimated, true, axis=0):
    if not is_iterable(axis):
        axis = {axis}
    axis = set(axis)

    reduction_axis = tuple(i for i in range(true.ndim) if i not in axis)
    return ((estimated - true)**2).sum(axis=reduction_axis)


def compute_slabwise_sse(estimated, true, axis=0):
    """Compute the slabwise SSE along the given mode(s).

    # TODO: Write description of slabwise SSE.

    Arguments
    ---------
    estimated : xarray or numpy array
        Estimated dataset, if this is an xarray, then the output is too.
    true : xarray or numpy array
        True dataset, if this is an xarray, then the output is too.
    axis : int
        Axis (or axes) that the SSE is computed across (i.e. these are not the ones summed over).
        The output will still have these axes.
    
    Returns
    -------
    slab_sse : xarray or numpy array
        The slabwise-SSE, if true tensor input is an xarray array, then the returned
        tensor is too.
    """
    # Check that dimensions match up.
    if hasattr(estimated, 'to_dataframe') and hasattr(true, 'to_dataframe'):
        if estimated.dims != true.dims:
            raise ValueError(
                f"Dimensions of estimated and true tensor must be equal,"
                f" they are {estimated.dims} and {true.dims}, respectively."
            )
        for dim in estimated.dims:
            if len(true.coords[dim]) != len(estimated.coords[dim]):
                raise ValueError(
                    f"The dimension {dim} has different length for the true and estiamted tensor. "
                    f"The true tensor has length {len(true.coords[dim])} and the estimated tensor "
                    f"has length {len(estimated.coords[dim])}."
                )
            if not all(true.coords[dim] == estimated.coords[dim]):
                raise ValueError(
                    f"The dimension {dim} has different coordinates for the true and estimated tensor."
                )

    slab_sse = _compute_slabwise_sse(estimated, true, axis=axis)
    if hasattr(slab_sse, 'to_dataframe'):
        slab_sse.name = _SLABWISE_SSE_NAME
    return slab_sse


def compute_leverage(factor_matrix):
    """Compute the leverage score of the given factor matrix.

    # TODO: Write description of leverage.

    If the factor matrix is a dataframe (i.e. has an index), then the output is
    also a dataframe with that index. Otherwise, the output is a NumPy array.

    Arguments
    ---------
    factor_matrix : DataFrame or numpy array
        The factor matrix whose leverage we compute
    
    Returns
    -------
    leverage : DataFrame or numpy array
        The leverage scores, if the input is a dataframe, then the index is preserved.
    """
    leverage = _compute_leverage(factor_matrix)
    
    if hasattr(factor_matrix, "index"):
        return pd.DataFrame(leverage.reshape(-1, 1), columns=[_LEVERAGE_NAME], index=factor_matrix.index)
    else:
        return leverage
    

def compute_outlier_info(cp_tensor, true_tensor, axis=0):
    f"""Compute the leverage score and Slabwise SSE along one axis.

    # TODO: Write description of how to use this.

    These metrics are often plotted against each other to discover outliers.

    Arguments
    ---------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument
    true_tensor : xarray or numpy array
        Dataset that cp_tensor is fitted against.
    axis : int

    Returns
    -------
    DataFrame
        Dataframe with two columns, "{_LEVERAGE_NAME}" and "{_SLABWISE_SSE_NAME}".
    """
    leverage = compute_leverage(cp_tensor[1][axis])

    estimated_tensor = construct_cp_tensor(cp_tensor)
    slab_sse = compute_slabwise_sse(estimated_tensor, true_tensor, axis=axis)
    if hasattr(slab_sse, 'to_dataframe'):
        slab_sse = pd.DataFrame(slab_sse.to_series())

    is_labelled = isinstance(leverage, pd.DataFrame)
    is_xarray = isinstance(slab_sse, pd.DataFrame)
    if (is_labelled and not is_xarray) or (not is_labelled and is_xarray):
        raise ValueError(
            "If `cp_tensor` is labelled (factor matrices are dataframes), then"
            "`true_tensor` should be an xarray object and vice versa."
        )
    elif not all(slab_sse.index == leverage.index):
        raise ValueError(
            "The indices of the labelled factor matrices does not match up with the xarray dataset"
        )

    results = pd.concat([leverage, slab_sse], axis=1)
    results.columns = [_LEVERAGE_NAME, _SLABWISE_SSE_NAME]
    return results

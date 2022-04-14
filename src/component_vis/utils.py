import inspect
from functools import wraps

import numpy as np
import xarray as xr

from ._module_utils import validate_cp_tensor
from .xarray_wrapper import _handle_labelled_dataset, is_labelled_cp, is_labelled_tucker

__all__ = ["extract_singleton", "unfold_tensor", "cp_to_tensor", "tucker_to_tensor", "normalise"]


def _alias_mode_axis():
    def decorator(func):
        func_sig = inspect.signature(func)
        if "axis" not in func_sig.parameters or "mode" not in func_sig.parameters:
            raise TypeError(f"Function {func} needs both ``mode`` and ``axis`` as possible arguments.")
        mode_default_value = func_sig.parameters["mode"].default
        if mode_default_value == inspect._empty:
            mode_default_value = None

        @wraps(func)
        def new_func(*args, **kwargs):
            bound_arguments = func_sig.bind_partial(*args, **kwargs)

            mode = bound_arguments.arguments.get("mode", mode_default_value)
            axis = bound_arguments.arguments.get("axis", None)
            if mode is None and axis is None:
                raise TypeError(
                    f"Function {func} needs either ``mode`` or ``axis`` to be set to a value different than None."
                )
            elif mode != mode_default_value and axis is not None:
                raise TypeError("Either ``mode`` or ``axis`` can be specified, not both.")
            elif axis is not None:
                bound_arguments.arguments["mode"] = axis
            return func(**bound_arguments.arguments)

        return new_func

    return decorator


def extract_singleton(x):
    """Extracts a singleton from an array.

    This is useful whenever XArray or Pandas is used, since many NumPy functions that
    return a number may return a singleton array instead.

    Parameters
    ----------
    x : float, numpy.ndarray, xarray.DataArray or pandas.DataFrame
        Singleton array to extract value from.

    Returns
    -------
    float
        Singleton value extracted from ``x``.
    """
    return np.asarray(x).reshape(-1).item()


@_handle_labelled_dataset("tensor", None)
@_alias_mode_axis()
def unfold_tensor(tensor, mode, axis=None):
    """Unfolds (matricises) a potentially labelled data tensor into a numpy array along given mode.

    Arguments
    ---------
    tensor : np.ndarray or xarray.DataArray
        Dataset to unfold
    mode : int
        Which mode (axis) to unfold the dataset along.
    axis : int (optional)
        Which mode (axis) to unfold the dataset along. If set, then the mode-argument is unused.

    Returns
    -------
    np.ndarray
        The unfolded dataset as a numpy array.
    """
    # TODO: return xarray or dataframe if tensor is labelled
    dataset = np.asarray(tensor)
    return np.moveaxis(dataset, mode, 0).reshape(dataset.shape[mode], -1)


def cp_to_tensor(cp_tensor):
    """Convert a CP tensor to a dense array.

    This function is equivalent to ``cp_to_tensor`` in TensorLy, but supports dataframes.

    If the factor matrices are data frames, then the tensor will be returned as a labelled
    xarray. Otherwise, it will be returned as a numpy array.

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument.

    Returns
    -------
    xarray or np.ndarray
        Dense tensor represented by the decomposition.
    """
    # TODOC: Example with and without labels
    cp_tensor = validate_cp_tensor(cp_tensor)

    if cp_tensor[0] is None:
        weights = np.ones(cp_tensor[1][0].shape[1])
    else:
        weights = cp_tensor[0].reshape(-1)

    einsum_input = "R"
    einsum_output = ""
    for mode in range(len(cp_tensor[1])):
        idx = chr(ord("a") + mode)

        # We cannot use einsum with letters outside the alphabet
        if ord(idx) > ord("z"):
            max_modes = ord("a") - ord("z") - 1
            raise ValueError(f"Cannot have more than {max_modes} modes. Current components have {len(cp_tensor[1])}.")

        einsum_input += f", {idx}R"
        einsum_output += idx

    tensor = np.einsum(f"{einsum_input} -> {einsum_output}", weights, *cp_tensor[1])

    if not is_labelled_cp(cp_tensor):
        return tensor

    # Convert to labelled xarray DataArray:
    coords_dict = {}
    dims = []
    for mode, fm in enumerate(cp_tensor[1]):
        mode_name = f"Mode {mode}"
        if fm.index.name is not None:
            mode_name = fm.index.name

        coords_dict[mode_name] = fm.index.values
        dims.append(mode_name)

    return xr.DataArray(tensor, dims=dims, coords=coords_dict)


def tucker_to_tensor(tucker_tensor):
    """Convert a Tucker tensor to a dense array.

    This function is equivalent to ``tucker_to_tensor`` in TensorLy, but supports dataframes.

    If the factor matrices are data frames, then the tensor will be returned as a labelled
    xarray. Otherwise, it will be returned as a numpy array.

    Parameters
    ----------
    tucker : CPTensor or tuple
        TensorLy-style TuckerTensor object or tuple with weights as first
        argument and a tuple of components as second argument.

    Returns
    -------
    xarray or np.ndarray
        Dense tensor represented by the decomposition.
    """
    einsum_core = ""
    einsum_input = ""
    einsum_output = ""
    if len(tucker_tensor[1]) > 16:
        raise ValueError("NumPy's einsum function doesn't support forming dense Tucker arrays with more than 16 modes.")

    for mode in range(len(tucker_tensor[1])):
        idx = chr(ord("a") + mode)
        rank_idx = chr(ord("A") + mode)

        einsum_core += rank_idx
        einsum_input += f", {idx}{rank_idx}"
        einsum_output += idx

    tensor = np.einsum(f"{einsum_core}{einsum_input} -> {einsum_output}", tucker_tensor[0], *tucker_tensor[1],)
    if not is_labelled_tucker(tucker_tensor):
        return tensor

    # Convert to labelled xarray DataArray:
    coords_dict = {}
    dims = []
    for mode, fm in enumerate(tucker_tensor[1]):
        mode_name = f"Mode {mode}"
        if fm.index.name is not None:
            mode_name = fm.index.name

        coords_dict[mode_name] = fm.index.values
        dims.append(mode_name)

    return xr.DataArray(tensor, dims=dims, coords=coords_dict)


@_alias_mode_axis()
def normalise(x, mode=0, axis=None):
    """Normalise a matrix (or tensor) so all columns (or fibers) have unit norm.

    Parameters
    ----------
    x : np.ndarray
        Matrix (or vector/tensor) to normalise.
    mode : int
        Axis along which to normalise, if 0, then all columns will have unit norm
        and if 1 then all rows will have unit norm. When normalising a tensor, then
        the axis represents the axis whose fibers should have unit norm.
    axis : int
        Alias for mode. If this is provided, then no value for mode can be provided.

    Returns
    -------
    np.ndarray
        Normalised matrix

    Examples
    --------
    >>> random_matrix = np.random.random_sample((3, 4))
    >>> matrix_normalized_cols = normalise(random_matrix, axis=0)
    >>> print(np.linalg.norm(matrix_normalized_cols, axis=0))
    [1. 1. 1. 1.]

    >>> random_matrix = np.random.random_sample((3, 4))
    >>> matrix_normalized_rows = normalise(random_matrix, axis=1)
    >>> print(np.linalg.norm(matrix_normalized_rows, axis=1))
    [1. 1. 1.]
    """
    norms = np.linalg.norm(x, axis=mode, keepdims=True)
    norms[norms == 0] = 1
    return x / norms

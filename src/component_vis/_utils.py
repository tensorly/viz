import inspect
from functools import wraps

import numpy as np
import xarray

from .xarray_wrapper import _handle_labelled_dataset


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


def is_iterable(x):
    """Check if variable is iterable

    Arguments
    ---------
    x
        Variable to check if is iterable
    
    Returns
    -------
    bool
        Whether ``x`` is iterable or not.
    """
    try:
        iter(x)
    except TypeError:
        return False
    else:
        return True


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

from functools import wraps
from inspect import signature

import numpy as np
import pandas as pd
import xarray as xr


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


def is_xarray(x):
    """Check if ``x`` is an xarray data array.

    Arguments
    ---------
    x
        Object to check

    Returns
    -------
    bool
        ``True`` if x is an xarray data array, ``False`` otherwise.
    """
    # TODO: Is this how we want to check?
    return isinstance(x, xr.DataArray)


def is_dataframe(x):
    """Check if ``x`` is a data frame.

    Arguments
    ---------
    x
        Object to check

    Returns
    -------
    bool
        ``True`` if x is a data frame, ``False`` otherwise.
    """
    return isinstance(x, pd.DataFrame)


def is_labelled_dataset(x):
    # TODOC: is_labelled_Dataset
    return is_dataframe(x) or is_xarray(x)


def _check_is_argument(func, arg_name):
    sig = signature(func)
    if arg_name in sig.parameters:
        return
    raise ValueError(f"{arg_name} is not an argument of {func}")


def _handle_none_weights_cp_tensor(cp_tensor_name, optional=False):
    def decorator(func):
        _check_is_argument(func, cp_tensor_name)

        @wraps(func)
        def func2(*args, **kwargs):
            bound_arguments = signature(func).bind(*args, **kwargs)

            if optional and cp_tensor_name not in bound_arguments.arguments:
                return func(*bound_arguments.args, **bound_arguments.kwargs)

            cp_tensor = bound_arguments.arguments.get(cp_tensor_name, None)  # TODO: validate cp_tensor?
            weights, factors = cp_tensor
            if weights is None:
                rank = factors[0].shape[1]
                cp_tensor = (np.ones(rank), factors)
                bound_arguments.arguments[cp_tensor_name] = cp_tensor

            out = func(*bound_arguments.args, **bound_arguments.kwargs)
            return out

        return func2

    return decorator

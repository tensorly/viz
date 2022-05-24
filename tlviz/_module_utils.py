# -*- coding: utf-8 -*-

__author__ = "Marie Roald & Yngve Mardal Moe"

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
    """Check if ``x`` is an xarray DataArray.

    Arguments
    ---------
    x
        Object to check

    Returns
    -------
    bool
        ``True`` if x is an xarray DataArray, ``False`` otherwise.
    """
    return isinstance(x, xr.DataArray)


def is_dataframe(x):
    """Check if ``x`` is a DataFrame.

    Arguments
    ---------
    x
        Object to check

    Returns
    -------
    bool
        ``True`` if x is a DataFrame, ``False`` otherwise.
    """
    return isinstance(x, pd.DataFrame)


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

            cp_tensor = bound_arguments.arguments.get(cp_tensor_name, None)
            weights, factors = validate_cp_tensor(cp_tensor)
            if weights is None:
                rank = factors[0].shape[1]
                cp_tensor = (np.ones(rank), factors)
                bound_arguments.arguments[cp_tensor_name] = cp_tensor

            out = func(*bound_arguments.args, **bound_arguments.kwargs)
            return out

        return func2

    return decorator


def validate_cp_tensor(cp_tensor):
    weights, factors = cp_tensor
    rank = factors[0].shape[1]
    for i, factor in enumerate(factors):
        if factor.ndim != 2:
            raise ValueError(
                f"All factor matrices should have two dimensions, but factor[{i}] has shape {factor.shape}."
            )

        # Check that all factor matrices have the same rank
        if factor.shape[1] != rank:
            raise ValueError(
                "All factor matrices should have the same number of columns. However factors[0] has shape"
                + f" {factors[0].shape} and factors[{i}] has shape {factor.shape}."
            )

    if weights is not None:
        if not isinstance(weights, np.ndarray):
            raise TypeError(f"The weights must be a numpy array, not {type(weights)}")
        if len(weights) != rank:
            raise ValueError("The weights should have the same length as the number of columns in the factor matrices.")
        if weights.ndim != 1:
            raise ValueError(f"The weights must be 1d array, {weights.ndim} != 1 ")
    return cp_tensor


_SINGLETON = object()

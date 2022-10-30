from functools import wraps
from inspect import signature
from textwrap import dedent

import numpy as np

try:
    import tensorly as tl
except ImportError:
    HAS_TENSORLY = False
else:
    HAS_TENSORLY = True

from ._module_utils import _SINGLETON, _check_is_argument
from ._xarray_wrapper import (
    is_labelled_cp,
    is_labelled_dataset,
)


def _all_not(iterable):
    return all(not x for x in iterable)


def is_tensorly_cp(cp_tensor, none_ok=False):
    if cp_tensor is None and none_ok:
        return False
    elif cp_tensor is None:
        raise TypeError("cp_tensor is None, but none_ok=False")

    if not HAS_TENSORLY:
        return False

    weights, factors = cp_tensor
    tl_weights = tl.is_tensor(weights) or weights is None
    tl_factors = [tl.is_tensor(factor) for factor in factors]

    if tl_weights and all(tl_factors):
        return True
    elif not tl_weights and _all_not(tl_factors):
        return False
    elif is_labelled_cp(cp_tensor):
        return False
    else:
        raise TypeError(
            dedent(
                f"""\
                CP tensor has mixed tensorly and non-tensorly types.
                The weights have type: {type(weights)}
                The factors have types: {[type(f) for f in factors]}
                """.rstrip()
            )
        )


def to_numpy_cp(cp_tensor, cast_labelled_cp=True):
    if is_labelled_cp(cp_tensor) and not cast_labelled_cp:
        return cp_tensor
    elif is_labelled_cp(cp_tensor):
        weights, factors = cp_tensor
        return weights, [factor.values for factor in factors]

    if not is_tensorly_cp(cp_tensor):
        return cp_tensor

    weights, factors = cp_tensor
    if weights is not None:
        weights = to_numpy(weights)

    factors = [to_numpy(factor_matrix) for factor_matrix in factors]

    return weights, factors


def to_numpy(x, cast_labelled=True):
    if is_labelled_dataset(x) and not cast_labelled:
        return x
    elif is_labelled_dataset(x):
        return x.values

    if not HAS_TENSORLY:
        return np.asarray(x)
    return tl.to_numpy(x)


def _handle_tensorly_backends_cp(cp_tensor_name, output_cp_tensor_index, optional=False):
    # TOTEST: _handle_tensorly_backends_cp
    def decorator(func):
        _check_is_argument(func, cp_tensor_name)

        @wraps(func)
        def func2(*args, **kwargs):
            if not HAS_TENSORLY:
                return func(*args, **kwargs)

            bound_arguments = signature(func).bind(*args, **kwargs)

            cp_tensor = bound_arguments.arguments.get(cp_tensor_name, None)
            if cp_tensor is None and optional:
                np_cp_tensor = None
            elif cp_tensor is None:
                raise ValueError("cp_tensor is None, but it is not optional.")
            else:
                np_cp_tensor = to_numpy_cp(cp_tensor, cast_labelled_cp=False)

            bound_arguments.arguments[cp_tensor_name] = np_cp_tensor
            out = func(*bound_arguments.args, **bound_arguments.kwargs)

            if output_cp_tensor_index is _SINGLETON and is_tensorly_cp(cp_tensor, none_ok=optional):
                weights, factors = out
                if weights is not None:
                    weights = tl.tensor(weights)
                return tl.cp_tensor.CPTensor((weights, [tl.tensor(factor) for factor in factors]))
            elif output_cp_tensor_index is not None and is_tensorly_cp(cp_tensor, none_ok=optional):
                weights, factors = out[output_cp_tensor_index]
                if weights is not None:
                    weights = tl.tensor(weights)
                out_cp_tensor = tl.cp_tensor.CPTensor((weights, [tl.tensor(factor) for factor in factors]))

                return (
                    *out[:output_cp_tensor_index],
                    out_cp_tensor,
                    *out[output_cp_tensor_index + 1 :],
                )
            return out

        return func2

    return decorator


def _handle_tensorly_backends_dataset(dataset_name, output_dataset_index):
    def decorator(func):
        _check_is_argument(func, dataset_name)

        @wraps(func)
        def func2(*args, **kwargs):
            if not HAS_TENSORLY:
                return func(*args, **kwargs)

            bound_arguments = signature(func).bind(*args, **kwargs)

            dataset = bound_arguments.arguments.get(dataset_name, None)
            if tl.is_tensor(dataset):
                is_tensorly = True
                np_dataset = tl.to_numpy(dataset)
            else:
                is_tensorly = False
                np_dataset = dataset

            bound_arguments.arguments[dataset_name] = np_dataset
            out = func(*bound_arguments.args, **bound_arguments.kwargs)

            if output_dataset_index is _SINGLETON and is_tensorly:
                return tl.tensor(out)
            elif output_dataset_index is not None and is_tensorly:
                out_dataset = tl.tensor(out[output_dataset_index])

                return (
                    *out[:output_dataset_index],
                    out_dataset,
                    *out[output_dataset_index + 1 :],
                )
            return out

        return func2

    return decorator

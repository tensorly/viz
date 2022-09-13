# -*- coding: utf-8 -*-
"""This module contains utilities for seamlessly handling DataFrames as factor matrices and xarray DataArrays as data
"""

__author__ = "Marie Roald & Yngve Mardal Moe"

from functools import wraps
from inspect import signature
from warnings import warn

import numpy as np
import pandas as pd
import xarray as xr

from ._module_utils import (
    _SINGLETON,
    _check_is_argument,
    is_dataframe,
    is_xarray,
    validate_cp_tensor,
)

__all__ = [
    "is_dataframe",
    "is_xarray",
    "is_labelled_dataset",
    "is_labelled_cp",
    "is_labelled_tucker",
    "label_cp_tensor",
]


def add_factor_metadata(cp_tensor, dataset):
    """Adds the additional coordinates along each dataset dimension as new columns in the factor matrices.

    The coordinates of xarray DataArrays can contain metadata. For each dimension, there may be additional
    coordinates that are not used for indexing purposes. This function will iterate over all modes of
    a dataset and a labelled CP tensor and add the additional coordinates as new columns in the factor
    matrices.

    Parameters
    ----------
    cp_tensor : labelled CP Tensor
    dataset : xarray.DataArray

    Returns
    -------
    tuple
        CP-tensor like tuple where the factor matrices are augmented with additional metadata.

    Examples
    --------
    >>> from tlviz.data import load_oslo_city_bike
    >>> from tlviz.postprocessing import postprocess, add_factor_metadata
    >>> from tensorly.decomposition import parafac
    >>> bikes = load_oslo_city_bike()
    >>> bikes.coords
    Coordinates:
      * End station name  (End station name) object '7 Juni Plassen' ... 'Økernve...
        lat               (End station name) float64 59.92 59.93 ... 59.93 59.92
        lon               (End station name) float64 10.73 10.75 ... 10.8 10.78
      * Hour              (Hour) int32 0 1 2 3 4 5 6 7 8 ... 16 17 18 19 20 21 22 23
      * Month             (Month) int32 1 2 3 4 5 6 7 8 9 10 11 12
      * Day of week       (Day of week) int32 0 1 2 3 4 5 6
      * Year              (Year) int32 2020 2021

    We see that the ``End station name`` dimension has two additional columns: ``lat`` and ``lon``.
    These contain metadata about the end station coordinates, and it can be useful to have these
    columns also in the factor matrices. To do this, we first fit a PARAFAC model to the dataset,
    then we postprocess it to label the CP tensor and finally, we add the metadata information

    >>> cp = parafac(bikes.data, 3, init="random")
    >>> cp_labelled = postprocess(cp, bikes)
    >>> print(cp_labelled[1][0].columns)
    RangeIndex(start=0, stop=3, step=1)
    >>> cp_with_metadata = add_factor_metadata(cp_labelled, bikes)
    >>> print(cp_with_metadata[1][0].columns)
    Index([0, 1, 2, 'lat', 'lon'], dtype='object')

    We see that when we add the metadata, then the latitude and longitude columns are added
    to the dataframe.
    """
    if not is_labelled_cp(cp_tensor):
        raise ValueError("The CP tensor must be labelled with the same labels as the dataset.")
    if not is_labelled_dataset(dataset):
        raise ValueError("The dataset must be labelled with the same labels as the CP tensor.")

    weights, factor_matrices = cp_tensor
    factors_with_metadata = [None] * len(factor_matrices)
    for mode, factor_matrix in enumerate(factor_matrices):
        dim_name = factor_matrix.index.name
        coords = dataset.coords[dim_name]
        metadata = pd.DataFrame({name: coords.coords[name].to_pandas() for name in coords.coords})
        metadata = metadata.drop(dim_name, axis=1)
        factors_with_metadata[mode] = factor_matrix.join(metadata)
    return weights, factors_with_metadata


def _label_factor_matrices(factor_matrices, dataset):
    if is_xarray(dataset):

        def xarray_to_pandas_index(dataset, dim_name):
            return dataset.coords[dim_name].xindexes[dim_name].to_pandas_index()

        factor_matrices = [
            pd.DataFrame(factor_matrix, index=xarray_to_pandas_index(dataset, dim_name))
            for factor_matrix, dim_name in zip(factor_matrices, dataset.dims)
        ]
    elif is_dataframe(dataset) and len(factor_matrices) == 2:
        factor_matrices = [
            pd.DataFrame(factor_matrices[0], index=dataset.index),
            pd.DataFrame(factor_matrices[1], index=dataset.columns),
        ]
    else:
        raise ValueError(
            "``dataset`` must be xarray.DataArray or, pandas.DataFrame "
            "(only possible if ``len(factor_matrices) == 2``)"
        )
    return factor_matrices


def label_cp_tensor(cp_tensor, dataset):
    """Label the CP tensor by converting the factor matrices into DataFrames with a sensible index.

    Convert the factor matrices into Pandas DataFrames where the DataFrame indices
    are given by the coordinate names of an xarray DataArray. If the dataset has only
    two modes, then it can also be a pandas DataFrame.

    Parameters
    ----------
    cp_tensor : CPTensor
        CP Tensor whose factor matrices should be labelled
    dataset : xarray.DataArray of pandas.DataFrame
        Dataset used to label the factor matrices

    Returns
    -------
    CPTensor
        Tuple on the CPTensor format, except that the factor matrices are DataFrames.
    """
    if is_labelled_cp(cp_tensor) and is_labelled_dataset(dataset):
        warn(
            "Both the CP tensor and the dataset is labelled, the labels from the cp tensor will be overwritten "
            + " with the labels from the dataset."
        )
        cp_tensor = _unlabel_cp_tensor(cp_tensor, optional=False, preserve_columns=True)[0]

    if is_xarray(dataset) or is_dataframe(dataset):
        return (cp_tensor[0], _label_factor_matrices(cp_tensor[1], dataset))
    elif isinstance(dataset, np.ndarray):
        return cp_tensor
    else:
        raise ValueError("Dataset must be either numpy array, xarray or pandas dataframe.")


def get_data(x):
    """Extract the numerical values from ``x`` as a numpy array.

    Arguments
    ---------
    x : np.ndarray or pd.DataFrame or xr.DataArray

    Returns
    -------
    np.ndarray
        The numerical values of ``x`` as a numpy array.
    """
    if is_xarray(x):
        return x.data
    if is_dataframe(x):
        return x.values
    return np.asarray(x)


def is_labelled_cp(cp_tensor):
    """Check if a cp tensor is labelled or not

    Arguments
    ---------
    cp_tensor : tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and an iterable of factor matrices as second argument

    Returns
    -------
    bool
        Whether the factor matrices are labelled or not

    Raises
    ------
    TypeError
        If only some of the factor matrices are labelled (i.e. not none or all).
    """
    num_dataframes = 0
    for factor_matrix in cp_tensor[1]:
        if is_dataframe(factor_matrix):
            num_dataframes += 1

    if num_dataframes == 0:
        return False
    elif num_dataframes == len(cp_tensor[1]):
        return True
    else:
        raise TypeError(
            f"{num_dataframes} out of {len(cp_tensor[1])} factor matrices are labelled (are DataFrames)."
            + " All or none should be labelled."
        )


def is_labelled_tucker(tucker_tensor):
    """Check if a Tucker tensor is labelled or not

    Arguments
    ---------
    tucker_tensor : tuple
        TensorLy-style TuckerTensor object or tuple with the core array as the
        first argument and an iterable of factor matrices as second argument

    Returns
    -------
    bool
        Whether the factor matrices are labelled or not

    Raises
    ------
    TypeError
        If only some of the factor matrices are labelled (i.e. not none or all).
    """
    return is_labelled_cp(tucker_tensor)  # The weights are not considered for cp, neither is the core array for tucker


def is_labelled_dataset(x):
    """Returns True if the dataset is labelled (is a DataFrame or DataArray).

    This function is the same as writing ``is_dataframe(x) or is_xarray(x)``.

    Parameters
    ----------
    x
        Variable to check

    Returns
    -------
    bool
        Whether ``x`` is labelled or not.

    """
    # TOTEST: is_labeled_dataset
    return is_dataframe(x) or is_xarray(x)


def _extract_df_metadata(df, preserve_columns=True):
    values = df.values
    if preserve_columns:
        metadata = {"index": df.index, "columns": df.columns}
    else:
        metadata = {"index": df.index}
    return values, metadata


# TODO: Make public?
def _unlabel_cp_tensor(cp_tensor, optional, preserve_columns):
    if cp_tensor is None and optional:
        return None, None
    elif cp_tensor is None:
        raise TypeError("cp_tensor cannot be None")
    weights, factors = cp_tensor

    # Check that factor matrices are valid
    is_labelled = is_dataframe(factors[0])
    for factor in factors:
        if is_dataframe(factor) != is_labelled:
            raise ValueError("All factor matrices must either be labelled or not labelled.")

    if not is_labelled:
        return (weights, factors), None

    unlabelled_factors = []
    factor_metadata = []
    for factor in factors:
        factor, metadata = _extract_df_metadata(factor, preserve_columns=preserve_columns)
        unlabelled_factors.append(factor)
        factor_metadata.append(metadata)
    return (weights, unlabelled_factors), factor_metadata


def _relabel_cp_tensor(cp_tensor, factor_metadata, optional):
    if cp_tensor is None and optional:
        return

    if factor_metadata is None:
        return cp_tensor

    weights, factors = cp_tensor
    labelled_factors = []
    for factor, metadata in zip(factors, factor_metadata):
        labelled_factors.append(pd.DataFrame(factor, **metadata))
    return weights, labelled_factors


def _unlabel_factor_matrix(factor_matrix, optional, preserve_columns):
    if factor_matrix is None and optional:
        return None, None
    if not is_dataframe(factor_matrix):
        return factor_matrix, None
    return _extract_df_metadata(factor_matrix, preserve_columns=preserve_columns)


def _relabel_factor_matrix(factor_matrix, factor_metadata, optional):
    if factor_matrix is None and optional:
        return

    if factor_metadata is None:
        return factor_matrix

    return pd.DataFrame(factor_matrix, **factor_metadata)


def _unlabel_dataset(dataset, optional):
    if optional and dataset is None:
        return None, None, None
    elif dataset is None:  # Not optional and dataset is None
        raise TypeError("Dataset cannot be None")
    if is_xarray(dataset):
        np_dataset = dataset.values
        dataset_constructor = xr.DataArray
        dataset_metadata = {
            "name": dataset.name,
            "coords": dataset.coords,
            "dims": dataset.dims,
            "attrs": dataset.attrs,
        }
    elif is_dataframe(dataset):
        np_dataset = dataset.values
        dataset_constructor = pd.DataFrame
        dataset_metadata = {
            "index": dataset.index,
            "columns": dataset.columns,
        }
    else:
        np_dataset = dataset
        dataset_constructor = np.array
        dataset_metadata = {}
    return np_dataset, dataset_constructor, dataset_metadata


def _relabel_dataset(np_dataset, dataset_constructor, dataset_metadata, optional):
    if optional and np_dataset is None:
        return
    return dataset_constructor(np_dataset, **dataset_metadata)


def _handle_labelled_cp(cp_tensor_name, output_cp_tensor_index, optional=False, preserve_columns=True):
    def decorator(func):
        _check_is_argument(func, cp_tensor_name)

        @wraps(func)
        def func2(*args, **kwargs):
            bound_arguments = signature(func).bind(*args, **kwargs)

            cp_tensor = bound_arguments.arguments.get(cp_tensor_name, None)
            if cp_tensor is not None:
                validate_cp_tensor(cp_tensor)

            cp_tensor_unlabelled, cp_tensor_metadata = _unlabel_cp_tensor(
                cp_tensor, optional=optional, preserve_columns=preserve_columns
            )

            bound_arguments.arguments[cp_tensor_name] = cp_tensor_unlabelled
            out = func(*bound_arguments.args, **bound_arguments.kwargs)

            if output_cp_tensor_index is _SINGLETON:
                out = _relabel_cp_tensor(out, cp_tensor_metadata, optional=optional)
            elif output_cp_tensor_index is not None:
                out_cp_tensor = _relabel_cp_tensor(out[output_cp_tensor_index], cp_tensor_metadata, optional=optional)
                out = (
                    *out[:output_cp_tensor_index],
                    out_cp_tensor,
                    *out[output_cp_tensor_index + 1 :],
                )
            return out

        return func2

    return decorator


def _handle_labelled_dataset(dataset_name, output_dataset_index, optional=False):
    def decorator(func):
        _check_is_argument(func, dataset_name)

        @wraps(func)
        def func2(*args, **kwargs):
            bound_arguments = signature(func).bind(*args, **kwargs)

            if optional and dataset_name not in bound_arguments.arguments:
                return func(*bound_arguments.args, **bound_arguments.kwargs)
            dataset = bound_arguments.arguments[dataset_name]
            dataset_unlabelled, dataset_constructor, dataset_metadata = _unlabel_dataset(dataset, optional=optional)

            bound_arguments.arguments[dataset_name] = dataset_unlabelled
            out = func(*bound_arguments.args, **bound_arguments.kwargs)

            if output_dataset_index is _SINGLETON:
                out = _relabel_dataset(out, dataset_constructor, dataset_metadata, optional=optional)
            elif output_dataset_index is not None:
                out_dataset = _relabel_dataset(
                    out[output_dataset_index], dataset_constructor, dataset_metadata, optional=optional
                )
                out = (
                    *out[:output_dataset_index],
                    out_dataset,
                    *out[output_dataset_index + 1 :],
                )
            return out

        return func2

    return decorator


def _handle_labelled_factor_matrix(
    factor_matrix_name, output_factor_matrix_index, optional=False, preserve_columns=True
):
    def decorator(func):
        _check_is_argument(func, factor_matrix_name)

        @wraps(func)
        def func2(*args, **kwargs):
            bound_arguments = signature(func).bind(*args, **kwargs)

            factor_matrix = bound_arguments.arguments.get(factor_matrix_name, None)

            factor_matrix_unlabelled, factor_matrix_metadata = _unlabel_factor_matrix(
                factor_matrix, optional=optional, preserve_columns=preserve_columns
            )

            bound_arguments.arguments[factor_matrix_name] = factor_matrix_unlabelled
            out = func(*bound_arguments.args, **bound_arguments.kwargs)

            if output_factor_matrix_index is _SINGLETON:
                out = _relabel_factor_matrix(out, factor_matrix_metadata, optional=optional)
            elif output_factor_matrix_index is not None:
                out_factor_matrix = _relabel_factor_matrix(
                    out[output_factor_matrix_index],
                    factor_matrix_metadata,
                    optional=optional,
                )
                out = (
                    *out[:output_factor_matrix_index],
                    out_factor_matrix,
                    *out[output_factor_matrix_index + 1 :],
                )
            return out

        return func2

    return decorator

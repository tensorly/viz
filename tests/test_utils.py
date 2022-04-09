from typing import Type

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import component_vis._utils as utils


@pytest.mark.parametrize(
    "iterable",
    [
        "a",
        [],
        (1,),
        set(),
        {},
        np.zeros(2),
        xr.DataArray(np.zeros((2, 2, 2))),
        pd.DataFrame(np.zeros((2, 2))),
        range(4),
        (i for i in range(2)),
    ],
)
def test_is_iterable_is_true_for_iterables(iterable):
    assert utils.is_iterable(iterable)


@pytest.mark.parametrize("non_iterable", [1, 1.0, object(), lambda x: x, np.float64(1), np.int64(1)])
def test_is_iterable_is_false_for_noniterables(non_iterable):
    assert not utils.is_iterable(non_iterable)


def test_extract_singleton(rng):
    scalar = rng.normal()

    singleton_list = [scalar]
    assert utils.extract_singleton(singleton_list) == scalar

    singleton_array_1d = np.array([scalar])
    assert utils.extract_singleton(singleton_array_1d) == scalar

    singleton_array_2d = np.array([[scalar]])
    assert utils.extract_singleton(singleton_array_2d) == scalar

    singleton_dataframe = pd.DataFrame([scalar])
    assert utils.extract_singleton(singleton_dataframe) == scalar

    singleton_series = pd.Series([scalar])
    assert utils.extract_singleton(singleton_series) == scalar

    singleton_xarray_1d = xr.DataArray([scalar])
    assert utils.extract_singleton(singleton_xarray_1d) == scalar

    singleton_xarray_2d = xr.DataArray([[scalar]])
    assert utils.extract_singleton(singleton_xarray_2d) == scalar

    singleton_xarray_3d = xr.DataArray([[[scalar]]])
    assert utils.extract_singleton(singleton_xarray_3d) == scalar


def test_alias_mode_axis_passes_arguments_correctly():
    @utils._alias_mode_axis()
    def func(x, mode, axis=None):
        return mode

    assert func(1, 2) == 2
    assert func(1, axis=3) == 3
    with pytest.raises(TypeError):
        func(1, 2, 3)

    with pytest.raises(TypeError):
        func(1)

    with pytest.raises(TypeError):
        func(1, None, None)

    @utils._alias_mode_axis()
    def func(x, mode=2, axis=None):
        return mode

    assert func(1, 1) == 1
    assert func(1, axis=3) == 3
    with pytest.raises(TypeError):
        func(1, 1, 3)  # If mode is not equal to its default value, then axis cannot be set


def test_alias_mode_axis_fails_with_incompatible_functions():
    with pytest.raises(TypeError):

        @utils._alias_mode_axis()
        def func(x, mode, not_axis=None):
            return x

    with pytest.raises(TypeError):

        @utils._alias_mode_axis()
        def func(x, not_mode, axis=None):
            return x

    with pytest.raises(TypeError):

        @utils._alias_mode_axis()
        def func(x, not_mode, not_axis=None):
            return x


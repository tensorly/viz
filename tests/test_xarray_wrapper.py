import itertools

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from component_vis._module_utils import is_dataframe
from component_vis.data import simulated_random_cp_tensor
from component_vis.xarray_wrapper import (
    _SINGLETON,
    _check_is_argument,
    _handle_labelled_cp,
    _handle_none_weights_cp_tensor,
    _relabel_cp_tensor,
    _relabel_dataset,
    _unlabel_cp_tensor,
    _unlabel_dataset,
    get_data,
    label_cp_tensor,
    is_labelled_cp,
)


def test_is_labelled_cp(rng):
    rank = 3
    shape = (10, 20, 30)
    weights = rng.random(size=rank)
    factors = [rng.random(size=(length, rank)) for length in shape]
    assert not is_labelled_cp((weights, factors))

    labelled_factors = [pd.DataFrame(factor) for factor in factors]
    assert is_labelled_cp((weights, labelled_factors))

    partially_labelled_factors = [rng.random(size=(length, rank)) for length in shape]
    partially_labelled_factors[0] = pd.DataFrame(partially_labelled_factors[0])
    with pytest.raises(TypeError):
        is_labelled_cp((weights, partially_labelled_factors))


@pytest.mark.parametrize("is_labelled", [True, False])
def test_handle_labelled_cp_no_return_wrapping(is_labelled, seed):
    cp_tensor, X = simulated_random_cp_tensor((3, 2, 3), 2, labelled=is_labelled, seed=seed)

    @_handle_labelled_cp("cp_tensor", None, optional=False)
    def no_return_wrapping(a, cp_tensor):  # Two inputs to check that the correct argument is handled
        b = cp_tensor[1][0][0, 0]  # This will not work if factors are dataframes
        return b, cp_tensor

    number, unlabelled = no_return_wrapping(1, cp_tensor)
    for mode in range(3):
        assert not is_dataframe(unlabelled[1][mode])
        assert np.all(cp_tensor[1][mode] == unlabelled[1][mode])


@pytest.mark.parametrize("is_labelled", [True, False])
def test_handle_labelled_cp_singleton_return_wrapping(is_labelled, seed):
    cp_tensor, X = simulated_random_cp_tensor((3, 2, 3), 2, labelled=is_labelled, seed=seed)

    @_handle_labelled_cp("cp_tensor", _SINGLETON, optional=False)
    def singelton_wrapping(a, cp_tensor):  # Two inputs to check that the correct argument is handled
        b = cp_tensor[1][0][0, 0]
        return cp_tensor

    labelled = singelton_wrapping(1, cp_tensor)
    for mode in range(3):
        assert np.all(cp_tensor[1][mode] == labelled[1][mode])
        if is_labelled:
            assert is_dataframe(labelled[1][mode])
            assert np.all(cp_tensor[1][mode].index == labelled[1][mode].index)


@pytest.mark.parametrize("is_labelled", [True, False])
def test_handle_labelled_cp_return_wrapping(is_labelled, seed):
    cp_tensor, X = simulated_random_cp_tensor((3, 2, 3), 2, labelled=is_labelled, seed=seed)

    @_handle_labelled_cp("cp_tensor", 1, optional=False)
    def output_wrapping(a, cp_tensor):  # Two inputs to check that the correct argument is handled
        b = cp_tensor[1][0][0, 0]
        return b, cp_tensor

    number, labelled = output_wrapping(1, cp_tensor)
    for mode in range(3):
        assert np.all(cp_tensor[1][mode] == labelled[1][mode])
        if is_labelled:
            assert is_dataframe(labelled[1][mode])
            assert np.all(cp_tensor[1][mode].index == labelled[1][mode].index)


@pytest.mark.parametrize("is_labelled", [True, False])
def test_handle_optional_input(is_labelled, seed):
    cp_tensor, X = simulated_random_cp_tensor((3, 2, 3), 2, labelled=is_labelled, seed=seed)

    @_handle_labelled_cp("cp_tensor", _SINGLETON, optional=True)
    def output_wrapping(a, cp_tensor=None):  # Two inputs to check that the correct argument is handled
        return cp_tensor

    assert output_wrapping(1, None) is None

    labelled = output_wrapping(1, cp_tensor)
    for mode in range(3):
        assert np.all(cp_tensor[1][mode] == labelled[1][mode])
        if is_labelled:
            assert is_dataframe(labelled[1][mode])
            assert np.all(cp_tensor[1][mode].index == labelled[1][mode].index)


def test_label_cp_tensor_xarray(seed):
    cp_tensor, X = simulated_random_cp_tensor((3, 2, 3), 2, labelled=True, seed=seed)

    unlabelled_cp = (cp_tensor[0], [factor.values for factor in cp_tensor[1]])
    labelled_cp = label_cp_tensor(unlabelled_cp, X)
    for mode in range(3):
        assert isinstance(labelled_cp[1][mode], pd.DataFrame)
        assert np.all(labelled_cp[1][mode].values == cp_tensor[1][mode].values)
        assert np.all(labelled_cp[1][mode].index == cp_tensor[1][mode].index)


def test_label_cp_tensor_dataframe(seed):
    cp_tensor, X = simulated_random_cp_tensor((3, 2), 2, labelled=True, seed=seed)
    X_df = pd.DataFrame(X.values, index=X.coords["Mode 0"], columns=X.coords["Mode 1"])

    unlabelled_cp = (cp_tensor[0], [factor.values for factor in cp_tensor[1]])
    labelled_cp = label_cp_tensor(unlabelled_cp, X_df)
    for mode in range(2):
        assert isinstance(labelled_cp[1][mode], pd.DataFrame)
        assert np.all(labelled_cp[1][mode].values == cp_tensor[1][mode].values)
        assert np.all(labelled_cp[1][mode].index == cp_tensor[1][mode].index)


def test_label_cp_tensor_array(seed):
    size = (3, 2, 3)
    cp_tensor, X = simulated_random_cp_tensor(size, 2, labelled=False, seed=seed)

    labelled_cp = label_cp_tensor(cp_tensor, X)
    np.testing.assert_array_equal(cp_tensor[0], labelled_cp[0])
    for mode in range(len(size)):
        assert isinstance(labelled_cp[1][mode], np.ndarray)
        np.testing.assert_array_equal(labelled_cp[1][mode], cp_tensor[1][mode])


def test_label_cp_tensor_invalid_dataset_type(seed):
    size = (3, 2, 3)
    cp_tensor, X = simulated_random_cp_tensor(size, 2, labelled=False, seed=seed)

    # Invalid dataset type
    invalid_datasets = [None, 1, [[1, 2], [2, 4]], {0: 1, 3: 2}]

    for invalid_dataset in invalid_datasets:
        with pytest.raises(ValueError):
            label_cp_tensor(cp_tensor, invalid_dataset)

    # Dataset type is dataframe, but wrong number of factor matrices
    cp_tensor_2d, X = simulated_random_cp_tensor((3, 2), 2, labelled=True, seed=seed)
    X_df = pd.DataFrame(X.values, index=X.coords["Mode 0"], columns=X.coords["Mode 1"])

    with pytest.raises(ValueError):
        label_cp_tensor(cp_tensor, X_df)


def test_get_data(rng):
    X = rng.random(size=(10, 20))
    X_df = pd.DataFrame(X)
    X_xr = xr.DataArray(X)

    for data in [X, X_df, X_xr]:
        np.testing.assert_array_equal(get_data(data), X)
        assert isinstance(get_data(data), np.ndarray)


def test_check_is_argument():
    def dummy_function(argument1, argument2):
        return argument1 + argument2

    assert _check_is_argument(dummy_function, "argument1") is None

    with pytest.raises(ValueError):
        _check_is_argument(dummy_function, "argument3")


def test_unlabel_cp_tensor_fails_for_mix_of_labelled_and_unlabelled_factor_matrices(seed):
    shape = (10, 20, 30)
    cp_tensor, X = simulated_random_cp_tensor(shape, 3, labelled=True, seed=seed)

    invalid_cp_tensor = (cp_tensor[0], (cp_tensor[1][0], cp_tensor[1][1].values, cp_tensor[1][2]))
    with pytest.raises(ValueError):
        _unlabel_cp_tensor(invalid_cp_tensor, False)
    with pytest.raises(TypeError):
        _unlabel_cp_tensor(None, False)


def test_unlabel_none_dataset():
    assert _unlabel_dataset(None, True) == (None, None, None)

    with pytest.raises(TypeError):
        _unlabel_dataset(None, False)


def test_relabel_none_dataset():
    assert _relabel_dataset(None, None, None, True) is None


# TODO: test for ndarray and dataframe also?
def test_relabel_dataset_xarray(rng):
    shape = (10, 20, 30)
    X = rng.random(size=shape)

    DatasetType = xr.DataArray
    metadata = {
        "name": "A dataset",
        "dims": ("first mode", "second mode", "third mode"),
        "coords": {
            "first mode": list(range(shape[0])),
            "second mode": list(range(shape[1])),
            "third mode": list(range(shape[2])),
        },
    }
    labelled_dataset = _relabel_dataset(X, DatasetType=DatasetType, dataset_metadata=metadata, optional=False)
    assert isinstance(labelled_dataset, DatasetType)
    np.testing.assert_array_equal(labelled_dataset, X)
    assert labelled_dataset.name == metadata["name"]
    assert labelled_dataset.dims == metadata["dims"]

    for key in metadata["coords"]:
        np.testing.assert_array_equal(labelled_dataset.coords[key], metadata["coords"][key])


def test_unlabel_and_relabel_cp_tensor_inverse(seed):
    # First unlabel, so relabel
    shape = (10, 20, 30)
    cp_tensor, X = simulated_random_cp_tensor(shape, 3, labelled=True, seed=seed)
    unlabelled_cp_tensor, cp_tensor_metadata = _unlabel_cp_tensor(cp_tensor, False)
    relabelled_cp_tensor = _relabel_cp_tensor(unlabelled_cp_tensor, cp_tensor_metadata, False)

    assert is_labelled_cp(relabelled_cp_tensor)
    np.testing.assert_array_equal(cp_tensor[0], relabelled_cp_tensor[0])
    for factor_matrix, relabelled_factor_matrix in zip(cp_tensor[1], relabelled_cp_tensor[1]):
        pd.testing.assert_frame_equal(factor_matrix, relabelled_factor_matrix)

    # Unlabelling relabeled cp_tensor
    unlabelled_cp_tensor_again, cp_tensor_metadata_again = _unlabel_cp_tensor(relabelled_cp_tensor, False)
    assert not is_labelled_cp(unlabelled_cp_tensor_again)
    np.testing.assert_array_equal(unlabelled_cp_tensor[0], unlabelled_cp_tensor_again[0])
    for unlabelled_factor_matrix1, unlabelled_factor_matrix2 in zip(
        unlabelled_cp_tensor[1], unlabelled_cp_tensor_again[1]
    ):
        np.testing.assert_array_equal(unlabelled_factor_matrix1, unlabelled_factor_matrix2)
    assert cp_tensor_metadata == cp_tensor_metadata_again


def test_unlabel_and_relabel_dataset_inverse(seed):
    # xarray dataset
    shape = (10, 20, 30)
    cp_tensor, labelled_dataset = simulated_random_cp_tensor(shape, 3, labelled=True, seed=seed)

    unlabelled_dataset, DatasetType, dataset_metadata = _unlabel_dataset(labelled_dataset, False)
    relabelled_dataset = _relabel_dataset(
        unlabelled_dataset, DatasetType=DatasetType, dataset_metadata=dataset_metadata, optional=False
    )
    assert isinstance(relabelled_dataset, xr.DataArray)
    xr.testing.assert_identical(labelled_dataset, relabelled_dataset)

    unlabelled_dataset_again, DatasetType_again, dataset_metadata_again = _unlabel_dataset(labelled_dataset, False)
    assert isinstance(unlabelled_dataset_again, np.ndarray)
    np.testing.assert_array_equal(unlabelled_dataset_again, unlabelled_dataset)

    # dataframe dataset
    shape = (10, 20)
    cp_tensor, X = simulated_random_cp_tensor(shape, 3, labelled=True, seed=seed)
    labelled_dataset = pd.DataFrame(X.values, index=X.coords["Mode 0"], columns=X.coords["Mode 1"])

    unlabelled_dataset, DatasetType, dataset_metadata = _unlabel_dataset(labelled_dataset, False)
    relabelled_dataset = _relabel_dataset(
        unlabelled_dataset, DatasetType=DatasetType, dataset_metadata=dataset_metadata, optional=False
    )
    assert isinstance(relabelled_dataset, pd.DataFrame)
    pd.testing.assert_frame_equal(labelled_dataset, relabelled_dataset)

    unlabelled_dataset_again, DatasetType_again, dataset_metadata_again = _unlabel_dataset(labelled_dataset, False)
    assert isinstance(unlabelled_dataset_again, np.ndarray)
    np.testing.assert_array_equal(unlabelled_dataset_again, unlabelled_dataset)


@pytest.mark.parametrize("is_labelled,rank", list(itertools.product((True, False), [1, 2, 3, 4])))
def test_handle_none_weights_cp_tensor(is_labelled, rank, seed):
    @_handle_none_weights_cp_tensor("cp_tensor")
    def return_weights(cp_tensor):
        assert cp_tensor[0] is not None
        return cp_tensor[0]

    shape = (10, 20, 30)
    cp_tensor, X = simulated_random_cp_tensor(shape, rank, labelled=is_labelled, seed=seed)

    out_weights = return_weights(cp_tensor)
    np.testing.assert_array_equal(out_weights, cp_tensor[0])

    out_weights = return_weights((None, cp_tensor[1]))
    np.testing.assert_array_equal(out_weights, np.ones(rank))

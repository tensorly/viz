import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tlviz._module_utils import is_dataframe, is_xarray
from tlviz._xarray_wrapper import (
    _SINGLETON,
    _handle_labelled_cp,
    _handle_labelled_dataset,
    _handle_labelled_factor_matrix,
    _relabel_cp_tensor,
    _relabel_dataset,
    _relabel_factor_matrix,
    _unlabel_cp_tensor,
    _unlabel_dataset,
    _unlabel_factor_matrix,
    add_factor_metadata,
    get_data,
    is_labelled_cp,
    is_labelled_dataset,
    is_labelled_tucker,
    label_cp_tensor,
)
from tlviz.data import simulated_random_cp_tensor


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


def test_is_labelled_tucker(rng):
    rank = 3
    shape = (10, 20, 30)
    core = rng.random(size=[rank, rank, rank])
    factors = [rng.random(size=(length, rank)) for length in shape]
    assert not is_labelled_tucker((core, factors))

    labelled_factors = [pd.DataFrame(factor) for factor in factors]
    assert is_labelled_tucker((core, labelled_factors))

    partially_labelled_factors = [rng.random(size=(length, rank)) for length in shape]
    partially_labelled_factors[0] = pd.DataFrame(partially_labelled_factors[0])
    with pytest.raises(TypeError):
        is_labelled_tucker((core, partially_labelled_factors))


@pytest.mark.parametrize("is_labelled", [True, False])
def test_handle_labelled_cp_no_return_wrapping(is_labelled, seed):
    cp_tensor, X = simulated_random_cp_tensor((3, 2, 3), 2, labelled=is_labelled, seed=seed)

    @_handle_labelled_cp("cp_tensor", None, optional=False)
    def no_return_wrapping(a, cp_tensor):  # Two inputs to check that the correct argument is handled
        assert not is_labelled_cp(cp_tensor)
        return 1, cp_tensor

    number, unlabelled = no_return_wrapping(1, cp_tensor)
    for mode in range(3):
        assert not is_dataframe(unlabelled[1][mode])
        assert np.all(cp_tensor[1][mode] == unlabelled[1][mode])


@pytest.mark.parametrize("is_labelled", [True, False])
def test_handle_labelled_cp_singleton_return_wrapping(is_labelled, seed):
    cp_tensor, X = simulated_random_cp_tensor((3, 2, 3), 2, labelled=is_labelled, seed=seed)

    @_handle_labelled_cp("cp_tensor", _SINGLETON, optional=False)
    def singelton_wrapping(a, cp_tensor):  # Two inputs to check that the correct argument is handled
        assert not is_labelled_cp(cp_tensor)
        return cp_tensor

    labelled = singelton_wrapping(1, cp_tensor)
    for mode in range(3):
        assert is_dataframe(labelled[1][mode]) == is_labelled
        if is_labelled:
            pd.testing.assert_frame_equal(cp_tensor[1][mode], labelled[1][mode])
        else:
            np.testing.assert_array_equal(cp_tensor[1][mode], labelled[1][mode])


@pytest.mark.parametrize("is_labelled", [True, False])
def test_handle_labelled_cp_return_wrapping(is_labelled, seed):
    cp_tensor, X = simulated_random_cp_tensor((3, 2, 3), 2, labelled=is_labelled, seed=seed)

    @_handle_labelled_cp("cp_tensor", 1, optional=False)
    def output_wrapping(a, cp_tensor):  # Two inputs to check that the correct argument is handled
        assert not is_labelled_cp(cp_tensor)
        return 1, cp_tensor

    number, labelled = output_wrapping(1, cp_tensor)
    for mode in range(3):
        assert is_dataframe(labelled[1][mode]) == is_labelled
        if is_labelled:
            pd.testing.assert_frame_equal(cp_tensor[1][mode], labelled[1][mode])
        else:
            np.testing.assert_array_equal(cp_tensor[1][mode], labelled[1][mode])


@pytest.mark.parametrize("is_labelled", [True, False])
def test_handle_labelled_cp_optional_input(is_labelled, seed):
    cp_tensor, X = simulated_random_cp_tensor((3, 2, 3), 2, labelled=is_labelled, seed=seed)

    @_handle_labelled_cp("cp_tensor", _SINGLETON, optional=True)
    def output_wrapping(a, cp_tensor=None):  # Two inputs to check that the correct argument is handled
        return cp_tensor

    assert output_wrapping(1, None) is None

    labelled = output_wrapping(1, cp_tensor)
    for mode in range(3):
        assert is_dataframe(labelled[1][mode]) == is_labelled
        if is_labelled:
            pd.testing.assert_frame_equal(cp_tensor[1][mode], labelled[1][mode])
        else:
            np.testing.assert_array_equal(cp_tensor[1][mode], labelled[1][mode])


def test_label_cp_tensor_with_xarray(seed):
    cp_tensor, X = simulated_random_cp_tensor((3, 2, 3), 2, labelled=True, seed=seed)

    unlabelled_cp = (cp_tensor[0], [factor.values for factor in cp_tensor[1]])
    labelled_cp = label_cp_tensor(unlabelled_cp, X)
    for mode in range(3):
        assert isinstance(labelled_cp[1][mode], pd.DataFrame)
        pd.testing.assert_frame_equal(labelled_cp[1][mode], cp_tensor[1][mode])


def test_label_cp_tensor_warns_when_cp_tensor_and_dataset_both_are_labelled(seed):
    cp_tensor, X = simulated_random_cp_tensor((3, 2, 3), 2, labelled=True, seed=seed)
    with pytest.warns(UserWarning):
        label_cp_tensor(cp_tensor, X)


def test_label_cp_tensor_with_dataframe(seed):
    cp_tensor, X = simulated_random_cp_tensor((3, 2), 2, labelled=True, seed=seed)
    X_df = pd.DataFrame(X.values, index=X.coords["Mode 0"], columns=X.coords["Mode 1"])
    X_df.index.name = "Mode 0"
    X_df.columns.name = "Mode 1"

    unlabelled_cp = (cp_tensor[0], [factor.values for factor in cp_tensor[1]])
    labelled_cp = label_cp_tensor(unlabelled_cp, X_df)
    for mode in range(2):
        assert isinstance(labelled_cp[1][mode], pd.DataFrame)
        pd.testing.assert_frame_equal(labelled_cp[1][mode], cp_tensor[1][mode])


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


def test_unlabel_cp_tensor_fails_for_mix_of_labelled_and_unlabelled_factor_matrices(seed):
    shape = (10, 20, 30)
    cp_tensor, X = simulated_random_cp_tensor(shape, 3, labelled=True, seed=seed)

    invalid_cp_tensor = (cp_tensor[0], (cp_tensor[1][0], cp_tensor[1][1].values, cp_tensor[1][2]))
    with pytest.raises(ValueError):
        _unlabel_cp_tensor(invalid_cp_tensor, False, preserve_columns=False)
    with pytest.raises(TypeError):
        _unlabel_cp_tensor(None, False, preserve_columns=False)


def test_unlabel_none_dataset():
    assert _unlabel_dataset(None, True) == (None, None, None)

    with pytest.raises(TypeError):
        _unlabel_dataset(None, False)


def test_relabel_none_dataset():
    assert _relabel_dataset(None, None, None, True) is None


def test_relabel_dataset_xarray(rng):
    shape = (10, 20, 30)
    X = rng.random(size=shape)

    dataset_constructor = xr.DataArray
    metadata = {
        "name": "A dataset",
        "dims": ("first mode", "second mode", "third mode"),
        "coords": {
            "first mode": list(range(shape[0])),
            "second mode": list(range(shape[1])),
            "third mode": list(range(shape[2])),
        },
    }
    labelled_dataset = _relabel_dataset(
        X, dataset_constructor=dataset_constructor, dataset_metadata=metadata, optional=False
    )
    assert isinstance(labelled_dataset, dataset_constructor)
    np.testing.assert_array_equal(labelled_dataset, X)
    assert labelled_dataset.name == metadata["name"]
    assert labelled_dataset.dims == metadata["dims"]

    for key in metadata["coords"]:
        np.testing.assert_array_equal(labelled_dataset.coords[key], metadata["coords"][key])


def test_relabel_dataset_ndarray(rng):
    shape = (10, 20, 30)
    X = rng.random(size=shape)

    labelled_dataset = _relabel_dataset(X, dataset_constructor=np.array, dataset_metadata={}, optional=False)
    assert isinstance(labelled_dataset, np.ndarray)
    np.testing.assert_array_equal(labelled_dataset, X)


def test_relabel_dataset_dataframe(rng):
    shape = (10, 20)
    X = rng.random(size=shape)

    metadata = {
        "index": list(range(shape[0])),
        "columns": list(range(shape[1])),
    }
    labelled_dataset = _relabel_dataset(X, dataset_constructor=pd.DataFrame, dataset_metadata=metadata, optional=False)
    assert isinstance(labelled_dataset, pd.DataFrame)
    np.testing.assert_array_equal(labelled_dataset, X)
    assert all(labelled_dataset.index == metadata["index"])
    assert all(labelled_dataset.columns == metadata["columns"])


def test_unlabel_and_relabel_cp_tensor_inverse(seed):
    # First unlabel, so relabel
    shape = (10, 20, 30)
    cp_tensor, X = simulated_random_cp_tensor(shape, 3, labelled=True, seed=seed)
    unlabelled_cp_tensor, cp_tensor_metadata = _unlabel_cp_tensor(cp_tensor, False, preserve_columns=True)
    relabelled_cp_tensor = _relabel_cp_tensor(unlabelled_cp_tensor, cp_tensor_metadata, False)

    assert is_labelled_cp(relabelled_cp_tensor)
    np.testing.assert_array_equal(cp_tensor[0], relabelled_cp_tensor[0])
    for factor_matrix, relabelled_factor_matrix in zip(cp_tensor[1], relabelled_cp_tensor[1]):
        pd.testing.assert_frame_equal(factor_matrix, relabelled_factor_matrix)

    # Unlabelling relabeled cp_tensor
    unlabelled_cp_tensor_again, cp_tensor_metadata_again = _unlabel_cp_tensor(
        relabelled_cp_tensor, False, preserve_columns=True
    )
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

    unlabelled_dataset, dataset_constructor, dataset_metadata = _unlabel_dataset(labelled_dataset, False)
    relabelled_dataset = _relabel_dataset(
        unlabelled_dataset, dataset_constructor=dataset_constructor, dataset_metadata=dataset_metadata, optional=False
    )
    assert isinstance(relabelled_dataset, xr.DataArray)
    xr.testing.assert_identical(labelled_dataset, relabelled_dataset)

    unlabelled_dataset_again, dataset_constructor_again, dataset_metadata_again = _unlabel_dataset(
        labelled_dataset, False
    )
    assert isinstance(unlabelled_dataset_again, np.ndarray)
    np.testing.assert_array_equal(unlabelled_dataset_again, unlabelled_dataset)

    # dataframe dataset
    shape = (10, 20)
    cp_tensor, X = simulated_random_cp_tensor(shape, 3, labelled=True, seed=seed)
    labelled_dataset = pd.DataFrame(X.values, index=X.coords["Mode 0"], columns=X.coords["Mode 1"])

    unlabelled_dataset, dataset_constructor, dataset_metadata = _unlabel_dataset(labelled_dataset, False)
    relabelled_dataset = _relabel_dataset(
        unlabelled_dataset, dataset_constructor=dataset_constructor, dataset_metadata=dataset_metadata, optional=False
    )
    assert isinstance(relabelled_dataset, pd.DataFrame)
    pd.testing.assert_frame_equal(labelled_dataset, relabelled_dataset)

    unlabelled_dataset_again, dataset_constructor_again, dataset_metadata_again = _unlabel_dataset(
        labelled_dataset, False
    )
    assert isinstance(unlabelled_dataset_again, np.ndarray)
    np.testing.assert_array_equal(unlabelled_dataset_again, unlabelled_dataset)


def test_unlabel_and_relabel_factor_matrix_inverse(rng):
    factor_matrix = pd.DataFrame(rng.standard_normal((10, 3)), index=rng.permutation(10))
    unlabelled_factor_matrix, factor_matrix_metadata = _unlabel_factor_matrix(
        factor_matrix, False, preserve_columns=True
    )
    relabelled_factor_matrix = _relabel_factor_matrix(unlabelled_factor_matrix, factor_matrix_metadata, False)

    assert is_dataframe(relabelled_factor_matrix)
    pd.testing.assert_frame_equal(factor_matrix, relabelled_factor_matrix)

    # Unlabelling relabeled cp_tensor
    unlabelled_factor_matrix_again, factor_matrix_metadata_again = _unlabel_factor_matrix(
        relabelled_factor_matrix, False, preserve_columns=True
    )
    assert not is_dataframe(unlabelled_factor_matrix_again)
    np.testing.assert_array_equal(unlabelled_factor_matrix, unlabelled_factor_matrix_again)
    assert factor_matrix_metadata == factor_matrix_metadata_again


@pytest.mark.parametrize("is_labelled", [True, False])
def test_handle_labelled_factor_matrix_singleton_return_wrapping(is_labelled, rng):
    factor_matrix = rng.uniform(size=(10, 4))
    if is_labelled:
        factor_matrix = pd.DataFrame(factor_matrix)

    @_handle_labelled_factor_matrix("factor_matrix", _SINGLETON, optional=False)
    def singleton_wrapping(a, factor_matrix):  # Two inputs to check that the correct argument is handled
        assert not is_dataframe(factor_matrix)
        return factor_matrix

    labelled = singleton_wrapping(1, factor_matrix)
    assert is_dataframe(labelled) == is_labelled

    if is_labelled:
        pd.testing.assert_frame_equal(factor_matrix, labelled)
    else:
        np.testing.assert_array_equal(factor_matrix, labelled)


@pytest.mark.parametrize("is_labelled", [True, False])
def test_handle_labelled_factor_matrix_no_return_wrapping(is_labelled, rng):
    factor_matrix = rng.uniform(size=(10, 4))
    if is_labelled:
        factor_matrix = pd.DataFrame(factor_matrix)

    @_handle_labelled_factor_matrix("factor_matrix", None, optional=False)
    def no_return_wrapping(a, factor_matrix):  # Two inputs to check that the correct argument is handled
        assert not is_labelled_cp(factor_matrix)
        return 1, factor_matrix

    number, unlabelled = no_return_wrapping(1, factor_matrix)
    for mode in range(3):
        assert not is_dataframe(unlabelled[1][mode])
        np.testing.assert_array_equal(factor_matrix, unlabelled)


@pytest.mark.parametrize("is_labelled", [True, False])
def test_handle_labelled_factor_matrix_return_wrapping(is_labelled, rng):
    factor_matrix = rng.uniform(size=(10, 4))
    if is_labelled:
        factor_matrix = pd.DataFrame(factor_matrix)

    @_handle_labelled_factor_matrix("factor_matrix", 1, optional=False)
    def output_wrapping(a, factor_matrix):  # Two inputs to check that the correct argument is handled
        assert not is_dataframe(factor_matrix)
        b = factor_matrix[0, 0]
        return b, factor_matrix

    number, labelled = output_wrapping(1, factor_matrix)
    assert is_dataframe(labelled) == is_labelled

    if is_labelled:
        pd.testing.assert_frame_equal(factor_matrix, labelled)
    else:
        np.testing.assert_array_equal(factor_matrix, labelled)


@pytest.mark.parametrize("is_labelled", [True, False])
def test_handle_labelled_factor_matrix_optional_input(is_labelled, rng):
    factor_matrix = rng.uniform(size=(10, 4))
    if is_labelled:
        factor_matrix = pd.DataFrame(factor_matrix)

    @_handle_labelled_factor_matrix("factor_matrix", _SINGLETON, optional=True)
    def output_wrapping(a, factor_matrix=None):  # Two inputs to check that the correct argument is handled
        return factor_matrix

    assert output_wrapping(1, None) is None

    labelled = output_wrapping(1, factor_matrix)
    assert is_dataframe(factor_matrix) == is_labelled
    if is_labelled:
        pd.testing.assert_frame_equal(factor_matrix, labelled)
    else:
        np.testing.assert_array_equal(factor_matrix, labelled)


@pytest.mark.parametrize("is_labelled", [True, False])
def test_handle_labelled_dataset_singleton_return_wrapping(is_labelled, seed):

    dataset = simulated_random_cp_tensor((3, 2, 3), 2, labelled=is_labelled, seed=seed)[1]

    @_handle_labelled_dataset("dataset", _SINGLETON, optional=False)
    def singleton_wrapping(a, dataset):  # Two inputs to check that the correct argument is handled
        assert not is_labelled_dataset(dataset)
        return dataset

    labelled = singleton_wrapping(1, dataset)
    assert is_labelled_dataset(dataset) == is_labelled

    if is_xarray(dataset):
        xr.testing.assert_identical(dataset, labelled)
    if is_dataframe(dataset):
        pd.testing.assert_frame_equal(dataset, labelled)
    else:
        np.testing.assert_array_equal(dataset, labelled)


@pytest.mark.parametrize("is_labelled", [True, False])
def test_handle_labelled_dataset_no_return_wrapping(is_labelled, seed):
    dataset = simulated_random_cp_tensor((3, 2, 3), 2, labelled=is_labelled, seed=seed)[1]

    @_handle_labelled_dataset("dataset", None, optional=False)
    def no_return_wrapping(a, dataset):  # Two inputs to check that the correct argument is handled
        assert not is_labelled_dataset(dataset)
        return 1, dataset

    number, unlabelled = no_return_wrapping(1, dataset)
    assert not is_labelled_dataset(unlabelled)
    np.testing.assert_array_equal(dataset, unlabelled)


@pytest.mark.parametrize("is_labelled", [True, False])
def test_handle_labelled_dataset_return_wrapping(is_labelled, seed):
    dataset = simulated_random_cp_tensor((3, 2, 3), 2, labelled=is_labelled, seed=seed)[1]

    @_handle_labelled_dataset("dataset", 1, optional=False)
    def output_wrapping(a, dataset):  # Two inputs to check that the correct argument is handled
        assert not is_labelled_dataset(dataset)
        return 1, dataset

    number, labelled = output_wrapping(1, dataset)
    assert is_labelled_dataset(labelled) == is_labelled

    if is_dataframe(labelled):
        pd.testing.assert_frame_equal(dataset, labelled)
    if is_xarray(labelled):
        xr.testing.assert_identical(dataset, labelled)
    else:
        np.testing.assert_array_equal(dataset, labelled)


def test_add_factor_metadata_raises_for_unlabelled_cp(seed):
    cp_tensor_no_labels, dataset_no_labels = simulated_random_cp_tensor((3, 2, 3), 2, labelled=False, seed=seed)
    cp_tensor_labels, dataset_labels = simulated_random_cp_tensor((3, 2, 3), 2, labelled=True, seed=seed)
    with pytest.raises(ValueError):
        add_factor_metadata(cp_tensor_no_labels, dataset_labels)


def test_add_factor_metadata_raises_for_unlabelled_dataset(seed):
    cp_tensor_no_labels, dataset_no_labels = simulated_random_cp_tensor((3, 2, 3), 2, labelled=False, seed=seed)
    cp_tensor_labels, dataset_labels = simulated_random_cp_tensor((3, 2, 3), 2, labelled=True, seed=seed)
    with pytest.raises(ValueError):
        add_factor_metadata(cp_tensor_labels, dataset_no_labels)

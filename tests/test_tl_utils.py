import numpy as np
import pandas as pd
import pytest
import tensorly as tl
import tensorly.testing  # noqa
import xarray as xa

from tlvis import _tl_utils
from tlvis.data import simulated_random_cp_tensor

TENSORLY_BACKENDS = ["numpy", "pytorch"]  # TODO: Do this programmatically


@pytest.fixture
def pytorch_backend():
    backend = tl.get_backend()
    tl.set_backend("pytorch")
    yield
    tl.set_backend(backend)


def test_convert_to_numpy_cp_works_with_pandas(seed):
    cp_tensor = simulated_random_cp_tensor((10, 20, 30), 3, labelled=True, seed=seed)[0]
    weights, factors = _tl_utils.to_numpy_cp(cp_tensor)
    np.testing.assert_allclose(weights, cp_tensor[0])
    for labelled_factor_matrix, unlabelled_factor_matrix in zip(cp_tensor[1], factors):
        np.testing.assert_allclose(labelled_factor_matrix.values, unlabelled_factor_matrix)
        assert isinstance(unlabelled_factor_matrix, np.ndarray)


@pytest.mark.parametrize("backend", TENSORLY_BACKENDS)
def test_convert_to_numpy_cp_works_with_tensorly(seed, backend):
    tl.set_backend(backend)
    cp_tensor = tl.random.random_cp((10, 20, 30), 3, random_state=seed)
    cp_tensor = tl.tensor([1, 2, 3]), cp_tensor[1]

    weights, factors = _tl_utils.to_numpy_cp(cp_tensor)
    assert type(weights) == np.ndarray
    np.testing.assert_array_equal(weights, cp_tensor[0])
    for fm_org, fm in zip(cp_tensor[1], factors):
        assert type(fm) == np.ndarray
        np.testing.assert_array_equal(fm, tl.to_numpy(fm_org))


@pytest.mark.parametrize("backend", TENSORLY_BACKENDS)
def test_to_numpy_works_with_tensorly(rng, backend):
    tl.set_backend(backend)
    array = rng.uniform(size=(10, 20)).astype("float32")
    tensor = tl.tensor(array)
    array_again = _tl_utils.to_numpy(tensor)

    assert type(array_again) == np.ndarray
    np.testing.assert_array_equal(array_again, array)


def test_to_numpy_works_with_pandas(rng):
    array = rng.uniform(size=(10, 20))
    dataframe = pd.DataFrame(array)
    array_again = _tl_utils.to_numpy(dataframe)

    assert type(array_again) == np.ndarray
    np.testing.assert_array_equal(array_again, array)


@pytest.mark.parametrize("cast_labelled", [True, False])
def test_to_numpy_works_with_xarray(rng, cast_labelled):
    array = rng.uniform(size=(10, 20))
    xarray = xa.DataArray(array)
    array_again = _tl_utils.to_numpy(xarray, cast_labelled=cast_labelled)

    assert (type(array_again) == np.ndarray) == cast_labelled
    np.testing.assert_array_equal(array_again, array)


@pytest.mark.parametrize("none_ok", [True, False])
def test_is_tensorly_cp_returns_true_with_tensorly(seed, none_ok):
    # Create a random tensorly cp_tensor
    cp_tensor = tl.random.random_cp((10, 20, 30), 3, random_state=seed)
    # Check that is_tensorly_cp returns True
    assert _tl_utils.is_tensorly_cp(cp_tensor, none_ok=none_ok)
    # Convert cp_tensor into tuple
    # Check that is_tensorly_cp returns True
    assert _tl_utils.is_tensorly_cp(tuple(cp_tensor), none_ok=none_ok)


@pytest.mark.parametrize("none_ok", [True, False])
def test_is_tensorly_cp_returns_false_with_numpy(seed, none_ok, pytorch_backend):
    # Create a random cp_tensor

    cp_tensor = simulated_random_cp_tensor((10, 20, 30), 4, seed=seed)[0]
    assert not _tl_utils.is_tensorly_cp(cp_tensor, none_ok=none_ok)


@pytest.mark.parametrize("none_ok", [True, False])
def test_is_tensorly_cp_returns_false_with_labelled(seed, none_ok):
    # Create a random cp_tensor
    # Check that is_tensorly_cp returns False
    cp_tensor = simulated_random_cp_tensor((10, 20, 30), 4, seed=seed, labelled=True)[0]
    assert not _tl_utils.is_tensorly_cp(cp_tensor, none_ok=none_ok)


def test_is_tensorly_cp_handles_none_correctly(seed):
    # Check that is_tensorly_cp returns False if optional=True
    assert not _tl_utils.is_tensorly_cp(None, none_ok=True)

    # Check that is_tensorly_cp raises TypeError if optional=False
    with pytest.raises(TypeError):
        _tl_utils.is_tensorly_cp(None, none_ok=False)


@pytest.mark.parametrize("optional", [True, False])
def test_is_tensorly_cp_raises_with_mixed_types(seed, optional, pytorch_backend):
    # Create a random tensorly cp_tensor
    cp_tensor = tl.random.random_cp((10, 20, 30), 3, random_state=seed)
    # Create a shallow copy where weights are numpy array
    cp_tensor_numpy_weights = np.array([1, 1, 1]), cp_tensor[1]
    # Check that is_tensorly_cp raises TypeErrovvv
    with pytest.raises(TypeError):
        _tl_utils.is_tensorly_cp(cp_tensor_numpy_weights, none_ok=optional)
    # Create a shallow copy where one factor matrix is a numpy array and weights are tensorly
    cp_tensor_numpy_factor = cp_tensor[0], (tl.to_numpy(cp_tensor[1][0]), cp_tensor[1][1], cp_tensor[1][2])
    # Check that is_tensorly_cp raises TypeError
    with pytest.raises(TypeError):
        _tl_utils.is_tensorly_cp(cp_tensor_numpy_factor, none_ok=optional)

    # Create a shallow copy where one factor matrix is a numpy array and weights are numpy
    cp_tensor_numpy_factor_weights = (
        np.array([1, 1, 1]),
        (tl.to_numpy(cp_tensor[1][0]), cp_tensor[1][1], cp_tensor[1][2]),
    )

    # Check that is_tensorly_cp raises TypeError
    with pytest.raises(TypeError):
        _tl_utils.is_tensorly_cp(cp_tensor_numpy_factor_weights, none_ok=optional)
    # Create a shallow copy where all factor matrices are numpy arrays and weights are tensorly

    cp_tensor_numpy_factors = cp_tensor[0], [tl.to_numpy(fm) for fm in cp_tensor[1]]
    # Check that is_tensorly_cp raises TypeError
    with pytest.raises(TypeError):
        _tl_utils.is_tensorly_cp(cp_tensor_numpy_factors, none_ok=optional)


def test_handle_tensorly_backends_cp_singleton_return_wrapping(seed, pytorch_backend):
    cp_tensor = tl.random.random_cp((10, 15, 20), 3, random_state=seed)

    @_tl_utils._handle_tensorly_backends_cp("cp_tensor", _tl_utils._SINGLETON, optional=False)
    def singleton_wrapping(a, cp_tensor):  # Two inputs to check that the correct argument is handled
        assert not _tl_utils.is_tensorly_cp(cp_tensor)
        return cp_tensor

    out = singleton_wrapping(1, cp_tensor)
    assert _tl_utils.is_tensorly_cp(out)

    tl.testing.assert_array_equal(cp_tensor[0], out[0])
    for fm1, fm2 in zip(cp_tensor[1], out[1]):
        tl.testing.assert_array_equal(fm1, fm2)


def test_handle_tensorly_backends_cp_no_return_wrapping(seed, pytorch_backend):
    cp_tensor = tl.random.random_cp((10, 15, 20), 3, random_state=seed)

    @_tl_utils._handle_tensorly_backends_cp("cp_tensor", None, optional=False)
    def no_return_wrapping(a, cp_tensor):  # Two inputs to check that the correct argument is handled
        assert not _tl_utils.is_tensorly_cp(cp_tensor)
        return 1, cp_tensor

    out = no_return_wrapping(1, cp_tensor)[1]
    assert not _tl_utils.is_tensorly_cp(out)
    np.testing.assert_array_equal(tl.to_numpy(cp_tensor[0]), out[0])
    for fm1, fm2 in zip(cp_tensor[1], out[1]):
        np.testing.assert_array_equal(tl.to_numpy(fm1), fm2)


def test_handle_tensorly_backends_cp_return_wrapping(seed, pytorch_backend):
    cp_tensor = tl.random.random_cp((10, 15, 20), 3, random_state=seed)

    @_tl_utils._handle_tensorly_backends_cp("cp_tensor", 1, optional=False)
    def output_wrapping(a, cp_tensor):  # Two inputs to check that the correct argument is handled
        assert not _tl_utils.is_tensorly_cp(cp_tensor)
        return 1, cp_tensor

    out = output_wrapping(1, cp_tensor)[1]
    assert _tl_utils.is_tensorly_cp(out)
    tl.testing.assert_array_equal(cp_tensor[0], out[0])
    for fm1, fm2 in zip(cp_tensor[1], out[1]):
        tl.testing.assert_array_equal(fm1, fm2)


def test_handle_tensorly_backends_cp_optional_input(seed, pytorch_backend):
    cp_tensor = tl.random.random_cp((10, 15, 20), 3, random_state=seed)

    @_tl_utils._handle_tensorly_backends_cp("cp_tensor", _tl_utils._SINGLETON, optional=True)
    def output_wrapping(a, cp_tensor=None):  # Two inputs to check that the correct argument is handled
        return cp_tensor

    assert output_wrapping(1, None) is None

    out = output_wrapping(1, cp_tensor)
    assert _tl_utils.is_tensorly_cp(out)
    tl.testing.assert_array_equal(cp_tensor[0], out[0])
    for fm1, fm2 in zip(cp_tensor[1], out[1]):
        tl.testing.assert_array_equal(fm1, fm2)


def test_handle_tensorly_dataset_singleton_return_wrapping(rng, pytorch_backend):
    dataset = tl.tensor(rng.uniform(size=(20, 30, 40)))

    @_tl_utils._handle_tensorly_backends_dataset("dataset", _tl_utils._SINGLETON)
    def singleton_wrapping(a, dataset):  # Two inputs to check that the correct argument is handled
        assert not tl.is_tensor(dataset)
        return dataset

    out = singleton_wrapping(1, dataset)
    assert tl.is_tensor(dataset)

    tl.testing.assert_array_equal(dataset, out)


def test_handle_tensorly_backends_dataset_no_return_wrapping(rng, pytorch_backend):
    dataset = tl.tensor(rng.uniform(size=(20, 30, 40)))

    @_tl_utils._handle_tensorly_backends_dataset("dataset", None)
    def no_return_wrapping(a, dataset):  # Two inputs to check that the correct argument is handled
        assert not tl.is_tensor(dataset)
        return 1, dataset

    number, np_data = no_return_wrapping(1, dataset)
    assert not tl.is_tensor(np_data)
    np.testing.assert_array_equal(tl.to_numpy(dataset), np_data)


def test_handle_labelled_dataset_return_wrapping(rng, pytorch_backend):
    dataset = tl.tensor(rng.uniform(size=(20, 30, 40)))

    @_tl_utils._handle_tensorly_backends_dataset("dataset", 1)
    def output_wrapping(a, dataset):  # Two inputs to check that the correct argument is handled
        assert not tl.is_tensor(dataset)
        return 1, dataset

    number, out = output_wrapping(1, dataset)
    assert tl.is_tensor(out)
    tl.testing.assert_array_equal(dataset, out)

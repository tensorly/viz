from cProfile import label

import numpy as np
import pandas as pd
import pytest

from component_vis.data import simulated_random_cp_tensor
from component_vis.xarray_wrapper import (
    _SINGLETON,
    _handle_labelled_cp,
    is_dataframe,
    is_labelled_cp,
    label_cp_tensor,
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

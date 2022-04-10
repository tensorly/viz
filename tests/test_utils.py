from cProfile import label

import numpy as np
import pandas as pd
import pytest
import tensorly as tl
import xarray as xr

import component_vis.factor_tools as factor_tools
import component_vis.utils as utils
from component_vis._module_utils import is_xarray
from component_vis.data import simulated_random_cp_tensor


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
        def func1(x, mode, not_axis=None):
            return x

    with pytest.raises(TypeError):

        @utils._alias_mode_axis()
        def func2(x, not_mode, axis=None):
            return x

    with pytest.raises(TypeError):

        @utils._alias_mode_axis()
        def func3(x, not_mode, not_axis=None):
            return x


def test_cp_tensors_equals(rng):
    # Generate random decomposition
    A = rng.standard_normal((30, 3))
    B = rng.standard_normal((20, 3))
    C = rng.standard_normal((10, 3))
    w = rng.uniform(size=(3,))

    cp_tensor1 = (w, (A, B, C))
    cp_tensor2 = (w.copy(), (A.copy(), B.copy(), C.copy()))

    # Check that a decomposition is equal to its copy
    assert factor_tools.check_cp_tensors_equals(cp_tensor1, cp_tensor2)

    # Check that the decompositions are not equal if one of the factor matrices differ
    cp_tensor3 = (w.copy(), (A.copy(), B.copy(), rng.standard_normal((15, 3))))
    assert not factor_tools.check_cp_tensors_equals(cp_tensor1, cp_tensor3)

    # Check that two equivalent, but permuted decompositions are not equal
    permutation = [2, 1, 0]
    cp_tensor4 = (
        w[permutation],
        (A[:, permutation], B[:, permutation], C[:, permutation]),
    )
    assert not factor_tools.check_cp_tensors_equals(cp_tensor1, cp_tensor4)

    # Check that two equivalent decompositions with different weight distributions are not equal
    cp_tensor5 = factor_tools.distribute_weights_evenly(cp_tensor1)
    assert not factor_tools.check_cp_tensors_equals(cp_tensor1, cp_tensor5)

    # Check that two completely different CP tensors are not equal
    A2 = rng.standard_normal((30, 3))
    B2 = rng.standard_normal((20, 3))
    C2 = rng.standard_normal((10, 3))
    w2 = rng.uniform(size=(3,))

    cp_tensor6 = (w2, (A2, B2, C2))
    assert not factor_tools.check_cp_tensors_equals(cp_tensor1, cp_tensor6)


def test_cp_tensors_equivalent(rng):
    # Generate random decomposition
    A = rng.standard_normal((30, 3))
    B = rng.standard_normal((20, 3))
    C = rng.standard_normal((10, 3))
    w = rng.uniform(size=(3,))
    cp_tensor1 = (w, (A, B, C))

    # Check that a decomposition is equivalent to its copy
    cp_tensor2 = (w, (A.copy(), B.copy(), C.copy()))
    assert factor_tools.check_cp_tensors_equivalent(cp_tensor1, cp_tensor2)

    # Check that the decompositions are not equivalent if one of the factor matrices differ
    cp_tensor3 = (w, (A.copy(), B.copy(), rng.standard_normal((15, 3))))
    assert not factor_tools.check_cp_tensors_equivalent(cp_tensor1, cp_tensor3)

    # Check that two permuted decompositions are equivalent
    permutation = [2, 1, 0]
    cp_tensor4 = (
        w[permutation],
        (A[:, permutation], B[:, permutation], C[:, permutation]),
    )
    assert factor_tools.check_cp_tensors_equivalent(cp_tensor1, cp_tensor4)

    # Check that two decompositions with different weight distributions are equivalent
    cp_tensor5 = factor_tools.normalise_cp_tensor(cp_tensor1)
    assert factor_tools.check_cp_tensors_equivalent(cp_tensor1, cp_tensor5)

    cp_tensor6 = factor_tools.distribute_weights_evenly(cp_tensor1)
    assert factor_tools.check_cp_tensors_equivalent(cp_tensor1, cp_tensor6)

    cp_tensor7 = factor_tools.distribute_weights_in_one_mode(cp_tensor1, mode=1)
    assert factor_tools.check_cp_tensors_equivalent(cp_tensor1, cp_tensor7)

    # Check that two completely different cp decompositions are not equivalent
    A2 = rng.standard_normal((30, 3))
    B2 = rng.standard_normal((20, 3))
    C2 = rng.standard_normal((10, 3))
    w2 = rng.uniform(size=(3,))
    cp_tensor8 = (w2, (A2, B2, C2))
    assert not factor_tools.check_cp_tensors_equivalent(cp_tensor1, cp_tensor8)


@pytest.mark.parametrize("labelled", [True, False])
def test_cp_to_tensor(seed, labelled):
    cp_tensor, X = simulated_random_cp_tensor((10, 20, 30), 3, seed=seed, labelled=labelled)
    weights, factors = cp_tensor
    if labelled:
        factors = [fm.values for fm in factors]

    np.testing.assert_allclose(tl.cp_tensor.cp_to_tensor((weights, factors)), utils.cp_to_tensor(cp_tensor))
    assert labelled == is_xarray(utils.cp_to_tensor(cp_tensor))


@pytest.mark.parametrize("labelled", [True, False])
def test_tucker_to_tensor(rng, seed, labelled):
    cp_tensor, X = simulated_random_cp_tensor((10, 20, 30), 3, seed=seed, labelled=labelled)
    core_array = rng.standard_normal((3, 3, 3))
    weights, factors = cp_tensor
    tucker_tensor = core_array, factors

    if labelled:
        factors = [fm.values for fm in factors]

    np.testing.assert_allclose(
        tl.tucker_tensor.tucker_to_tensor((core_array, factors)), utils.tucker_to_tensor(tucker_tensor)
    )
    assert labelled == is_xarray(utils.tucker_to_tensor(tucker_tensor))

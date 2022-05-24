import numpy as np
import pandas as pd
import pytest
import xarray as xr

import tlviz._module_utils as cv_utils
from tlviz._xarray_wrapper import is_labelled_cp
from tlviz.data import simulated_random_cp_tensor
from tlviz.factor_tools import check_cp_tensor_equal
from tlviz.utils import cp_to_tensor


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
    assert cv_utils.is_iterable(iterable)


@pytest.mark.parametrize("non_iterable", [1, 1.0, object(), lambda x: x, np.float64(1), np.int64(1)])
def test_is_iterable_is_false_for_noniterables(non_iterable):
    assert not cv_utils.is_iterable(non_iterable)


def test_check_is_argument():
    def dummy_function(argument1, argument2):
        return argument1 + argument2

    assert cv_utils._check_is_argument(dummy_function, "argument1") is None

    with pytest.raises(ValueError):
        cv_utils._check_is_argument(dummy_function, "argument3")


@pytest.mark.parametrize("is_labelled", [True, False])
@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_handle_none_weights_cp_tensor(is_labelled, rank, seed):
    @cv_utils._handle_none_weights_cp_tensor("cp_tensor")
    def return_weights(cp_tensor):
        assert cp_tensor[0] is not None
        return cp_tensor[0]

    shape = (10, 20, 30)
    cp_tensor, X = simulated_random_cp_tensor(shape, rank, labelled=is_labelled, seed=seed)

    out_weights = return_weights(cp_tensor)
    np.testing.assert_array_equal(out_weights, cp_tensor[0])

    out_weights = return_weights((None, cp_tensor[1]))
    np.testing.assert_array_equal(out_weights, np.ones(rank))


@pytest.mark.parametrize("is_labelled", [True, False])
@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_handle_none_weights_cp_tensor_works_with_optional_argument(is_labelled, rank, seed):
    @cv_utils._handle_none_weights_cp_tensor("cp_tensor", optional=True)
    def return_weights(cp_tensor=None):
        return cp_tensor

    shape = (10, 20, 30)
    cp_tensor, X = simulated_random_cp_tensor(shape, rank, labelled=is_labelled, seed=seed)

    assert return_weights() is None
    assert return_weights(cp_tensor) == cp_tensor

    # Check that it fails if optional is false
    @cv_utils._handle_none_weights_cp_tensor("cp_tensor", optional=False)
    def return_weights(cp_tensor=None):
        return cp_tensor

    with pytest.raises(TypeError):
        return_weights()


@pytest.mark.parametrize("is_labelled", [True, False])
@pytest.mark.parametrize("rank", [1, 2, 3, 4])
@pytest.mark.parametrize("shape", [(10, 20), (10, 20, 30), (10, 20, 30, 40), (10, 20, 30, 40, 50)])
def test_validate_cp_tensor_accepts_valid_decompositions(is_labelled, rank, shape, seed):
    cp_tensor, X = simulated_random_cp_tensor(shape, rank, labelled=is_labelled, seed=seed)
    validated_cp_tensor = cv_utils.validate_cp_tensor(cp_tensor)

    assert check_cp_tensor_equal(cp_tensor, validated_cp_tensor)
    assert is_labelled_cp(cp_tensor) == is_labelled_cp(validated_cp_tensor)
    if is_labelled:
        xr.testing.assert_identical(cp_to_tensor(cp_tensor), cp_to_tensor(validated_cp_tensor))


@pytest.mark.parametrize("is_labelled", [True, False])
def test_validate_cp_tensor_accepts_none_weights(is_labelled, seed):
    shape = (10, 20, 30)
    rank = 3
    cp_tensor, X = simulated_random_cp_tensor(shape, rank, labelled=is_labelled, seed=seed)
    cp_tensor_none_weight = None, cp_tensor[1]
    validated_cp_tensor = cv_utils.validate_cp_tensor(cp_tensor_none_weight)

    assert check_cp_tensor_equal(cp_tensor_none_weight, validated_cp_tensor)
    assert is_labelled_cp(cp_tensor_none_weight) == is_labelled_cp(validated_cp_tensor)
    if is_labelled:
        xr.testing.assert_identical(cp_to_tensor(cp_tensor_none_weight), cp_to_tensor(validated_cp_tensor))


@pytest.mark.parametrize("num_columns", [(2, 2, 3), (2, 3, 2), (3, 2, 2, 3)])
@pytest.mark.parametrize("is_labelled", [True, False])
def test_validate_cp_tensor_raises_with_different_ranks(rng, is_labelled, num_columns):
    factor_matrices = []
    for i, rank in enumerate(num_columns):
        factor_matrices.append(rng.uniform(size=(10 * (i + 1), rank)))

    cp_tensor = np.ones(len(num_columns)), factor_matrices
    with pytest.raises(ValueError):
        cv_utils.validate_cp_tensor(cp_tensor)


@pytest.mark.parametrize("is_labelled", [True, False])
def test_validate_cp_tensor_raises_with_wrong_weight_shape(seed, is_labelled):
    shape = (10, 20, 30)
    rank = 3
    cp_tensor, X = simulated_random_cp_tensor(shape, rank, labelled=is_labelled, seed=seed)

    invalid_value_weights = [np.ones(2), np.ones(4), np.ones((3, 2))]
    for weights in invalid_value_weights:
        invalid_values_cp_tensor = weights, cp_tensor[1]
        with pytest.raises(ValueError):
            cv_utils.validate_cp_tensor(invalid_values_cp_tensor)

    invalid_type_weights = [[1, 2, 3], 1.0, 10, (2, 3)]
    for weights in invalid_type_weights:
        invalid_type_cp_tensor = weights, cp_tensor[1]
        with pytest.raises(TypeError):
            cv_utils.validate_cp_tensor(invalid_type_cp_tensor)


@pytest.mark.parametrize("factor_matrix_shape", [(3, 2, 1), (3, 3, 3)])
@pytest.mark.parametrize("is_labelled", [True, False])
def test_validate_cp_tensor_raises_with_wrong_factor_matrix_array_dim(rng, seed, is_labelled, factor_matrix_shape):
    shape = (10, 20, 30)
    rank = 3
    cp_tensor, X = simulated_random_cp_tensor(shape, rank, labelled=is_labelled, seed=seed)
    invalid_factor_matrix = rng.uniform(size=factor_matrix_shape)
    invalid_cp_tensor = cp_tensor[0], (invalid_factor_matrix, cp_tensor[1][1], cp_tensor[1][2])
    with pytest.raises(ValueError):
        cv_utils.validate_cp_tensor(invalid_cp_tensor)

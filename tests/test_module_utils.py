import numpy as np
import pandas as pd
import pytest
import xarray as xr

import component_vis._module_utils as cv_utils
from component_vis.data import simulated_random_cp_tensor


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

import numpy as np
import pytest

from component_vis import factor_tools
from component_vis.data import simulated_random_cp_tensor
from component_vis.outliers import (
    compute_leverage,
    compute_slabwise_sse,
    get_leverage_outlier_threshold,
    get_slabwise_sse_outlier_threshold,
)


def test_leverage_length(rng):
    N, R = 10, 3
    A = rng.standard_normal(size=(N, R))
    leverage = compute_leverage(A)
    assert len(leverage) == N


def test_leverage_known_matrix():
    A = np.array([[0, 0, -1], [0, 2, 1], [1, -2, 1], [2, -2, 0]])
    leverage = compute_leverage(A)
    supposed_leverage = [0.4, 0.9 + 1 / 30, 0.7 + 1 / 30, 0.9 + 1 / 30]
    assert np.allclose(leverage, supposed_leverage)


def test_leverages_sum_to_R(rng):
    N, R = 10, 3
    A = rng.standard_normal(size=(N, R))
    leverages = compute_leverage(A)
    assert sum(leverages) == pytest.approx(R)


def test_leverages_one_datapoint(rng):
    N, R = 10, 3
    A = rng.standard_normal(size=(N, R))
    datapoint = 3
    A[:3, 0] = 0
    A[3 + 1 :, 0] = 0

    leverages = compute_leverage(A)
    assert leverages[datapoint] == pytest.approx(1)


def test_slabwise_sse_sum_equals_total_sse(rng):
    tensor1 = rng.standard_normal(size=(10, 20, 30, 40))
    tensor2 = rng.standard_normal(size=(10, 20, 30, 40))
    slab_sse = compute_slabwise_sse(tensor1, tensor2, normalise=False)
    total_sse = np.sum((tensor1 - tensor2) ** 2)
    assert np.sum(slab_sse) == pytest.approx(total_sse)


def test_slab_leverage_threshold_invalid_method(rng):
    # Test that value error is raised when method is not among:
    # 'huber lower', 'huber higher', 'hw lower', 'hw higher', 'p-value', and 'hotelling'

    N, R = 10, 3
    A = rng.standard_normal(size=(N, R))
    leverage = compute_leverage(A)
    with pytest.raises(ValueError):
        get_leverage_outlier_threshold(leverage, method="invalid method")


def test_slab_leverage_threshold_p_value_one_component(rng):
    # Test that value error is raised when rank is 1 and method is p_value
    N, R = 10, 1
    A = rng.standard_normal(size=(N, R))
    leverage = compute_leverage(A)
    with pytest.raises(ValueError):
        get_leverage_outlier_threshold(leverage, method="p-value")


def test_slab_leverage_threshold_p_value_too_large_rank(rng):
    # Test that value error is raised when I=10, rank is 10 and method is p_value
    N, R = 10, 10
    A = rng.standard_normal(size=(N, R))
    leverage = compute_leverage(A)
    with pytest.raises(ValueError):
        get_leverage_outlier_threshold(leverage, method="p-value")


def test_slab_leverage_threshold_hotelling_too_large_rank(rng):
    # Create random 10 component CP tensor with I=11
    # Compute leverages in first mode
    # Test that value error is raised when method is p_value
    N, R = 11, 10
    A = rng.standard_normal(size=(N, R))
    leverage = compute_leverage(A)
    with pytest.raises(ValueError):
        get_leverage_outlier_threshold(leverage, method="hotelling")


@pytest.mark.parametrize("labelled", [True, False])
def test_slab_sse_threshold_invalid_method(seed, labelled):
    # Test that value error is raised when method is not 'two sigma' or 'p-value'
    X1 = simulated_random_cp_tensor((10, 11, 12), 4, seed=seed, labelled=labelled)[1]
    X2 = simulated_random_cp_tensor((10, 11, 12), 4, seed=seed, labelled=labelled)[1]
    sse = compute_slabwise_sse(X1, X2)
    with pytest.raises(ValueError):
        get_slabwise_sse_outlier_threshold(sse, method="invalid method")


@pytest.mark.parametrize("labelled", [True, False])
def test_check_cp_tensors_equivalent_same_weights_different_components(seed, labelled):

    cp_tensor1 = simulated_random_cp_tensor((10, 11, 12), 4, seed=seed, labelled=labelled)[0]
    cp_tensor2 = simulated_random_cp_tensor((10, 11, 12), 4, seed=seed + 1, labelled=labelled)[0]
    cp_tensor2 = factor_tools.permute_cp_tensor(cp_tensor2, reference_cp_tensor=cp_tensor1)
    cp_tensor1 = factor_tools.normalise_cp_tensor(cp_tensor1)
    cp_tensor2 = factor_tools.normalise_cp_tensor(cp_tensor2)

    cp_tensor3 = cp_tensor1[0], cp_tensor2[1]  # same weights as cp_tensor1, but different components
    assert not factor_tools.check_cp_tensors_equivalent(cp_tensor1, cp_tensor3)

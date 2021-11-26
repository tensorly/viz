import numpy as np
from component_vis.outliers import compute_leverage, compute_slabwise_sse
import pytest

# TODO: test with xarrays as well
def test_leverage(rng):
    # TODO: Find more properties of leverages
    # TODO: Manually check for small set
    N, R = 10, 3
    A = rng.standard_normal(size=(N, R))
    compute_leverage(A)
    pass


def test_leverage_known_matrix():
    A = np.array([[0, 0, -1], [0, 2, 1], [1, -2, 1], [2, -2, 0],])
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

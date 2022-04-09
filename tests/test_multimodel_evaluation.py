from random import shuffle

import numpy as np
import pytest
from pytest import approx

import component_vis.multimodel_evaluation as multimodel_evaluation
from component_vis.data import simulated_random_cp_tensor
from component_vis.factor_tools import check_cp_tensors_equals, construct_cp_tensor
from component_vis.postprocessing import permute_cp_tensor


@pytest.mark.parametrize("labelled", [True, False])
def test_similarity_evaluation(rng, labelled):
    rank = 5
    shape = (10, 20, 30)
    cp_tensors = [
        simulated_random_cp_tensor(shape, rank, labelled=labelled, seed=rng.randint(0, 1000))[0] for _ in range(5)
    ]
    similarity = multimodel_evaluation.similarity_evaluation(cp_tensors[0], cp_tensors)
    assert similarity[0] == approx(1)  # Check that it is equal to itself
    for sim in similarity[1:]:
        assert sim != approx(1)  # Check that it is not equal to others

    similarity = multimodel_evaluation.similarity_evaluation(
        cp_tensors[0], cp_tensors, similarity_metric=lambda x, y: 1
    )
    for sim in similarity:
        assert sim == 1

    cp_tensors = [permute_cp_tensor(cp_tensors[0], permutation=rng.permutation(rank)) for _ in range(5)]
    similarity = multimodel_evaluation.similarity_evaluation(cp_tensors[0], cp_tensors)
    for sim in similarity:
        assert sim == approx(1)


# TODO: Use the simulated dataset function to test both with and without labels
def test_get_model_with_lowest_error(rng):
    cp_tensors = [
        (None, (rng.standard_normal((10, 3)), rng.standard_normal((20, 3)), rng.standard_normal((30, 3)),),)
        for i in range(5)
    ]

    for i, cp_tensor in enumerate(cp_tensors):
        dense_tensor = construct_cp_tensor(cp_tensor)
        (selected_cp_tensor, selected_index, all_sse,) = multimodel_evaluation.get_model_with_lowest_error(
            cp_tensors, dense_tensor, return_index=True, return_errors=True
        )
        assert check_cp_tensors_equals(selected_cp_tensor, cp_tensor)
        assert i == selected_index
        assert all_sse[i] == approx(0)

    A = rng.standard_normal((30, 3))
    B = rng.standard_normal((20, 3))
    C = rng.standard_normal((10, 3))
    w = rng.uniform(size=(3,))
    cp_tensors = []
    for scale in range(1, 10):
        cp_tensors.append((w, (scale * A.copy(), B.copy(), C.copy())))

    X = construct_cp_tensor(cp_tensors[0])
    selected_cp_tensor, selected_index, errors = multimodel_evaluation.get_model_with_lowest_error(
        cp_tensors, X, return_errors=True, return_index=True
    )

    assert check_cp_tensors_equals(selected_cp_tensor, (w, (A, B, C)))
    for cp_tensor, error in zip(cp_tensors, errors):
        rel_sse = np.sum((X - construct_cp_tensor(cp_tensor)) ** 2) / np.sum(X ** 2)
        assert error == approx(rel_sse)

    out = multimodel_evaluation.get_model_with_lowest_error(cp_tensors, X, return_errors=False, return_index=False)
    assert len(out) == 2
    out = multimodel_evaluation.get_model_with_lowest_error(cp_tensors, X, return_errors=False, return_index=True)
    assert len(out) == 2
    assert out[1] == 0

    out = multimodel_evaluation.get_model_with_lowest_error(cp_tensors, X, return_errors=True, return_index=False)
    assert len(out) == 2
    np.testing.assert_allclose(out[1], errors)

    out = multimodel_evaluation.get_model_with_lowest_error(cp_tensors, X, return_errors=True, return_index=True)
    assert len(out) == 3
    assert out[1] == 0
    np.testing.assert_allclose(out[2], errors)


def test_sort_models_by_error(rng):
    A = rng.standard_normal((30, 3))
    B = rng.standard_normal((20, 3))
    C = rng.standard_normal((10, 3))
    w = rng.uniform(size=(3,))
    cp_tensors = []
    for scale in range(1, 10):
        cp_tensors.append((w, (scale * A.copy(), B.copy(), C.copy())))

    X = construct_cp_tensor(cp_tensors[0])

    shuffled_cp_tensors = cp_tensors.copy()
    shuffle(shuffled_cp_tensors)

    sorted_cp_tensors, errors = multimodel_evaluation.sort_models_by_error(shuffled_cp_tensors, X)
    assert cp_tensors == sorted_cp_tensors
    assert errors == sorted(errors)

    for cp_tensor, error in zip(sorted_cp_tensors, errors):
        rel_sse = np.sum((X - construct_cp_tensor(cp_tensor)) ** 2) / np.sum(X ** 2)
        assert error == rel_sse


def test_sort_models_by_error_with_identical_decompositions(rng):
    A = rng.standard_normal((30, 3))
    B = rng.standard_normal((20, 3))
    C = rng.standard_normal((10, 3))
    w = rng.uniform(size=(3,))
    cp_tensors = []
    for scale in range(1, 10):
        cp_tensors.append((w, (scale * A.copy(), B.copy(), C.copy())))

    cp_tensors *= 2  # Repeat list twice to have duplicate entries of the same decomposition
    X = construct_cp_tensor(cp_tensors[0])

    sorted_cp_tensors, errors = multimodel_evaluation.sort_models_by_error(cp_tensors, X)
    assert cp_tensors[:9] == sorted_cp_tensors[0::2]
    assert cp_tensors[9:] == sorted_cp_tensors[1::2]

    assert errors == sorted(errors)


from audioop import mul
from random import shuffle

import numpy as np
from pytest import approx

import component_vis.multimodel_evaluation as multimodel_evaluation
from component_vis.factor_tools import check_cp_tensors_equals, construct_cp_tensor


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
    assert len(out) == 2  # TODO CHECK: better things to test?
    out = multimodel_evaluation.get_model_with_lowest_error(cp_tensors, X, return_errors=False, return_index=True)
    assert len(out) == 2
    assert out[1] == 0

    out = multimodel_evaluation.get_model_with_lowest_error(cp_tensors, X, return_errors=True, return_index=False)
    assert len(out) == 2
    np.testing.assert_allclose(out[1], errors)  # TODO: CHECK does this make sense?

    # TODO NEXT: Test that index is only returned if return_index=True
    # TODO NEXT: Test that errors is only returned if return_errors=True


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


# TODO: Regression test, input list of cp tensors where two of the cp tensors are identical. This failed before.
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


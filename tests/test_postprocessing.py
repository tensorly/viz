import numpy as np
import pytest
from component_vis import postprocessing
from component_vis import factor_tools


def test_normalise_cp_tensor_normalises(rng):
    A = rng.standard_normal((10, 3))
    B = rng.standard_normal((20, 3))
    C = rng.standard_normal((30, 3))
    D = rng.standard_normal((40, 3))

    cp_tensor = (None, (A, B, C, D))
    normalised_cp_tensor = postprocessing.normalise_cp_tensor(cp_tensor)

    for factor_matrix in normalised_cp_tensor[1]:
        np.testing.assert_allclose(np.linalg.norm(factor_matrix, axis=0), 1)

    w = rng.standard_normal((3,))
    cp_tensor = (w, (A, B, C, D))
    normalised_cp_tensor = postprocessing.normalise_cp_tensor(cp_tensor)

    for factor_matrix in normalised_cp_tensor[1]:
        np.testing.assert_allclose(np.linalg.norm(factor_matrix, axis=0), 1)


def test_normalise_cp_tensor_does_not_change_tensor(rng):
    A = rng.standard_normal((10, 4))
    B = rng.standard_normal((11, 4))
    C = rng.standard_normal((12, 4))
    w = rng.standard_normal((4,))

    cp_tensor = (w, (A, B, C))
    dense_tensor = factor_tools.construct_cp_tensor(cp_tensor)

    normalised_cp_tensor = postprocessing.normalise_cp_tensor(cp_tensor)
    normalised_dense_tensor = factor_tools.construct_cp_tensor(normalised_cp_tensor)

    np.testing.assert_allclose(dense_tensor, normalised_dense_tensor)


def test_distribute_weights_in_one_mode_does_not_change_tensor(rng):
    A = rng.standard_normal(size=(10, 4))
    B = rng.standard_normal(size=(11, 4))
    C = rng.standard_normal(size=(12, 4))
    w = rng.uniform(size=(4,))

    cp_tensor = (w, (A, B, C))
    dense_tensor = factor_tools.construct_cp_tensor(cp_tensor)

    for mode in range(3):
        redistributed_cp_tensor = postprocessing.distribute_weights_in_one_mode(
            cp_tensor, mode
        )
        redistributed_dense_tensor = factor_tools.construct_cp_tensor(
            redistributed_cp_tensor
        )
        np.testing.assert_allclose(dense_tensor, redistributed_dense_tensor)


def test_distribute_weights_in_one_mode_distributes_correctly(rng):
    A = rng.standard_normal(size=(10, 4))
    B = rng.standard_normal(size=(11, 4))
    C = rng.standard_normal(size=(12, 4))
    w = rng.uniform(size=(4,))

    cp_tensor = (w, (A, B, C))
    dense_tensor = factor_tools.construct_cp_tensor(cp_tensor)

    for mode in range(3):
        new_weights, new_factors = postprocessing.distribute_weights_in_one_mode(
            cp_tensor, mode
        )
        np.testing.assert_allclose(new_weights, np.ones_like(new_weights))

        for i, new_factor_matrix in enumerate(new_factors):
            if i != mode:
                np.testing.assert_allclose(
                    np.linalg.norm(new_factor_matrix, axis=0),
                    np.ones_like(new_factor_matrix[0]),
                )


def test_distribute_weights_evenly_does_not_change_tensor(rng):
    A = rng.standard_normal((10, 4))
    B = rng.standard_normal((11, 4))
    C = rng.standard_normal((12, 4))
    w = rng.uniform(size=(4,))

    cp_tensor = (w, (A, B, C))
    dense_tensor = factor_tools.construct_cp_tensor(cp_tensor)

    redistributed_cp_tensor = postprocessing.distribute_weights_evenly(cp_tensor)
    redistributed_cp_tensor = factor_tools.construct_cp_tensor(redistributed_cp_tensor)

    np.testing.assert_allclose(dense_tensor, redistributed_cp_tensor)


def test_distribute_weights_evenly(rng):
    A = rng.standard_normal((10, 4))
    B = rng.standard_normal((11, 4))
    C = rng.standard_normal((12, 4))
    w = rng.uniform(size=(4,))

    cp_tensor = (w, (A, B, C))
    dense_tensor = factor_tools.construct_cp_tensor(cp_tensor)

    new_weights, new_factors = postprocessing.distribute_weights_evenly(cp_tensor)
    np.testing.assert_allclose(
        np.linalg.norm(new_factors[0], axis=0), np.linalg.norm(new_factors[1], axis=0)
    )
    np.testing.assert_allclose(
        np.linalg.norm(new_factors[0], axis=0), np.linalg.norm(new_factors[2], axis=0)
    )
    np.testing.assert_allclose(new_weights, np.ones_like(new_weights))


def test_resolve_cp_sign_indeterminacy_does_not_change_tensor(rng):
    A = rng.standard_normal((10, 4))
    B = rng.standard_normal((11, 4))
    C = rng.standard_normal((12, 4))
    w = rng.uniform(size=(4,))

    cp_tensor = (w, (A, B, C))
    dense_tensor = factor_tools.construct_cp_tensor(cp_tensor)

    sign_flipped_cp_tensor = postprocessing.resolve_cp_sign_indeterminacy(
        cp_tensor, dense_tensor
    )
    sign_flipped_dense_tensor = factor_tools.construct_cp_tensor(sign_flipped_cp_tensor)

    np.testing.assert_allclose(dense_tensor, sign_flipped_dense_tensor)


@pytest.mark.parametrize("method", ["transpose", "positive_coord"])
def test_resolve_cp_sign_indeterminacy_flips_negative_components_for_nonnegative_tensor(
    rng, method
):
    A = rng.uniform(size=(10, 4))
    B = rng.uniform(size=(11, 4))
    C = rng.uniform(size=(12, 4))
    w = rng.uniform(size=(4,))

    factor_matrices = [A, B, C]
    cp_tensor = (w, factor_matrices)
    dense_tensor = factor_tools.construct_cp_tensor(cp_tensor)

    for flip1 in range(3):
        for flip2 in range(3):
            if flip1 == flip2:
                continue
            signs = np.ones(3)
            signs[flip1] = -1
            signs[flip2] = -1
            wrong_flip_factor_matrices = [
                factor_matrix * sign
                for sign, factor_matrix in zip(signs, factor_matrices)
            ]
            wrong_flip_cp_tensor = (w, wrong_flip_factor_matrices)

            sign_flipped_cp_tensor = postprocessing.resolve_cp_sign_indeterminacy(
                wrong_flip_cp_tensor,
                dense_tensor,
                flip_mode=flip1,
                resolve_mode=flip2,
                method=method,
            )
            assert np.all(sign_flipped_cp_tensor[1][0] >= 0)
            assert np.all(sign_flipped_cp_tensor[1][1] >= 0)
            assert np.all(sign_flipped_cp_tensor[1][2] >= 0)
            assert np.all(
                wrong_flip_cp_tensor[1][flip1] <= 0
            ), "Did not setup test correctly"
            assert np.all(
                wrong_flip_cp_tensor[1][flip2] <= 0
            ), "Did not setup test correctly"


def test_permute_cp_tensor(rng):
    # Create a rank-3 CP tensor and a copy of it that is permuted. Test if postprocessing.permute_cp_tensor permutes it back
    # Modify the permuted copy so that one of its components consists of random vectors, check again that the permutation is correct
    # Create a copy of the rank-3 CP tensor, permute it and remove a component. Align the 3-component to the 2-component model and check that the first two components are the two components present in the two-component model
    A = rng.standard_normal((10, 4))
    B = rng.standard_normal((11, 4))
    C = rng.standard_normal((12, 4))
    w = rng.uniform(size=(4,))

    cp_tensor = (w, (A, B, C))

    permutation = [2, 1, 3, 0]
    cp_tensor_permuted = (
        w[permutation],
        (A[:, permutation], B[:, permutation], C[:, permutation]),
    )
    cp_tensor_permuted_back = postprocessing.permute_cp_tensor(
        cp_tensor_permuted, cp_tensor
    )
    assert factor_tools.check_cp_tensors_equals(cp_tensor_permuted_back, cp_tensor)

    # Check permutation comparing against fewer components
    permutation_2comp = [1, 3]
    cp_tensor_permuted2 = (
        w[permutation_2comp],
        (A[:, permutation_2comp], B[:, permutation_2comp], C[:, permutation_2comp]),
    )
    aligned_cp_tensor = postprocessing.permute_cp_tensor(cp_tensor, cp_tensor_permuted2)

    aligned_weights, aligned_factors = aligned_cp_tensor

    assert np.all(aligned_weights[:2] == cp_tensor_permuted2[0])
    for factor1, factor2 in zip(cp_tensor_permuted2[1], aligned_factors):
        np.testing.assert_allclose(factor1, factor2[:, :2])

    # Check that the permutation is equivalent to the unpermuted decomposition
    assert factor_tools.check_cp_tensors_equivalent(cp_tensor, aligned_cp_tensor)

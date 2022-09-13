import warnings

import numpy as np
import pandas as pd
import pytest

from tlviz import factor_tools, postprocessing
from tlviz.data import simulated_random_cp_tensor
from tlviz.factor_tools import check_cp_tensor_equal, distribute_weights
from tlviz.utils import cp_to_tensor


def test_resolve_cp_sign_indeterminacy_does_not_change_tensor(rng):
    A = rng.standard_normal((10, 4))
    B = rng.standard_normal((11, 4))
    C = rng.standard_normal((12, 4))
    w = rng.uniform(size=(4,))

    cp_tensor = (w, (A, B, C))
    dense_tensor = cp_to_tensor(cp_tensor)

    sign_flipped_cp_tensor = postprocessing.resolve_cp_sign_indeterminacy(cp_tensor, dense_tensor)
    sign_flipped_dense_tensor = cp_to_tensor(sign_flipped_cp_tensor)

    np.testing.assert_allclose(dense_tensor, sign_flipped_dense_tensor)


@pytest.mark.parametrize("method", ["transpose", "positive_coord"])
def test_resolve_cp_sign_indeterminacy_flips_negative_components_for_nonnegative_tensor(rng, method):
    A = rng.uniform(size=(10, 4))
    B = rng.uniform(size=(11, 4))
    C = rng.uniform(size=(12, 4))
    w = rng.uniform(size=(4,))

    factor_matrices = [A, B, C]
    cp_tensor = (w, factor_matrices)
    dense_tensor = cp_to_tensor(cp_tensor)

    for flip1 in range(3):
        for flip2 in range(3):
            if flip1 == flip2:
                continue
            signs = np.ones(3)
            signs[flip1] = -1
            signs[flip2] = -1
            wrong_flip_factor_matrices = [factor_matrix * sign for sign, factor_matrix in zip(signs, factor_matrices)]
            wrong_flip_cp_tensor = (w, wrong_flip_factor_matrices)

            sign_flipped_cp_tensor = postprocessing.resolve_cp_sign_indeterminacy(
                wrong_flip_cp_tensor,
                dense_tensor,
                resolve_mode=flip1,
                unresolved_mode=flip2,
                method=method,
            )
            assert np.all(sign_flipped_cp_tensor[1][0] >= 0)
            assert np.all(sign_flipped_cp_tensor[1][1] >= 0)
            assert np.all(sign_flipped_cp_tensor[1][2] >= 0)
            assert np.all(wrong_flip_cp_tensor[1][flip1] <= 0), "Did not setup test correctly"
            assert np.all(wrong_flip_cp_tensor[1][flip2] <= 0), "Did not setup test correctly"


@pytest.mark.parametrize("labelled", [True, False])
def test_resolve_cp_sign_indeterminacy_invalid_method(seed, labelled):
    cp_tensor, X = simulated_random_cp_tensor((10, 20, 30), 3, seed=seed, labelled=labelled)
    with pytest.raises(ValueError):
        postprocessing.resolve_cp_sign_indeterminacy(cp_tensor, X, method="invalid method")


@pytest.mark.parametrize("labelled", [True, False])
@pytest.mark.parametrize("unresolved_mode", [-5, 3])
def test_resolve_cp_sign_indeterminacy_unresolved_mode_outside_bounds(seed, labelled, unresolved_mode):
    cp_tensor, X = simulated_random_cp_tensor((10, 20, 30), 3, seed=seed, labelled=labelled)
    with pytest.raises(ValueError):
        postprocessing.resolve_cp_sign_indeterminacy(cp_tensor, X, unresolved_mode=unresolved_mode)


@pytest.mark.parametrize("labelled", [True, False])
@pytest.mark.parametrize("unresolved_mode,resolve_mode", [(2, 2), (2, (1, 2))])
def test_resolve_cp_sign_indeterminacy_unresolved_mode_unresolvable(seed, labelled, unresolved_mode, resolve_mode):
    cp_tensor, X = simulated_random_cp_tensor((10, 20, 30), 3, seed=seed, labelled=labelled)
    with pytest.raises(ValueError):
        postprocessing.resolve_cp_sign_indeterminacy(
            cp_tensor, X, unresolved_mode=unresolved_mode, resolve_mode=resolve_mode
        )


@pytest.mark.parametrize("labelled", [True, False])
def test_postprocess_warns_if_reference_is_given_and_permute_is_false(labelled, seed):
    cp_tensor1, X1 = simulated_random_cp_tensor((10, 20, 30), 3, seed=seed, labelled=labelled)
    cp_tensor2, X2 = simulated_random_cp_tensor((10, 20, 30), 3, seed=seed + 1, labelled=labelled)

    with pytest.warns(UserWarning):
        postprocessing.postprocess(cp_tensor1, reference_cp_tensor=cp_tensor2, permute=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        postprocessing.postprocess(cp_tensor1, reference_cp_tensor=cp_tensor2, permute=True)
        postprocessing.postprocess(cp_tensor1, reference_cp_tensor=cp_tensor2)


@pytest.mark.parametrize("labelled", [True, False])
def test_postprocess_only_permutes_when_it_should(labelled, seed):
    # Create a rank 4 CP tensor

    cp_tensor, X = simulated_random_cp_tensor((10, 20, 30), 4, seed=seed, labelled=labelled)

    # We want a decomposition where the weights are in increasing order (i.e. oposite of what we get from postprocess)
    # To do this, we first permute the decomposition so the weights are in decreasing order
    # Permute it without giving permutation. This should place the components in decreasing weight order and
    # give all component vectors unit norm. Then, we transform the weights by 1/x, thus making the smallest numbers the
    # largest and the largest numbers the smallest. This will place the components in increasing weight order.
    cp_tensor = postprocessing.postprocess(cp_tensor, permute=True, weight_behaviour="normalise")
    cp_tensor = 1 / cp_tensor[0], cp_tensor[1]  # Flip weights so they are in increasing weight order

    # Set permute=False and check that the decomposition is permuted by postprocess
    postprocessed_cp_tensor = postprocessing.postprocess(cp_tensor, weight_behaviour="ignore", permute=False)
    assert check_cp_tensor_equal(postprocessed_cp_tensor, cp_tensor)

    # Set permute=True and compare with manually permuted
    postprocessed_cp_tensor = postprocessing.postprocess(cp_tensor, weight_behaviour="ignore", permute=True)
    manually_permuted_cp_tensor = factor_tools.permute_cp_tensor(cp_tensor, permutation=[3, 2, 1, 0])
    assert check_cp_tensor_equal(postprocessed_cp_tensor, manually_permuted_cp_tensor)

    # Compare with reference CP tensor
    manually_permuted_cp_tensor = factor_tools.permute_cp_tensor(cp_tensor, permutation=[1, 0, 3, 2])
    postprocessed_cp_tensor = postprocessing.postprocess(
        cp_tensor, reference_cp_tensor=manually_permuted_cp_tensor, weight_behaviour="ignore"
    )
    assert check_cp_tensor_equal(postprocessed_cp_tensor, manually_permuted_cp_tensor)

    # Compare with reference CP tensor, setting permute=False
    with pytest.warns(UserWarning):
        postprocessed_cp_tensor = postprocessing.postprocess(
            cp_tensor, reference_cp_tensor=manually_permuted_cp_tensor, weight_behaviour="ignore", permute=False
        )
    assert check_cp_tensor_equal(postprocessed_cp_tensor, manually_permuted_cp_tensor)


@pytest.mark.parametrize("labelled", [True, False])
@pytest.mark.parametrize("weight_behaviour", ["ignore", "normalise", "evenly", "one_mode"])
def test_postprocess_distributes_weights_correctly(weight_behaviour, labelled, seed):
    cp_tensor, X = simulated_random_cp_tensor((10, 20, 30), 4, seed=seed, labelled=labelled)
    postprocessed_cp_tensor = postprocessing.postprocess(cp_tensor, weight_behaviour=weight_behaviour, permute=False)
    distributed_cp_tensor = factor_tools.distribute_weights(cp_tensor, weight_behaviour=weight_behaviour)

    assert check_cp_tensor_equal(postprocessed_cp_tensor, distributed_cp_tensor)

import numpy as np
import pytest

from component_vis import postprocessing
from component_vis.utils import cp_to_tensor


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
                wrong_flip_cp_tensor, dense_tensor, resolve_mode=flip1, unresolved_mode=flip2, method=method,
            )
            assert np.all(sign_flipped_cp_tensor[1][0] >= 0)
            assert np.all(sign_flipped_cp_tensor[1][1] >= 0)
            assert np.all(sign_flipped_cp_tensor[1][2] >= 0)
            assert np.all(wrong_flip_cp_tensor[1][flip1] <= 0), "Did not setup test correctly"
            assert np.all(wrong_flip_cp_tensor[1][flip2] <= 0), "Did not setup test correctly"

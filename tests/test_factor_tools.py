import numpy as np
import pandas as pd
import pytest
from pytest import approx

import component_vis.utils as utils
from component_vis import factor_tools
from component_vis._module_utils import is_dataframe
from component_vis.data import simulated_random_cp_tensor


def safe_permute(arr, permutation):
    if is_dataframe(arr):
        out = arr.copy()[permutation]
        out.columns = arr.columns[: len(permutation)]
        return out
    return arr[:, permutation]


@pytest.mark.parametrize("labelled", [True, False])
def test_factor_match_score(seed, labelled):
    cp_tensor = simulated_random_cp_tensor((20, 31), 3, seed=seed, labelled=labelled)[0]
    w, (A, B) = cp_tensor

    assert factor_tools.factor_match_score((None, (A, B)), (None, (A, B))) == approx(1)
    assert factor_tools.factor_match_score((None, (A, B)), (None, (2.0 * A, 0.5 * B)),) == approx(1)
    assert factor_tools.factor_match_score(
        (None, (A, B)), (None, (0.5 * A, 0.5 * B)), consider_weights=False
    ) == approx(1)
    assert factor_tools.factor_match_score((None, (A, B)), (None, (0.1 * A, 0.1 * B)), consider_weights=True) < 0.5


def test_factor_match_score_against_tensortoolbox_three_modes():
    A1 = np.array(
        [
            [-3.17188834, 0.12037674, -0.52441857],
            [0.77006048, -1.46279475, 1.15829157],
            [-1.18581816, 0.48408977, -1.09991184],
            [1.16631215, 1.04360917, 1.36915614],
        ]
    )

    A2 = np.array(
        [
            [-1.30262887, -0.4169056, -0.93705999],
            [0.6507304, -0.35287475, -1.3054718],
            [2.3391102, -1.62249815, -2.44364441],
            [0.03060683, 1.1635985, -0.96374088],
        ]
    )

    B1 = np.array(
        [
            [1.53796467, -0.63422707, 0.45490094],
            [-1.22797090, -1.00880186, -0.68148948],
            [0.99478163, 0.56521819, 1.77120808],
            [0.32201656, 0.38783583, -0.80020482],
            [0.57868839, -1.25683464, 0.45556118],
        ]
    )

    B2 = np.array(
        [
            [2.59245268, -2.27144787, -0.2187818],
            [-0.81465855, -0.35188988, -1.64408907],
            [-0.20094789, 1.10495721, 0.72345472],
            [0.43747334, 1.24612605, -0.94359963],
            [-0.32968311, 1.29134727, -0.73294861],
        ]
    )
    C1 = np.array(
        [
            [0.9931878, -0.95281053, -0.76501819],
            [0.36991261, -1.14379257, -1.25567542],
            [1.19267987, -0.34327929, 0.5883692],
            [-1.76280424, -0.04712039, 0.12633968],
            [-0.21826917, 0.4038183, -0.96475298],
            [0.45657526, 1.82831411, -0.49339521],
        ]
    )

    C2 = np.array(
        [
            [1.09267308, -1.05671575, 0.19105565],
            [-0.54742553, -0.18846809, 0.12229789],
            [0.75474572, 0.71870513, 0.73885282],
            [1.0305555, -0.02279294, 1.57613977],
            [-0.2184589, -0.48151848, 2.38306396],
            [1.85820322, -2.10472093, -0.02799594],
        ]
    )

    assert factor_tools.factor_match_score(
        (None, (A1, B1, C1)), (None, (A2, B2, C2)), consider_weights=False
    ) == pytest.approx(0.027696956568833, rel=1e-8, abs=1e-10)
    assert factor_tools.factor_match_score(
        (None, (A1, B1, C1)), (None, (A2, B2, C2)), consider_weights=True
    ) == pytest.approx(0.019965399034340, rel=1e-8, abs=1e-10)


def test_factor_match_score_against_tensortoolbox_four_modes():
    A1 = np.array(
        [
            [-1.25305822, 0.41480863, 0.67937122, -1.41094398],
            [-0.61751106, -1.07501907, 1.53552703, 1.73289436],
            [-1.10167628, 0.10916708, 0.15440049, 0.71191913],
            [-0.75397339, -0.53277329, -0.31800362, -1.39385172],
        ]
    )
    A2 = np.array(
        [
            [-0.91375871, 1.25690037, 2.02901623, 1.23133122],
            [0.21117567, -0.04855492, -0.33869225, 0.24722038],
            [-1.04180422, 1.43493527, -0.98951756, 1.30852824],
            [-0.88917527, -1.15392866, 1.38868925, 0.51818143],
        ]
    )

    B1 = np.array(
        [
            [0.55524731, 0.18657303, 0.56951858, -0.45689014],
            [-0.45996235, 0.46204667, -1.14346653, -1.17044652],
            [-0.43544214, 2.11642726, -2.53224555, 0.37626344],
            [0.62684971, -1.46490152, -0.72183586, -0.5028133],
            [1.07613603, 0.28500245, 0.59924702, -0.74853447],
        ]
    )

    B2 = np.array(
        [
            [0.11685685, 0.00346736, -1.14081191, 0.03363361],
            [0.06782851, -0.12775747, -0.54999814, -0.73082095],
            [1.99588507, 1.07894308, 0.58721622, 0.56918949],
            [1.40235112, 2.1899298, 0.62988048, -1.3539888],
            [1.8810379, 0.37515909, 1.25157966, 0.20167407],
        ]
    )
    C1 = np.array(
        [
            [-0.40255827, 1.66766184, 0.45864578, 0.74544796],
            [-0.74191489, 0.22902213, -0.03567836, 1.20862804],
            [-2.09713766, -1.62251038, 0.00501596, 0.76610473],
            [0.35926428, 0.55441067, 1.57895469, 0.14168695],
            [-0.02182733, -0.04726342, -0.53751347, 0.4963245],
            [1.26656772, -0.08609847, 1.49541366, 0.23812769],
        ]
    )
    C2 = np.array(
        [
            [-1.35248615, 0.28876374, -0.48831322, 1.30607284],
            [-1.01727243, 1.70950614, -0.6432707, 2.27651227],
            [-0.07297823, -0.0599663, -1.54827796, -0.90780148],
            [2.18938174, 0.6736972, -0.21463208, 0.84751937],
            [-0.80608285, 0.0103886, 1.39146945, -0.09375545],
            [-1.08319207, -0.65801888, -1.46349547, -0.40689636],
        ]
    )
    D1 = np.array(
        [
            [0.16535113, 0.53760458, 1.42702119, 0.82964388],
            [1.05943545, 0.62382289, 0.99240215, -0.61245248],
            [-0.03280214, -0.21058093, -1.23180961, -1.17422252],
            [-1.79503608, -0.61318651, 0.38374694, -0.70810762],
            [-0.96153239, 1.21283919, 0.3295456, -0.40590376],
            [-0.0443726, -0.62473263, -0.9840032, 0.66008108],
            [0.48558508, 1.7994932, 1.23246065, -0.02151854],
        ]
    )
    D2 = np.array(
        [
            [-1.74672431, 0.73026699, -0.25879295, -1.9992047],
            [-0.92546903, 0.13632921, -1.0511824, 2.06251471],
            [-0.36672464, -0.19523376, -1.0453565, -0.31365466],
            [-0.83074898, 0.50937367, -0.18895524, 0.57345223],
            [-1.43804184, 0.02874416, 0.02366292, 0.09117656],
            [-0.40793959, 0.6383231, -0.71778188, -0.62907123],
            [-0.22739631, 0.62798532, 0.87659409, 1.03663001],
        ]
    )

    assert factor_tools.factor_match_score(
        (None, (A1, B1, C1, D1)), (None, (A2, B2, C2, D2)), consider_weights=False
    ) == pytest.approx(0.015619878684950, rel=1e-8, abs=1e-10)
    assert factor_tools.factor_match_score(
        (None, (A1, B1, C1, D1)), (None, (A2, B2, C2, D2)), consider_weights=True
    ) == pytest.approx(0.007867447488467, rel=1e-8, abs=1e-10)


@pytest.mark.parametrize("labelled", [True, False])
def test_factor_match_score_permutation(rng, seed, labelled):
    num_components = 4
    cp_tensor = simulated_random_cp_tensor((20, 31), num_components, seed=seed, labelled=labelled)[0]
    w, (A, B) = cp_tensor
    permutation = rng.permutation(num_components)

    if labelled:
        A_permuted = A[permutation]
        B_permuted = B[permutation]
    else:
        A_permuted = A[:, permutation]
        B_permuted = B[:, permutation]

    fms, p = factor_tools.factor_match_score((None, (A_permuted, B_permuted)), (None, (A, B)), return_permutation=True)
    assert fms == approx(1)
    assert np.allclose(permutation, p)


def test_degeneracy_on_degenerate_components():
    A = np.array([[1, 1, 3], [-1, -1, 0], [1, 1, 2], [2, 2, 6]])
    B = np.array([[4, 4, 6], [-3, -3, 2], [0, 0, -8]])
    C = np.array([[1, -1, 3], [2, -2, 4], [-1, 1, 2], [2, -2, -3]])
    assert factor_tools.degeneracy_score((None, (A, B, C))) == pytest.approx(-1)


def test_degeneracy_on_orthogonal_components_arrays(rng):
    A = rng.standard_normal(size=(4, 4))
    A_orthogonal = np.linalg.qr(A)[0]
    B = rng.standard_normal(size=(4, 4))
    B_orthogonal = np.linalg.qr(B)[0]
    assert factor_tools.degeneracy_score((None, (A_orthogonal, B))) == pytest.approx(0)
    assert factor_tools.degeneracy_score((None, (A, B_orthogonal))) == pytest.approx(0)


def test_degeneracy_on_orthogonal_components_dataframes(rng):
    A = rng.standard_normal(size=(4, 4))
    A_orthogonal = np.linalg.qr(A)[0]
    B = rng.standard_normal(size=(4, 4))
    B_orthogonal = np.linalg.qr(B)[0]
    A = pd.DataFrame(A)
    A_orthogonal = pd.DataFrame(A_orthogonal)
    B = pd.DataFrame(B)
    B_orthogonal = pd.DataFrame(B_orthogonal)
    assert factor_tools.degeneracy_score((None, (A_orthogonal, B))) == pytest.approx(0)
    assert factor_tools.degeneracy_score((None, (A, B_orthogonal))) == pytest.approx(0)


def test_degeneracy_one_mode(rng):
    A = rng.standard_normal(size=(5, 3))
    min_crossproduct = (utils.normalise(A).T @ utils.normalise(A)).min()
    assert factor_tools.degeneracy_score((None, (A,))) == pytest.approx(min_crossproduct)


@pytest.mark.parametrize("labelled", [True, False])
def test_get_permutation(rng, labelled):
    A = rng.standard_normal((30, 5))
    if labelled:
        A = pd.DataFrame(A)

    permutation = [2, 1, 0, 4, 3]
    out_permutation = factor_tools.get_factor_matrix_permutation(A, safe_permute(A, permutation))
    assert out_permutation == permutation

    subset = [1, 2, 3]
    out_permutation = factor_tools.get_factor_matrix_permutation(safe_permute(A, subset), A)
    assert out_permutation == [1, 2, 3, 0, 4]

    out_permutation = factor_tools.get_factor_matrix_permutation(A, safe_permute(A, subset), allow_smaller_rank=True)
    assert out_permutation == [factor_tools.NO_COLUMN, 0, 1, 2, factor_tools.NO_COLUMN]

    subset = [0, 2]
    out_permutation = factor_tools.get_factor_matrix_permutation(safe_permute(A, subset), A)
    assert out_permutation == [0, 2, 1, 3, 4]

    out_permutation = factor_tools.get_factor_matrix_permutation(A, safe_permute(A, subset), allow_smaller_rank=True)
    assert out_permutation == [0, factor_tools.NO_COLUMN, 1, factor_tools.NO_COLUMN, factor_tools.NO_COLUMN]


@pytest.mark.parametrize("num_modes", [2, 3, 4])
@pytest.mark.parametrize("labelled", [True, False])
def test_normalise_cp_tensor_normalises(seed, num_modes, labelled):
    cp_tensor, X = simulated_random_cp_tensor([10 + i for i in range(num_modes)], 3, labelled=labelled, seed=seed)
    normalised_cp_tensor = factor_tools.normalise_cp_tensor(cp_tensor)

    for factor_matrix in normalised_cp_tensor[1]:
        np.testing.assert_allclose(np.linalg.norm(factor_matrix, axis=0), 1)

    cp_tensor = (None, cp_tensor[1])
    normalised_cp_tensor = factor_tools.normalise_cp_tensor(cp_tensor)

    for factor_matrix in normalised_cp_tensor[1]:
        np.testing.assert_allclose(np.linalg.norm(factor_matrix, axis=0), 1)


@pytest.mark.parametrize("num_modes", [2, 3, 4])
@pytest.mark.parametrize("labelled", [True, False])
def test_normalise_cp_tensor_works_with_zero_valued_column(seed, num_modes, labelled):
    cp_tensor, X = simulated_random_cp_tensor([10 + i for i in range(num_modes)], 3, labelled=labelled, seed=seed)
    factors = cp_tensor[1]
    if labelled:
        factors[0][0] = 0
    else:
        factors[0][:, 0] = 0

    normalised_cp_tensor = factor_tools.normalise_cp_tensor(cp_tensor)

    assert normalised_cp_tensor[0][0] == 0
    if labelled:
        np.testing.assert_allclose(normalised_cp_tensor[1][0][0], 0)
    else:
        np.testing.assert_allclose(normalised_cp_tensor[1][0][:, 0], 0)


@pytest.mark.parametrize("labelled", [True, False])
@pytest.mark.parametrize("num_modes", [2, 3, 4])
def test_normalise_cp_tensor_does_not_change_tensor(seed, num_modes, labelled):
    cp_tensor, X = simulated_random_cp_tensor([10 + i for i in range(num_modes)], 3, labelled=labelled, seed=seed)

    dense_tensor = utils.cp_to_tensor(cp_tensor)

    normalised_cp_tensor = factor_tools.normalise_cp_tensor(cp_tensor)
    normalised_dense_tensor = utils.cp_to_tensor(normalised_cp_tensor)

    np.testing.assert_allclose(dense_tensor, normalised_dense_tensor)


@pytest.mark.parametrize("labelled", [True, False])
@pytest.mark.parametrize("num_modes", [2, 3, 4])
def test_distribute_weights_in_one_mode_does_not_change_tensor(seed, num_modes, labelled):
    cp_tensor, X = simulated_random_cp_tensor([10 + i for i in range(num_modes)], 3, labelled=labelled, seed=seed)

    dense_tensor = utils.cp_to_tensor(cp_tensor)

    for mode in range(num_modes):
        redistributed_cp_tensor = factor_tools.distribute_weights_in_one_mode(cp_tensor, mode)
        redistributed_dense_tensor = utils.cp_to_tensor(redistributed_cp_tensor)
        np.testing.assert_allclose(dense_tensor, redistributed_dense_tensor)


@pytest.mark.parametrize("labelled", [True, False])
@pytest.mark.parametrize("num_modes", [2, 3, 4])
def test_distribute_weights_in_one_mode_distributes_correctly(seed, num_modes, labelled):
    cp_tensor, X = simulated_random_cp_tensor([10 + i for i in range(num_modes)], 3, labelled=labelled, seed=seed)

    for mode in range(num_modes):
        new_weights, new_factors = factor_tools.distribute_weights_in_one_mode(cp_tensor, mode)
        np.testing.assert_allclose(new_weights, np.ones_like(new_weights))

        for i, new_factor_matrix in enumerate(new_factors):
            if i != mode:
                np.testing.assert_allclose(
                    np.linalg.norm(new_factor_matrix, axis=0), 1,
                )


@pytest.mark.parametrize("labelled", [True, False])
@pytest.mark.parametrize("num_modes", [2, 3, 4])
def test_distribute_weights_evenly_does_not_change_tensor(seed, num_modes, labelled):
    cp_tensor, X = simulated_random_cp_tensor([10 + i for i in range(num_modes)], 3, labelled=labelled, seed=seed)

    dense_tensor = utils.cp_to_tensor(cp_tensor)

    redistributed_cp_tensor = factor_tools.distribute_weights_evenly(cp_tensor)
    redistributed_cp_tensor = utils.cp_to_tensor(redistributed_cp_tensor)

    np.testing.assert_allclose(dense_tensor, redistributed_cp_tensor)


@pytest.mark.parametrize("labelled", [True, False])
@pytest.mark.parametrize("num_modes", [2, 3, 4])
def test_distribute_weights_evenly(seed, num_modes, labelled):
    cp_tensor, X = simulated_random_cp_tensor([10 + i for i in range(num_modes)], 3, labelled=labelled, seed=seed)

    new_weights, new_factors = factor_tools.distribute_weights_evenly(cp_tensor)
    for i in range(1, num_modes):
        np.testing.assert_allclose(np.linalg.norm(new_factors[0], axis=0), np.linalg.norm(new_factors[i], axis=0))
    np.testing.assert_allclose(new_weights, np.ones_like(new_weights))


@pytest.mark.parametrize("labelled", [True, False])
def test_permute_cp_tensor(seed, labelled):
    """Test for permutation of CP tensors

    Create a rank-3 CP tensor and a copy of it that is permuted, then test if postprocessing.permute_cp_tensor
    permutes it back.

    Next, modify the permuted copy so that one of its components consists of random vectors, check again that
    the permutation is correct.

    Finally, create a copy of the rank-3 CP tensor, permute it and remove a component. Align the 3-component
    to the 2-component model and check that the first two components are the two components present in the
    two-component model
    """
    cp_tensor = simulated_random_cp_tensor((10, 11, 12), 4, seed=seed, labelled=labelled)[0]
    w, (A, B, C) = cp_tensor

    permutation = [2, 1, 3, 0]
    cp_tensor_permuted = (
        w[permutation],
        (safe_permute(A, permutation), safe_permute(B, permutation), safe_permute(C, permutation)),
    )
    cp_tensor_permuted_back = factor_tools.permute_cp_tensor(cp_tensor_permuted, reference_cp_tensor=cp_tensor)
    assert factor_tools.check_cp_tensors_equals(cp_tensor_permuted_back, cp_tensor)

    # Check permutation comparing against fewer components
    permutation_2comp = [1, 3]
    cp_tensor_permuted2 = (
        w[permutation_2comp],
        (safe_permute(A, permutation_2comp), safe_permute(B, permutation_2comp), safe_permute(C, permutation_2comp)),
    )
    aligned_cp_tensor = factor_tools.permute_cp_tensor(cp_tensor, reference_cp_tensor=cp_tensor_permuted2)

    aligned_weights, aligned_factors = aligned_cp_tensor

    assert np.all(aligned_weights[:2] == cp_tensor_permuted2[0])
    for factor1, factor2 in zip(cp_tensor_permuted2[1], aligned_factors):
        if labelled:
            factor2 = factor2[[0, 1]]
        else:
            factor2 = factor2[:, :2]
        np.testing.assert_allclose(factor1, factor2)

    # Check that the permutation is equivalent to the unpermuted decomposition
    assert factor_tools.check_cp_tensors_equivalent(cp_tensor, aligned_cp_tensor)


@pytest.mark.parametrize("labelled", [True, False])
def test_permute_cp_tensor_pads_factors_correctly(seed, labelled):
    cp_tensor = simulated_random_cp_tensor((10, 11, 12), 4, seed=seed, labelled=labelled)[0]
    w, (A, B, C) = cp_tensor

    permutation = [1, 2, 3]
    cp_tensor_permuted = (
        w[permutation],
        (safe_permute(A, permutation), safe_permute(B, permutation), safe_permute(C, permutation)),
    )
    aligned_cp_tensor = factor_tools.permute_cp_tensor(
        cp_tensor_permuted, reference_cp_tensor=cp_tensor, allow_smaller_rank=True
    )

    aligned_weights, aligned_factors = aligned_cp_tensor
    assert np.all(aligned_weights[1:] == w[1:])
    for factor1, factor2 in zip(cp_tensor_permuted[1], aligned_factors):
        if labelled:
            factor2 = factor2[[1, 2, 3]]
        else:
            factor2 = factor2[:, 1:]
        np.testing.assert_allclose(factor1, factor2)


@pytest.mark.parametrize("labelled", [True, False])
def test_get_cp_permutation(seed, labelled):
    cp_tensor = simulated_random_cp_tensor((10, 11, 12), 4, seed=seed, labelled=labelled)[0]
    w, (A, B, C) = cp_tensor

    permutation = [2, 1, 3, 0]
    cp_tensor_permuted = (
        w[permutation],
        (safe_permute(A, permutation), safe_permute(B, permutation), safe_permute(C, permutation)),
    )
    np.testing.assert_equal(factor_tools.get_cp_permutation(cp_tensor, cp_tensor_permuted), permutation)

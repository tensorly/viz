import numpy as np
import pandas as pd
import pytest
import xarray as xr

import component_vis.data
from component_vis.utils import cp_to_tensor


@pytest.mark.parametrize("is_labelled", [True, False])
def test_simulated_data_labels_correctly(is_labelled, seed):
    cp_tensor, X = component_vis.data.simulated_random_cp_tensor((5, 6, 7), 3, labelled=is_labelled, seed=seed)

    assert isinstance(X, xr.DataArray) == is_labelled
    for mode in range(3):
        assert isinstance(cp_tensor[1][mode], pd.DataFrame) == is_labelled


@pytest.mark.parametrize("noise_level", [0, 0.1, 0.5, 1])
def test_simulated_data_has_correct_noise(noise_level, seed):
    cp_tensor, X = component_vis.data.simulated_random_cp_tensor((5, 6, 7), 3, noise_level=noise_level, seed=seed)

    true_X = cp_to_tensor(cp_tensor)
    relative_error = np.linalg.norm(X - true_X) / np.linalg.norm(true_X)
    assert relative_error == pytest.approx(noise_level)


def test_simulated_data_is_seeded(seed):
    shape = (5, 6, 7)
    rank = 3
    cp_tensor1, X1 = component_vis.data.simulated_random_cp_tensor(
        shape, rank, noise_level=0.3, labelled=False, seed=seed
    )
    cp_tensor2, X2 = component_vis.data.simulated_random_cp_tensor(
        shape, rank, noise_level=0.3, labelled=False, seed=seed
    )

    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(cp_tensor1[0], cp_tensor2[0])

    for fm1, fm2 in zip(cp_tensor1[1], cp_tensor2[1]):
        np.testing.assert_array_equal(fm1, fm2)

    cp_tensor3, X3 = component_vis.data.simulated_random_cp_tensor(
        shape, rank, noise_level=0.3, labelled=False, seed=seed + 1
    )

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(X1, X3)

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(cp_tensor1[0], cp_tensor3[0])

    for fm1, fm3 in zip(cp_tensor1[1], cp_tensor3[1]):
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(fm1, fm3)

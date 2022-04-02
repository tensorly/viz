import numpy as np
import pandas as pd
import pytest
import xarray as xr

import component_vis.data
from component_vis.factor_tools import construct_cp_tensor


@pytest.mark.parametrize("is_labelled", [True, False])
def test_simulated_data_labels_correctly(is_labelled, seed):
    cp_tensor, X = component_vis.data.simulated_random_cp_tensor((5, 6, 7), 3, labelled=is_labelled, seed=seed)

    assert isinstance(X, xr.DataArray) == is_labelled
    for mode in range(3):
        assert isinstance(cp_tensor[1][mode], pd.DataFrame) == is_labelled


@pytest.mark.parametrize("noise_level", [0, 0.1, 0.5, 1])
def test_simulated_data_has_correct_noise(noise_level, seed):
    cp_tensor, X = component_vis.data.simulated_random_cp_tensor((5, 6, 7), 3, noise_level=noise_level, seed=seed)

    true_X = construct_cp_tensor(cp_tensor)
    relative_error = np.linalg.norm(X - true_X) / np.linalg.norm(true_X)
    assert relative_error == pytest.approx(noise_level)


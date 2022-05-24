import numpy as np
import pandas as pd
import pytest
import xarray as xr

import tlviz.data as data
from tlviz.utils import cp_to_tensor


@pytest.mark.parametrize("is_labelled", [True, False])
def test_simulated_data_labels_correctly(is_labelled, seed):
    cp_tensor, X = data.simulated_random_cp_tensor((5, 6, 7), 3, labelled=is_labelled, seed=seed)

    assert isinstance(X, xr.DataArray) == is_labelled
    for mode in range(3):
        assert isinstance(cp_tensor[1][mode], pd.DataFrame) == is_labelled


@pytest.mark.parametrize("noise_level", [0, 0.1, 0.5, 1])
def test_simulated_data_has_correct_noise(noise_level, seed):
    cp_tensor, X = data.simulated_random_cp_tensor((5, 6, 7), 3, noise_level=noise_level, seed=seed)

    true_X = cp_to_tensor(cp_tensor)
    relative_error = np.linalg.norm(X - true_X) / np.linalg.norm(true_X)
    assert relative_error == pytest.approx(noise_level)


def test_simulated_data_is_seeded(seed):
    shape = (5, 6, 7)
    rank = 3
    cp_tensor1, X1 = data.simulated_random_cp_tensor(shape, rank, noise_level=0.3, labelled=False, seed=seed)
    cp_tensor2, X2 = data.simulated_random_cp_tensor(shape, rank, noise_level=0.3, labelled=False, seed=seed)

    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(cp_tensor1[0], cp_tensor2[0])

    for fm1, fm2 in zip(cp_tensor1[1], cp_tensor2[1]):
        np.testing.assert_array_equal(fm1, fm2)

    cp_tensor3, X3 = data.simulated_random_cp_tensor(shape, rank, noise_level=0.3, labelled=False, seed=seed + 1)

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(X1, X3)

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(cp_tensor1[0], cp_tensor3[0])

    for fm1, fm3 in zip(cp_tensor1[1], cp_tensor3[1]):
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(fm1, fm3)


def test_load_aminoacids_prints_citation(capfd):
    data.load_aminoacids()
    assert (
        "Bro, R, PARAFAC: Tutorial and applications, Chemometrics and Intelligent Laboratory Systems, 1997, 38, 149-171"
        in capfd.readouterr()[0]
    )


def test_load_aminoacids_gives_correct_shape():
    aminoacids = data.load_aminoacids()
    assert aminoacids.shape == (5, 61, 201)


def test_load_oslo_city_bike_gives_correct_shape():
    bike_data = data.load_oslo_city_bike()
    num_stations = 270
    num_years = 2
    months_per_year = 12
    days_per_week = 7
    hours_per_day = 24
    assert bike_data.shape == (num_stations, num_years, months_per_year, days_per_week, hours_per_day)


def test_load_oslo_city_bike_is_nonnegative():
    bike_data = data.load_oslo_city_bike()

    assert np.all(bike_data.data >= 0)

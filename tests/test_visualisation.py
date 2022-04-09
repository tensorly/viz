from distutils import core

import matplotlib
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from component_vis import model_evaluation, visualisation
from component_vis.data import simulated_random_cp_tensor


@pytest.mark.parametrize("labelled", [True, False])
def test_histogram_of_residuals_works_labelled_and_unlabelled(seed, labelled):
    cp_tensor, X = simulated_random_cp_tensor((10, 20, 30), 3, noise_level=0.2, labelled=labelled, seed=seed)
    ax = visualisation.histogram_of_residuals(cp_tensor, X)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_histogram_of_resiudals_standardised_flag(seed):
    # With standardized = True
    cp_tensor, X = simulated_random_cp_tensor((10, 20, 30), 3, noise_level=0.2, seed=seed)
    ax = visualisation.histogram_of_residuals(cp_tensor, X, standardised=True)
    assert isinstance(ax, matplotlib.axes.Axes)
    assert ax.get_xlabel() == "Standardised residuals"

    # With standardized = False
    cp_tensor, X = simulated_random_cp_tensor((10, 20, 30), 3, noise_level=0.2, seed=seed)
    ax = visualisation.histogram_of_residuals(cp_tensor, X, standardised=False)
    assert isinstance(ax, matplotlib.axes.Axes)
    assert ax.get_xlabel() == "Residuals"
    # TODO: Check if it actually is standardised


@pytest.mark.parametrize("labelled", [True, False])
def test_residual_qq_works_labelled_and_unlabelled(seed, labelled):
    cp_tensor, X = simulated_random_cp_tensor((10, 20, 30), 3, noise_level=0.2, labelled=labelled, seed=seed)
    ax = visualisation.residual_qq(cp_tensor, X)
    assert isinstance(ax, matplotlib.axes.Axes)


@pytest.mark.parametrize("labelled", [True, False])
def test_outlier_plot_works_labelled_and_unlabelled(seed, labelled):
    cp_tensor, X = simulated_random_cp_tensor((10, 20, 30), 3, noise_level=0.2, labelled=labelled, seed=seed)
    ax = visualisation.outlier_plot(cp_tensor, X)
    assert isinstance(ax, matplotlib.axes.Axes)


@pytest.mark.parametrize("labelled", [True, False])
def test_core_element_plot_works_labelled_and_unlabelled(seed, labelled):
    cp_tensor, X = simulated_random_cp_tensor((10, 20, 30), 3, noise_level=0.2, labelled=labelled, seed=seed)
    ax = visualisation.core_element_plot(cp_tensor, X)
    assert isinstance(ax, matplotlib.axes.Axes)


@pytest.mark.parametrize("normalised", [True, False])
def test_core_element_plot_normalised_flag(seed, normalised):
    rank = 3
    cp_tensor, X = simulated_random_cp_tensor((10, 20, 30), rank, noise_level=0.2, seed=seed)
    # If not normalised
    ax = visualisation.core_element_plot(cp_tensor, X, normalised=normalised)
    title = ax.get_title()
    title_core_consistency = float(title.split(": ")[1])
    core_consistency = model_evaluation.core_consistency(cp_tensor, X, normalised=normalised)
    assert title_core_consistency == pytest.approx(core_consistency, abs=0.1)

    superdiag_x, superdiag_y = ax.lines[-2].get_data()
    offdiag_x, offdiag_y = ax.lines[-1].get_data()
    squared_core_error = np.sum(offdiag_y ** 2) + np.sum((superdiag_y - 1) ** 2)
    if normalised:
        denom = np.sum(superdiag_y ** 2) + np.sum(offdiag_y ** 2)
    else:
        denom = rank
    assert 100 - 100 * (squared_core_error / denom) == pytest.approx(core_consistency)


@pytest.mark.parametrize("labelled", [True, False])
def test_scree_plot_works_labelled_and_unlabelled(seed, labelled):
    shape = (10, 20, 30)
    cp_tensor, X = simulated_random_cp_tensor(shape, 3, labelled=labelled, seed=seed)
    cp_tensors = {seed: cp_tensor}
    for i in range(5):
        seed += 1
        cp_tensors[seed] = simulated_random_cp_tensor(shape, 3, labelled=labelled, seed=seed)[0]

    ax = visualisation.scree_plot(cp_tensors, X)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_scree_plot_works_with_given_errors(seed):
    shape = (10, 20, 30)
    rank = 4
    cp_tensor, X = simulated_random_cp_tensor(shape, rank, seed=seed)
    cp_tensors = {seed: cp_tensor}
    errors = {seed: model_evaluation.relative_sse(cp_tensor, X)}
    for i in range(5):
        seed += 1
        new_cp_tensor = simulated_random_cp_tensor(shape, rank, seed=seed)[0]
        cp_tensors[seed] = new_cp_tensor
        errors[seed] = model_evaluation.relative_sse(new_cp_tensor, X)

    ax = visualisation.scree_plot(cp_tensors, X, errors=errors)
    assert isinstance(ax, matplotlib.axes.Axes)
    line = ax.lines[-1]
    line_x, line_y = line.get_data()
    assert_array_equal(line_x, list(errors.keys()))
    assert_array_equal(line_y, list(errors.values()))

import matplotlib
import pytest

from component_vis import visualisation
from component_vis.data import simulated_random_cp_tensor


@pytest.mark.parametrize("labelled", [True, False])
def test_histogram_of_residuals(seed, labelled):
    cp_tensor, X = simulated_random_cp_tensor((10, 20, 30), 3, noise_level=0.2, labelled=labelled, seed=seed)
    ax = visualisation.histogram_of_residuals(cp_tensor, X)
    assert isinstance(ax, matplotlib.axes.Axes)


@pytest.mark.parametrize("labelled", [True, False])
def test_residual_qq(seed, labelled):
    cp_tensor, X = simulated_random_cp_tensor((10, 20, 30), 3, noise_level=0.2, labelled=labelled, seed=seed)
    ax = visualisation.residual_qq(cp_tensor, X)
    assert isinstance(ax, matplotlib.axes.Axes)


@pytest.mark.parametrize("labelled", [True, False])
def test_outlier_plot(seed, labelled):
    cp_tensor, X = simulated_random_cp_tensor((10, 20, 30), 3, noise_level=0.2, labelled=labelled, seed=seed)
    ax = visualisation.outlier_plot(cp_tensor, X)
    assert isinstance(ax, matplotlib.axes.Axes)


@pytest.mark.parametrize("labelled", [True, False])
def test_core_element_plot(seed, labelled):
    cp_tensor, X = simulated_random_cp_tensor((10, 20, 30), 3, noise_level=0.2, labelled=labelled, seed=seed)
    ax = visualisation.core_element_plot(cp_tensor, X)
    assert isinstance(ax, matplotlib.axes.Axes)


@pytest.mark.parametrize("labelled", [True, False])
def test_scree_plot_works_labelled_and_unlabelled(seed, labelled):
    shape = (10, 20, 30)
    cp_tensor, X = simulated_random_cp_tensor(shape, 3, labelled=labelled, seed=seed)
    cp_tensors = [cp_tensor]
    for i in range(5):
        seed += 1
        cp_tensors.append(simulated_random_cp_tensor(shape, 3, labelled=labelled, seed=seed)[0])

    ax = visualisation.scree_plot(cp_tensors, X)
    assert isinstance(ax, matplotlib.axes.Axes)


import matplotlib
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import component_vis.outliers as outliers
from component_vis import model_evaluation, visualisation
from component_vis.data import simulated_random_cp_tensor
from component_vis.utils import cp_to_tensor


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


def test_component_scatterplot_labels_dataframes_correctly(seed):
    shape = (10, 5, 15)
    cp_tensor, X = simulated_random_cp_tensor(shape, 3, labelled=True, seed=seed)

    cp_tensor[1][1].index = ["A", "B", "C", "D", "E"]
    ax = visualisation.component_scatterplot(cp_tensor, mode=1, x_component=0, y_component=1)

    coord_to_label = {(row[0], row[1]): index for index, row in cp_tensor[1][1].iterrows()}
    for text in ax.texts:
        pos = text.get_position()
        label = text.get_text()
        assert label == coord_to_label[pos]


def test_component_scatterplot_labels_arrays_correctly(seed):
    shape = (10, 5, 15)
    cp_tensor, X = simulated_random_cp_tensor(shape, 3, labelled=False, seed=seed)

    ax = visualisation.component_scatterplot(cp_tensor, mode=1, x_component=0, y_component=1)

    coord_to_label = {(row[0], row[1]): str(i) for i, row in enumerate(cp_tensor[1][1])}
    for text in ax.texts:
        pos = text.get_position()
        label = text.get_text()
        assert label == coord_to_label[pos]


def test_component_scatterplot_has_correct_length(seed):
    shape = (10, 5, 15)
    cp_tensor, X = simulated_random_cp_tensor(shape, 3, labelled=False, seed=seed)

    ax = visualisation.component_scatterplot(cp_tensor, mode=1, x_component=0, y_component=1)
    assert len(ax.texts) == 5


@pytest.mark.parametrize("labelled", [True, False])
def test_component_scatterplot_has_correct_point_locations(seed, labelled):
    shape = (10, 5, 15)
    cp_tensor, X = simulated_random_cp_tensor(shape, 3, labelled=labelled, seed=seed)

    ax = visualisation.component_scatterplot(cp_tensor, mode=1, x_component=0, y_component=1)

    if labelled:
        data = cp_tensor[1][1].values[:, :2]
    else:
        data = cp_tensor[1][1][:, :2]

    patch_collection = ax.collections[0]
    np.testing.assert_allclose(data, patch_collection.get_offsets())


@pytest.mark.parametrize("labelled", [True, False])
def test_outlier_plot_has_correct_scatter_point_locations(seed, labelled):
    shape = (10, 5, 15)
    rank = 3
    cp_tensor, X = simulated_random_cp_tensor(shape, rank, labelled=labelled, seed=seed)

    leverage = outliers.compute_leverage(cp_tensor[1][0])
    sse = outliers.compute_slabwise_sse(estimated=cp_to_tensor(cp_tensor), true=X, mode=0)
    if labelled:
        leverage = leverage.values.ravel()
        sse = sse.values.ravel()

    ax = visualisation.outlier_plot(cp_tensor, dataset=X, mode=0)
    x, y = ax.lines[-1].get_data()
    np.testing.assert_allclose(leverage, x)
    np.testing.assert_allclose(sse, y)


def test_outlier_plot_has_correct_text_labels_with_dataframe(seed):
    # Use the leverage and slab sse functions to compute leverage and slab sse values
    # Iterate over texts and check that the label is correct based on the leverage, slab sse and dataframe index
    shape = (10, 5, 15)
    rank = 3
    cp_tensor, X = simulated_random_cp_tensor(shape, rank, labelled=True, seed=seed)

    leverage = outliers.compute_leverage(cp_tensor[1][0]).values.ravel()
    sse = outliers.compute_slabwise_sse(estimated=cp_to_tensor(cp_tensor), true=X, mode=0).values.ravel()

    ax = visualisation.outlier_plot(cp_tensor, dataset=X, mode=0)
    for text, leverage_value, sse_value, index in zip(ax.texts, leverage, sse, cp_tensor[1][0].index):
        x, y = text.get_position()
        assert x == pytest.approx(leverage_value)
        assert y == pytest.approx(sse_value)
        assert str(index) == text.get_text()


def test_outlier_plot_has_correct_text_labels_with_array(seed):
    # Use the leverage and slab sse functions to compute leverage and slab sse values
    # Iterate over texts and check that the label is correct based on the leverage, slab sse and array index
    shape = (10, 5, 15)
    rank = 3
    cp_tensor, X = simulated_random_cp_tensor(shape, rank, labelled=False, seed=seed)

    leverage = outliers.compute_leverage(cp_tensor[1][0])
    sse = outliers.compute_slabwise_sse(estimated=cp_to_tensor(cp_tensor), true=X, mode=0)

    ax = visualisation.outlier_plot(cp_tensor, dataset=X, mode=0)
    for text, leverage_value, sse_value, index in zip(ax.texts, leverage, sse, range(len(cp_tensor[1][0]))):
        x, y = text.get_position()
        assert x == pytest.approx(leverage_value)
        assert y == pytest.approx(sse_value)
        assert index == int(text.get_text())


@pytest.mark.parametrize("labelled", [True, False])
@pytest.mark.parametrize("method", ["huber lower", "huber higher", "hw lower", "hw higher", "p-value", "hotelling"])
@pytest.mark.parametrize("p_value", [0.1, 0.5])
def test_outlier_plot_has_correct_leverage_threshold(seed, labelled, method, p_value):
    shape = (10, 5, 15)
    rank = 3
    cp_tensor, X = simulated_random_cp_tensor(shape, rank, labelled=labelled, seed=seed)

    leverage = outliers.compute_leverage(cp_tensor[1][0])
    if labelled:
        leverage = leverage.values.ravel()
    threshold = outliers.get_leverage_outlier_threshold(leverage, method=method, p_value=p_value)

    ax = visualisation.outlier_plot(cp_tensor, X, mode=0, leverage_rule_of_thumbs=method, p_value=p_value)
    assert all(x == pytest.approx(threshold) for x in ax.lines[-1].get_xdata())


@pytest.mark.parametrize("labelled", [True, False])
@pytest.mark.parametrize(
    "methods", [["huber lower", "p-value"], ["huber higher", "hw lower"], ["p-value", "hotelling"]]
)
@pytest.mark.parametrize("p_value", [0.1, 0.5])
def test_outlier_plot_has_multiple_correct_leverage_threshold_methods(seed, labelled, methods, p_value):
    shape = (10, 5, 15)
    rank = 3
    cp_tensor, X = simulated_random_cp_tensor(shape, rank, labelled=labelled, seed=seed)

    leverage = outliers.compute_leverage(cp_tensor[1][0])
    if labelled:
        leverage = leverage.values.ravel()

    ax = visualisation.outlier_plot(cp_tensor, X, mode=0, leverage_rule_of_thumbs=methods, p_value=p_value)
    for i, method in enumerate(methods[::-1]):
        threshold = outliers.get_leverage_outlier_threshold(leverage, method=method, p_value=p_value)
        assert all(x == pytest.approx(threshold) for x in ax.lines[-1 - i].get_xdata())


@pytest.mark.parametrize("labelled", [True, False])
@pytest.mark.parametrize("method", ["p-value", "hotelling"])
@pytest.mark.parametrize("p_values", [[0.1, 0.5], [0.1, 0.2, 0.3]])
def test_outlier_plot_has_multiple_correct_leverage_threshold_p_values(seed, labelled, method, p_values):
    shape = (10, 5, 15)
    rank = 3
    cp_tensor, X = simulated_random_cp_tensor(shape, rank, labelled=labelled, seed=seed)

    leverage = outliers.compute_leverage(cp_tensor[1][0])
    if labelled:
        leverage = leverage.values.ravel()

    ax = visualisation.outlier_plot(cp_tensor, X, mode=0, leverage_rule_of_thumbs=method, p_value=p_values)
    for i, p_value in enumerate(p_values[::-1]):
        threshold = outliers.get_leverage_outlier_threshold(leverage, method=method, p_value=p_value)
        assert all(x == pytest.approx(threshold) for x in ax.lines[-1 - i].get_xdata())


# TOTEST: test with multiple methods and p_values
@pytest.mark.parametrize("labelled", [True, False])
@pytest.mark.parametrize("method", ["p-value", "two sigma"])
@pytest.mark.parametrize("p_value", [0.1, 0.5])
def test_outlier_plot_has_correct_residual_thresholds(seed, labelled, method, p_value):
    shape = (10, 5, 15)
    rank = 3
    cp_tensor, X = simulated_random_cp_tensor(shape, rank, labelled=labelled, seed=seed)

    sse = outliers.compute_slabwise_sse(estimated=cp_to_tensor(cp_tensor), true=X, mode=0)
    threshold = outliers.get_slab_sse_outlier_threshold(sse, method, p_value=p_value)
    ax = visualisation.outlier_plot(cp_tensor, X, mode=0, p_value=p_value, residual_rule_of_thumbs=method)
    assert all(y == pytest.approx(threshold) for y in ax.lines[-1].get_ydata())


def test_components_plot_unlabelled(seed):
    # Postprocess CP tensor with each possible weight behaviour
    # Check that the plots have the same values as the postprocessed CP tensor
    assert False, "Test not written yet"


def test_components_plot_labelled(seed):
    # Postprocess CP tensor with each possible weight behaviour
    # Check that the plots have the same values as the postprocessed CP tensor
    # Check that the plots have logical x-labels
    assert False, "Test not written yet"

from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from matplotlib.lines import Line2D
from scipy import stats

from component_vis import model_evaluation

from . import model_evaluation, postprocessing
from .factor_tools import construct_cp_tensor, factor_match_score, get_permutation
from .model_evaluation import estimate_core_tensor
from .outliers import (
    _LEVERAGE_NAME,
    _SLABWISE_SSE_NAME,
    compute_leverage,
    compute_outlier_info,
    compute_slabwise_sse,
    get_leverage_outlier_threshold,
    get_slab_sse_outlier_threshold,
)

# TODO: Examples in docstrings

# TODO: Scree plot
# TODO: Test this function
def scree_plot(cp_tensors, dataset, errors=None, metric="fit", ax=None):
    if ax is None:
        ax = plt.gca()

    if isinstance(metric, str):
        metric = getattr(model_evaluation, metric)
    cp_tensors = dict(cp_tensors)

    if errors is None:
        # compute error using the metric function
        errors = {
            model: metric(cp_tensor, dataset) for model, cp_tensor in cp_tensors.items()
        }
    else:
        errors = dict(errors)

    ax.plot(errors.keys(), errors.values(), "-o")
    return ax


# TODO: Plotly version of these plots
def histogram_of_residuals(cp_tensor, dataset, ax=None, standardised=True, **kwargs):
    """Create a histogram of model residuals.

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument
    dataset : np.ndarray or xarray.DataArray
        Dataset to compare with
    ax : Matplotlib axes (Optional)
        Axes to plot the histogram in
    standardised : bool
        If true, then the residuals are divided by their standard deviation
    **kwargs
        Additional keyword arguments passed to the histogram function
    
    Returns
    -------
    ax : Matplotlib axes
    """
    # TODO: Handle if only one is labelled
    if hasattr(dataset, "data"):
        dataset = dataset.data
        cp_tensor = (cp_tensor[0], [fm.values for fm in cp_tensor[1]])

    estimated_dataset = construct_cp_tensor(cp_tensor)
    residuals = (estimated_dataset - dataset).ravel()

    if ax is None:
        ax = plt.gca()
    if standardised:
        residuals = residuals / np.std(residuals)
        ax.set_xlabel("Standardised residuals")
    else:
        ax.set_xlabel("Residuals")

    ax.hist(residuals, **kwargs)
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of residuals")

    return ax


def residual_qq(cp_tensor, dataset, ax=None, use_pingouin=False, **kwargs):
    """QQ-plot of the model residuals.

    By default, ``statsmodels`` is used to create the QQ-plot. However,
    if ``use_pingouin=True``, then we import the GPL-3 lisenced Pingouin
    library to create a more informative QQ-plot.

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument
    dataset : np.ndarray or xarray.DataArray
        Dataset to compare with
    ax : Matplotlib axes (Optional)
        Axes to plot the qq-plot in
    use_pingouin : bool
        If true, then the GPL-3 licensed ``pingouin``-library will be used
        for generating an enhanced QQ-plot (with error bars), at the cost
        of changing the license of component-vis into a GPL-license too.
    **kwargs
        Additional keyword arguments passed to the qq-plot function 
        (``statsmodels.api.qqplot`` or ``pingouin.qqplot``)
    
    Returns
    -------
    ax : Matplotlib axes
    """
    # TODO: Handle if only one is labelled
    if hasattr(dataset, "data"):
        dataset = dataset.data
        cp_tensor = (cp_tensor[0], [fm.values for fm in cp_tensor[1]])

    estimated_dataset = construct_cp_tensor(cp_tensor)
    residuals = (estimated_dataset - dataset).ravel()

    if ax is None:
        ax = plt.gca()

    if use_pingouin:
        from pingouin import qqplot

        warn(
            "GPL-3 Lisenced code is loaded, so this code also follows the GPL-3 license."
        )
        qqplot(residuals, ax=ax, **kwargs)
    else:
        sm.qqplot(residuals, ax=ax, **kwargs)

    ax.set_title("QQ-plot of residuals")
    return ax


# TODO: mode or axis?
def outlier_plot(
    cp_tensor,
    dataset,
    mode=0,
    leverage_rule_of_thumbs=None,
    residual_rule_of_thumbs=None,
    leverage_p_value=0.05,
    ax=None,
):
    # TODO: rule of thumbs
    weights, factor_matrices = cp_tensor
    factor_matrix = factor_matrices[mode]

    outlier_info = compute_outlier_info(cp_tensor, dataset, axis=mode)

    if ax is None:
        ax = plt.gca()

    ax.plot(
        outlier_info[f"{_LEVERAGE_NAME}"], outlier_info[f"{_SLABWISE_SSE_NAME}"], "o"
    )
    ax.set_xlabel("Leverage score")
    ax.set_ylabel("Slabwise SSE")
    if hasattr(factor_matrices[mode], "index") and factor_matrices[
        mode
    ].index.name not in {None, ""}:
        title = f"Outlier plot for {factor_matrices[mode].index.name}"
    else:
        title = f"Outlier plot for mode {mode}"
    ax.set_title(title)

    for x, y, s in zip(
        outlier_info[f"{_LEVERAGE_NAME}"],
        outlier_info[f"{_SLABWISE_SSE_NAME}"],
        outlier_info.index,
    ):
        ax.text(x, y, s)

    # Vertical lines for leverage based rule-of-thumb thresholds
    leverage_thresholds = {}
    if leverage_rule_of_thumbs is not None:
        if isinstance(leverage_rule_of_thumbs, str):
            leverage_rule_of_thumbs = [leverage_rule_of_thumbs]

        for leverage_rule_of_thumb in leverage_rule_of_thumbs:
            threshold = get_leverage_outlier_threshold(
                outlier_info[f"{_LEVERAGE_NAME}"],
                method=leverage_rule_of_thumb,
                p_value=leverage_p_value,
            )
            if leverage_rule_of_thumb == "p-value":
                leverage_rule_of_thumb = f"p-value: {leverage_p_value}"
            leverage_thresholds[leverage_rule_of_thumb] = threshold
    for key, value in leverage_thresholds.items():
        ax.axvline(value, label=key, **next(ax._get_lines.prop_cycler))

    residual_thresholds = {}
    if residual_rule_of_thumbs is not None:
        if isinstance(residual_rule_of_thumbs, str):
            residual_rule_of_thumbs = [residual_rule_of_thumbs]

        for residual_rule_of_thumb in residual_rule_of_thumbs:
            threshold = get_slab_sse_outlier_threshold(
                outlier_info[f"{_SLABWISE_SSE_NAME}"], method=residual_rule_of_thumb,
            )
            residual_thresholds[residual_rule_of_thumb] = threshold
    for key, value in residual_thresholds.items():
        ax.axhline(value, label=key, **next(ax._get_lines.prop_cycler))

    if len(leverage_thresholds) > 0 or len(residual_thresholds) > 0:
        ax.legend()
    return ax


def component_scatterplot(
    cp_tensor, mode, x_component=0, y_component=1, ax=None, **kwargs
):
    """Scatterplot of two columns in a factor matrix.

    Create a scatterplot with the columns of a factor matrix as feature-vectors.
    Note that since factor matrices are not orthogonal, the distances between points
    can be misleading. The lack of orthogonality means that distances and angles
    are "skewed", and two slabs with vastly different locations in the scatter plot
    can be very similar (in the case of collinear components). For more information
    about this phenomenon, see :cite:p:`kiers2000some` and example 8.3 in
    :cite:p:`smilde2005multi`.

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument
    mode : int
        Mode for the factor matrix whose columns are plotted
    x_component : int
        Component plotted on the x-axis
    y_component : int
        Component plotted on the y-axis
    ax : Matplotlib axes (Optional)
        Axes to plot the scatterplot in
    **kwargs
        Additional keyword arguments passed to ``ax.scatter``.
    
    Returns
    -------
    ax : Matplotlib axes
    """
    # TODO: handle weight
    # TODO: component scatterplot?
    # TODO: Handle dataframes
    if ax is None:
        ax = plt.gca()

    factor_matrix = cp_tensor[1][mode]
    relevant_factors = factor_matrix[:, [x_component, y_component]]

    ax.set_xlabel(f"Component {x_component}")
    ax.set_ylabel(f"Component {y_component}")
    ax.set_title("Component plot")
    ax.scatter(relevant_factors[:, 0], relevant_factors[:, 1], **kwargs)

    if hasattr(factor_matrix, "index"):
        index = factor_matrix.index
    else:
        index = np.arange(relevant_factors.shape[0])
    for x, y, s in zip(relevant_factors[:, 0], relevant_factors[:, 1], index):
        ax.text(x, y, s)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.text(
        xmax - 0.05 * (xmax - xmin),
        0,
        f"Component {x_component}",
        horizontalalignment="left",
    )

    return ax


# TODO: Core element heatmaps
def core_element_plot(cp_tensor, dataset, normalised=False, ax=None):
    """Scatter plot with the elements of the optimal core tensor for a given CP tensor.

    If the CP-model is appropriate for the data, then the core tensor should
    be superdiagonal, and all off-superdiagonal entries should be zero. This plot
    shows the core elements, sorted so the first R scatter-points correspond to the
    superdiagonal and the subsequent scatter-points correspond to off-diagonal entries
    in the optimal core tensor.

    Together with the scatter plot, there is a line-plot that indicate where the scatter-points
    should be if the CP-model perfectly describes the data.

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument
    dataset : np.ndarray or xarray.DataArray
        The dataset the CP tensor models.
    normalised : bool
        If true then the normalised core consistency will be estimated
        (see ``component_vis.model_evaluation.core_consistency``)
    ax : Matplotlib axes
        Axes to plot the core element plot within
    
    Returns:
    --------
    ax : Matplotlib axes
    """
    weights, factors = cp_tensor
    rank = weights.shape[0]

    A = factors[0].copy()
    if weights is not None:
        A *= weights
    factors = tuple((A, *factors[1:]))

    # Estimate core and compute core consistency
    core_tensor = estimate_core_tensor(factors, dataset)
    T = np.zeros([rank] * dataset.ndim)
    np.fill_diagonal(T, 1)
    if normalised:
        denom = np.linalg.norm(core_tensor, "fro") ** 2
    else:
        denom = rank

    core_consistency = 100 - 100 * np.sum((core_tensor - T) ** 2) / denom

    # Extract superdiagonal and offdiagonal elements
    core_elements = np.zeros_like(core_tensor.ravel())
    diagonal_mask = np.zeros([rank] * dataset.ndim, dtype=np.bool)
    np.fill_diagonal(diagonal_mask, 1)

    core_elements[:rank] = core_tensor[diagonal_mask]
    core_elements[rank:] = core_tensor[~diagonal_mask]

    # Plot core elements
    if ax is None:
        ax = plt.gca()

    x = np.arange(len(core_elements))
    y = np.zeros_like(x)
    y[:rank] = 1
    ax.plot(x, y, "-", label="Target")
    ax.plot(x[:rank], core_elements[:rank], "o", label="Superdiagonal")
    ax.plot(x[rank:], core_elements[rank:], "x", label="Off diagonal")
    ax.legend()
    ax.set_xlabel("Core element")
    ax.set_ylabel("Value")
    ax.set_title(f"Core consistency: {core_consistency:.1f}")

    return ax


def components_plot(
    cp_tensor, weight_behaviour="normalise", weight_mode=0, plot_kwargs=None
):
    """Plot the component vectors of a CP model.
    
    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument
    weight_behaviour : {'ignore', 'normalise', 'evenly', 'one_mode'}
        How to handle the component weights.
        
         * ignore - Do nothing, just plot the factor matrices
         * normalise - Plot all components after normalising them
         * evenly - Distribute the weight evenly across all modes
         * one_mode - Move all the weight into one factor matrix
    
    weight_mode : int
        The mode that the weight should be placed in (only used if ``weight_behaviour='one_mode'``)
    plot_kwargs : list of dictionaries
        List of same length as the number of modes. Each element is a kwargs-dict passed to
        the plot function for that mode.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray(dtype=matplotlib.axes.Axes)
    
    Examples
    --------
    Small example with a simulated CP tensor
    >>> from tensorly.random import random_cp
    ... from component_vis.visualisation import plot_components
    ... cp_tensor = random_cp(shape=(5,10,15), rank=3)
    ... plot_components(cp_tensor)

    Full example with PCA of a real stock dataset
    >>> import pandas as pd
    ... import numpy as np
    ... from plotly.data import stocks
    ... from component_vis.xarray_wrapper import label_cp_tensor
    ... from component_vis.visualisation import plot_components
    ... 
    ... # Load data and convert to xarray
    ... stocks = px.data.stocks().set_index("date").stack()
    ... stocks.index.names = ["Date", "Stock"]
    ... stocks = stocks.to_xarray()
    ... 
    ... # Compute PCA via SVD of centered data
    ... stocks -= stocks.mean(axis=0)
    ... U, s, Vh = np.linalg.svd(stocks, full_matrices=False)
    ... 
    ... # Extract two components and convert to cp_tensor
    ... num_components = 2
    ... cp_tensor = s[:num_components], (U[:, :num_components], Vh.T[:, :num_components])
    ... cp_tensor = component_vis.xarray_wrapper.label_cp_tensor(cp_tensor, stocks)
    ... 
    ... # Visualise the components with plot_components
    ... fig, axes = plot_components(cp_tensor, weight_behaviour="one_mode", weight_mode=1, plot_kwargs=[{}, {'marker': 'o', 'linewidth': 0}])
    """
    if weight_behaviour == "ignore":
        weights, factor_matrices = cp_tensor
    elif weight_behaviour == "normalise":
        weights, factor_matrices = postprocessing.normalise_cp_tensor(cp_tensor)
    elif weight_behaviour == "evenly":
        weights, factor_matrices = postprocessing.distribute_weights_evenly(cp_tensor)
    elif weight_behaviour == "one_mode":
        weights, factor_matrices = postprocessing.distribute_weights_in_one_mode(
            cp_tensor, weight_mode
        )
    else:
        raise ValueError(
            "weight_behaviour must be either 'ignore', 'normalise' or 'one_mode'"
        )

    num_components = len(weights.reshape(-1))
    num_modes = len(factor_matrices)

    if plot_kwargs is None:
        plot_kwargs = [{}] * num_modes

    fig, axes = plt.subplots(1, num_modes, figsize=(16, 9 / num_modes))

    for mode, factor_matrix in enumerate(factor_matrices):
        if hasattr(factor_matrix, "plot"):
            factor_matrix.plot(ax=axes[mode], **plot_kwargs[mode])
        else:
            axes[mode].plot(factor_matrix, **plot_kwargs[mode])
            axes[mode].set_xlabel(f"Mode {mode}")
            axes[mode].legend([str(i) for i in range(num_components)])
    return fig, axes


def component_comparison_plot(
    cp_tensors,
    row="model",
    weight_behaviour="normalise",
    weight_mode=0,
    plot_kwargs=None,
):
    """Create a plot to compare different CP tensors.

    This function creates a figure with either D columns and R rows or D columns and N rows,
    where D is the number of modes, R is the number of components and N is the number of cp tensors
    to compare.

    Parameters
    ----------
    cp_tensors : dict (str -> CPTensor)
        Dictionary with model names mapping to decompositions. The model names
        are used for labels. The components of all CP tensors will be aligned
        to maximise the factor match score with the components of the first CP
        tensor in the dictionary (from Python 3.7, dictionaries are sorted by
        insertion order, and dictionaries were sorted by insertion order already
        in CPython 3.6).
    row : {"model", "component"}
    weight_behaviour : {"ignore", "normalise", "evenly", "one_mode"} (default="normalise")
        How to handle the component weights.

         * ``"ignore"`` - Do nothing
         * ``"normalise"`` - Normalise all factor matrices
         * ``"evenly"`` - All factor matrices have equal norm
         * ``"one_mode"`` - The weight is allocated in one mode, all other factor matrices have unit norm columns.

    weight_mode : int (optional)
        Which mode to have the component weights in (only used if ``weight_behaviour="one_mode"``)
    plot_kwargs : list of list of dicts
        Nested list of dictionaries, one dictionary with keyword arguments for each subplot.
    
    Returns:
    --------
    fig : matplotlib figure
    axes : array of matplotlib axes
    """
    main_cp_tensor = next(iter(cp_tensors.values()))
    weights, factor_matrices = main_cp_tensor
    main_legend = next(iter(cp_tensors.keys()))

    num_components = len(weights.reshape(-1))
    num_modes = len(factor_matrices)
    num_models = len(cp_tensors)

    if row == "model":
        num_rows = num_models
    elif row == "component":
        num_rows = num_components
    else:
        raise ValueError("Row must be either 'model' or 'component'")

    fig, axes = plt.subplots(
        num_rows, num_modes, figsize=(16, num_rows * 9 / num_modes)
    )
    for i, (model_name, cp_tensor) in enumerate(cp_tensors.items()):
        # TODO: Function for weight_behaviour?
        if weight_behaviour == "ignore":
            weights, factor_matrices = cp_tensor
        elif weight_behaviour == "normalise":
            weights, factor_matrices = postprocessing.normalise_cp_tensor(cp_tensor)
        elif weight_behaviour == "evenly":
            weights, factor_matrices = postprocessing.distribute_weights_evenly(
                cp_tensor
            )
        elif weight_behaviour == "one_mode":
            weights, factor_matrices = postprocessing.distribute_weights_in_one_mode(
                cp_tensor, weight_mode
            )
        else:
            raise ValueError(
                "weight_behaviour must be either 'ignore', 'normalise', 'evenly', or 'one_mode'"
            )

        fms, permutation = factor_match_score(
            cp_tensor, main_cp_tensor, consider_weights=False, return_permutation=True
        )

        for mode, factor_matrix in enumerate(factor_matrices):
            for component_num, r in enumerate(permutation):
                if row == "model":
                    row_idx = i
                elif row == "component":
                    row_idx = component_num

                if plot_kwargs is None:
                    kwargs = {}
                else:
                    kwargs = plot_kwargs[row_idx][mode]

                if hasattr(factor_matrix, "plot") and hasattr(factor_matrix, "iloc"):
                    factor_matrix.iloc[:, r].plot(ax=axes[row_idx, mode], **kwargs)
                    legend = axes[row_idx, mode].get_legend()
                    if legend is not None:
                        legend.remove()
                else:
                    axes[row_idx, mode].plot(factor_matrix[:, r], **kwargs)
                    axes[row_idx, mode].set_xlabel(f"Mode {mode}")

    if row == "model":
        fig.legend(
            [str(i) for i in range(num_components)],
            loc="upper center",
            ncol=num_components,
        )
        for row_idx, model_name in enumerate(cp_tensors):
            axes[row_idx, 0].set_ylabel(model_name)
    elif row == "component":
        fig.legend(cp_tensors.keys(), loc="upper center", ncol=len(cp_tensors))
        for row_idx in range(num_components):
            axes[row_idx, 0].set_ylabel(row_idx)

    for row_idx in range(num_rows - 1):
        for mode in range(num_modes):
            ax = axes[row_idx, mode]
            ax.set_xticklabels(["" for _ in ax.get_xticks()])
            ax.set_xlabel("")
    return fig, axes


def optimisation_diagnostic_plots(error_logs, n_iter_max):
    """Diagnostic plots for the optimisation problem.

    This function creates two plots. One plot that shows the loss value for each initialisation
    and whether or not that initialisation converged or ran until the maximum number of iterations.
    The other plot shows the error log for each initialisation, with the initialisation with lowest
    final error in a different colour (orange).

    These plots can be helpful for understanding how stable the model is with respect to initialisation.
    Ideally, we should see that many initialisations converged and obtained the same, low, error.
    If models converge, but with different errors, then this can indicate that indicates that a stricter
    convergence tolerance is required, and if no models converge, then more iterations may be required.

    Parameters
    ----------
    error_logs : list of arrays
        List of arrays, each containing the error per iteration for an initialisation.
    n_iter_max : int
        Maximum number of iterations for the fitting procedure. Used to determine if the
        models converged or not.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : array(dtype=matplotlib.axes.Axes)

    Examples
    --------
    Fit the wrong number of components to show local minima problems
    >>> import numpy as np
    ... from tensorly.random import random_cp
    ... from tensorly.decomposition import parafac
    ... from component_vis.visualisation import optimisation_diagnostic_plots
    ... 
    ... # Generate random tensor and add noise
    ... rng = np.random.RandomState(1)
    ... cp_tensor = random_cp((5, 6, 7), 2, random_state=rng)
    ... dataset = cp_tensor.to_tensor() + rng.standard_normal((5, 6, 7))
    ... 
    ... # Fit 10 models
    ... errs = []
    ... for i in range(10):
    ...     errs.append(parafac(dataset, 3, n_iter_max=500, return_errors=True, init="random", random_state=rng)[1])
    ... 
    ... # Plot the diganostic plots
    ... optimisation_diagnostic_plots(errs, 500)


    Fit a model with too few iterations
    >>> import numpy as np
    ... from tensorly.random import random_cp
    ... from tensorly.decomposition import parafac
    ... from component_vis.visualisation import optimisation_diagnostic_plots
    ... 
    ... # Generate random tensor and add noise
    ... rng = np.random.RandomState(1)
    ... cp_tensor = random_cp((5, 6, 7), 3, random_state=rng)
    ... dataset = cp_tensor.to_tensor() + rng.standard_normal((5, 6, 7))
    ... 
    ... # Fit 10 models
    ... errs = []
    ... for i in range(10):
    ...     errs.append(parafac(dataset, 3, n_iter_max=50, return_errors=True, init="random", random_state=rng)[1])
    ... 
    ... # Plot the diganostic plots
    ... optimisation_diagnostic_plots(errs, 50)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))

    selected_init = None
    lowest_error = np.inf
    for init, error in enumerate(error_logs):
        if error[-1] < lowest_error:
            selected_init = init
            lowest_error = error[-1]

    ymax = 0
    for init, error in enumerate(error_logs):
        if init == selected_init:
            alpha = 1
            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]
            zorder = 10
        else:
            alpha = 0.5
            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
            zorder = 0

        if len(error) == n_iter_max:
            axes[0].scatter([init], [error[-1]], color=color, alpha=alpha, marker="x")
        else:
            axes[0].scatter([init], [error[-1]], color=color, alpha=alpha, marker="o")

        axes[1].semilogy(error, color=color, alpha=alpha, zorder=zorder)
        ymax = max(error[1], ymax)

    axes[0].set_xlabel("Initialisation")
    axes[0].set_ylabel("Error")
    axes[1].set_ylim(top=ymax)
    axes[1].set_ylim(top=ymax)

    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Error (Log scale)")

    custom_lines = [
        Line2D([0], [0], marker="o", alpha=1, color="k", linewidth=0),
        Line2D([0], [0], marker="x", alpha=1, color="k", linewidth=0),
        Line2D(
            [0],
            [0],
            marker="s",
            alpha=1,
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][1],
            linewidth=0,
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            alpha=0.5,
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
            linewidth=0,
        ),
    ]

    fig.legend(
        custom_lines,
        ["Converged", "Did not converge", "Lowest final error", "Other runs",],
        ncol=2,
        bbox_to_anchor=(0.5, -0.1),
        loc="lower center",
    )
    return fig, axes

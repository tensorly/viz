from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from matplotlib.lines import Line2D

from . import model_evaluation, postprocessing
from ._utils import is_iterable
from .factor_tools import construct_cp_tensor, factor_match_score
from .model_evaluation import estimate_core_tensor, percentage_variation
from .outliers import (
    _LEVERAGE_NAME,
    _SLABWISE_SSE_NAME,
    compute_outlier_info,
    get_leverage_outlier_threshold,
    get_slab_sse_outlier_threshold,
)
from .xarray_wrapper import _handle_labelled_cp, _handle_labelled_dataset, is_dataframe

__all__ = [
    "scree_plot",
    "histogram_of_residuals",
    "residual_qq",
    "outlier_plot",
    "component_scatterplot",
    "core_element_plot",
    "components_plot",
    "component_comparison_plot",
    "optimisation_diagnostic_plots",
]


# TODO: Examples in docstrings
# TODO NEXT: Use all of these functions in an example notebook


# TODO: Test this function - Ok with list of cp tensors and dict of cp tensors
def scree_plot(cp_tensors, dataset, errors=None, metric="Fit", ax=None):
    """Create scree plot for the given cp tensors.

    A scree plot is a plot with the model on the x-axis and a metric (often fit)
    on the y-axis. It is commonly plotted as a line plot with a scatter point located at each
    model.

    Parameters
    ----------
    cp_tensor: dict[Any, CPTensor]
        Dictionary mapping model names (often just the number of components as an int) to a
        model.
    dataset: numpy.ndarray or xarray.DataArray
        Dataset to compare the model against.
    errors: dict[Any, float] (optional)
        The metric to plot. If given, then the cp_tensor and dataset-arguments are ignored.
        This is useful to save computation time if, for example, the fit is computed beforehand.
    metric: str or Callable
        Which metric to plot, should have the signature ``metric(cp_tensor, dataset)`` and return
        a float. If it is a string, then this will be used as the y-label and metric will be set to 
        ``metric = getattr(component_vis.model_evaluation, metric)``.
        Also, if ``metric`` is a string, then it is converted to lower-case letters and spaces
        are converted to underlines before getting the metric from the ``model_evaluation`` module.
    ax: matplotlib axes
        Matplotlib axes that the plot will be placed in. If ``None``, then ``plt.gca()`` will be used.
    
    Returns
    -------
    ax
        Matplotlib axes object with the scree plot


    Examples
    --------
    Simple scree plot of fit

    .. plot ::
        :context: close-figs
        :include-source:

        >>> from component_vis.data import simulated_random_cp_tensor
        >>> from component_vis.visualisation import scree_plot
        >>> import matplotlib.pyplot as plt
        >>> from tensorly.decomposition import parafac
        >>> 
        >>> dataset = simulated_random_cp_tensor((10, 20, 30), rank=3, noise_level=0.2, seed=42)[1]
        >>> cp_tensors = {}
        >>> for rank in range(1, 5):
        ...     cp_tensors[f"{rank} components"] = parafac(dataset, rank, random_state=1)
        >>>     
        >>> scree_plot(cp_tensors, dataset)
        >>> plt.show()

    Scree plots for fit and core consistency in the same figure

    .. plot ::
        :context: close-figs
        :include-source:

        >>> from component_vis.data import simulated_random_cp_tensor
        >>> from component_vis.visualisation import scree_plot
        >>> import matplotlib.pyplot as plt
        >>> from tensorly.decomposition import parafac
        >>> 
        >>> dataset = simulated_random_cp_tensor((10, 20, 30), rank=3, noise_level=0.2, seed=42)[1]
        >>> cp_tensors = {}
        >>> for rank in range(1, 5):
        ...     cp_tensors[rank] = parafac(dataset, rank, random_state=1)
        >>> 
        >>> fig, axes = plt.subplots(1, 2, figsize=(8, 2), tight_layout=True)
        >>> scree_plot(cp_tensors, dataset, ax=axes[0])
        >>> scree_plot(cp_tensors, dataset, metric="Core consistency", ax=axes[1])
        >>> # Names are converted to lowercase and spaces are converted to underlines when fetching metric-function,
        >>> # so "Core consistency" becomes getattr(component_vis.model_evaluation, "core_consistency")
        >>>
        >>> for ax in axes:
        ...     ax.set_xlabel("Number of components")
        ...     ax.set_xticks(list(cp_tensors.keys()))
        >>> axes[1].set_ylim((0, 105))
        >>> plt.show()
    """
    if ax is None:
        ax = plt.gca()

    if isinstance(metric, str):
        ax.set_ylabel(metric.replace("_", " "))
        metric = getattr(model_evaluation, metric.lower().replace(" ", "_"))
    cp_tensors = dict(enumerate(cp_tensors))

    if errors is None:
        # compute error using the metric function
        errors = {model: metric(cp_tensor, dataset) for model, cp_tensor in cp_tensors.items()}
    else:
        errors = dict(errors)

    ax.plot(errors.keys(), errors.values(), "-o")
    return ax


@_handle_labelled_dataset("dataset", None)
@_handle_labelled_cp("cp_tensor", None)
def histogram_of_residuals(cp_tensor, dataset, ax=None, standardised=True, **kwargs):
    """Create a histogram of model residuals (:math:`\hat{\mathbf{\mathcal{X}}} - \mathbf{\mathcal{X}}`).

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

    Examples
    --------

    .. plot::
        :context: close-figs
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> from tensorly.decomposition import parafac
        >>> from component_vis.data import simulated_random_cp_tensor
        >>> from component_vis.visualisation import histogram_of_residuals
        >>> true_cp, X = simulated_random_cp_tensor((10, 20, 30), 3, seed=0)
        >>> est_cp = parafac(X, 3)
        >>> histogram_of_residuals(est_cp, X)
        >>> plt.show()
    """
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


@_handle_labelled_dataset("dataset", None)
@_handle_labelled_cp("cp_tensor", None)
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

    Examples
    --------

    .. plot::
        :context: close-figs
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> from tensorly.decomposition import parafac
        >>> from component_vis.data import simulated_random_cp_tensor
        >>> from component_vis.visualisation import residual_qq
        >>> true_cp, X = simulated_random_cp_tensor((10, 20, 30), 3, seed=0)
        >>> est_cp = parafac(X, 3)
        >>> residual_qq(est_cp, X)
        >>> plt.show()
    """
    estimated_dataset = construct_cp_tensor(cp_tensor)
    residuals = (estimated_dataset - dataset).ravel()

    if ax is None:
        ax = plt.gca()

    if use_pingouin:
        from pingouin import qqplot

        warn("GPL-3 Lisenced code is loaded, so this code also follows the GPL-3 license.")
        qqplot(residuals, ax=ax, **kwargs)
    else:
        sm.qqplot(residuals, ax=ax, **kwargs)

    ax.set_title("QQ-plot of residuals")
    return ax


# TODO: mode or axis?
# TODO: Write more docs, include see also
def outlier_plot(
    cp_tensor, dataset, mode=0, leverage_rule_of_thumbs=None, residual_rule_of_thumbs=None, p_value=0.05, ax=None,
):
    """Create the leverage-residual scatterplot to detect outliers.

    Detecting outliers can be a difficult task, and a common way to do this is by making a scatter-plot where the
    leverage score is plotted against the slabwise SSE (or residual).

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument
    dataset : np.ndarray or xarray.DataArray
        Dataset to compare with
    mode : int
        Which mode to create the outlier plot for
    leverage_rule_of_thumbs : str or iterable of str
        Rule of thumb(s) used to create lines for detecting outliers based on leverage score. Must be a supported
        argument for ``method`` with :meth:`component_vis.outliers.get_leverage_outlier_threshold`. If
        ``leverage_rule_of_thumbs`` is an iterable of strings, then multiple lines will be drawn, one for each
        method.
    residual_rule_of_thumbs : str or iterable of str
        Rule of thumb(s) used to create lines for detecting outliers based on residuals. Must be a supported
        argument for ``method`` with :meth:`component_vis.outliers.get_slab_sse_outlier_threshold`. If
        ``residual_rule_of_thumbs`` is an iterable of strings, then multiple lines will be drawn, one for each
        method.
    p_value : float or iterable of float
        p-value(s) to use for both the leverage and residual rule of thumbs. If an iterable of floats is used,
        then there will be drawn lines for each p-value.
    ax : Matplotlib axes
        Axes to plot outlier plot in. If ``None``, then ``plt.gca()`` is used.
    
    Returns
    -------
    ax : Matplotlib axes
        Axes with outlier plot in
    
    Examples
    --------
    Here is a simple example demonstrating how to use the outlier plot to detect outliers based on the
    Oslo bike sharing data.

    .. plot::
        :context: close-figs
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> from tensorly.decomposition import non_negative_parafac_hals
        >>> from component_vis.data import load_oslo_city_bike
        >>> from component_vis.postprocessing import postprocess
        >>> from component_vis.visualisation import outlier_plot
        >>> 
        >>> data = load_oslo_city_bike()
        >>> X = data.data
        >>> cp = non_negative_parafac_hals(X, 3, init="random")
        >>> cp = postprocess(cp, dataset=data, )
        >>> 
        >>> outlier_plot(cp, data, leverage_rule_of_thumbs='p-value', residual_rule_of_thumbs='p-value', p_value=[0.05, 0.01])
        >>> plt.show()
    
    We can also provide multiple types of rule-of-thumbs

    .. plot::
        :context: close-figs
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> from tensorly.decomposition import non_negative_parafac_hals
        >>> from component_vis.data import load_oslo_city_bike
        >>> from component_vis.postprocessing import postprocess
        >>> from component_vis.visualisation import outlier_plot
        >>> 
        >>> data = load_oslo_city_bike()
        >>> X = data.data
        >>> cp = non_negative_parafac_hals(X, 3, init="random")
        >>> cp = postprocess(cp, dataset=data, )
        >>> 
        >>> outlier_plot(cp, data, leverage_rule_of_thumbs=['huber lower', 'hw higher'], residual_rule_of_thumbs='two sigma')
        >>> plt.show()
    """
    weights, factor_matrices = cp_tensor

    outlier_info = compute_outlier_info(cp_tensor, dataset, axis=mode)

    if ax is None:
        ax = plt.gca()

    ax.plot(outlier_info[f"{_LEVERAGE_NAME}"], outlier_info[f"{_SLABWISE_SSE_NAME}"], "o", zorder=1, alpha=0.8)
    ax.set_xlabel("Leverage score")
    ax.set_ylabel("Slabwise SSE")
    if hasattr(factor_matrices[mode], "index") and factor_matrices[mode].index.name not in {None, ""}:
        title = f"Outlier plot for {factor_matrices[mode].index.name}"
    else:
        title = f"Outlier plot for mode {mode}"
    ax.set_title(title)

    for x, y, s in zip(outlier_info[f"{_LEVERAGE_NAME}"], outlier_info[f"{_SLABWISE_SSE_NAME}"], outlier_info.index,):
        ax.text(x, y, s, zorder=0)
    # Vertical lines for leverage based rule-of-thumb thresholds
    leverage_thresholds = {}
    if leverage_rule_of_thumbs is not None:
        if isinstance(leverage_rule_of_thumbs, str):
            leverage_rule_of_thumbs = [leverage_rule_of_thumbs]

        for leverage_rule_of_thumb in leverage_rule_of_thumbs:
            if leverage_rule_of_thumb in {"p-value", "hotelling"} and not is_iterable(p_value):
                p_values = [p_value]
            elif leverage_rule_of_thumb in {"p-value", "hotelling"}:
                p_values = p_value
            else:
                p_values = [None]  # We still need something to iterate over even if it doesn't use the p-value

            for p in p_values:
                threshold = get_leverage_outlier_threshold(
                    outlier_info[f"{_LEVERAGE_NAME}"], method=leverage_rule_of_thumb, p_value=p,
                )

                if leverage_rule_of_thumb == "p-value":
                    name = f"p-value: {p}"
                elif leverage_rule_of_thumb == "hotelling":
                    name = f"Hotelling T2 p-value: {p}"
                else:
                    name = leverage_rule_of_thumb
                leverage_thresholds[name] = threshold

    # Draw the lines
    for key, value in leverage_thresholds.items():
        ax.axvline(value, label=key, **next(ax._get_lines.prop_cycler))

    residual_thresholds = {}
    if residual_rule_of_thumbs is not None:
        if isinstance(residual_rule_of_thumbs, str):
            residual_rule_of_thumbs = [residual_rule_of_thumbs]

        for residual_rule_of_thumb in residual_rule_of_thumbs:
            if residual_rule_of_thumb == "p-value" and not is_iterable(p_value):
                p_values = [p_value]
            elif residual_rule_of_thumb == "p-value":
                p_values = p_value
            else:
                p_values = [None]  # We still need something to iterate over even if it doesn't use the p-value

            for p in p_values:
                threshold = get_slab_sse_outlier_threshold(
                    outlier_info[f"{_SLABWISE_SSE_NAME}"], method=residual_rule_of_thumb, p_value=p
                )
                if residual_rule_of_thumb == "p-value":
                    name = f"p-value: {p}"
                else:
                    name = residual_rule_of_thumb
                residual_thresholds[name] = threshold
    for key, value in residual_thresholds.items():
        ax.axhline(value, label=key, **next(ax._get_lines.prop_cycler))

    if len(leverage_thresholds) > 0 or len(residual_thresholds) > 0:
        ax.legend()
    return ax


def component_scatterplot(cp_tensor, mode, x_component=0, y_component=1, ax=None, **kwargs):
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
    
    Examples
    --------
    Small example with a simulated third order CP tensor

    .. plot::
        :context: close-figs
        :include-source:

        >>> from tensorly.random import random_cp
        >>> from component_vis.visualisation import component_scatterplot
        >>> import matplotlib.pyplot as plt
        >>> cp_tensor = random_cp(shape=(5,10,15), rank=2)
        >>> component_scatterplot(cp_tensor, mode=0)
        >>> plt.show()

    Eexample with PCA of a real stock dataset

    .. plot::
        :context: close-figs
        :include-source:

        >>> import pandas as pd
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import plotly.express as px
        >>> from component_vis.xarray_wrapper import label_cp_tensor
        >>> from component_vis.visualisation import component_scatterplot
        >>>
        >>> # Load data and convert to xarray
        >>> stocks = px.data.stocks().set_index("date").stack()
        >>> stocks.index.names = ["Date", "Stock"]
        >>> stocks = stocks.to_xarray()
        >>>
        >>> # Compute PCA via SVD of centered data
        >>> stocks -= stocks.mean(axis=0)
        >>> U, s, Vh = np.linalg.svd(stocks, full_matrices=False)
        >>>
        >>> # Extract two components and convert to cp_tensor
        >>> num_components = 2
        >>> cp_tensor = s[:num_components], (U[:, :num_components], Vh.T[:, :num_components])
        >>> cp_tensor = label_cp_tensor(cp_tensor, stocks)
        >>>
        >>> # Visualise the components with components_plot
        >>> component_scatterplot(cp_tensor, mode=1)
        >>> plt.show()
    """
    # TODO: example with svd also maybe?
    # TODO: handle weight
    # TODO: component scatterplot?
    # TODO: Handle dataframes
    if ax is None:
        ax = plt.gca()

    factor_matrix = cp_tensor[1][mode]
    if is_dataframe(factor_matrix):
        index = factor_matrix.index
        factor_matrix = factor_matrix.values
    else:
        index = np.arange(factor_matrix.shape[0])

    relevant_factors = factor_matrix[:, [x_component, y_component]]

    ax.set_xlabel(f"Component {x_component}")
    ax.set_ylabel(f"Component {y_component}")
    ax.set_title("Component plot")
    ax.scatter(relevant_factors[:, 0], relevant_factors[:, 1], **kwargs)

    for x, y, s in zip(relevant_factors[:, 0], relevant_factors[:, 1], index):
        ax.text(x, y, s)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

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
    
    Returns
    -------
    ax : Matplotlib axes

    
    Examples
    --------

    .. plot::
        :context: close-figs
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> from tensorly.decomposition import parafac
        >>> from component_vis.data import simulated_random_cp_tensor
        >>> from component_vis.visualisation import core_element_plot
        >>> true_cp, X = simulated_random_cp_tensor((10, 20, 30), 3, seed=42)
        >>> est_cp = parafac(X, 3)
        >>> core_element_plot(est_cp, X)
        >>> plt.show()
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


def components_plot(cp_tensor, weight_behaviour="normalise", weight_mode=0, plot_kwargs=None):
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

    .. plot::
        :context: close-figs
        :include-source:

        >>> from tensorly.random import random_cp
        >>> from component_vis.visualisation import components_plot
        >>> import matplotlib.pyplot as plt
        >>> cp_tensor = random_cp(shape=(5,10,15), rank=3)
        >>> components_plot(cp_tensor)
        >>> plt.show()

    Full example with PCA of a real stock dataset

    .. plot::
        :context: close-figs
        :include-source:

        >>> import pandas as pd
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import plotly.express as px
        >>> from component_vis.xarray_wrapper import label_cp_tensor
        >>> from component_vis.visualisation import components_plot
        >>>
        >>> # Load data and convert to xarray
        >>> stocks = px.data.stocks().set_index("date").stack()
        >>> stocks.index.names = ["Date", "Stock"]
        >>> stocks = stocks.to_xarray()
        >>>
        >>> # Compute PCA via SVD of centered data
        >>> stocks -= stocks.mean(axis=0)
        >>> U, s, Vh = np.linalg.svd(stocks, full_matrices=False)
        >>>
        >>> # Extract two components and convert to cp_tensor
        >>> num_components = 2
        >>> cp_tensor = s[:num_components], (U[:, :num_components], Vh.T[:, :num_components])
        >>> cp_tensor = label_cp_tensor(cp_tensor, stocks)
        >>>
        >>> # Visualise the components with components_plot
        >>> fig, axes = components_plot(cp_tensor, weight_behaviour="one_mode", weight_mode=1,
        ...                             plot_kwargs=[{}, {'marker': 'o', 'linewidth': 0}])
        >>> plt.show()
    """
    if weight_behaviour == "ignore":
        weights, factor_matrices = cp_tensor
    elif weight_behaviour == "normalise":
        weights, factor_matrices = postprocessing.normalise_cp_tensor(cp_tensor)
    elif weight_behaviour == "evenly":
        weights, factor_matrices = postprocessing.distribute_weights_evenly(cp_tensor)
    elif weight_behaviour == "one_mode":
        weights, factor_matrices = postprocessing.distribute_weights_in_one_mode(cp_tensor, weight_mode)
    else:
        raise ValueError("weight_behaviour must be either 'ignore', 'normalise' or 'one_mode'")

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
    cp_tensors, row="model", weight_behaviour="normalise", weight_mode=0, plot_kwargs=None,
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
    
    Returns
    -------
    fig : matplotlib figure
    axes : array of matplotlib axes

    Examples
    --------

    .. plot::
        :context: close-figs
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> from tensorly.decomposition import parafac, non_negative_parafac_hals
        >>> from component_vis.data import simulated_random_cp_tensor
        >>> from component_vis.visualisation import component_comparison_plot
        >>> from component_vis.postprocessing import postprocess
        >>>
        >>> true_cp, X = simulated_random_cp_tensor((10, 20, 30), 3, noise_level=0.5, seed=42)
        >>> cp_tensors = {
        ...     "True": true_cp,
        ...     "CP": parafac(X, 3),
        ...     "NN CP": non_negative_parafac_hals(X, 3),
        ... }
        >>> component_comparison_plot(cp_tensors, row="component")
        >>> plt.show()
    """
    main_cp_tensor = next(iter(cp_tensors.values()))
    weights, factor_matrices = main_cp_tensor
    main_legend = next(iter(cp_tensors.keys()))  # TODO: is this not used?

    cp_tensors = {
        key: postprocessing.postprocess(
            value, reference_cp_tensor=main_cp_tensor, weight_behaviour=weight_behaviour, weight_mode=weight_mode
        )
        for key, value in cp_tensors.items()
    }

    num_components = len(weights.reshape(-1))
    num_modes = len(factor_matrices)
    num_models = len(cp_tensors)

    if row == "model":
        num_rows = num_models
    elif row == "component":
        num_rows = num_components
    else:
        raise ValueError("Row must be either 'model' or 'component'")

    fig, axes = plt.subplots(num_rows, num_modes, figsize=(16, num_rows * 9 / num_modes))
    for model_num, (model_name, cp_tensor) in enumerate(cp_tensors.items()):
        weights, factor_matrices = cp_tensor
        for mode, factor_matrix in enumerate(factor_matrices):
            for component_num in range(num_components):
                if row == "model":
                    row_idx = model_num
                elif row == "component":
                    row_idx = component_num

                if plot_kwargs is None:
                    kwargs = {}
                else:
                    kwargs = plot_kwargs[row_idx][mode]

                if is_dataframe(factor_matrix):
                    factor_matrix[component_num].plot(ax=axes[row_idx, mode], **kwargs)
                    legend = axes[row_idx, mode].get_legend()
                    if legend is not None:
                        legend.remove()
                else:
                    axes[row_idx, mode].plot(factor_matrix[:, component_num], **kwargs)
                    axes[row_idx, mode].set_xlabel(f"Mode {mode}")

    if row == "model":
        fig.legend(
            [f"Component {i}" for i in range(num_components)], loc="upper center", ncol=num_components,
        )
        for row_idx, model_name in enumerate(cp_tensors):
            axes[row_idx, 0].set_ylabel(model_name)
    elif row == "component":
        fig.legend(cp_tensors.keys(), loc="upper center", ncol=len(cp_tensors))
        for row_idx in range(num_components):
            axes[row_idx, 0].set_ylabel(f"Component {row_idx}")

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
   
    .. plot::
        :context: close-figs
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from tensorly.random import random_cp
        >>> from tensorly.decomposition import parafac
        >>> from component_vis.visualisation import optimisation_diagnostic_plots
        >>> 
        >>> # Generate random tensor and add noise
        >>> rng = np.random.RandomState(1)
        >>> cp_tensor = random_cp((5, 6, 7), 2, random_state=rng)
        >>> dataset = cp_tensor.to_tensor() + rng.standard_normal((5, 6, 7))
        >>> 
        >>> # Fit 10 models
        >>> errs = []
        >>> for i in range(10):
        ...     errs.append(parafac(dataset, 3, n_iter_max=500, return_errors=True, init="random", random_state=rng)[1])
        >>> 
        >>> # Plot the diganostic plots
        >>> optimisation_diagnostic_plots(errs, 500)
        >>> plt.show()


    Fit a model with too few iterations

    .. plot::
        :context: close-figs
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from tensorly.random import random_cp
        >>> from tensorly.decomposition import parafac
        >>> from component_vis.visualisation import optimisation_diagnostic_plots
        >>> 
        >>> # Generate random tensor and add noise
        >>> rng = np.random.RandomState(1)
        >>> cp_tensor = random_cp((5, 6, 7), 3, random_state=rng)
        >>> dataset = cp_tensor.to_tensor() + rng.standard_normal((5, 6, 7))
        >>> 
        >>> # Fit 10 models
        >>> errs = []
        >>> for i in range(10):
        ...     errs.append(parafac(dataset, 3, n_iter_max=50, return_errors=True, init="random", random_state=rng)[1])
        >>> 
        >>> # Plot the diganostic plots
        >>> optimisation_diagnostic_plots(errs, 50)
        >>> plt.show()
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
        Line2D([0], [0], marker="s", alpha=1, color=plt.rcParams["axes.prop_cycle"].by_key()["color"][1], linewidth=0,),
        Line2D(
            [0], [0], marker="s", alpha=0.5, color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0], linewidth=0,
        ),
    ]

    fig.legend(
        custom_lines,
        ["Converged", "Did not converge", "Lowest final error", "Other runs"],
        ncol=2,
        bbox_to_anchor=(0.5, -0.1),
        loc="lower center",
    )
    return fig, axes


def percentage_variation_plots(
    cp_tensor, dataset=None, method="data", ax=None,
):
    """Bar chart showing the percentage of variation explained by each of the components.
    """
    # TODO: Write docstring
    if ax is None:
        ax = plt.gca()

    labels = {"data": "Percentage of data", "model": "Percentage of model"}
    variation = percentage_variation(cp_tensor, dataset, method=method)
    if method == "both":
        data_var, model_var = variation
        ax.bar(np.arange(len(data_var)) - 0.2, data_var, width=0.4, label=labels["data"])
        ax.bar(np.arange(len(model_var)) + 0.2, model_var, width=0.4, label=labels["model"])

    else:
        ax.bar(range(len(variation)), variation, label=labels[method])

    ax.legend()
    ax.set_xlabel("Component number")
    ax.set_ylabel("Percentage variation explained [%]")
    return ax

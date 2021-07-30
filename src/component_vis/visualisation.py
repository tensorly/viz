from .model_evaluation import estimate_core_tensor
from .outliers import compute_leverage, compute_slabwise_sse, compute_outlier_info
from .factor_tools import construct_cp_tensor, factor_match_score
from .outliers import _LEVERAGE_NAME, _SLABWISE_SSE_NAME
from . import postprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from matplotlib.lines import Line2D


#TODO: visualisation or visualisation?
def histogram_of_residuals(cp_tensor, X, ax=None, standardised=True, **kwargs):
    # TODO: docstring
    estimated_X = construct_cp_tensor(cp_tensor)
    residuals = (estimated_X - X).ravel()

    if ax is None:
        ax = plt.gca()
    if standardised:
        residuals = residuals/np.std(residuals)
        ax.set_xlabel("Standardised residuals")
    else:
        ax.set_xlabel("Residuals")

    ax.hist(residuals, **kwargs)
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of residuals")
    
    return ax


def residual_qq(cp_tensor, X, ax=None, **kwargs):
    #TODO: qq plot or prob plot?
    #TODO: pingouin plot
    estimated_X = construct_cp_tensor(cp_tensor)
    residuals = (estimated_X - X).ravel()

    if ax is None:
        ax = plt.gca()

    #res = stats.probplot(residuals, plot=ax)
    sm.qqplot(residuals, ax=ax, **kwargs)

    return ax


#TODO: mode or axis?
def outlier_plot(cp_tensor, X, mode=0, rule_of_thumbs=None, ax=None):
    # TODO: rule of thumbs
    weights, factor_matrices = cp_tensor
    factor_matrix = factor_matrices[mode]

    outlier_info = compute_outlier_info(cp_tensor, X, axis=mode)
    
    if ax is None:
        ax = plt.gca()

    ax.plot(outlier_info[f"{_LEVERAGE_NAME}"], outlier_info[f"{_SLABWISE_SSE_NAME}"], 'o')
    ax.set_xlabel("Leverage score")
    ax.set_ylabel("Slabwise SSE")
    ax.set_title("Outlier plot")
    for x, y, s in zip(outlier_info[f"{_LEVERAGE_NAME}"], outlier_info[f"{_SLABWISE_SSE_NAME}"], outlier_info.index):
        ax.text(x,y,s)
    return ax


def factor_scatterplot(cp_tensor, mode, x_component=0, y_component=1, orthogonalise=False, ax=None):
    #TODO: component scatterplot?
    if ax is None:
        ax = plt.gca()

    factor_matrix = cp_tensor[1][mode]
    relevant_factors = factor_matrix[:, [x_component, y_component]]

    if orthogonalise:
        relevant_factors, R = np.linalg.qr(relevant_factors)
        ax.axline((0, 0), R[:, 1], 'k')
        ax.axhline(0, 'k')
    else:
        ax.axvline(0, 'k')
        ax.axhline(0, 'k')

    ax.set_xlabel(f"Component {x_component}")
    ax.set_ylabel(f"Component {y_component}")
    ax.set_title("Component plot")
    ax.plot(relevant_factors[:, 0], relevant_factors[:, 1], 'o')

    if hasattr(factor_matrix, 'index'):
        index = factor_matrix.index
    else:
        index = np.arange(relevant_factors.shape[0])
    for x, y, s in zip(relevant_factors[:, 0], relevant_factors[:, 1], index):
        ax.text(x,y,s)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    
    ax.text(xmax - 0.05*(xmax - xmin), 0, f"Component {x_component}", horizontalalignment="left")
    if orthogonalise:
        y_pos = (xmax - 0.05*(xmax - xmin)) * R[1, 1]/R[0, 1]
        x_pos = (ymax - 0.05*(ymax - ymin)) * R[0, 1]/R[1, 1]
        if y_pos <= ymax:
            ax.text(xmax - 0.05*(xmax - xmin), y_pos, f"Component {y_component}", horizontalalignment="left")
        else:
            ax.text(x_pos, ymax - 0.05*(ymax - ymin), f"Component {y_component}", horizontalalignment="left")
    else:
        ax.text(0, ymax - 0.05*(ymax - ymin), f"Component {y_component}", horizontalalignment="left")

    return ax


# TODO: Core element image plot
def core_element_plot(cp_tensor, X, normalised=False, ax=None):
    # TODO: docstring
    
    weights, factors = cp_tensor
    rank = weights.shape[0]

    A = factors[0].copy()
    if weights is not None:
        A *= weights
    factors = tuple((A, *factors[1:]))

    # Estimate core and compute core consistency
    core_tensor = estimate_core_tensor(factors, X)
    T = np.zeros([rank]*X.ndim)
    np.fill_diagonal(T, 1)
    if normalised:
        denom = np.linalg.norm(core_tensor, 'fro')**2 
    else:
        denom = rank

    core_consistency = 100 - 100*np.sum((core_tensor - T)**2)/denom

    # Extract superdiagonal and offdiagonal elements
    core_elements = np.zeros_like(core_tensor.ravel())
    diagonal_mask = np.zeros([rank]*X.ndim, dtype=np.bool)
    np.fill_diagonal(diagonal_mask, 1)

    core_elements[:rank] = core_tensor[diagonal_mask]
    core_elements[rank:] = core_tensor[~diagonal_mask]

    # Plot core elements
    if ax is None:
        ax = plt.gca()

    x = np.arange(len(core_elements))
    y = np.zeros_like(x)
    y[:rank] = 1
    ax.plot(x, y, '-', label='Target')
    ax.plot(x[:rank], core_elements[:rank], 'o', label="Superdiagonal")
    ax.plot(x[rank:], core_elements[rank:], 'x', label="Off diagonal")
    ax.legend()
    ax.set_xlabel("Core element")
    ax.set_ylabel("Value")
    ax.set_title(f"Core consistency: {core_consistency:.1f}")

    return ax


def components_plot(cp_tensor, weight_behaviour="normalise", weight_mode=0, plot_kwargs=None):
    """Plot the component vectors of a CP model.
    
    Arguments
    ---------
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
        weights, factor_matrices = postprocessing.distribute_weights_in_one_mode(cp_tensor, weight_mode)
    else:
        raise ValueError("weight_behaviour must be either 'ignore', 'normalise' or 'one_mode'")
    
    num_components = len(weights.reshape(-1))
    num_modes = len(factor_matrices)
    
    if plot_kwargs is None:
        plot_kwargs = [{}]*num_modes
    
    fig, axes = plt.subplots(1, num_modes, figsize=(16, 9/num_modes))

    for mode, factor_matrix in enumerate(factor_matrices):
        if hasattr(factor_matrix, 'plot'):
            factor_matrix.plot(ax=axes[mode], **plot_kwargs[mode])
        else:
            axes[mode].plot(factor_matrix, **plot_kwargs[mode])
            axes[mode].set_xlabel(f"Mode {mode}")
            axes[mode].legend([str(i) for i in range(num_components)])
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
    
    Arguments
    ---------
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
    ... X = cp_tensor.to_tensor() + rng.standard_normal((5, 6, 7))
    ... 
    ... # Fit 10 models
    ... errs = []
    ... for i in range(10):
    ...     errs.append(parafac(X, 3, n_iter_max=500, return_errors=True, init="random", random_state=rng)[1])
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
    ... X = cp_tensor.to_tensor() + rng.standard_normal((5, 6, 7))
    ... 
    ... # Fit 10 models
    ... errs = []
    ... for i in range(10):
    ...     errs.append(parafac(X, 3, n_iter_max=50, return_errors=True, init="random", random_state=rng)[1])
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
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
            zorder = 10
        else:
            alpha = 0.5
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
            zorder = 0

        if len(error) == n_iter_max:
            axes[0].scatter([init], [error[-1]], color=color, alpha=alpha, marker='x')
        else:
            axes[0].scatter([init], [error[-1]], color=color, alpha=alpha, marker='o')

        axes[1].semilogy(error, color=color, alpha=alpha, zorder=zorder)
        ymax = max(error[1], ymax)

    axes[0].set_xlabel("Initialisation")
    axes[0].set_ylabel("Error")
    axes[1].set_ylim(top=ymax)
    axes[1].set_ylim(top=ymax)

    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Error (Log scale)")

    custom_lines = [Line2D([0], [0], marker='o', alpha=1, color='k', linewidth=0),
                    Line2D([0], [0], marker='x', alpha=1, color='k', linewidth=0),
                    Line2D([0], [0], marker='s', alpha=1, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], linewidth=0),
                    Line2D([0], [0], marker='s', alpha=0.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0], linewidth=0),]

    fig.legend(custom_lines, [ "Converged", "Did not converge", "Lowest final error", "Other runs",],
               ncol=2, bbox_to_anchor=(0.5, -0.1), loc="lower center")
    return fig, axes
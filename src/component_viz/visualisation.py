from .model_evaluation import estimate_core_tensor
from .outliers import compute_leverage, compute_slabwise_sse, compute_outlier_info
from .factor_tools import construct_cp_tensor
from .outliers import _LEVERAGE_NAME, _SLABWISE_SSE_NAME
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

#TODO: visualisation or vizualisation?
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

from .model_evaluation import estimate_core_tensor
import matplotlib.pyplot as plt
import numpy as np


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

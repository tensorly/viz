import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt
from itertools import product


def estimate_core_tensor(factors, X):
    # FAST EFFICIENT AND SCALABLE CORE CONSISTENCY DIAGNOSTIC FOR THE PARAFAC DECOMPOSITION FOR BIG SPARSE TENSORS
    # Efficient Vector and Parallel Manipulation Tensor Products
    svds = [sla.svd(factor, full_matrices=False) for factor in factors]
    for U, s, Vh in svds[::-1]:
        X = np.tensordot(U.T, X, (1, X.ndim - 1))
    for U, s, Vh in svds[::-1]:
        s_pinv = s.copy()
        mask = s_pinv != 0
        s_pinv[mask] = 1/s_pinv[mask]
        X = np.tensordot(np.diag(s_pinv), X, (1, X.ndim - 1))
    for U, s, Vh in svds[::-1]:
        X = np.tensordot(Vh.T, X, (1, X.ndim - 1))
    return np.ascontiguousarray(X)


def core_consistency(cp_tensor, X, normalised=False):
    # Distribute weights
    weights, factors = cp_tensor
    rank = factors[0].shape[1]

    A = factors[0].copy()
    if weights is not None:
        A *= weights
    
    factors = tuple((A, *factors[1:]))

    # Estimate core and compare
    G = estimate_core_tensor(factors, X)
    T = np.zeros([rank]*X.ndim)
    np.fill_diagonal(T, 1)
    if normalised:
        denom = np.linalg.norm(G, 'fro')**2 
    else:
        denom = rank

    return 100 - 100*np.sum((G - T)**2)/denom


def core_element_plot(cp_tensor, X, ax=None):
    weights, factors = cp_tensor
    rank = weights.shape[0]

    A = factors[0].copy()
    if weights is not None:
        A *= weights
    
    core_tensor = estimate_core_tensor(factors, X)

    if ax is None:
        ax = plt.gca()
    
    core_elements = np.zeros_like(core_tensor.ravel())
    core_elements[:rank] = np.diag(core_tensor)
    for i, (r1, r2, r3) in product(range(rank), repeat=3):
        if r1 != r2 and r2 != r3:
            core_elements[i+rank] = core_tensor[r1, r2, r3]
    
    x = np.arange(len(core_elements))
    ax.plot(x[:rank], core_elements[:rank], 'o', label="Superdiagonal")
    ax.plot(x[rank:], core_elements[rank:], 'x', label="Off diagonal")
    # TODO: labels
    return ax
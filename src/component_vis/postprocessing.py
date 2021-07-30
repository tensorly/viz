import numpy as np
import scipy.linalg as sla
from ._utils import unfold_tensor


def resolve_cp_sign_indeterminacy(cp_tensor, dataset, flip_mode=-1, resolve_mode=None, method="transpose"):
    """Resolve the sign indeterminacy of CP models.
    """
    # TODO: More documentation
    if flip_mode < 0:
        flip_mode = dataset.ndim + flip_mode
    if flip_mode > dataset.ndim or flip_mode < 0:
        raise ValueError("`flip_mode` must be between `-dataset.ndim` and `dataset.ndim-1`.")
    
    if resolve_mode is None:
        for mode in range(dataset.ndim):
            if mode != flip_mode:
                cp_tensor = resolve_cp_sign_indeterminacy(
                    cp_tensor, dataset, flip_mode=flip_mode, resolve_mode=mode, method=method,
                )
        
        return cp_tensor

    unfolded_dataset = unfold_tensor(dataset, resolve_mode)
    factor_matrix = cp_tensor[1][resolve_mode]

    if method == "transpose":
        sign_scores = factor_matrix.T @ unfolded_dataset
    elif method == "positive_coord":
        sign_scores = sla.lstsq(factor_matrix, unfolded_dataset)[0]
    else:
        raise ValueError("Method must be either `transpose` or `positive_coord`")

    signs = np.sign(np.sum(sign_scores**2 * np.sign(sign_scores), axis=1))
    signs = np.asarray(signs).reshape(1, -1)

    factor_matrices = list(cp_tensor[1])
    factor_matrices[resolve_mode] = factor_matrices[resolve_mode] * signs
    factor_matrices[flip_mode] = factor_matrices[flip_mode] * signs
    return cp_tensor[0], tuple(factor_matrices)

def normalise_cp_tensor(cp_tensor):
    # TODO: documentation
    # TODO: test
    weights, factors = cp_tensor
    if weights is None:
        weights = np.ones(factors[0].shape[1])

    weights = weights.copy()
    new_factors = []
    for factor in factors:
        norm = np.linalg.norm(factor, axis=0, keepdims=True)
        weights *= norm.ravel()
        new_factors.append(factor / norm)
    return weights, tuple(new_factors)

def distribute_weights_evenly(cp_tensor):
    # TODO: documentation
    # TODO: test
    weights, factors = normalise_cp_tensor(cp_tensor)
    weights = weights**(1/3)
    for factor in factors:
        factor[:] *= weights
    weights = np.ones_like(weights)
    return weights, factors

def distribute_weights_in_one_mode(cp_tensor, mode):
    # TODO: documentation
    # TODO: test
    weights, factors = normalise_cp_tensor(cp_tensor)
    factors[mode][:] *= weights
    return np.ones_like(weights), factors
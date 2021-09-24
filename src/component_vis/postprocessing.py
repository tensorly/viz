import numpy as np
import scipy.linalg as sla
from ._utils import unfold_tensor
from .factor_tools import factor_match_score
from .xarray_wrapper import label_cp_tensor


# TODO: Fix naming, what is resolve mode and what is flip mode?
def resolve_cp_sign_indeterminacy(cp_tensor, dataset, resolve_mode=None, flip_mode=-1, method="transpose"):
    """Resolve the sign indeterminacy of CP models.
    """
    # TODO: More documentation for resolve_cp_sign_indeterminacy
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
    # TODO: documentation for normalise_cp_tensor
    # TODO: test for normalise_cp_tensor
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
    # TODO: documentation for distribute_weights_evenly
    # TODO: test for distribute_weights_evenly
    weights, factors = normalise_cp_tensor(cp_tensor)
    weights = weights**(1/3)
    for factor in factors:
        factor[:] *= weights
    weights = np.ones_like(weights)
    return weights, factors

def distribute_weights_in_one_mode(cp_tensor, mode):
    # TODO: documentation for distribute_weights_in_one_mode
    # TODO: test for distribute_weights_in_one_mode
    weights, factors = normalise_cp_tensor(cp_tensor)
    factors[mode][:] *= weights
    return np.ones_like(weights), factors


# TODO: Should we name this reference_cp_tensor or target_cp_tensor?
def permute_cp_tensor(cp_tensor, reference_cp_tensor, consider_weights=True):
    # TODO: docstring for permute_cp_tensor
    # TODO: test for permute_cp_tensor
    fms, permutation = factor_match_score(reference_cp_tensor, cp_tensor, consider_weights=consider_weights, return_permutation=True)
    weights, factors = cp_tensor
    
    if weights is not None:
        new_weights = weights.copy()[permutation]
    else:
        new_weights = None

    new_factors = [None]*len(factors)
    for mode, factor in enumerate(factors):
        new_factor = factor.copy()
        if hasattr(factor, 'values'):
            new_factor.values[:] = new_factor.values[:, permutation]
        else:
            new_factor[:] = new_factor[:, permutation]
        new_factors[mode] = new_factor

    return new_weights, new_factors
        

def postprocess(cp_tensor, reference_cp_tensor=None, dataset=None, resolve_mode=None, flip_mode=-1, flip_method="transpose"):
    # TODO: Docstring for postprocess
    # TODO: Unit test for postprocess
    if reference_cp_tensor is not None:
        cp_tensor = permute_cp_tensor(cp_tensor, reference_cp_tensor)
    cp_tensor = normalise_cp_tensor(cp_tensor)

    if dataset is not None:
        cp_tensor = label_cp_tensor(cp_tensor, dataset)
        cp_tensor = resolve_cp_sign_indeterminacy(
            cp_tensor, dataset, resolve_mode=resolve_mode, flip_mode=flip_mode, method=flip_method
        )
    
    return cp_tensor
    
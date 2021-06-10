import numpy as np
from scipy.optimize import linear_sum_assignment
import scipy.linalg as sla

from ._utils import unfold_tensor


def normalise(x, axis=0):
    return x / np.linalg.norm(x, axis=axis, keepdims=True)


def tucker_congruence(factor_matrix1, factor_matrix2):
    congruence = normalise(factor_matrix1).T @ normalise(factor_matrix2)
    permutation = linear_sum_assignment(-congruence)
    return congruence[permutation].mean()


def get_permutation(factor_matrix1, factor_matrix2, ignore_sign=True):
    congruence_product = normalise(factor_matrix1).T@normalise(factor_matrix2)
    if ignore_sign:
        congruence_product = np.abs(congruence_product)
    row_index, column_index = linear_sum_assignment(-congruence_product)
    permutation = np.zeros_like(row_index)
    permutation[row_index] = column_index
    return permutation


def factor_match_score(
    cp_tensor1,
    cp_tensor2,
    consider_weights=False,
    skip_axis=None,
    return_permutation=False,
    absolute_value=True,
):
    r"""Compute the factor match score.

    The factor match score is used to measure the similarity between two
    sets of components. There are many definitions of the FMS, but one 
    common definition for third order tensors is given by:

    .. math::

        \sum_{r=1}^R \frac{\vec{a}_r^T \hat{\vec{a}}_r}{\|\vec{a}_r^T\|\|\hat{\vec{a}}_r\|}
                     \frac{\vec{b}_r^T \hat{\vec{b}}_r]{\|\vec{b}_r^T\|\|\hat{\vec{b}}_r\|}
                     \frac{\vec{c}_r^T \hat{\vec{c}}_r]{\|\vec{c}_r^T\|\|\hat{\vec{c}}_r\|},

    where :math:`\vec{a}, \vec{b}` and :math:`\vec{c}` are the component vectors for
    one of the decompositions and :math:`\hat{\vec{a}}, \hat{\vec{b}}` and :math:`\hat{\vec{c}}`
    are the component vectors for the other decomposition. 

    The above definition does not take the norm of the component vectors into account.
    However, sometimes, we also wish to compare their norm. In that case, set the
    ``consider_weights`` argument to ``True`` to compute

    .. math::

        \sum_{r=1}^R \left(1 - \frac{w_r \hat{w}_r}{\max\left( w_r \hat{w}_r \right)}\right)
                     \frac{\vec{a}_r^T \hat{\vec{a}}_r}{\|\vec{a}_r^T\|\|\hat{\vec{a}}_r\|}
                     \frac{\vec{b}_r^T \hat{\vec{b}}_r]{\|\vec{b}_r^T\|\|\hat{\vec{b}}_r\|}
                     \frac{\vec{c}_r^T \hat{\vec{c}}_r]{\|\vec{c}_r^T\|\|\hat{\vec{c}}_r\|}
    
    instead, where :math:`w_r = \|\vec{a}_r\| \|\vec{b}_r\| \|\vec{c}_r\|` and
    :math:`\hat{w}_r = \|\hat{\vec{a}}_r\| \|\hat{\vec{b}}_r\| \|\hat{\vec{c}}_r\|`.

    For both definitions above, there is a permutation determinacy. Two equivalent decompositions
    can have the same component vectors, but in a different order. To resolve this determinacy,
    we use linear sum assignment solver available in SciPy to efficiently find the optimal
    permutation.

    LSAP https://doi.org/10.1109/TAES.2016.140952
    """
    # Extract weights and components from decomposition
    weights1, factors1 = cp_tensor1
    weights2, factors2 = cp_tensor2

    norms1 = np.ones(factors1[0].shape[1])
    norms2 = np.ones(factors2[0].shape[1])
    if weights1 is not None:
        norms1 *= weights1
    if weights2 is not None:
        norms2 *= weights2

    congruence_product = 1
    for i, (factor1, factor2) in enumerate(zip(factors1, factors2)):
        if i == skip_axis:
            continue
        if consider_weights:
            norms1 *= np.linalg.norm(factor1, axis=0)
            norms2 *= np.linalg.norm(factor2, axis=0)
        congruence_product *= normalise(factor1).T @ normalise(factor2)
    
    if absolute_value:
        congruence_product = np.abs(congruence_product)
    row_index, column_index = linear_sum_assignment(-congruence_product)
    norms1 = norms1[row_index]
    norms2 = norms2[column_index]
    congruence_product = congruence_product[row_index, column_index]

    if consider_weights:
        congruence_product *= (1 - np.abs(norms1 - norms2)/np.maximum(norms1, norms2))
    
    if not return_permutation:
        return congruence_product.mean()
    
    permutation = np.zeros_like(row_index)
    permutation[row_index] = column_index
    return congruence_product.mean(), permutation


def degeneracy_score(cp_tensor):
    # TODO: docstring for degeneracy_score
    weights, factors = cp_tensor
    rank = factors[0].shape[1]
    tucker_congruence_scores = np.ones(shape=(rank,rank))

    for factor in factors:
        tucker_congruence_scores *= normalise(factor).T@normalise(factor)
    
    return tucker_congruence_scores.min()

def construct_cp_tensor(cp_tensor):
    #TODO: reconsider name
    #TODO: move to utils?
    if cp_tensor[0] is None:
        weights = np.ones(cp_tensor[1][0].shape[1])
    else:
        weights = cp_tensor[0].squeeze()
    
    einsum_input = 'R'
    einsum_output = ''
    for mode in range(len(cp_tensor[1])):
        idx = chr(ord('a') + mode)

        # We cannot use einsum with letters outside the alphabet
        if ord(idx) > ord("z"):
            max_modes = ord("a") - ord("z") - 1
            raise ValueError(f"Cannot have more than {max_modes} modes. Current components have {len(cp_tensor[1])}.")

        einsum_input += f', {idx}R'
        einsum_output += idx
    
    return np.einsum(f'{einsum_input} -> {einsum_output}', weights, *cp_tensor[1])


def construct_tucker_tensor(tucker_tensor):
    einsum_core = ''
    einsum_input = ''
    einsum_output = ''
    
    for mode in range(len(tucker_tensor[1])):
        idx = chr(ord('a') + mode)
        rank_idx = chr(ord('A') + mode)

        # We cannot use einsum with letters outside the alphabet
        if ord(idx) > ord("z"):
            max_modes = ord("a") - ord("z")
            raise ValueError(f"Cannot have more than {max_modes} modes. Current components have {len(tucker_tensor[1])}.")

        einsum_core += rank_idx
        einsum_input += f', {idx}{rank_idx}'
        einsum_output += idx
        
    
    return np.einsum(f'{einsum_core}{einsum_input} -> {einsum_output}', tucker_tensor[0], *tucker_tensor[1])


def resolve_cp_sign_indeterminacy(cp_tensor, dataset, flip_mode=-1, resolve_mode=None, method="transpose"):
    """Resolve the sign indeterminacy of CP models.
    """
    if flip_mode < 0:
        flip_mode = dataset.ndim - flip_mode
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
    signs = signs.reshape(1, -1)

    factor_matrices = list(cp_tensor[1])
    factor_matrices[resolve_mode] = factor_matrices[resolve_mode] * signs
    factor_matrices[flip_mode] = factor_matrices[resolve_mode] * signs
    return cp_tensor[0], tuple(factor_matrices)


def classification_accuracy(factor_matrix, labels, classifier, metric=None):
    classifier.fit(factor_matrix, labels)
    if metric is None:
        return classifier.score(factor_matrix, labels)
    return metric(labels, classifier.predict(factor_matrix))

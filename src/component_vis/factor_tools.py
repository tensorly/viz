import numpy as np
from scipy.optimize import linear_sum_assignment
import scipy.linalg as sla

from ._utils import unfold_tensor, extract_singleton


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
    consider_weights=True,
    skip_axis=None,
    return_permutation=False,
    absolute_value=True,
):
    r"""Compute the factor match score between ``cp_tensor1`` and ``cp_tensor2``.

    The factor match score is used to measure the similarity between two
    sets of components. There are many definitions of the FMS, but one 
    common definition for third order tensors is given by:

    .. math::

        \sum_{r=1}^R \frac{\vec{a}_r^T \hat{\vec{a}}_r}{\|\vec{a}_r^T\| \|\hat{\vec{a}}_r\|}
                     \frac{\vec{b}_r^T \hat{\vec{b}}_r}{\|\vec{b}_r^T\| \|\hat{\vec{b}}_r\|}
                     \frac{\vec{c}_r^T \hat{\vec{c}}_r}{\|\vec{c}_r^T\| \|\hat{\vec{c}}_r\|},

    where :math:`\vec{a}, \vec{b}` and :math:`\vec{c}` are the component vectors for
    one of the decompositions and :math:`\hat{\vec{a}}, \hat{\vec{b}}` and :math:`\hat{\vec{c}}`
    are the component vectors for the other decomposition. Often, the absolute value of the inner
    products is used instead of just the inner products (i.e. :math:`|\vec{a}_r^T \hat{\vec{a}}_r|`).

    The above definition does not take the norm of the component vectors into account.
    However, sometimes, we also wish to compare their norm. In that case, set the
    ``consider_weights`` argument to ``True`` to compute

    .. math::

        \sum_{r=1}^R \left(1 - \frac{w_r \hat{w}_r}{\max\left( w_r \hat{w}_r \right)}\right)
                     \frac{\vec{a}_r^T \hat{\vec{a}}_r}{\|\vec{a}_r^T\|\|\hat{\vec{a}}_r\|}
                     \frac{\vec{b}_r^T \hat{\vec{b}}_r}{\|\vec{b}_r^T\|\|\hat{\vec{b}}_r\|}
                     \frac{\vec{c}_r^T \hat{\vec{c}}_r}{\|\vec{c}_r^T\|\|\hat{\vec{c}}_r\|}
    
    instead, where :math:`w_r = \|\vec{a}_r\| \|\vec{b}_r\| \|\vec{c}_r\|` and
    :math:`\hat{w}_r = \|\hat{\vec{a}}_r\| \|\hat{\vec{b}}_r\| \|\hat{\vec{c}}_r\|`.

    For both definitions above, there is a permutation determinacy. Two equivalent decompositions
    can have the same component vectors, but in a different order. To resolve this determinacy,
    we use linear sum assignment solver available in SciPy :cite:p:`crouse2016implementing` to 
    efficiently find the optimal permutation.

    Parameters
    ----------
    cp_tensor1 : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument
    cp_tensor2 : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument
    consider_weights : bool (default=True)
        If False, then the weight-penalty is used (second equation above).
    skip_axis : int or None (default=None)
        Which axis to skip when computing the FMS. Useful if cross validation
        or split-half analysis is used.
    return_permutation : bool (default=False)
        Whether or not to return the optimal permutation of the factors
    absolute_value : bool (default=True)
        If True, then only magnitude of the congruence is considered, not the
        sign.
    
    Examples
    --------
    >>> import numpy as np
    ... from component_vis.factor_tools import factor_match_score
    ... from tensorly.decomposition import parafac
    ... from tensorly.random import random_cp
    ... # Construct random cp tensor with TensorLy
    ... cp_tensor = random_cp(shape=(4,5,6), rank=3, random_state=42)
    ... X = cp_tensor.to_tensor()
    ... # Add noise
    ... X_noisy = X + 0.05*np.random.RandomState(0).standard_normal(size=X.shape)
    ... # Decompose with TensorLy and compute FMS
    ... estimated_cp_tensor = parafac(X_noisy, rank=3, random_state=42)
    ... fms_with_weight_penalty = factor_match_score(cp_tensor, estimated_cp_tensor, consider_weights=True)
    ... fms_without_weight_penalty = factor_match_score(cp_tensor, estimated_cp_tensor, consider_weights=False)
    ... print(f"Factor match score (with weight penalty): {fms_with_weight_penalty:.2f}")
    ... print(f"Factor match score (without weight penalty): {fms_without_weight_penalty:.2f}")
    Factor match score (with weight penalty): 0.95
    Factor match score (without weight penalty): 0.99
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
        if hasattr(factor1, 'values'):
            factor1 = factor1.values
        if hasattr(factor2, 'values'):
            factor2 = factor2.values
        
        if i == skip_axis:
            continue
        if consider_weights:
            norms1 *= np.linalg.norm(factor1, axis=0)
            norms2 *= np.linalg.norm(factor2, axis=0)
        congruence_product *= normalise(factor1).T @ normalise(factor2)
    
    if consider_weights:
        congruence_product *= 1 - np.abs(norms1[:, np.newaxis] - norms2[np.newaxis, :])/np.maximum(norms1[:, np.newaxis], norms2[np.newaxis, :])

    if absolute_value:
        congruence_product = np.abs(congruence_product)

    row_index, column_index = linear_sum_assignment(-congruence_product)
    congruence_product = congruence_product[row_index, column_index]

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
    
    return np.asarray(tucker_congruence_scores).min()

def construct_cp_tensor(cp_tensor):
    #TODO: reconsider name
    #TODO: move to utils?
    #TODO: Tests (1 component for example)
    if cp_tensor[0] is None:
        weights = np.ones(cp_tensor[1][0].shape[1])
    else:
        weights = cp_tensor[0].reshape(-1)
    
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
    # TODO: Rename
    # TODO: Documentation
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


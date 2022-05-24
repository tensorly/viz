# -*- coding: utf-8 -*-

__author__ = "Marie Roald & Yngve Mardal Moe"

"""
This module contains most functions that only work on tensor factorisation models, not data
tensors. The module contains functions that are useful for inspecting tensor factorisation
models. For example, computing how similar two factorisations are, checking if two decompositions
are equivalent, or simply generating a dense tensor from a (possibly) labelled decomposition.
"""

__author__ = "Marie Roald & Yngve Mardal Moe"

from warnings import warn

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ._module_utils import (
    _handle_none_weights_cp_tensor,
    is_dataframe,
    validate_cp_tensor,
)
from ._tl_utils import (
    _handle_tensorly_backends_cp,
    _handle_tensorly_backends_dataset
)
from ._xarray_wrapper import (
    _SINGLETON,
    _handle_labelled_cp,
    _handle_labelled_dataset,
    _handle_labelled_factor_matrix,
)
from .utils import _alias_mode_axis, cp_norm, extract_singleton, normalise

__all__ = [
    "normalise_cp_tensor",
    "distribute_weights_evenly",
    "distribute_weights_in_one_mode",
    "distribute_weights",
    "cosine_similarity",
    "get_factor_matrix_permutation",
    "factor_match_score",
    "degeneracy_score",
    "get_cp_permutation",
    "permute_cp_tensor",
    "check_factor_matrix_equal",
    "check_cp_tensor_equal",
    "check_factor_matrix_close",
    "check_cp_tensors_equivalent",
    "percentage_variation",
]


@_handle_tensorly_backends_cp("cp_tensor", _SINGLETON)
@_handle_labelled_cp("cp_tensor", _SINGLETON)
def normalise_cp_tensor(cp_tensor):
    """Ensure that the all factor matrices have unit norm, and all weight is stored in the weight-vector

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument.

    Returns
    -------
    tuple
        The scaled CP tensor.
    """
    weights, factors = cp_tensor
    if weights is None:
        weights = np.ones(factors[0].shape[1])

    weights = weights.copy()
    new_factors = []
    for factor in factors:
        norm = np.linalg.norm(factor, axis=0, keepdims=True)
        weights *= norm.ravel()

        # If a component vector is zero, then we do not want to divide by zero, and zero / 1 is equal to zero.
        norm[norm == 0] = 1
        new_factors.append(factor / norm)
    return weights, tuple(new_factors)


@_handle_tensorly_backends_cp("cp_tensor", _SINGLETON)
@_handle_labelled_cp("cp_tensor", _SINGLETON)
def distribute_weights_evenly(cp_tensor):
    """Ensure that the weight-vector consists of ones and all factor matrices have equal norm

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument.

    Returns
    -------
    tuple
        The scaled CP tensor.
    """
    weights, factors = normalise_cp_tensor(cp_tensor)
    weights = weights ** (1 / len(factors))
    for factor in factors:
        factor[:] *= weights
    weights = np.ones_like(weights)
    return weights, factors


@_handle_tensorly_backends_cp("cp_tensor", _SINGLETON)
@_handle_labelled_cp("cp_tensor", _SINGLETON)
@_alias_mode_axis()
def distribute_weights_in_one_mode(cp_tensor, mode, axis=None):
    """Normalise all factors and multiply the weights into one mode.

    The CP tensor is scaled so all factor matrices except one have unit norm
    columns and the weight-vector contains only ones.

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument.
    mode : int
        Which mode (axis) to store the weights in
    axis : int (optional)
        Alias for mode. If this is set, then no value is needed for mode

    Returns
    -------
    tuple
        The scaled CP tensor.
    """
    weights, factors = normalise_cp_tensor(cp_tensor)
    factors[mode][:] *= weights
    return np.ones_like(weights), factors


def distribute_weights(cp_tensor, weight_behaviour, weight_mode=0):
    """Utility to distribute the weights of a CP tensor.

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument.
    weight_behaviour : {"ignore", "normalise", "evenly", "one_mode"} (default="normalise")
        How to handle the component weights.

         * ``"ignore"`` - Do nothing
         * ``"normalise"`` - Normalise all factor matrices
         * ``"evenly"`` - All factor matrices have equal norm
         * ``"one_mode"`` - The weight is allocated in one mode, all other factor matrices have unit norm columns.

    weight_mode : int (optional)
        Which mode to have the component weights in (only used if ``weight_behaviour="one_mode"``)

    Returns
    -------
    tuple
        The scaled CP tensor.

    See Also
    --------
    normalise_cp_tensor : Give all component vectors unit norm
    distribute_weights_evenly : Give all component vectors the same norm and set the weight-array to one.
    distribute_weights_in_one_mode : Keep all the weights in one factor matrix and set the weight-array to one.

    Raises
    ------
    ValueError
        If ``weight_behaviour`` is not one of ``"ignore"``, ``"normalise"``, ``"evenly"`` or ``"one_mode"``.
    """
    if weight_behaviour == "ignore":
        return cp_tensor
    elif weight_behaviour == "normalise":
        return normalise_cp_tensor(cp_tensor)
    elif weight_behaviour == "evenly":
        return distribute_weights_evenly(cp_tensor)
    elif weight_behaviour == "one_mode":
        return distribute_weights_in_one_mode(cp_tensor, weight_mode)
    else:
        raise ValueError("weight_behaviour must be either 'ignore', 'normalise', 'evenly', or 'one_mode'")


@_handle_tensorly_backends_dataset("factor_matrix1", None)
@_handle_tensorly_backends_dataset("factor_matrix2", None)
@_handle_labelled_factor_matrix("factor_matrix2", None)
@_handle_labelled_factor_matrix("factor_matrix1", None)
def cosine_similarity(factor_matrix1, factor_matrix2):
    r"""The average cosine similarity (Tucker congruence) with optimal column permutation.

    The cosine similarity between two vectors is computed as

    .. math::

        \cos (\mathbf{x}, \mathbf{y}) =
        \frac{\mathbf{x}^\mathsf{T}}{\|\mathbf{x}\|}\frac{\mathbf{y}}{\|\mathbf{y}\|}

    This function returns the average cosine similarity between the columns vectors of
    the two factor matrices, using the optimal column permutation.

    Parameters
    ----------
    factor_matrix1 : np.ndarray or pd.DataFrame
        First factor matrix
    factor_matrix2 : np.ndarray or pd.DataFrame
        Second factor matrix

    Returns
    -------
    float
        The average cosine similarity.
    """
    congruence = normalise(factor_matrix1).T @ normalise(factor_matrix2)
    permutation = linear_sum_assignment(-congruence)
    return extract_singleton(congruence[permutation].mean())


NO_COLUMN = slice(0, 0, 1)


def _get_linear_sum_assignment_permutation(cost_matrix, allow_smaller_rank):
    row_index, column_index = linear_sum_assignment(-cost_matrix)

    R1, R2 = cost_matrix.shape
    if R1 > R2 and not allow_smaller_rank:
        raise ValueError(
            f"Cannot permute a {R2}-column matrix against a {R1}-column matrix unless ``allow_smaller_rank=True``."
        )

    permutation = [None] * max(R1, R2)
    for row_idx, col_idx in zip(row_index, column_index):
        permutation[row_idx] = col_idx

    missing_entries = sorted(set(range(R2)) - set(permutation))
    for i, missing in enumerate(missing_entries):
        permutation[i + len(row_index)] = missing

    for i, p, in enumerate(permutation):
        if p is None:
            permutation[i] = NO_COLUMN

    return row_index, column_index, permutation


@_handle_tensorly_backends_dataset("factor_matrix1", None)
@_handle_tensorly_backends_dataset("factor_matrix2", None)
def get_factor_matrix_permutation(factor_matrix1, factor_matrix2, ignore_sign=True, allow_smaller_rank=False):
    r"""Find optimal permutation of the factor matrices

    Efficient estimation of the optimal permutation for two factor matrices.
    To find the optimal permutation, :math:`\sigma`, we solve the following
    optimisation problem:

    .. math::

        \max_\sigma \sum_{r} \frac{\left|\mathbf{a}_{r}^\mathsf{T}\hat{\mathbf{a}}_{\sigma(r)}\right|}
                                  {\|\mathbf{a}_{r}\| \|\hat{\mathbf{a}}_{\sigma(r)}\|}

    where :math:`\mathbf{a}_r` is the :math:`r`-th component vector for the
    first factor matrix and :math:`\hat{\mathbf{a}}_{\sigma(r)}` is :math:`r`-th
    component vector of the second factor matrix after permuting the columns.

    Parameters
    ----------
    factor_matrix1 : np.ndarray or pd.DataFrame
        First factor matrix
    factor_matrix2 : np.ndarray or pd.DataFrame
        Second factor matrix
    ignore_sign : bool
        Whether to take the absolute value of the inner products before
        computing the permutation. This is usually done because of the sign
        indeterminacy of component models.
    allow_smaller_rank : bool (default=False)
        If ``True``, then the function can align a smaller matrix onto a larger one. Missing
        columns are aligned with ``tlviz.factor_tools.NO_COLUMN`` (a slice that slices nothing).

    Returns
    -------
    permutation : list[int | slice]
        List of ints used to permute ``factor_matrix2`` so its columns optimally align with ``factor_matrix1``.
        If the ``factor_matrix1`` has a column with no corresponding column in ``factor_matrix2`` (i.e. there
        are fewer columns in ``factor_matrix2`` than in ``factor_matrix1``), then
        ``tlviz.factor_tools.NO_COLUMN`` (a slice that slices nothing) is used to indicate missing columns.

    Raises
    ------
    ValueError
        If ``allow_smaller_rank=False`` and ``factor_matrix2`` has fewer columns than ``factor_matrix1``.
    """
    congruence_product = normalise(factor_matrix1).T @ normalise(factor_matrix2)
    if ignore_sign:
        congruence_product = np.abs(congruence_product)

    return _get_linear_sum_assignment_permutation(congruence_product, allow_smaller_rank=allow_smaller_rank)[-1]


@_handle_tensorly_backends_cp("cp_tensor1", None)
@_handle_tensorly_backends_cp("cp_tensor2", None)
def factor_match_score(
    cp_tensor1,
    cp_tensor2,
    consider_weights=True,
    skip_mode=None,
    return_permutation=False,
    absolute_value=True,
    allow_smaller_rank=False,
):
    r"""Compute the factor match score between ``cp_tensor1`` and ``cp_tensor2``.

    The factor match score is used to measure the similarity between two
    sets of components. There are many definitions of the FMS, but one
    common definition for third order tensors is given by:

    .. math::

        \sum_{r=1}^R \frac{\mathbf{a}_r^T \hat{\mathbf{a}}_r}{\|\mathbf{a}_r^T\| \|\hat{\mathbf{a}}_r\|}
                     \frac{\mathbf{b}_r^T \hat{\mathbf{b}}_r}{\|\mathbf{b}_r^T\| \|\hat{\mathbf{b}}_r\|}
                     \frac{\mathbf{c}_r^T \hat{\mathbf{c}}_r}{\|\mathbf{c}_r^T\| \|\hat{\mathbf{c}}_r\|},

    where :math:`\mathbf{a}, \mathbf{b}` and :math:`\mathbf{c}` are the component vectors for
    one of the decompositions and :math:`\hat{\mathbf{a}}, \hat{\mathbf{b}}` and :math:`\hat{\mathbf{c}}`
    are the component vectors for the other decomposition. Often, the absolute value of the inner
    products is used instead of just the inner products (i.e. :math:`|\mathbf{a}_r^T \hat{\mathbf{a}}_r|`).

    The above definition does not take the norm of the component vectors into account.
    However, sometimes, we also wish to compare their norm. In that case, set the
    ``consider_weights`` argument to ``True`` to compute

    .. math::

        \sum_{r=1}^R \left(1 - \frac{w_r \hat{w}_r}{\max\left( w_r \hat{w}_r \right)}\right)
                     \frac{\mathbf{a}_r^T \hat{\mathbf{a}}_r}{\|\mathbf{a}_r^T\|\|\hat{\mathbf{a}}_r\|}
                     \frac{\mathbf{b}_r^T \hat{\mathbf{b}}_r}{\|\mathbf{b}_r^T\|\|\hat{\mathbf{b}}_r\|}
                     \frac{\mathbf{c}_r^T \hat{\mathbf{c}}_r}{\|\mathbf{c}_r^T\|\|\hat{\mathbf{c}}_r\|}

    instead, where :math:`w_r = \|\mathbf{a}_r\| \|\mathbf{b}_r\| \|\mathbf{c}_r\|` and
    :math:`\hat{w}_r = \|\hat{\mathbf{a}}_r\| \|\hat{\mathbf{b}}_r\| \|\hat{\mathbf{c}}_r\|`.

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
    skip_mode : int or None (default=None)
        Which mode to skip when computing the FMS. Useful if cross validation
        or split-half analysis is used.
    return_permutation : bool (default=False)
        Whether or not to return the optimal permutation of the factors
    absolute_value : bool (default=True)
        If True, then only magnitude of the congruence is considered, not the
        sign.
    allow_smaller_rank : bool (default=False)
        Only relevant if ``return_permutation=True``. If ``True``, then ``cp_tensor2``
        can have fewer components than ``cp_tensor2``. Missing components are aligned
        with ``tlviz.factor_tools.tlviz.factor_tools.NO_COLUMN`` (a slice that slices nothing).

    Returns
    -------
    fms : float
        The factor match score
    permutation : list[int | object] (only if return_permutation=True)
        List of ints used to permute ``cp_tensor2`` so its components optimally align with ``cp_tensor1``.
        If the ``cp_tensor1`` has a component with no corresponding component in ``cp_tensor2`` (i.e. there
        are fewer components in ``cp_tensor2`` than in ``cp_tensor1``), then
        ``tlviz.factor_tools.NO_COLUMN`` (a slice that slices nothing) is used to indicate missing components.

    Raises
    ------
    ValueError
        If ``allow_smaller_rank=False`` and ``cp_tensor2`` has fewer components than ``cp_tensor1``.

    Examples
    --------
    >>> import numpy as np
    >>> from tlviz.factor_tools import factor_match_score
    >>> from tensorly.decomposition import parafac
    >>> from tensorly.random import random_cp
    >>> # Construct random cp tensor with TensorLy
    >>> cp_tensor = random_cp(shape=(4,5,6), rank=3, random_state=42)
    >>> X = cp_tensor.to_tensor()
    >>> # Add noise
    >>> X_noisy = X + 0.05*np.random.RandomState(0).standard_normal(size=X.shape)
    >>> # Decompose with TensorLy and compute FMS
    >>> estimated_cp_tensor = parafac(X_noisy, rank=3, random_state=42)
    >>> fms_with_weight_penalty = factor_match_score(cp_tensor, estimated_cp_tensor, consider_weights=True)
    >>> print(f"Factor match score (with weight penalty): {fms_with_weight_penalty:.2f}")
    Factor match score (with weight penalty): 0.95
    >>> fms_without_weight_penalty = factor_match_score(cp_tensor, estimated_cp_tensor, consider_weights=False)
    >>> print(f"Factor match score (without weight penalty): {fms_without_weight_penalty:.2f}")
    Factor match score (without weight penalty): 0.99
    """
    if skip_mode is not None and consider_weights:
        warn(
            "Cannot consider weights when a mode is skipped due to the scaling indeterminacy of PARAFAC models."
            + " consider_weights will therefore be set to False. To supress this warning, specify"
            + " consider_weights=False when calling factor_match_score with skip_mode not equal to None."
        )
        consider_weights = False

    # Extract weights and components from decomposition
    weights1, factors1 = normalise_cp_tensor(cp_tensor1)
    weights2, factors2 = normalise_cp_tensor(cp_tensor2)

    congruence_product = 1
    for i, (factor1, factor2) in enumerate(zip(factors1, factors2)):
        if hasattr(factor1, "values"):
            factor1 = factor1.values
        if hasattr(factor2, "values"):
            factor2 = factor2.values

        if i == skip_mode:
            continue
        congruence_product *= factor1.T @ factor2

    if consider_weights:
        congruence_product *= 1 - np.abs(weights1[:, np.newaxis] - weights2[np.newaxis, :]) / np.maximum(
            weights1[:, np.newaxis], weights2[np.newaxis, :]
        )

    if absolute_value:
        congruence_product = np.abs(congruence_product)

    # If permutation is not returned, then smaller rank is OK
    allow_smaller_rank = allow_smaller_rank or not return_permutation
    row_index, column_index, permutation = _get_linear_sum_assignment_permutation(
        congruence_product, allow_smaller_rank=allow_smaller_rank
    )
    congruence_product = congruence_product[row_index, column_index]

    if not return_permutation:
        return congruence_product.mean()

    return congruence_product.mean(), permutation


@_handle_tensorly_backends_cp("cp_tensor", None)
def degeneracy_score(cp_tensor):
    r"""Compute the degeneracy score for a given decomposition.

    PARAFAC models can be degenerate, which is a sign that we should
    be careful before interpreting that model. For a third order tensor,
    this generally manifests in a triple cosine of two components that
    approach -1. That is

    .. math::

        \cos(\mathbf{a}_{r}, \mathbf{a}_{s})
        \cos(\mathbf{b}_{r}, \mathbf{b}_{s})
        \cos(\mathbf{c}_{r}, \mathbf{c}_{s})
        \approx -1

    for some :math:`r \neq s`, where :math:`\mathbf{A}, \mathbf{B}`
    and :math:`\mathbf{C}` are factor matrices and

    .. math::

        \cos(\mathbf{x}, \mathbf{y}) =
        \frac{\mathbf{x}^\mathsf{T} \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}.

    Furthermore, the magnitude of the degenerate components are unbounded and
    will approach infinity as the number of iterations increase.

    Degenerate solutions typically signify that the decomposition is unreliable,
    and one should take care before interpreting the components. Degeneracy
    can, in fact, be a sign that the PARAFAC problem is ill-posed. There are certain
    tensors where there are no solutions to the least squares problem to needed to fit
    PARAFAC models. And in those cases, the "optimal" but unobtainable PARAFAC
    decomposition will have component vectors with infinite norm that point in
    opposite directions :cite:p:`krijnen2008non`.

    There are several strategies to avoid degenerate solutions:

     * Fitting models with more random initialisations
     * Decreasing the convergence tolerance or increasing the number of iterations
     * Imposing non-negativity constraints in all modes
     * Imposing orthogonality constraints in at least one mode
     * Changing the number of components

    Both non-negativity constraints and orthogonality constraints will
    remove the potential ill-posedness of the CP model. We can, in fact,
    not obtain degenerate solutions when we impose such constriants
    :cite:p:`krijnen2008non`

    To measure degeneracy, we compute the degeneracy score, which is the
    minimum triple cosine (for a third-order tensor). A score close to
    -1 signifies a degenerate solution. A score of -0.85 is an indication
    of a troublesome model :cite:p:`krijnen1993analysis` (as cited in
    :cite:p:`bro1997parafac`).

    For more information about degeneracy for component models see
    :cite:p:`zijlstra2002degenerate` and :cite:p:`bro1997parafac`.


    .. note::

        There are other kinds of degeneracies too. For example three-component
        degeneracies, which manifests as two components of increasing magnitude
        and one other component equal to the negative sum of the former
        two :cite:p:`paatero2000construction,stegeman2006degeneracy`. However, it
        is the two-component degeneracy that is most commonly discussed in the
        litterature :cite:p:`bro1997parafac,zijlstra2002degenerate,krijnen2008non`.
        Still, if three or more components display weights that have a much higher
        magnitude than the data, there is a reason to be concerned.

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument.

    Returns
    -------
    degeneracy_score : float
        Degeneracy score, between 1 and -1. A score close to -1 signifies
        a degenerate solution. A score of -0.85 is an indication of a
        troublesome model :cite:p:`krijnen1993analysis` (as cited in
        :cite:p:`bro1997parafac`).

    Examples
    --------
    We begin by constructing a random simulated cp tensor and compute the degeneracy score

    >>> from tlviz.data import simulated_random_cp_tensor
    >>> from tlviz.factor_tools import degeneracy_score
    >>> cp_tensor = simulated_random_cp_tensor((10, 11, 12), rank=3, seed=0)[0]
    >>> print(f"Degeneracy score: {degeneracy_score(cp_tensor):.2f}")
    Degeneracy score: 0.35

    We see that (as expected) the random cp_tensor is not very degenerate. To simulate
    a tensor with two-component degeneracy, we can, for example, replace one of the
    components with a flipped copy of another component

    >>> w, (A, B, C) = cp_tensor
    >>> A[:,1] = -A[:, 0]
    >>> B[:,1] = -B[:, 0]
    >>> C[:,1] = -C[:, 0]
    >>> print(f"Degeneracy score: {degeneracy_score(cp_tensor):.2f}")
    Degeneracy score: -1.00

    We see that this modified cp_tensor is degenerate.
    """
    # TODOC: There may be some more relevant cites in Paatero 2000
    weights, factors = cp_tensor
    rank = factors[0].shape[1]
    tucker_congruence_scores = np.ones(shape=(rank, rank))

    for factor in factors:
        tucker_congruence_scores *= normalise(factor).T @ normalise(factor)

    return np.asarray(tucker_congruence_scores).min()


def _permute_cp_tensor(cp_tensor, permutation):
    """Internal function, does not handle labelled cp tensors. Use ``permute_cp_tensor`` instead.
    """
    weights, factors = cp_tensor

    if weights is not None:
        new_weights = np.zeros(len(permutation))
        for i, p in enumerate(permutation):
            if p == NO_COLUMN:
                new_weights[i] = np.nan
            else:
                new_weights[i] = weights[p]
    else:
        new_weights = None

    new_factors = [None] * len(factors)
    for mode, factor in enumerate(factors):
        new_factor = np.zeros((factor.shape[0], len(permutation)))
        for i, p in enumerate(permutation):
            if p == NO_COLUMN:
                new_factor[:, i] = np.nan
            else:
                new_factor[:, i] = factor[:, p]

        new_factors[mode] = new_factor

    return new_weights, new_factors


@_handle_tensorly_backends_cp("cp_tensor", None)
@_handle_labelled_cp("cp_tensor", None)
def get_cp_permutation(cp_tensor, reference_cp_tensor=None, consider_weights=True, allow_smaller_rank=False):
    """Find the optimal permutation between two CP tensors.

    This function supports two ways of finding the permutation of a CP tensor: Aligning the components
    with those of a reference CP tensor (if ``reference_cp_tensor`` is not ``None``), or finding the
    permutation so the components are in descending order with respect to their explained variation
    (if both ``reference_cp_tensor`` and ``permutation`` is ``None``).

    This function uses the factor match score to compute the optimal permutation between
    two CP tensors. This is useful for comparison purposes, as CP two identical CP tensors
    may have permuted columns.

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument.
    reference_cp_tensor : CPTensor or tuple (optional)
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument. The tensor
        that ``cp_tensor`` is aligned with. Either this or the ``permutation``
        argument must be passed, not both.
    consider_weights : bool
        Whether to consider the factor weights when the factor match score is computed.

    Returns
    -------
    tuple
        The permutation to use when permuting ``cp_tensor``.
    """
    if reference_cp_tensor is not None:
        fms, permutation = factor_match_score(
            reference_cp_tensor,
            cp_tensor,
            consider_weights=consider_weights,
            return_permutation=True,
            allow_smaller_rank=allow_smaller_rank,
        )
        rank = cp_tensor[1][0].shape[1]
        target_rank = reference_cp_tensor[1][0].shape[1]

        if rank > target_rank:  # There are more components in the tensor to permute than the reference tensor
            remaining_indices = sorted(set(range(rank)) - set(permutation))
            permutation = list(permutation) + remaining_indices
    else:
        variation = percentage_variation(cp_tensor, method="model")
        permutation = sorted(range(len(variation)), key=lambda i: -variation[i])

    return permutation


@_handle_tensorly_backends_cp("cp_tensor", _SINGLETON)
@_handle_labelled_cp("cp_tensor", _SINGLETON, preserve_columns=False)
def permute_cp_tensor(
    cp_tensor, permutation=None, reference_cp_tensor=None, consider_weights=True, allow_smaller_rank=False
):
    """Permute the CP tensor

    This function supports three ways of permuting a CP tensor: Aligning the components
    with those of a reference CP tensor (if ``reference_cp_tensor`` is not ``None``),
    permuting the components according to a given permutation (if ``permutation`` is not ``None``)
    or so the components are in descending order with respect to their explained variation
    (if both ``reference_cp_tensor`` and ``permutation`` is ``None``).

    This function uses the factor match score to compute the optimal permutation between
    two CP tensors. This is useful for comparison purposes, as CP two identical CP tensors
    may have permuted columns.

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument.
    permutation : tuple (optional)
        Tuple with the column permutations. Either this or the ``reference_cp_tensor``
        argument must be passed, not both.
    reference_cp_tensor : CPTensor or tuple (optional)
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument. The tensor
        that ``cp_tensor`` is aligned with. Either this or the ``permutation``
        argument must be passed, not both.
    consider_weights : bool
        Whether to consider the factor weights when the factor match score is computed.

    Returns
    -------
    tuple
        Tuple representing ``cp_tensor`` optimally permuted.

    Raises
    ------
    ValueError
        If neither ``permutation`` nor ``reference_cp_tensor`` is provided
    ValueError
        If both ``permutation`` and ``reference_cp_tensor`` is provided
    """
    if permutation is not None and reference_cp_tensor is not None:
        raise ValueError("Must either provide a permutation, a reference CP tensor or neither. Both is provided")

    if permutation is None:
        permutation = get_cp_permutation(
            cp_tensor=cp_tensor,
            reference_cp_tensor=reference_cp_tensor,
            consider_weights=consider_weights,
            allow_smaller_rank=allow_smaller_rank,
        )

    return _permute_cp_tensor(cp_tensor, permutation)


@_handle_tensorly_backends_dataset("factor_matrix1", None)
@_handle_tensorly_backends_dataset("factor_matrix2", None)
def check_factor_matrix_equal(factor_matrix1, factor_matrix2, ignore_labels=False):
    """Check that all entries in a factor matrix are close, if labelled, then label equality is also checked.

    This function is similar to ``numpy.allclose``, but works on both labelled and unlabelled factor
    matrices. If the factor matrices are labelled, then the DataFrame index and columns are also
    compared (unless ``ignore_labels=True``).

    Parameters
    ----------
    factor_matrix1 : numpy.ndarray or pandas.DataFrame
        Labelled or unlabelled factor matrix
    cp_tensor2 : CPTensor or tuple
        Labelled or unlabelled factor matrix
    rtol : float
        Relative tolerance (see ``numpy.allclose``)
    atol : float
        Absolute tolerance (see ``numpy.allclose``)
    ignore_labels : bool
        If True, then labels (i.e. DataFrame column names and indices) can differ.

    Returns
    -------
    bool
        Whether the decompositions are equivalent.

    Examples
    --------
    ``check_factor_matrix_equal`` checks if two factor matrices are exactly the same.

    >>> from tlviz.data import simulated_random_cp_tensor
    >>> import numpy as np
    >>> A = np.arange(6).reshape(3, 2).astype(float)
    >>> B = A.copy()
    >>> check_factor_matrix_equal(A, B)
    True

    If they are only the same up to round off errors, then this function returns ``False``

    >>> check_factor_matrix_equal(A, B + 1e-10)
    False

    If we make only one of them into a DataFrame, then the factor matrices are not equal

    >>> import pandas as pd
    >>> A_labelled = pd.DataFrame(A)
    >>> check_factor_matrix_equal(A_labelled, B)
    False
    >>> check_factor_matrix_equal(B, A_labelled)
    False

    If we turn B into a DataFrame too, it passes again

    >>> B_labelled = pd.DataFrame(A)
    >>> check_factor_matrix_equal(A_labelled, B_labelled)
    True

    The index is checked for equality, so if we change the index of ``B_labelled``, then
    the factor matrices are not equal

    >>> B_labelled.index += 1
    >>> check_factor_matrix_equal(A_labelled, B_labelled)
    False

    However, we can disable checking the labels by using the ``ignore_labels`` argument

    >>> check_factor_matrix_equal(A_labelled, B_labelled, ignore_labels=True)
    True
    """
    if is_dataframe(factor_matrix1) != is_dataframe(factor_matrix2) and not ignore_labels:
        return False
    if ignore_labels and is_dataframe(factor_matrix1):
        factor_matrix1 = factor_matrix1.values
    if ignore_labels and is_dataframe(factor_matrix2):
        factor_matrix2 = factor_matrix2.values

    if is_dataframe(factor_matrix1):
        return factor_matrix1.equals(factor_matrix2)

    return np.array_equal(factor_matrix1, factor_matrix2)


@_handle_tensorly_backends_cp("cp_tensor1", None)
@_handle_tensorly_backends_cp("cp_tensor2", None)
def check_cp_tensor_equal(cp_tensor1, cp_tensor2, ignore_labels=False):
    """Check if the factor matrices and weights are equal.

    This will check if the factor matrices and weights are exactly equal
    to one another. It will not check if the two decompositions are equivalent.
    For example, if ``cp_tensor2`` contain the same factors as ``cp_tensor1``,
    but permuted, or with the weights distributed differently between the
    modes, then this function will return False. To check for equivalence,
    use ``check_cp_tensors_equivalent``.

    Parameters
    ----------
    cp_tensor1 : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument
    cp_tensor2 : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument
    ignore_labels : bool
        If True, then labels (i.e. DataFrame column names and indices) can differ.

    Returns
    -------
    bool
        Whether the decompositions are equal.

    Examples
    --------
    ``check_cp_tensor_equal`` checks for strict equality of the factor matrices and
    weights.

    >>> from tlviz.data import simulated_random_cp_tensor
    >>> from tlviz.factor_tools import check_cp_tensor_equal
    >>> cp_tensor, dataset = simulated_random_cp_tensor((10, 20, 30), 3, seed=0)
    >>> check_cp_tensor_equal(cp_tensor, cp_tensor)
    True

    But it does not check the identity of the decompositions, only their numerical values

    >>> cp_tensor2, dataset2 = simulated_random_cp_tensor((10, 20, 30), 3, seed=0)
    >>> check_cp_tensor_equal(cp_tensor, cp_tensor2)
    True

    Normalising a ``cp_tensor`` changes its values, so then we do not have strict equality
    of the factor matrices, even though the decomposition is equivalent

    >>> from tlviz.factor_tools import normalise_cp_tensor
    >>> normalised_cp_tensor = normalise_cp_tensor(cp_tensor)
    >>> check_cp_tensor_equal(cp_tensor, normalised_cp_tensor)
    False

    Permutations will also make the numerical values of the``cp_tensor`` change

    >>> from tlviz.factor_tools import permute_cp_tensor
    >>> check_cp_tensor_equal(cp_tensor, permute_cp_tensor(cp_tensor, permutation=[1, 2, 0]))
    False

    See Also
    --------
    check_cp_tensors_equivalent : Function for checking if two CP tensors represent the same dense tensor.
    """
    validate_cp_tensor(cp_tensor1)
    validate_cp_tensor(cp_tensor2)

    rank = cp_tensor1[1][0].shape[1]
    num_modes = len(cp_tensor1[1])

    if rank != cp_tensor2[1][0].shape[1]:
        return False
    if num_modes != len(cp_tensor2[1]):
        return False

    # Check weights
    if cp_tensor1[0] is None and cp_tensor2[0] is not None:
        return False
    if cp_tensor1[0] is not None and cp_tensor2[0] is None:
        return False
    if not np.all(cp_tensor1[0] == cp_tensor2[0]):
        return False

    for mode in range(num_modes):
        if not check_factor_matrix_equal(cp_tensor1[1][mode], cp_tensor2[1][mode], ignore_labels=ignore_labels):
            return False
    return True


@_handle_tensorly_backends_dataset("factor_matrix1", None)
@_handle_tensorly_backends_dataset("factor_matrix2", None)
def check_factor_matrix_close(factor_matrix1, factor_matrix2, rtol=1e-5, atol=1e-8, ignore_labels=False):
    """Check that all entries in a factor matrix are close, if labelled, then label equality is also checked.

    This function is similar to ``numpy.allclose``, but works on both labelled and unlabelled factor
    matrices. If the factor matrices are labelled, then the DataFrame index and columns are also
    compared (unless ``ignore_labels=True``).

    Parameters
    ----------
    factor_matrix1 : numpy.ndarray or pandas.DataFrame
        Labelled or unlabelled factor matrix
    cp_tensor2 : CPTensor or tuple
        Labelled or unlabelled factor matrix
    rtol : float
        Relative tolerance (see ``numpy.allclose``)
    atol : float
        Absolute tolerance (see ``numpy.allclose``)
    ignore_labels : bool
        If True, then labels (i.e. DataFrame column names and indices) can differ.

    Returns
    -------
    bool
        Whether the decompositions are equivalent.

    Examples
    --------
    ``check_factor_matrix_close`` checks if two factor matrices are close up to round off errors.

    >>> from tlviz.data import simulated_random_cp_tensor
    >>> import numpy as np
    >>> A = np.arange(6).reshape(3, 2).astype(float)
    >>> B = A + 1e-10
    >>> check_factor_matrix_close(A, B)
    True

    If we make only one of them into a DataFrame, then the factor matrices are not close

    >>> import pandas as pd
    >>> A_labelled = pd.DataFrame(A)
    >>> check_factor_matrix_close(A_labelled, B)
    False
    >>> check_factor_matrix_close(B, A_labelled)
    False

    If we turn B into a DataFrame too, it passes again

    >>> B_labelled = pd.DataFrame(A)
    >>> check_factor_matrix_close(A_labelled, B_labelled)
    True

    The index is checked for equality, so if we change the index of ``B_labelled``, then
    the factor matrices are not close

    >>> B_labelled.index += 1
    >>> check_factor_matrix_close(A_labelled, B_labelled)
    False

    However, we can disable checking the labels by using the ``ignore_labels`` argument

    >>> check_factor_matrix_close(A_labelled, B_labelled, ignore_labels=True)
    True
    """
    if is_dataframe(factor_matrix1) != is_dataframe(factor_matrix2) and not ignore_labels:
        return False
    if ignore_labels and is_dataframe(factor_matrix1):
        factor_matrix1 = factor_matrix1.values
    if ignore_labels and is_dataframe(factor_matrix2):
        factor_matrix2 = factor_matrix2.values

    if is_dataframe(factor_matrix1):
        try:
            pd.testing.assert_frame_equal(factor_matrix1, factor_matrix2, rtol=rtol, atol=atol)
        except AssertionError:
            return False
        else:
            return True

    return np.allclose(factor_matrix1, factor_matrix2, rtol=rtol, atol=atol)


@_handle_tensorly_backends_cp("cp_tensor1", None)
@_handle_tensorly_backends_cp("cp_tensor2", None)
@_handle_none_weights_cp_tensor("cp_tensor1")
@_handle_none_weights_cp_tensor("cp_tensor2")
def check_cp_tensors_equivalent(cp_tensor1, cp_tensor2, rtol=1e-5, atol=1e-8, ignore_labels=False):
    """Check if the decompositions are equivalent

    This will check if the factor matrices and weights are equivalent. That is
    if they represent the same tensor. This differs from checking equality in
    the sense that if ``cp_tensor2`` contain the same factors as ``cp_tensor1``,
    but permuted, or with the weights distributed differently between the
    modes, then they are not equal, but equivalent.

    Parameters
    ----------
    cp_tensor1 : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument
    cp_tensor2 : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument
    rtol : float
        Relative tolerance (see ``numpy.allclose``)
    atol : float
        Absolute tolerance (see ``numpy.allclose``)
    ignore_labels : bool
        If True, then labels (i.e. DataFrame column names and indices) can differ.

    Returns
    -------
    bool
        Whether the decompositions are equivalent.

    Examples
    --------
    ``check_cp_tensors_equivalent`` checks if two CP tensors represent the same dense tensor

    >>> from tlviz.data import simulated_random_cp_tensor
    >>> from tlviz.factor_tools import check_cp_tensors_equivalent
    >>> cp_tensor, dataset = simulated_random_cp_tensor((10, 20, 30), 3, seed=0)
    >>> cp_tensor2, dataset2 = simulated_random_cp_tensor((10, 20, 30), 3, seed=0)
    >>> check_cp_tensors_equivalent(cp_tensor, cp_tensor2)
    True

    Normalising a ``cp_tensor`` changes its values, but not which dense tensor it represents

    >>> from tlviz.factor_tools import normalise_cp_tensor
    >>> normalised_cp_tensor = normalise_cp_tensor(cp_tensor)
    >>> check_cp_tensors_equivalent(cp_tensor, normalised_cp_tensor)
    True

    Permutations will also make the numerical values of the``cp_tensor`` change but not the
    dense tensor it represents

    >>> from tlviz.factor_tools import permute_cp_tensor
    >>> check_cp_tensors_equivalent(cp_tensor, permute_cp_tensor(cp_tensor, permutation=[1, 2, 0]))
    True

    See Also
    --------
    check_cp_tensor_equivalent : Function for checking if two CP tensors have the same
        numerical value (have equal weights and factor matrices)
    """
    validate_cp_tensor(cp_tensor1)
    validate_cp_tensor(cp_tensor2)

    rank = cp_tensor1[1][0].shape[1]
    num_modes = len(cp_tensor1[1])

    if rank != cp_tensor2[1][0].shape[1]:
        return False
    if num_modes != len(cp_tensor2[1]):
        return False

    for mode in range(num_modes):
        if not cp_tensor1[1][mode].shape == cp_tensor2[1][mode].shape:
            return False

    cp_tensor2 = permute_cp_tensor(cp_tensor2, reference_cp_tensor=cp_tensor1)

    cp_tensor1 = normalise_cp_tensor(cp_tensor1)
    cp_tensor2 = normalise_cp_tensor(cp_tensor2)

    if not np.allclose(cp_tensor1[0], cp_tensor2[0], rtol=rtol, atol=atol):
        return False
    for mode in range(num_modes):
        if not check_factor_matrix_close(
            cp_tensor1[1][mode], cp_tensor2[1][mode], rtol=rtol, atol=atol, ignore_labels=ignore_labels
        ):
            return False

    return True


@_handle_tensorly_backends_cp("cp_tensor", None)
@_handle_tensorly_backends_dataset("dataset", None)
@_handle_labelled_cp("cp_tensor", None)
@_handle_labelled_dataset("dataset", None, optional=True)
@_handle_none_weights_cp_tensor("cp_tensor")
def percentage_variation(cp_tensor, dataset=None, method="model"):
    r"""Compute the percentage of variation captured by each component.

    The (possible) non-orthogonality of CP factor matrices makes it less straightforward
    to estimate the amount of variation captured by each component, compared to a model with
    orthogonal factors. To estimate the amount of variation captured by a single component,
    we therefore use the following formula:

    .. math::

        \text{fit}_i = \frac{\text{SS}_i}{SS_\mathbf{\mathcal{X}}}

    where :math:`\text{SS}_i` is the squared norm of the tensor constructed using only the
    i-th component, and :math:`SS_\mathbf{\mathcal{X}}` is the squared norm of the data
    tensor :cite:p:`plstoolbox.varcap`. If ``method="data"``, then :math:`SS_\mathbf{\mathcal{X}}`
    is the squared norm of the tensor constructed from the CP tensor using all factor matrices.

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument
    dataset : np.ndarray
        Data tensor that the cp_tensor is fitted against
    method : {"data", "model", "both"} (default="model")
        Which method to use for computing the fit.

    Returns
    -------
    fit : float or tuple
        The fit (depending on the method). If ``method="both"``, then a tuple is returned
        where the first element is the fit computed against the data tensor and the second
        element is the fit computed against the model.

    Examples
    --------
    There are two ways of computing the percentage variation. One method is to divide by the variation
    in the data, giving us the percentage variation of the data captured by each component. This
    approach will not necessarily sum to 100 since

     1. the model will not explain all the variation.
     2. the components are likely not orthogonal

    Alternatively, we can divide by the variation in the model, which will give us the contribution
    of each component to the model. However, this may also not sum to 100 since the components may
    not be orthogonal.

    >>> from tlviz.data import simulated_random_cp_tensor
    >>> from tlviz.factor_tools import percentage_variation
    >>> cp_tensor, X = simulated_random_cp_tensor((30, 10, 10), 5, noise_level=0.3, seed=0)
    >>> print(percentage_variation(cp_tensor).astype(int))
    [11  2  0  0 39]
    >>> print(percentage_variation(cp_tensor, X, method="data").astype(int))
    [11  2  0  0 37]

    We see that the variation captured for each component sums to 50 when we compare with the
    data and 52 when we compare with the model. These low numbers are because the components
    are not orthogonal, which means that the magnitude of the data is not equal to the sum
    of the magnitudes of each component. We can also compute the percentage variation with
    the model and the data simultaneously:

    >>> percent_var_data, percent_var_model = percentage_variation(cp_tensor, X, method="both")
    >>> print(percent_var_data.astype(int))
    [11  2  0  0 37]
    >>> print(percent_var_model.astype(int))
    [11  2  0  0 39]

    If noise level is 0, both methods should give the same variantion percentages:

    >>> cp_tensor, X = simulated_random_cp_tensor((30, 10, 10), 5, noise_level=0.0, seed=1)
    >>> percent_var_data, percent_var_model = percentage_variation(cp_tensor, X, method="both")
    >>> print(percent_var_data.astype(int))
    [ 3 11  0 34  1]
    >>> print(f"Sum of variation: {percent_var_data.sum():.0f}")
    Sum of variation: 51
    >>> print(percent_var_model.astype(int))
    [ 3 11  0 34  1]
    >>> print(f"Sum of variation: {percent_var_model.sum():.0f}")
    Sum of variation: 51
    """
    weights, factor_matrices = cp_tensor
    ssc = weights ** 2

    if dataset is not None and method == "model":
        warn(
            'Dataset provided but method="model", so it is not used. To compute the variation'
            + ' captured in the data, use method="data" or method="both".'
        )

    for factor_matrix in factor_matrices:
        ssc = ssc * np.sum(np.abs(factor_matrix) ** 2, axis=0)

    if method == "data":
        if dataset is None:
            raise TypeError("The dataset must be provided if method='data'")
        return 100 * ssc / np.sum(dataset ** 2)
    elif method == "model":
        return 100 * ssc / (cp_norm(cp_tensor) ** 2)
    elif method == "both":
        return 100 * ssc / np.sum(dataset ** 2), 100 * ssc / (cp_norm(cp_tensor) ** 2)
    else:
        raise ValueError("Method must be either 'data', 'model' or 'both")

"""Utility functions for analysing tensor factorisation models

This module contains functions that are useful when inspecting tensor factorisation
models. For example, comparing two factorisations, or constructing
a CP tensor.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from ._utils import extract_singleton


def normalise(x, axis=0):
    """Normalise a matrix so all columns have unit norm.

    Parameters
    ----------
    x : np.ndarray
        Matrix (or vector/tensor) to normalise.
    axis : int
        Axis along which to normalise, if 0, then all columns will have unit norm
        and if 1 then all rows will have unit norm.

    Returns
    -------
    np.ndarray
        Normalised matrix
    """
    return x / np.linalg.norm(x, axis=axis, keepdims=True)


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


def get_permutation(factor_matrix1, factor_matrix2, ignore_sign=True):
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

    Returns
    -------
    permutation : np.ndarray(dtype=int)
    """
    congruence_product = normalise(factor_matrix1).T @ normalise(factor_matrix2)
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

    Returns
    -------
    fms : float
        The factor match score
    permutation : list of ints (only if return_permutation=True)
        The permutation of cp_tensor2

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
        if hasattr(factor1, "values"):
            factor1 = factor1.values
        if hasattr(factor2, "values"):
            factor2 = factor2.values

        if i == skip_axis:
            continue
        if consider_weights:
            norms1 *= np.linalg.norm(factor1, axis=0)
            norms2 *= np.linalg.norm(factor2, axis=0)
        congruence_product *= normalise(factor1).T @ normalise(factor2)

    if consider_weights:
        congruence_product *= 1 - np.abs(
            norms1[:, np.newaxis] - norms2[np.newaxis, :]
        ) / np.maximum(norms1[:, np.newaxis], norms2[np.newaxis, :])

    if absolute_value:
        congruence_product = np.abs(congruence_product)

    row_index, column_index = linear_sum_assignment(-congruence_product)
    congruence_product = congruence_product[row_index, column_index]

    if not return_permutation:
        return congruence_product.mean()

    permutation = np.zeros_like(row_index)
    permutation[row_index] = column_index
    return congruence_product.mean(), permutation


# TODO: Move all functions below to different modules
def degeneracy_score(cp_tensor):
    r"""Compute the degeneracy score for a given decomposition.

    PARAFAC models can be degenerate. For a third order tensor, this
    means that the triple cosine of two components can approach -1.
    That is

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
    may approach infinity.

    Degenerate solutions typically signify that the decomposition is unreliable,
    and that one should take care before interpreting the components.

    There are several strategies to avoid degenerate solutions:

     * Fitting models with more random initialisations
     * Decreasing the convergence tolerance or increasing the number of iterations
     * Imposing non-negativity constraints
     * Change number of components

    The latter of these strategies work well for data where non-negativity
    constraints are sensible, as non-negative tensor decompositions cannot
    be degenerate.

    To measure degeneracy, we compute the degeneracy score, which is the
    minimum triple cosine (for a third-order tensor). A score close to
    -1 signifies a degenerate solution. A score of -0.85 is an indication
    of a troublesome model :cite:p:`krijnen1993analysis` (as cited in
    :cite:p:`bro1997parafac`).

    For more information about degeneracy for component models see
    :cite:p:`zijlstra2002degenerate` and :cite:p:`bro1997parafac`.

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
    """
    # TODO: Example
    # TODO: Find rule of thumbs for degenerate solutions
    # TODO: Find cites
    # TODO: Move to model evaluation?
    weights, factors = cp_tensor
    rank = factors[0].shape[1]
    tucker_congruence_scores = np.ones(shape=(rank, rank))

    for factor in factors:
        tucker_congruence_scores *= normalise(factor).T @ normalise(factor)

    return np.asarray(tucker_congruence_scores).min()


# TODO: Rename these to be named cp_to_tensor and tucker_to_tensor? or cp_to_dense and tucker_to_dense?
def construct_cp_tensor(cp_tensor):
    """Construct a CP tensor, equivalent to ``cp_to_tensor`` in TensorLy, but supports dataframes.

    If the factor matrices are data frames, then the tensor will be returned as a labelled
    xarray. Otherwise, it will be returned as a numpy array.

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument.

    Returns
    -------
    xarray or np.ndarray
        Dense tensor represented by the decomposition.
    """
    # TODO: Handle dataframes
    # TODO: Docstring for construct_cp_tensor
    # TODO: Reconsider name
    # TODO: Move to utils?
    # TODO: Tests (1 component for example)
    if cp_tensor[0] is None:
        weights = np.ones(cp_tensor[1][0].shape[1])
    else:
        weights = cp_tensor[0].reshape(-1)

    einsum_input = "R"
    einsum_output = ""
    for mode in range(len(cp_tensor[1])):
        idx = chr(ord("a") + mode)

        # We cannot use einsum with letters outside the alphabet
        if ord(idx) > ord("z"):
            max_modes = ord("a") - ord("z") - 1
            raise ValueError(
                f"Cannot have more than {max_modes} modes. Current components have {len(cp_tensor[1])}."
            )

        einsum_input += f", {idx}R"
        einsum_output += idx

    return np.einsum(f"{einsum_input} -> {einsum_output}", weights, *cp_tensor[1])


def construct_tucker_tensor(tucker_tensor):
    """Construct a CP tensor, equivalent to ``tucker_to_tensor`` in TensorLy, but supports dataframes.

    If the factor matrices are data frames, then the tensor will be returned as a labelled
    xarray. Otherwise, it will be returned as a numpy array.

    Parameters
    ----------
    tucker : CPTensor or tuple
        TensorLy-style TuckerTensor object or tuple with weights as first
        argument and a tuple of components as second argument.

    Returns
    -------
    xarray or np.ndarray
        Dense tensor represented by the decomposition.
    """
    # TODO: Rename
    # TODO: Docstring for construct_tucker_tensor
    # TODO: Reconsider name
    # TODO: Move to utils?
    # TODO: Handle dataframes
    einsum_core = ""
    einsum_input = ""
    einsum_output = ""

    for mode in range(len(tucker_tensor[1])):
        idx = chr(ord("a") + mode)
        rank_idx = chr(ord("A") + mode)

        # We cannot use einsum with letters outside the alphabet
        if ord(idx) > ord("z"):
            max_modes = ord("a") - ord("z")
            raise ValueError(
                f"Cannot have more than {max_modes} modes. Current components have {len(tucker_tensor[1])}."
            )

        einsum_core += rank_idx
        einsum_input += f", {idx}{rank_idx}"
        einsum_output += idx

    return np.einsum(
        f"{einsum_core}{einsum_input} -> {einsum_output}",
        tucker_tensor[0],
        *tucker_tensor[1],
    )


# TODO: Test for check_cp_tensors_equals
# TODO: Move check_cp_tensors_equals and check_cp_tensors_equivalent to different module
def check_cp_tensors_equals(cp_tensor1, cp_tensor2):
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

    Returns
    -------
    bool
        Whether the decompositions are equal.
    """
    # (e.g. if factors are permuted, then)
    # TODO: Handle dataframes
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
        if not cp_tensor1[1][mode].shape == cp_tensor2[1][mode].shape:
            return False
        if not np.all(cp_tensor1[1][mode] == cp_tensor2[1][mode]):
            return False
    return True


def check_cp_tensors_equivalent(cp_tensor1, cp_tensor2, rtol=1e-5, atol=1e-8):
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

    Returns
    -------
    bool
        Whether the decompositions are equivalent.
    """
    from . import postprocessing  # HACK: Avoiding circular dependencies

    # TODO: Handle dataframes

    rank = cp_tensor1[1][0].shape[1]
    num_modes = len(cp_tensor1[1])

    if rank != cp_tensor2[1][0].shape[1]:
        return False
    if num_modes != len(cp_tensor2[1]):
        return False

    for mode in range(num_modes):
        if not cp_tensor1[1][mode].shape == cp_tensor2[1][mode].shape:
            return False

    cp_tensor2 = postprocessing.permute_cp_tensor(
        cp_tensor2, reference_cp_tensor=cp_tensor1
    )

    cp_tensor1 = postprocessing.normalise_cp_tensor(cp_tensor1)
    cp_tensor2 = postprocessing.normalise_cp_tensor(cp_tensor2)

    if not np.allclose(cp_tensor1[0], cp_tensor2[0], rtol=rtol, atol=atol):
        return False
    for mode in range(num_modes):
        if not np.allclose(
            cp_tensor1[1][mode], cp_tensor2[1][mode], rtol=rtol, atol=atol
        ):
            return False

    return True

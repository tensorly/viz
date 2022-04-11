"""
This module contains most functions that only work on tensor factorisation models, not data
tensors. The module contains functions that are useful for inspecting tensor factorisation
models. For example, computing how similar two factorisations are, checking if two decompositions
are equivalent, or simply generating a dense tensor from a (possibly) labelled decomposition.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from component_vis.xarray_wrapper import _SINGLETON, _handle_labelled_cp

from .model_evaluation import percentage_variation
from .utils import _alias_mode_axis, extract_singleton, normalise


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

    Arguments
    ---------
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
    # TOTEST: distribute_weights
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
        columns are aligned with ``component_vis.factor_tools.NO_COLUMN`` (a slice that slices nothing).

    Returns
    -------
    permutation : list[int | slice]
        List of ints used to permute ``factor_matrix2`` so its columns optimally align with ``factor_matrix1``.
        If the ``factor_matrix1`` has a column with no corresponding column in ``factor_matrix2`` (i.e. there
        are fewer columns in ``factor_matrix2`` than in ``factor_matrix1``), then
        ``component_vis.factor_tools.NO_COLUMN`` (a slice that slices nothing) is used to indicate missing columns.

    Raises
    ------
    ValueError
        If ``allow_smaller_rank=False`` and ``factor_matrix2`` has fewer columns than ``factor_matrix1``.
    """
    congruence_product = normalise(factor_matrix1).T @ normalise(factor_matrix2)
    if ignore_sign:
        congruence_product = np.abs(congruence_product)

    return _get_linear_sum_assignment_permutation(congruence_product, allow_smaller_rank=allow_smaller_rank)[-1]


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
        with ``component_vis.factor_tools.component_vis.factor_tools.NO_COLUMN`` (a slice that slices nothing).

    Returns
    -------
    fms : float
        The factor match score
    permutation : list[int | object] (only if return_permutation=True)
        List of ints used to permute ``cp_tensor2`` so its components optimally align with ``cp_tensor1``.
        If the ``cp_tensor1`` has a component with no corresponding component in ``cp_tensor2`` (i.e. there
        are fewer components in ``cp_tensor2`` than in ``cp_tensor1``), then
        ``component_vis.factor_tools.NO_COLUMN`` (a slice that slices nothing) is used to indicate missing components.

    Raises
    ------
    ValueError
        If ``allow_smaller_rank=False`` and ``cp_tensor2`` has fewer components than ``cp_tensor1``.

    Examples
    --------
    >>> import numpy as np
    >>> from component_vis.factor_tools import factor_match_score
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
    # TODOC: Example for degeneracy_score
    # TODOC: Add cite about ill-posedness and how it spawns degenerate solutions
    weights, factors = cp_tensor
    rank = factors[0].shape[1]
    tucker_congruence_scores = np.ones(shape=(rank, rank))

    for factor in factors:
        tucker_congruence_scores *= normalise(factor).T @ normalise(factor)

    return np.asarray(tucker_congruence_scores).min()


# TODO: Handle labelled cp?
# TODO: Add nan columns
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
    # TOTEST: get_cp_permutation
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


@_handle_labelled_cp("cp_tensor", _SINGLETON, preserve_columns=False)
def permute_cp_tensor(
    cp_tensor, reference_cp_tensor=None, permutation=None, consider_weights=True, allow_smaller_rank=False
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
    reference_cp_tensor : CPTensor or tuple (optional)
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument. The tensor
        that ``cp_tensor`` is aligned with. Either this or the ``permutation``
        argument must be passed, not both.
    permutation : tuple (optional)
        Tuple with the column permutations. Either this or the ``reference_cp_tensor``
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


    Examples
    --------
    ``check_cp_tensors_equivalent`` checks if two CP tensors represent the same dense tensor

    >>> from component_vis.data import simulated_random_cp_tensor
    >>> from component_vis.factor_tools import check_cp_tensors_equivalent
    >>> cp_tensor, dataset = simulated_random_cp_tensor((10, 20, 30), 3, seed=0)
    >>> cp_tensor2, dataset2 = simulated_random_cp_tensor((10, 20, 30), 3, seed=0)
    >>> check_cp_tensors_equivalent(cp_tensor, cp_tensor2)
    True

    Normalising a ``cp_tensor`` changes its values, but not which dense tensor it represents

    >>> from component_vis.factor_tools import normalise_cp_tensor
    >>> normalised_cp_tensor = normalise_cp_tensor(cp_tensor)
    >>> check_cp_tensors_equivalent(cp_tensor, normalised_cp_tensor)
    True

    Permutations will also make the numerical values of the``cp_tensor`` change but not the
    dense tensor it represents

    >>> from component_vis.factor_tools import permute_cp_tensor
    >>> check_cp_tensors_equivalent(cp_tensor, permute_cp_tensor(cp_tensor, permutation=[1, 2, 0]))
    True

    See Also
    --------
    check_cp_tensors_equals : Function for checking if two CP tensors have the same
	numerical value (have equal weights and factor matrices)
    """
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


    Examples
    --------
    ``check_cp_tensors_equals`` checks for strict equality of the factor matrices and
    weights.

    >>> from component_vis.data import simulated_random_cp_tensor
    >>> from component_vis.factor_tools import check_cp_tensors_equals
    >>> cp_tensor, dataset = simulated_random_cp_tensor((10, 20, 30), 3, seed=0)
    >>> check_cp_tensors_equals(cp_tensor, cp_tensor)
    True

    But it does not check the identity of the decompositions, only their numerical values

    >>> cp_tensor2, dataset2 = simulated_random_cp_tensor((10, 20, 30), 3, seed=0)
    >>> check_cp_tensors_equals(cp_tensor, cp_tensor2)
    True

    Normalising a ``cp_tensor`` changes its values, so then we do not have strict equality
    of the factor matrices, even though the decomposition is equivalent

    >>> from component_vis.factor_tools import normalise_cp_tensor
    >>> normalised_cp_tensor = normalise_cp_tensor(cp_tensor)
    >>> check_cp_tensors_equals(cp_tensor, normalised_cp_tensor)
    False

    Permutations will also make the numerical values of the``cp_tensor`` change

    >>> from component_vis.factor_tools import permute_cp_tensor
    >>> check_cp_tensors_equals(cp_tensor, permute_cp_tensor(cp_tensor, permutation=[1, 2, 0]))
    False

    See Also
    --------
    check_cp_tensors_equivalent : Function for checking if two CP tensors represent the same dense tensor.
    """
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

    cp_tensor2 = permute_cp_tensor(cp_tensor2, reference_cp_tensor=cp_tensor1)

    cp_tensor1 = normalise_cp_tensor(cp_tensor1)
    cp_tensor2 = normalise_cp_tensor(cp_tensor2)

    if not np.allclose(cp_tensor1[0], cp_tensor2[0], rtol=rtol, atol=atol):
        return False
    for mode in range(num_modes):
        if not np.allclose(cp_tensor1[1][mode], cp_tensor2[1][mode], rtol=rtol, atol=atol):
            return False

    return True

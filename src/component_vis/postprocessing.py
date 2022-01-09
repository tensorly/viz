import numpy as np
import scipy.linalg as sla

from ._utils import is_iterable, unfold_tensor
from .factor_tools import factor_match_score
from .model_evaluation import percentage_variation
from .xarray_wrapper import label_cp_tensor


# TODO: Fix naming, what is resolve mode and what is flip mode?
def resolve_cp_sign_indeterminacy(
    cp_tensor, dataset, resolve_mode=None, unresolved_mode=-1, method="transpose"
):
    r"""Resolve the sign indeterminacy of CP models.

    Tensor factorisations have a sign indeterminacy that allows any change in the
    sign of the component vectors in one mode, under the condition that the sign
    of a component vector in another mode changes as well. This means that we can
    "flip" any component vector so long as the corresponding component vector in
    another mode is also flipped. This flipping can hurt the model's interpretability.
    For example, if a factor represents a chemical spectrum, then this flipping may lead
    to it being negative instead of positive.

    To illustrate the sign indeterminacy, we start with the SVD, which is on the form

    .. math::

        \mathbf{X} = \mathbf{U} \mathbf{S} \mathbf{V}^\mathsf{T}.

    The factorisation above is equivalent with the following factorisation:

    .. math::
        \mathbf{X} = (\mathbf{U} \text{diag}(\mathbf{f})) \mathbf{S} (\mathbf{V} \text{diag}(\mathbf{f}))^\mathsf{T},

    where :math:`\mathbf{f}` is a vector containing only ones or zeros. Similarly,
    a CP factorisation with factor matrices :math:`\mathbf{A}, \mathbf{B}` and :math:`\mathbf{C}`
    is equivalent to the CP factorisations with the following factor matrices:

     * :math:`(\mathbf{A} \text{diag}(\mathbf{f})), (\mathbf{B} \text{diag}(\mathbf{f}))` and :math:`\mathbf{C}`
     * :math:`(\mathbf{A} \text{diag}(\mathbf{f})), \mathbf{B}` and :math:`(\mathbf{C} \text{diag}(\mathbf{f}))`
     * :math:`\mathbf{A}, (\mathbf{B} \text{diag}(\mathbf{f}))` and :math:`(\mathbf{C} \text{diag}(\mathbf{f}))`

    One way to circumvent the sign indeterminacy is by imposing non-negativity. However,
    that is not always a reasonable choice (e.g. if the data also contains negative entries).
    When we don't want to impose non-negativity constraints, then we need some other way to
    resolve the sign indeterminacy (which this function provides). The idea is easiest described
    in the two-way (matrix) case.

    Consider a data matrix, :math:`\mathbf{X}` whose columns represent samples and rows represent
    measurements. Then, we want the measurement-mode component-vectors to be mostly aligned with
    the data matrix. The components should describe what the data is, not what it is not.
    For example, if the data is non-negative, then the measurement-mode component vectors should
    be mostly non-negative. With the SVD, we can compute whether we should flip the :math:`r`-th
    column of :math:`\mathbf{U}` by computing

    .. math::

        f_r = \sum_{i=1^I} v_{ir}^2 \text{sign}{v_{ir}}

    if :math:`f_r` is negative, then we should flip the sign of the :math:`r-th` column of
    :math:`\mathbf{U}` and :math:`\mathbf{V}` :cite:p:`bro2008resolving`.

    The methodology above works well in practice, and is rooted in the fact that the
    :math:`i`-th row of :math:`\mathbf{V}` can be interpreted as the coordinates of the
    :math:`i`-th row of :math:`\mathbf{X}` in a vector space spanned by the columns of
    :math:`\mathbf{U}`. Then, the above equation will give us component vectors where the data
    points is mainly located in the non-negative orthant.

    The above interpretation is correct under the assumption: :math:`\mathbf{U}^\mathsf{T}\mathbf{U} = \mathbf{I}`.
    However, the heuristic still works well when this is not the case :cite:p:`bro2013solving`.
    Still, we also include a modification of the above scheme where the same interpretation
    holds with non-orthogonal factors.

    .. math::

        f_r = \sum_{i=1}^I h_{ir}^2 \text{sign}(h_{ir}),

    where :math:`\mathbf{H} = \mathbf{U}(\mathbf{U}^\mathsf{T}\mathbf{U})^{-1} \mathbf{X}`.
    That is the rows of :math:`\mathbf{H}` represent the rows of :math:`\mathbf{X}` as
    described by the column basis of :math:`\mathbf{U}`.

    In the multiway case, when :math:`\mathcal{X}` is a tensor instead of a matrix, we can
    apply the same logic :cite:p:`bro2013solving`. If we have the factor matrices 
    :math:`\mathbf{A}, \mathbf{B}` and :math:`\mathbf{C}`, then we flip the sign of any 
    factor matrix (e.g. :math:`\mathbf{A}`) by computing

    .. math::

        f_r^{(\mathbf{A})} = \sum_{i=1}^I {h_{ir}^{(\mathbf{A})}}^2 \text{sign}({h_{ir}^{(\mathbf{A})}}),

    where :math:`\mathbf{H}^{(\mathbf{A})} = \mathbf{A}^\mathsf{T} \mathbf{X}_{(0)}` or
    :math:`\mathbf{H}^{(\mathbf{A})} = \mathbf{A}(\mathbf{A}^\mathsf{T}\mathbf{A})^{-1} \mathbf{X}_{(0)}`,
    depending on whether the scheme based on the SVD scheme :cite:p:`bro2013solving` or the
    corrected scheme. :math:`\mathbf{X}_{(0)} \in \mathbb{R}^{I \times JK}` is the tensor,
    :math:`\mathcal{X}`, unfolded along the first mode. We can then correct the sign of 
    :math:`\mathbf{A}` by multiplying and one of the other factor matrices by
    :math:`\text{diag}(\mathbf{f}^{(\mathbf{A})})`. By using this procedure, we can align all
    factor matrices except for one (the unresolved mode) with the "direction of the data".

    Note that this sign indeterminacy comes as a direct consequence of the scaling indeterminacy
    of component models, since :math:`\text{diag}(\mathbf{f})^{-1} = \text{diag}(\mathbf{f})`.

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument.
    resolve_mode : int, iterable or None
        Mode(s) whose factor matrix should be aligned with the data. If
        None, then the sign should be corrected for all modes except the
        ``unresolved_mode``.
    unresolved_mode : int
        Mode used to correct the sign indeterminacy in other mode(s). The
        factor matrix in this mode may not be aligned with the data. 
    method : "transpose" or "positive_coord"
        Which method to use when computing the signs. Use ``"transpose"``
        for the method in :cite:p:`bro2008resolving,bro2013solving`, and
        ``"positive_coord"`` for the method corrected for non-orthogonal
        factor matrices described above.

    Returns
    -------
    CPTensor or tuple
        The CP tensor after correcting the signs.

    Raises
    ------
    ValueError
        If ``unresolved_mode`` is not between ``-dataset.ndim`` and ``dataset.ndim-1``.
    ValueError
        If ``unresolved_mode`` is in ``resolve_mode``

    Notes
    -----
    For more information, see :cite:p:`bro2008resolving,bro2013solving`
    """
    if unresolved_mode < 0:
        unresolved_mode = dataset.ndim + unresolved_mode
    if unresolved_mode > dataset.ndim or unresolved_mode < 0:
        raise ValueError(
            "`unresolved_mode` must be between `-dataset.ndim` and `dataset.ndim-1`."
        )
    if is_iterable(resolve_mode) and unresolved_mode in resolve_mode:
        raise ValueError("The unresolved mode cannot be resolved.")

    if resolve_mode is None:
        resolve_mode = range(dataset.ndim)

    if is_iterable(resolve_mode):
        for mode in resolve_mode:
            if mode != unresolved_mode:
                cp_tensor = resolve_cp_sign_indeterminacy(
                    cp_tensor,
                    dataset,
                    unresolved_mode=unresolved_mode,
                    resolve_mode=mode,
                    method=method,
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

    signs = np.sign(np.sum(sign_scores ** 2 * np.sign(sign_scores), axis=1))
    signs = np.asarray(signs).reshape(1, -1)

    factor_matrices = list(cp_tensor[1])
    factor_matrices[resolve_mode] = factor_matrices[resolve_mode] * signs
    factor_matrices[unresolved_mode] = factor_matrices[unresolved_mode] * signs
    return cp_tensor[0], tuple(factor_matrices)


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
    # TODO: test for distribute_weights_evenly
    weights, factors = normalise_cp_tensor(cp_tensor)
    weights = weights ** (1 / 3)
    for factor in factors:
        factor[:] *= weights
    weights = np.ones_like(weights)
    return weights, factors


def distribute_weights_in_one_mode(cp_tensor, mode):
    """Normalise all factors and multiply the weights into one mode.

    The CP tensor is scaled so all factor matrices except one have unit norm
    columns and the weight-vector contains only ones.

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument.
    mode : int
        Which mode to store the weights in

    Returns
    -------
    tuple
        The scaled CP tensor.
    """
    # TODO: test for distribute_weights_in_one_mode
    weights, factors = normalise_cp_tensor(cp_tensor)
    factors[mode][:] *= weights
    return np.ones_like(weights), factors


# TODO: Should we name this reference_cp_tensor or target_cp_tensor?
def _permute_cp_tensor(cp_tensor, permutation):
    # TODO: Handle dataframes

    weights, factors = cp_tensor

    if weights is not None:
        new_weights = weights.copy()[permutation]
    else:
        new_weights = None

    new_factors = [None] * len(factors)
    for mode, factor in enumerate(factors):
        new_factor = factor.copy()
        if hasattr(factor, "values"):
            new_factor.values[:] = new_factor.values[:, permutation]
        else:
            new_factor[:] = new_factor[:, permutation]
        new_factors[mode] = new_factor

    return new_weights, new_factors


# TODO: Should we name this reference_cp_tensor or target_cp_tensor?
def permute_cp_tensor(
    cp_tensor, reference_cp_tensor=None, permutation=None, consider_weights=True
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
        raise ValueError(
            "Must either provide a permutation, a reference CP tensor or neither. Both is provided"
        )
    # TODO: test for permute_cp_tensor

    if permutation is None and reference_cp_tensor is not None:
        fms, permutation = factor_match_score(
            reference_cp_tensor,
            cp_tensor,
            consider_weights=consider_weights,
            return_permutation=True,
        )
    elif permutation is None:
        variation = percentage_variation(cp_tensor, method="model")
        permutation = sorted(range(len(variation)), key=lambda i: -variation[i])

    rank = cp_tensor[1][0].shape[1]
    if len(permutation) != rank:
        remaining_indices = sorted(set(range(rank)) - set(permutation))
        permutation = list(permutation) + remaining_indices

    return _permute_cp_tensor(cp_tensor, permutation)


def postprocess(
    cp_tensor,
    reference_cp_tensor=None,
    dataset=None,
    resolve_mode=None,
    unresolved_mode=-1,
    flip_method="transpose",
):
    """Standard postprocessing of a CP decomposition.

    This function will perform standard postprocessing of a CP decomposition.
    If a reference CP tensor is provided, then the columns of ``cp_tensor``'s
    factor matrices are aligned with the columns of ``reference_cp_tensor``'s 
    factor matrices.

    Next, the CP tensor is normalised so the columns of all factor matrices have
    unit norm.

    If a dataset is provided, then the sign indeterminacy is resolved and if the
    dataset is labelled (i.e. is an xarray), then the factor matrices of the CP
    tensor is labelled too.

    This function is equivalent to calling

     * ``permute_cp_tensor``
     * ``normalise_cp_tensor``
     * ``resolve_cp_sign_inditerminacy``
     * ``label_cp_tensor``

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        CPTensor to postprocess
    reference_cp_tensor : CPTensor or tuple (optional)
        If provided, then the tensor whose factors we align the CP tensor's
        columns with.
    dataset : ndarray or xarray (optional)
        Dataset the CP tensor represents
    resolve_mode : int, iterable or None
        Mode(s) whose factor matrix should be aligned with the data. If
        None, then the sign should be corrected for all modes except the
        ``unresolved_mode``.
    unresolved_mode : int
        Mode used to correct the sign indeterminacy in other mode(s). The
        factor matrix in this mode may not be aligned with the data. 
    method : "transpose" or "positive_coord"
        Which method to use when computing the signs. Use ``"transpose"``
        for the method in :cite:p:`bro2008resolving,bro2013solving`, and
        ``"positive_coord"`` for the method corrected for non-orthogonal
        factor matrices described above.
    
    Returns
    -------
    CPTensor
        The post processed CPTensor.
    """
    # TODO: Docstring for postprocess
    # TODO: Unit test for postprocess
    cp_tensor = permute_cp_tensor(cp_tensor, reference_cp_tensor=reference_cp_tensor)
    cp_tensor = normalise_cp_tensor(cp_tensor)

    if dataset is not None:
        cp_tensor = resolve_cp_sign_indeterminacy(
            cp_tensor,
            dataset,
            resolve_mode=resolve_mode,
            unresolved_mode=unresolved_mode,
            method=flip_method,
        )
        cp_tensor = label_cp_tensor(cp_tensor, dataset)

    return cp_tensor

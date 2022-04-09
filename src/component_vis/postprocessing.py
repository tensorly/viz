import numpy as np
import scipy.linalg as sla

from . import factor_tools
from ._module_utils import is_iterable
from .utils import unfold_tensor
from .xarray_wrapper import (
    _SINGLETON,
    _handle_labelled_cp,
    _handle_labelled_dataset,
    label_cp_tensor,
)


# TODO: Fix naming, what is resolve mode and what is flip mode?
@_handle_labelled_dataset("dataset", None)
@_handle_labelled_cp("cp_tensor", _SINGLETON)
def resolve_cp_sign_indeterminacy(cp_tensor, dataset, resolve_mode=None, unresolved_mode=-1, method="transpose"):
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
        raise ValueError("`unresolved_mode` must be between `-dataset.ndim` and `dataset.ndim-1`.")
    if is_iterable(resolve_mode) and unresolved_mode in resolve_mode:
        raise ValueError("The unresolved mode cannot be resolved.")

    if resolve_mode is None:
        resolve_mode = range(dataset.ndim)

    if is_iterable(resolve_mode):
        for mode in resolve_mode:
            if mode != unresolved_mode:
                cp_tensor = resolve_cp_sign_indeterminacy(
                    cp_tensor, dataset, unresolved_mode=unresolved_mode, resolve_mode=mode, method=method,
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


@_handle_labelled_cp("reference_cp_tensor", None, optional=True)
@_handle_labelled_cp("cp_tensor", None)
def postprocess(
    cp_tensor,
    dataset=None,
    reference_cp_tensor=None,
    resolve_mode=None,
    unresolved_mode=-1,
    flip_method="transpose",
    weight_behaviour="normalise",
    weight_mode=-1,
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
     * ``resolve_cp_sign_indeterminacy``
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
    CPTensor
        The post processed CPTensor.
    """
    # TODO: Docstring example for postprocess
    # TODO: Unit test for postprocess
    cp_tensor = factor_tools.permute_cp_tensor(cp_tensor, reference_cp_tensor=reference_cp_tensor)

    if weight_behaviour == "ignore":
        pass
    elif weight_behaviour == "normalise":
        cp_tensor = factor_tools.normalise_cp_tensor(cp_tensor)
    elif weight_behaviour == "evenly":
        cp_tensor = factor_tools.distribute_weights_evenly(cp_tensor)
    elif weight_behaviour == "one_mode":
        cp_tensor = factor_tools.distribute_weights_in_one_mode(cp_tensor, weight_mode)
    else:
        raise ValueError("weight_behaviour must be either 'ignore', 'normalise', 'evenly', or 'one_mode'")

    if dataset is not None:
        cp_tensor = resolve_cp_sign_indeterminacy(
            cp_tensor, dataset, resolve_mode=resolve_mode, unresolved_mode=unresolved_mode, method=flip_method,
        )
        cp_tensor = label_cp_tensor(cp_tensor, dataset)

    return cp_tensor

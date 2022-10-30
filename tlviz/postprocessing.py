# -*- coding: utf-8 -*-

__author__ = "Marie Roald & Yngve Mardal Moe"

from warnings import warn

import numpy as np
import scipy.linalg as sla

from . import factor_tools
from ._module_utils import is_iterable
from ._tl_utils import _handle_tensorly_backends_cp, _handle_tensorly_backends_dataset
from ._xarray_wrapper import (
    _SINGLETON,
    _handle_labelled_cp,
    _handle_labelled_dataset,
    add_factor_metadata,
    label_cp_tensor,
)
from .utils import unfold_tensor

__all__ = [
    "resolve_cp_sign_indeterminacy",
    "postprocess",
    "factor_matrix_to_tidy",
    "add_factor_metadata",
    "label_cp_tensor",
]


@_handle_tensorly_backends_dataset("dataset", None)
@_handle_tensorly_backends_cp("cp_tensor", _SINGLETON)
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

    where :math:`\mathbf{f}` is a vector containing only ones or negative ones. Similarly,
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
    if unresolved_mode >= dataset.ndim or unresolved_mode < 0:
        raise ValueError("`unresolved_mode` must be between `-dataset.ndim` and `dataset.ndim-1`.")
    if resolve_mode == unresolved_mode or (is_iterable(resolve_mode) and unresolved_mode in resolve_mode):
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

    signs = np.sign(np.sum(sign_scores**2 * np.sign(sign_scores), axis=1))
    signs = np.asarray(signs).reshape(1, -1)

    factor_matrices = list(cp_tensor[1])
    factor_matrices[resolve_mode] = factor_matrices[resolve_mode] * signs
    factor_matrices[unresolved_mode] = factor_matrices[unresolved_mode] * signs
    return cp_tensor[0], tuple(factor_matrices)


@_handle_tensorly_backends_cp("reference_cp_tensor", None, optional=True)
@_handle_tensorly_backends_cp("cp_tensor", None)
@_handle_labelled_cp("reference_cp_tensor", None, optional=True)
def postprocess(
    cp_tensor,
    dataset=None,
    reference_cp_tensor=None,
    permute=True,
    resolve_mode=None,
    unresolved_mode=-1,
    flip_method="transpose",
    weight_behaviour="normalise",
    weight_mode=0,
    allow_smaller_rank=False,
    include_metadata=False,
):
    """Standard postprocessing of a CP decomposition.

    This function will perform standard postprocessing of a CP decomposition.
    If a reference CP tensor is provided, then the columns of ``cp_tensor``'s
    factor matrices are aligned with the columns of ``reference_cp_tensor``'s
    factor matrices.

    Next, the factor matrices of the CP tensor are scaled according the the specified
    weight behaviour (default is normalised).

    If a dataset is provided, then the sign indeterminacy is resolved and if the
    dataset is labelled (i.e. is an xarray or a dataframe), then the factor matrices
    of the CP tensor is labelled too.

    This function is equivalent to calling

     * ``permute_cp_tensor``
     * ``distribute_weights``
     * ``resolve_cp_sign_indeterminacy``
     * ``label_cp_tensor``

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        CPTensor to postprocess
    dataset : ndarray or xarray (optional)
        Dataset the CP tensor represents
    reference_cp_tensor : CPTensor or tuple (optional)
        If provided, then the tensor whose factors we align the CP tensor's
        columns with.
    permute : bool
        If ``True``, then the factors are permuted in descending weight order if
        ``reference_cp_tensor`` is ``None``.
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
    allow_smaller_rank : bool (default=False)
        If ``True``, then a low rank decomposition can be permuted against one with higher rank. The "missing columns"
        are padded by nan values
    include_metadata : bool (default=True)
        If ``True``, then the factor metadata will be added as columns in the factor matrices.


    Returns
    -------
    CPTensor
        The post processed CPTensor.

    See Also
    --------
    tlviz.factor_tools.permute_cp_tensor
    tlviz.factor_tools.distribute_weights
    tlviz.postprocessing.resolve_cp_sign_indeterminacy
    tlviz.postprocessing.label_cp_tensor

    Examples
    --------

    Here is an example were we use postprocess on a decomposition of aminoacid data

    .. plot::
        :context: close-figs
        :include-source:

        >>> import tlviz
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from tensorly.decomposition import parafac
        >>> dataset = tlviz.data.load_aminoacids()
        Loading Aminoacids dataset from:
        Bro, R, PARAFAC: Tutorial and applications, Chemometrics and Intelligent Laboratory Systems, 1997, 38, 149-171

        The dataset is an xarray DataArray and it contains relevant side information

        >>> print(type(dataset))
        <class 'xarray.core.dataarray.DataArray'>

        We see that after postprocessing, the cp_tensor contains pandas DataFrames

        >>> cp_tensor = parafac(dataset.data, 3, init="random", random_state=0)
        >>> cp_tensor_postprocessed = tlviz.postprocessing.postprocess(cp_tensor, dataset)
        >>> print(type(cp_tensor[1][0]))
        <class 'numpy.ndarray'>
        >>> print(type(cp_tensor_postprocessed[1][0]))
        <class 'pandas.core.frame.DataFrame'>


        We see that after postprocessing, the factor matrix has unit norm

        >>> print(np.linalg.norm(cp_tensor[1][0], axis=0))
        [160.82985402 182.37338941 125.3689186 ]
        >>> print(np.linalg.norm(cp_tensor_postprocessed[1][0], axis=0))
        [1. 1. 1.]

        When we construct a dense tensor from a postprocessed cp_tensor it is constructed
        as an xarray DataArray

        >>> print(type(tlviz.utils.cp_to_tensor(cp_tensor)))
        <class 'numpy.ndarray'>
        >>> print(type(tlviz.utils.cp_to_tensor(cp_tensor_postprocessed)))
        <class 'xarray.core.dataarray.DataArray'>

        The visualisation of the postprocessed cp_tensor shows that the scaling and sign indeterminacy
        is taken care of and x-xaxis has correct labels and ticks

        >>> fig, ax = tlviz.visualisation.components_plot(cp_tensor)
        >>> plt.show()

        >>> fig, ax = tlviz.visualisation.components_plot(cp_tensor_postprocessed)
        >>> plt.show()

    """
    if not permute and reference_cp_tensor is not None:
        warn("``permute=False`` is ignored if a reference CP tensor is provided.")

    if permute or reference_cp_tensor is not None:
        cp_tensor = factor_tools.permute_cp_tensor(
            cp_tensor, reference_cp_tensor=reference_cp_tensor, allow_smaller_rank=allow_smaller_rank
        )

    cp_tensor = factor_tools.distribute_weights(cp_tensor, weight_behaviour=weight_behaviour, weight_mode=weight_mode)

    if dataset is not None:
        cp_tensor = resolve_cp_sign_indeterminacy(
            cp_tensor,
            dataset,
            resolve_mode=resolve_mode,
            unresolved_mode=unresolved_mode,
            method=flip_method,
        )

        cp_tensor = label_cp_tensor(cp_tensor, dataset)

    if include_metadata and dataset is not None:
        cp_tensor = add_factor_metadata(cp_tensor, dataset)
    elif include_metadata:
        warn("Cannot include metadata when there is no provided dataset")

    return cp_tensor


def factor_matrix_to_tidy(factor_matrix, var_name="Component", value_name="Signal", id_vars=None):
    """Convert a factor matrix into a tidy dataset, for use with Plotly Express.

    If we convert a factor matrix into a tidy dataset (or long table), then we get a table on the form

    .. list-table:: Tidy format factor matrix
        :widths: 25 25 25
        :header-rows: 1

        * - Index
          - Component
          - Signal
        * - 0
          - 0
          - 0.1
        * - 1
          - 0
          - 0.5
        * - ...
          - ...
          - ...
        * - 38
          - 2
          - 0.7
        * - 39
          - 2
          - 0.2

    The component vectors are all stacked on top of each other, with a separate column that specifies which
    component each row belongs to. This function can also preserve metadata, which is signified by columns
    that have non-integer column names. For example, if we have a dataframe on the form

    .. list-table:: Factor matrix with metadata
        :widths: 25 25 25 25 25 25
        :header-rows: 1

        * - Index
          - 0
          - 1
          - 2
          - lat
          - lon
        * - 0
          - 0.1
          - 0.2
          - 0.5
          - 59.91273
          - 10.74609
        * - 1
          - 0.5
          - 0.2
          - 0.1
          - 63.43049
          - 10.39506
        * - ...
          - ...
          - ...
          - ...
          - ...
          - ...
        * - 5
          - 0.2
          - 0.1
          - 0.3
          - 60.39299
          - 5.32415
        * - 5
          - 0.0
          - 0.2
          - 0.1
          - 58.97005
          - 5.73332

    and convert it into a tidy format factor matrix, then we get a table on the form

    .. list-table:: Tidy format factor matrix with metadata
        :widths: 25 25 25 25 25
        :header-rows: 1

        * - Index
          - lat
          - lon
          - Component
          - Signal
        * - 0
          - 59.91273
          - 10.74609
          - 0
          - 0.1
        * - 1
          - 63.43049
          - 10.39506
          - 0
          - 0.5
        * - ...
          - ...
          - ...
          - ...
          - ...
        * - 4
          - 69.6489
          - 18.95508
          - 2
          - 0.0
        * - 5
          - 58.97005
          - 5.73332
          - 2
          - 0.1

    Parameters
    ----------
    factor_matrix : pd.DataFrame
        A labelled factor matrix potentially with metadata columns
    var_name : str
        Name of the column that specifies which component each row belongs to
    value_name : str
        Name of the column that holds the magnitude of each component entry
    id_vars : iterable or None (default=None)
        Which columns to interpret as metadata. The columns not specified here are considered as the components.
        If ``id_vars is None``, then all columns with non-integer names are considered metadata columns.

    Returns
    -------
    pd.DataFrame
        Tidy format factor matrix

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from tlviz.postprocessing import factor_matrix_to_tidy
    >>> rng = np.random.default_rng(0)
    >>> factor_matrix = pd.DataFrame(rng.uniform(size=(10, 3)))
    >>> factor_matrix.head()
              0         1         2
    0  0.636962  0.269787  0.040974
    1  0.016528  0.813270  0.912756
    2  0.606636  0.729497  0.543625
    3  0.935072  0.815854  0.002739
    4  0.857404  0.033586  0.729655
    >>> tidy_factor_matrix = factor_matrix_to_tidy(factor_matrix)
    >>> tidy_factor_matrix.head()
       index Component    Signal
    0      0         0  0.636962
    1      1         0  0.016528
    2      2         0  0.606636
    3      3         0  0.935072
    4      4         0  0.857404
    >>> factor_matrix_with_metadata = factor_matrix.copy()
    >>> factor_matrix_with_metadata["Metadata"] = rng.uniform(size=10)
    >>> factor_matrix_with_metadata.head()
              0         1         2  Metadata
    0  0.636962  0.269787  0.040974  0.688447
    1  0.016528  0.813270  0.912756  0.388921
    2  0.606636  0.729497  0.543625  0.135097
    3  0.935072  0.815854  0.002739  0.721488
    4  0.857404  0.033586  0.729655  0.525354
    >>> tidy_factor_matrix_with_metadata = factor_matrix_to_tidy(factor_matrix_with_metadata)
    >>> tidy_factor_matrix_with_metadata.head()
       Metadata  index Component    Signal
    0  0.688447      0         0  0.636962
    1  0.388921      1         0  0.016528
    2  0.135097      2         0  0.606636
    3  0.721488      3         0  0.935072
    4  0.525354      4         0  0.857404
    """
    factor_matrix = factor_matrix.reset_index()
    if id_vars is None:
        id_vars = set()
        for column in factor_matrix.columns:
            if type(column) != int:
                id_vars.add(column)
    id_vars = sorted(id_vars)

    return factor_matrix.melt(var_name=var_name, value_name=value_name, id_vars=id_vars)

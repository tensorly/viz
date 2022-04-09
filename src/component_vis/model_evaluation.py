"""
This module contains functions used to evaluate a single tensor factorisation model
by comparing it to a data tensor.
"""
import numpy as np
import scipy.linalg as sla

from .utils import _alias_mode_axis, cp_to_tensor
from .xarray_wrapper import (
    _handle_labelled_cp,
    _handle_labelled_dataset,
    _handle_none_weights_cp_tensor,
)


@_handle_labelled_dataset("X", None)
def estimate_core_tensor(factors, X):
    """Efficient estimation of the Tucker core from a factor matrices and a data tensor.

    Parameters
    ----------
    factors : tuple
        Tuple of factor matrices used to estimate the core tensor from
    X : np.ndarray
        The data tensor that the core tensor is estimated from

    Notes
    -----
    In the original paper, :cite:t:`papalexakis2015fast` present an algorithm
    for 3-way tensors. However, it is straightforward to generalise it to N-way tensors
    by using the inverse tensor product formula in :cite:p:`buis1996efficient`.
    """
    svds = [sla.svd(factor, full_matrices=False) for factor in factors]
    for U, s, Vh in svds[::-1]:
        X = np.tensordot(U.T, X, (1, X.ndim - 1))
    for U, s, Vh in svds[::-1]:
        s_pinv = s.copy()
        mask = s_pinv != 0
        s_pinv[mask] = 1 / s_pinv[mask]
        X = np.tensordot(np.diag(s_pinv), X, (1, X.ndim - 1))
    for U, s, Vh in svds[::-1]:
        X = np.tensordot(Vh.T, X, (1, X.ndim - 1))
    return np.ascontiguousarray(X)


@_handle_labelled_dataset("X", None)
@_handle_labelled_cp("cp_tensor", None)
def core_consistency(cp_tensor, X, normalised=False):
    r"""Computes the core consistency :cite:p:`bro2003new`

    A CP model can be interpreted as a restricted Tucker model, where the
    core tensor is constrained to be superdiagonal. For a third order tensor,
    this means that the core tensor, :math:`\mathcal{G}`, satisfy :math:`g_{ijk}\neq0`
    only if :math:`i = j = k`. To compute the core consistency of a CP decomposition,
    we use this property, and calculate the optimal Tucker core tensor given
    the factor matrices of the CP model.

    The key observation is that if the data tensor follows the assumptions
    of the CP model, then the optimal core tensor should be similar to that
    of the CP model, i. e. superdiagonal. However, if the data can be better
    described by allowing for interactions between the components across modes,
    then the core tensor will have non-zero off-diagonal. The core consistency
    quantifies this measure and is defined as:

    .. math::

        \text{CC} = 100 - 100 \frac{\| \mathcal{G} - \mathcal{I} \|_F^2}{N}

    where :math:`\mathcal{G}` is the estimated core tensor, :math:`\mathcal{I}`
    is a superdiagonal tensor only ones on the superdiagonal and :math:`N`
    is a normalising factor, either equal to the number of components or the
    squared frobenius norm of the estimated core tensor. A core consistency
    score close to 100 indicates that the CP model is likely valid. If the
    core consistency is low, however, then the model either has components
    that describe noise or the data does not follow the model's assumptions.
    So the core consistency can help determine if the chosen number of
    components is suitable.

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument
    X : np.ndarray
        Data tensor that the cp_tensor is fitted against
    normalised : Bool (default=False)
        If True, then the squared frobenius norm of the estimated core tensor
        is used to normalise the core consistency. Otherwise, the number of
        components is used.

        If ``normalised=False``, then the core consistency formula coincides
        with :cite:p:`bro2003new`, and if ``normalised=True``, the core consistency
        formula coincides with that used in the `N-Way toolbox <http://models.life.ku.dk/nwaytoolbox>`_,
        and is unlikely to be less than 0. For core consistencies close to
        100, the formulas approximately coincide.

    Returns
    -------
    float
        The core consistency

    Examples
    --------
    We can use the core consistency diagonstic to determine the correct number of components
    for a CP model. Here, we only fit one model, but in practice, you should fit multiple models
    and select the one with the lowest SSE (to account for local minima) before computing the
    core consistency.

    >>> cp_tensor = tensorly.random.random_cp(shape=(4,5,6), rank=3, random_state=42)
    ... X = cp_tensor.to_tensor()
    ... # Fit many CP models with different number of components
    ... for rank in range(1, 5):
    ...     decomposition = tl.decomposition.parafac(X, rank=rank, random_state=42)
    ...     cc = core_consistency(decomposition, X, normalised=True)
    ...     print(f"No. components: {rank} - core consistency: {cc}")
    No. components: 1 - core consistency: 100.0
    No. components: 2 - core consistency: 99.99971253658768
    No. components: 3 - core consistency: 99.99977773119056
    No. components: 4 - core consistency: -1.4210854715202004e-14

    Notes
    -----
    This implementation uses the fast method of estimating the core tensor
    :cite:p:`papalexakis2015fast,buis1996efficient`
    """
    # Distribute weights
    weights, factors = cp_tensor
    rank = factors[0].shape[1]

    A = factors[0].copy()
    if weights is not None:
        A *= weights.reshape(1, -1)

    factors = tuple((A, *factors[1:]))

    # Estimate core and compare
    G = estimate_core_tensor(factors, X)
    T = np.zeros([rank] * X.ndim)
    np.fill_diagonal(T, 1)
    if normalised:
        denom = np.sum(G ** 2)
    else:
        denom = rank

    return 100 - 100 * np.sum((G - T) ** 2) / denom


@_handle_labelled_dataset("X", None)
@_handle_labelled_cp("cp_tensor", None)
def sse(cp_tensor, X):
    """Compute the sum of squared error for a given cp_tensor.

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument
    X : ndarray
        Tensor approximated by ``cp_tensor``

    Returns
    -------
    float
        The sum of squared error, ``sum((X_hat - X)**2)``, where ``X_hat``
        is the dense tensor represented by ``cp_tensor``

    Examples
    --------
    Below, we create a random CP tensor and a random tensor and compute
    the sum of squared error for these two tensors.

    >>> import tensorly as tl
    >>> from tensorly.random import random_cp
    >>> from component_vis.model_evaluation import sse
    >>> rng = tl.check_random_state(0)
    >>> cp = random_cp((4, 5, 6), 3, random_state=rng)
    >>> X = rng.random_sample((4, 5, 6))
    >>> sse(cp, X)
    18.948918157419186
    """
    X_hat = cp_to_tensor(cp_tensor)
    return np.sum((X - X_hat) ** 2)


@_handle_labelled_dataset("X", None)
@_handle_labelled_cp("cp_tensor", None)
def relative_sse(cp_tensor, X, sum_squared_X=None):
    """Compute the relative sum of squared error for a given cp_tensor.

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument
    X : ndarray
        Tensor approximated by ``cp_tensor``
    sum_squared_X: float (optional)
        If ``sum(X**2)`` is already computed, you can optionally provide it
        using this argument to avoid unnecessary recalculation.

    Returns
    -------
    float
        The relative sum of squared error, ``sum((X_hat - X)**2)/sum(X**2)``,
        where ``X_hat`` is the dense tensor represented by ``cp_tensor``

    Examples
    --------
    Below, we create a random CP tensor and a random tensor and compute
    the sum of squared error for these two tensors.

    >>> import tensorly as tl
    >>> from tensorly.random import random_cp
    >>> from component_vis.model_evaluation import relative_sse
    >>> rng = tl.check_random_state(0)
    >>> cp = random_cp((4, 5, 6), 3, random_state=rng)
    >>> X = rng.random_sample((4, 5, 6))
    >>> relative_sse(cp, X)
    0.4817407254961442
    """
    if sum_squared_X is None:
        sum_squared_x = np.sum(X ** 2)
    return sse(cp_tensor, X) / sum_squared_x


@_handle_labelled_dataset("X", None)
@_handle_labelled_cp("cp_tensor", None)
def fit(cp_tensor, X, sum_squared_X=None):
    """Compute the fit (1-relative sum squared error) for a given cp_tensor.

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument
    X : ndarray
        Tensor approximated by ``cp_tensor``
    sum_squared_X: float (optional)
        If ``sum(X**2)`` is already computed, you can optionally provide it
        using this argument to avoid unnecessary recalculation.

    Returns
    -------
    float
        The relative sum of squared error, ``sum((X_hat - X)**2)/sum(X**2)``,
        where ``X_hat`` is the dense tensor represented by ``cp_tensor``

    Examples
    --------
    Below, we create a random CP tensor and a random tensor and compute
    the sum of squared error for these two tensors.

    >>> import tensorly as tl
    >>> from tensorly.random import random_cp
    >>> from component_vis.model_evaluation import fit
    >>> rng = tl.check_random_state(0)
    >>> cp = random_cp((4, 5, 6), 3, random_state=rng)
    >>> X = rng.random_sample((4, 5, 6))
    >>> fit(cp, X)
    0.5182592745038558

    We can see that it is equal to 1 - relative SSE

    >>> from component_vis.model_evaluation import relative_sse
    >>> 1 - relative_sse(cp, X)
    0.5182592745038558
    """
    return 1 - relative_sse(cp_tensor, X, sum_squared_X=sum_squared_X)


@_alias_mode_axis()
def predictive_power(cp_tensor, y, sklearn_estimator, mode=0, metric=None, axis=None):
    """Use scikit-learn estimator to evaluate the predictive power of a factor matrix.

    This is useful if you evaluate the components based on their predictive
    power with respect to some task.

    Parameters
    ----------
    factor_matrix : ndarray(ndim=2)
        Factor matrix from a tensor decomposition model
    y : ndarray(ndim=1)
        Prediction target for each row of the factor matrix in the given mode.
        ``y`` should have same length as the first dimension of this factor
        matrix (i.e. the length of the tensor along the given mode).
    sklearn_estimator : scikit learn estimator
        Scikit learn estimator. Must have the ``fit`` and ``predict`` methods,
        and if ``metric`` is ``None``, then it should also have the ``score``
        method. See https://scikit-learn.org/stable/developers/develop.html.
    mode : int
        Which mode to perform the scoring along
    metric : Callable
        Callable (typically function) with the signature ``metric(y_true, y_pred)``,
        where ``y_true=labels`` and ``y_pred`` is the predicted values
        obtained from ``sklearn_estimator``. See
        https://scikit-learn.org/stable/developers/develop.html#specific-models.
    axis : int (optional)
        Alias for mode, if set, then mode cannot be set.

    Returns
    -------
    float
        Score based on the estimator's performance.
    """
    # TOTEST: test for predictive_power
    # TODOC: example for predictive_power
    factor_matrix = cp_tensor[1][mode]
    sklearn_estimator.fit(factor_matrix, y)
    if metric is None:
        return sklearn_estimator.score(factor_matrix, y)
    return metric(y, sklearn_estimator.predict(factor_matrix))


@_handle_labelled_cp("cp_tensor", None)
@_handle_labelled_dataset("X", None, optional=True)
@_handle_none_weights_cp_tensor("cp_tensor")
def percentage_variation(cp_tensor, X=None, method="data"):
    r"""Compute the percentage of variation captured by each component.

    The (possible) non-orthogonality of CP factor matrices makes it less straightforward
    to estimate the amount of variation captured by each component, compared to a model with
    orthogonal factors. To estimate the amount of variation captured by a single component,
    we therefore use the following formula:

    .. math::

        \text{fit}_i = \frac{\text{SS}_i}{SS_\mathbf{\mathcal{X}}}

    where :math:`\text{SS}_i` is the squared norm of the tensor constructed using only the
    i-th component, and :math:`SS_\mathbf{\mathcal{X}}` is the squared norm of the data
    tensor. If ``method="data"``, then :math:`SS_\mathbf{\mathcal{X}}` is the squared
    norm of the tensor constructed from the CP tensor using all factor matrices.

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument
    X : np.ndarray
        Data tensor that the cp_tensor is fitted against
    method : {"data", "model", "both"}
        Which method to use for computing the fit.

    Returns
    -------
    fit : float or tuple
        The fit (depending on the method). If ``method="both"``, then a tuple is returned
        where the first element is the fit computed against the data tensor and the second
        element is the fit computed against the model.
    """
    # TODOC: Examples for percentage_variation
    # TOTEST: Unit tests for percentage_variation. Use orthogonal components in all modes
    # TOTEST: Unit test for percentage_variation - Should sum to 100
    # TODO: There is something wrong here...
    weights, factor_matrices = cp_tensor
    rank = factor_matrices[0].shape[1]
    if weights is not None:
        ssc = weights.copy()
    else:
        ssc = np.ones(rank)

    for factor_matrix in factor_matrices:
        ssc = np.sum(ssc * np.abs(factor_matrix) ** 2)

    if method == "data":
        if X is None:
            raise TypeError("The dataset must be provided if ``method='data'``")
        return 100 * ssc / np.sum(X ** 2)
    elif method == "model":
        return 100 * ssc / np.sum(ssc)
    elif method == "both":
        return 100 * ssc / np.sum(X ** 2), 100 * ssc / np.sum(ssc)
    else:
        raise ValueError("Method must be either 'data', 'model' or 'both")

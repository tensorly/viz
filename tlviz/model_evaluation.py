# -*- coding: utf-8 -*-

__author__ = "Marie Roald & Yngve Mardal Moe"

"""
This module contains functions used to evaluate a single tensor factorisation model
by comparing it to a data tensor.
"""
import numpy as np
import scipy.linalg as sla

from ._tl_utils import _handle_tensorly_backends_cp, _handle_tensorly_backends_dataset, to_numpy
from ._xarray_wrapper import _handle_labelled_cp, _handle_labelled_dataset
from .utils import _alias_mode_axis, cp_to_tensor

__all__ = [
    "estimate_core_tensor",
    "core_consistency",
    "sse",
    "relative_sse",
    "fit",
    "predictive_power",
]


@_handle_tensorly_backends_dataset("dataset", None)
@_handle_labelled_dataset("dataset", None)
def estimate_core_tensor(factors, dataset):
    """Efficient estimation of the Tucker core from a factor matrices and a data tensor.

    Parameters
    ----------
    factors : tuple
        Tuple of factor matrices used to estimate the core tensor from
    dataset : np.ndarray
        The data tensor that the core tensor is estimated from

    Notes
    -----
    In the original paper, :cite:t:`papalexakis2015fast` present an algorithm
    for 3-way tensors. However, it is straightforward to generalise it to N-way tensors
    by using the inverse tensor product formula in :cite:p:`buis1996efficient`.
    """
    factors = [to_numpy(factor, cast_labelled=True) for factor in factors]

    svds = [sla.svd(factor, full_matrices=False) for factor in factors]
    for U, s, Vh in svds[::-1]:
        dataset = np.tensordot(U.T, dataset, (1, dataset.ndim - 1))
    for U, s, Vh in svds[::-1]:
        s_pinv = s.copy()
        mask = s_pinv != 0
        s_pinv[mask] = 1 / s_pinv[mask]
        dataset = np.tensordot(np.diag(s_pinv), dataset, (1, dataset.ndim - 1))
    for U, s, Vh in svds[::-1]:
        dataset = np.tensordot(Vh.T, dataset, (1, dataset.ndim - 1))
    return np.ascontiguousarray(dataset)


@_handle_tensorly_backends_dataset("dataset", None)
@_handle_tensorly_backends_cp("cp_tensor", None)
@_handle_labelled_dataset("dataset", None)
@_handle_labelled_cp("cp_tensor", None)
def core_consistency(cp_tensor, dataset, normalised=False):
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
    dataset : np.ndarray
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

    >>> from tlviz.data import simulated_random_cp_tensor
    >>> from tensorly.decomposition import parafac
    >>> cp_tensor, dataset = simulated_random_cp_tensor((10,11,12), 3, seed=42)
    >>> # Fit many CP models with different number of components
    >>> for rank in range(1, 5):
    ...     decomposition = parafac(dataset, rank=rank, random_state=42)
    ...     cc = core_consistency(decomposition, dataset, normalised=True)
    ...     print(f"No. components: {rank} - core consistency: {cc:.0f}")
    No. components: 1 - core consistency: 100
    No. components: 2 - core consistency: 100
    No. components: 3 - core consistency: 81
    No. components: 4 - core consistency: 0

    .. note::

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
    G = estimate_core_tensor(factors, dataset)
    T = np.zeros([rank] * dataset.ndim)
    np.fill_diagonal(T, 1)
    if normalised:
        denom = np.sum(G**2)
    else:
        denom = rank

    return 100 - 100 * np.sum((G - T) ** 2) / denom


@_handle_tensorly_backends_dataset("dataset", None)
@_handle_tensorly_backends_cp("cp_tensor", None)
@_handle_labelled_dataset("dataset", None)
@_handle_labelled_cp("cp_tensor", None)
def sse(cp_tensor, dataset):
    """Compute the sum of squared error for a given cp_tensor.

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument
    dataset : ndarray
        Tensor approximated by ``cp_tensor``

    Returns
    -------
    float
        The sum of squared error, ``sum((X_hat - dataset)**2)``, where ``X_hat``
        is the dense tensor represented by ``cp_tensor``

    Examples
    --------
    Below, we create a random CP tensor and a random tensor and compute
    the sum of squared error for these two tensors.

    >>> import tensorly as tl
    >>> from tensorly.random import random_cp
    >>> from tlviz.model_evaluation import sse
    >>> rng = tl.check_random_state(0)
    >>> cp = random_cp((4, 5, 6), 3, random_state=rng)
    >>> X = rng.random_sample((4, 5, 6))
    >>> sse(cp, X)
    18.948918157419186
    """
    X_hat = cp_to_tensor(cp_tensor)
    return np.sum((dataset - X_hat) ** 2)


@_handle_tensorly_backends_dataset("dataset", None)
@_handle_tensorly_backends_cp("cp_tensor", None)
@_handle_labelled_dataset("dataset", None)
@_handle_labelled_cp("cp_tensor", None)
def relative_sse(cp_tensor, dataset, sum_squared_dataset=None):
    """Compute the relative sum of squared error for a given cp_tensor.

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument
    dataset : ndarray
        Tensor approximated by ``cp_tensor``
    sum_squared_dataset: float (optional)
        If ``sum(dataset**2)`` is already computed, you can optionally provide it
        using this argument to avoid unnecessary recalculation.

    Returns
    -------
    float
        The relative sum of squared error, ``sum((X_hat - dataset)**2)/sum(dataset**2)``,
        where ``X_hat`` is the dense tensor represented by ``cp_tensor``

    Examples
    --------
    Below, we create a random CP tensor and a random tensor and compute
    the sum of squared error for these two tensors.

    >>> import tensorly as tl
    >>> from tensorly.random import random_cp
    >>> from tlviz.model_evaluation import relative_sse
    >>> rng = tl.check_random_state(0)
    >>> cp = random_cp((4, 5, 6), 3, random_state=rng)
    >>> X = rng.random_sample((4, 5, 6))
    >>> relative_sse(cp, X)
    0.4817407254961442
    """
    if sum_squared_dataset is None:
        sum_squared_x = np.sum(dataset**2)
    return sse(cp_tensor, dataset) / sum_squared_x


@_handle_tensorly_backends_dataset("dataset", None)
@_handle_tensorly_backends_cp("cp_tensor", None)
@_handle_labelled_dataset("dataset", None)
@_handle_labelled_cp("cp_tensor", None)
def fit(cp_tensor, dataset, sum_squared_dataset=None):
    """Compute the fit (1-relative sum squared error) for a given cp_tensor.

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument
    dataset : ndarray
        Tensor approximated by ``cp_tensor``
    sum_squared_dataset: float (optional)
        If ``sum(dataset**2)`` is already computed, you can optionally provide it
        using this argument to avoid unnecessary recalculation.

    Returns
    -------
    float
        The relative sum of squared error, ``sum((X_hat - dataset)**2)/sum(dataset**2)``,
        where ``X_hat`` is the dense tensor represented by ``cp_tensor``

    Examples
    --------
    Below, we create a random CP tensor and a random tensor and compute
    the sum of squared error for these two tensors.

    >>> import tensorly as tl
    >>> from tensorly.random import random_cp
    >>> from tlviz.model_evaluation import fit
    >>> rng = tl.check_random_state(0)
    >>> cp = random_cp((4, 5, 6), 3, random_state=rng)
    >>> X = rng.random_sample((4, 5, 6))
    >>> fit(cp, X)
    0.5182592745038558

    We can see that it is equal to 1 - relative SSE

    >>> from tlviz.model_evaluation import relative_sse
    >>> 1 - relative_sse(cp, X)
    0.5182592745038558
    """
    return 1 - relative_sse(cp_tensor, dataset, sum_squared_dataset=sum_squared_dataset)


@_alias_mode_axis()
@_handle_tensorly_backends_cp("cp_tensor", None)
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

    Examples
    --------
    ``predictive_power`` can be useful to evaluate the predictive power of a CP decomposition.
    To illustrate this, we start by creating a simulated CP tensor and a variable we want to
    predict that is linearly related to one of the factor matrices.

    >>> from tlviz.data import simulated_random_cp_tensor
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> cp_tensor, X = simulated_random_cp_tensor((30, 10, 10), 3, noise_level=0.1, seed=rng)
    >>> weights, (A, B, C) = cp_tensor
    >>> regression_coefficients = rng.standard_normal((3, 1))
    >>> Y = A @ regression_coefficients

    Next, we fit a PARAFAC model to this data

    >>> from tensorly.decomposition import parafac
    >>> est_cp_tensor = parafac(X, 3)

    Finally, we see how well the estimated decomposition can describe our target variable, ``Y``.
    This will use the :math:`R^2`-coefficient for scoring, as that is the default scoring method
    for linear models.

    >>> from sklearn.linear_model import LinearRegression
    >>> from tlviz.model_evaluation import predictive_power
    >>> linear_regression = LinearRegression()
    >>> r_squared = predictive_power(cp_tensor, Y, linear_regression)
    >>> print(f"The R^2 coefficient is {r_squared:.2f}")
    The R^2 coefficient is 1.00

    We can also specify our own scoring function

    >>> from sklearn.metrics import max_error
    >>> highest_error = predictive_power(cp_tensor, Y, linear_regression, metric=max_error)
    >>> print(f"The maximum error is {highest_error:.2f}")
    The maximum error is 0.00
    """
    factor_matrix = cp_tensor[1][mode]
    sklearn_estimator.fit(factor_matrix, y)
    if metric is None:
        return sklearn_estimator.score(factor_matrix, y)
    return metric(y, sklearn_estimator.predict(factor_matrix))

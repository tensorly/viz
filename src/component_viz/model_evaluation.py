"""Methods to evaluate a single tensor factorisation model.
"""
import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt
from itertools import product


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
        s_pinv[mask] = 1/s_pinv[mask]
        X = np.tensordot(np.diag(s_pinv), X, (1, X.ndim - 1))
    for U, s, Vh in svds[::-1]:
        X = np.tensordot(Vh.T, X, (1, X.ndim - 1))
    return np.ascontiguousarray(X)


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
    This implementation uses the fast method of estimating the core tensor :cite:p:`papalexakis2015fast,buis1996efficient`
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
    T = np.zeros([rank]*X.ndim)
    np.fill_diagonal(T, 1)
    if normalised:
        denom = np.sum(G**2)
    else:
        denom = rank

    return 100 - 100*np.sum((G - T)**2)/denom

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import brentq

from ._utils import is_iterable
from .factor_tools import construct_cp_tensor
from .xarray_wrapper import is_dataframe, is_xarray

_LEVERAGE_NAME = "Leverage score"
_SLABWISE_SSE_NAME = "Slabwise SSE"


def _compute_leverage(factor_matrix):
    A = factor_matrix
    leverage = A @ np.linalg.solve(A.T @ A, A.T)
    return np.diag(leverage)


def _compute_slabwise_sse(estimated, true, normalise=True, axis=0):
    if not is_iterable(axis):
        axis = {axis}
    axis = set(axis)

    reduction_axis = tuple(i for i in range(true.ndim) if i not in axis)
    SSE = ((estimated - true) ** 2).sum(axis=reduction_axis)
    if normalise:
        return SSE / SSE.sum()
    else:
        return SSE


def compute_slabwise_sse(estimated, true, normalise=True, axis=0):
    r"""Compute the (normalised) slabwise SSE along the given mode(s).

    For a tensor, :math:`\mathcal{X}`, and an estimated tensor :math:`\hat{\mathcal{X}}`,
    we compute the :math:`i`-th normalised slabwise residual as

    .. math::
        r_i = \frac{\sum_{jk} \left(x_{ijk} - \hat{x}_{ijk}\right)^2}
                   {\sum_{ijk} \left(x_{ijk} - \hat{x}_{ijk}\right)^2}.

    The residuals can measure how well our decomposition fits the different
    sample. If a sample, :math:`i`, has a high residual, then that indicates that
    the model is not able to describe its behaviour.

    Parameters
    ----------
    estimated : xarray or numpy array
        Estimated dataset, if this is an xarray, then the output is too.
    true : xarray or numpy array
        True dataset, if this is an xarray, then the output is too.
    normalise : bool
        Whether the SSE should be scaled so the vector sums to one.
    axis : int
        Axis (or axes) that the SSE is computed across (i.e. these are not the ones summed over).
        The output will still have these axes.

    Returns
    -------
    slab_sse : xarray or numpy array
        The (normalised) slabwise SSE, if true tensor input is an xarray array,
        then the returned tensor is too.

    TODO: example for compute_slabwise_sse
    """
    # Check that dimensions match up.
    if is_xarray(estimated) and is_xarray(true):
        if estimated.dims != true.dims:
            raise ValueError(
                f"Dimensions of estimated and true tensor must be equal,"
                f" they are {estimated.dims} and {true.dims}, respectively."
            )
        for dim in estimated.dims:
            if len(true.coords[dim]) != len(estimated.coords[dim]):
                raise ValueError(
                    f"The dimension {dim} has different length for the true and estiamted tensor. "
                    f"The true tensor has length {len(true.coords[dim])} and the estimated tensor "
                    f"has length {len(estimated.coords[dim])}."
                )
            if not all(true.coords[dim] == estimated.coords[dim]):
                raise ValueError(
                    f"The dimension {dim} has different coordinates for the true and estimated tensor."
                )
    elif is_dataframe(estimated) and is_dataframe(true):
        if estimated.columns != true.columns:
            raise ValueError("Columns of true and estimated matrix must be equal")
        if estimated.index != true.index:
            raise ValueError("Index of true and estimated matrix must be equal")

    slab_sse = _compute_slabwise_sse(estimated, true, normalise=normalise, axis=axis)
    if hasattr(slab_sse, "to_dataframe"):
        slab_sse.name = _SLABWISE_SSE_NAME
    return slab_sse


def compute_leverage(factor_matrix):
    r"""Compute the leverage score of the given factor matrix.

    The leverage score is a measure of how much "influence" a slab (often representing a sample)
    has on a tensor factorisation model. For example, if we have a CP model, :math:`[A, B, C]`,
    where the :math:`A`-matrix represents the samples, then the sample-mode leverage score is
    defined as

    .. math::

        l_i = \left[A \left(A^T A\right)^{-1} A^T\right]_{ii},

    that is, the :math:`i`-th diagonal entry of the matrix :math:`\left[A \left(A^T A\right)^{-1} A^T\right]`.
    If a given sample, :math:`i`, has a high leverage score, then it likely has a strong
    influence on the model.

    # TODO: More description with some mathematical properties (e.g. sums to the rank) and interpretations

    If the factor matrix is a dataframe (i.e. has an index), then the output is
    also a dataframe with that index. Otherwise, the output is a NumPy array.

    Parameters
    ----------
    factor_matrix : DataFrame or numpy array
        The factor matrix whose leverage we compute

    Returns
    -------
    leverage : DataFrame or numpy array
        The leverage scores, if the input is a dataframe, then the index is preserved.

    #TODO: example for compute_leverage
    """
    leverage = _compute_leverage(factor_matrix)

    if hasattr(factor_matrix, "index"):
        return pd.DataFrame(
            leverage.reshape(-1, 1), columns=[_LEVERAGE_NAME], index=factor_matrix.index
        )
    else:
        return leverage


def compute_outlier_info(cp_tensor, true_tensor, normalise_sse=True, axis=0):
    f"""Compute the leverage score and (normalised) slabwise SSE along one axis.

    # TODO: Write description of how to use compute_outlier_info.

    These metrics are often plotted against each other to discover outliers.

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument
    true_tensor : xarray or numpy array
        Dataset that cp_tensor is fitted against.
    normalise_sse : bool
        If true, the slabwise SSE is scaled so it sums to one.
    axis : int

    Returns
    -------
    DataFrame
        Dataframe with two columns, "{_LEVERAGE_NAME}" and "{_SLABWISE_SSE_NAME}".
    """
    # Add whether suspicious based on rule-of-thumb cutoffs as boolean columns
    leverage = compute_leverage(cp_tensor[1][axis])

    estimated_tensor = construct_cp_tensor(cp_tensor)
    slab_sse = compute_slabwise_sse(
        estimated_tensor, true_tensor, normalise=normalise_sse, axis=axis
    )
    if is_xarray(slab_sse):
        slab_sse = pd.DataFrame(slab_sse.to_series())

    leverage_is_labelled = isinstance(leverage, pd.DataFrame)
    sse_is_labelled = isinstance(slab_sse, pd.DataFrame)  # TODO: isxarray function?
    if (leverage_is_labelled and not sse_is_labelled) or (
        not leverage_is_labelled and sse_is_labelled
    ):
        raise ValueError(
            "If `cp_tensor` is labelled (factor matrices are dataframes), then"
            "`true_tensor` should be an xarray object and vice versa."
        )
    elif not leverage_is_labelled and not sse_is_labelled:
        return pd.DataFrame({_LEVERAGE_NAME: leverage, _SLABWISE_SSE_NAME: slab_sse})
    elif leverage_is_labelled and not all(slab_sse.index == leverage.index):
        raise ValueError(
            "The indices of the labelled factor matrices does not match up with the xarray dataset"
        )

    results = pd.concat([leverage, slab_sse], axis=1)
    results.columns = [_LEVERAGE_NAME, _SLABWISE_SSE_NAME]
    return results


# TODO: Leverage and SSE rule of thumbs
# TODO: Unit tests for get_leverage_outlier_threshold: Monte-carlo experiment with p-value
def get_leverage_outlier_threshold(leverage_scores, method, p_value=0.05):
    """Compute threshold for detecting possible outliers based on leverage.

    Huber's heuristic for selecting outliers
    ----------------------------------------

    In Robust Statistics, Huber :cite:p:`huber2009robust`shows that that if the leverage
    score, :math:`h_i`, of a sample is equal to :math:`1/r` and we duplicate that sample,
    then its leverage score will be equal to :math:`1/(1+r)`. We can therefore, think of
    of the reciprocal of the leverage score, :math:`1/h_i`, as the number of similar samples
    in the dataset. Following this logic, Huber recommends two thresholds for selecting
    outliers: 0.2 (which we name ``"huber low"``) and 0.5 (which we name ``"huber high"``).

    Hoaglin and Welch's heuristic for selecting outliers
    ----------------------------------------------------

    In :cite:p:`belsley1980regression` (page 17), :citeauthor:p:`belsley1980regression`,
    show that if the factor matrix is normally distributed, then we can scale leverage,
    we obtain a Fisher-distributed random variable. Specifically, we have that
    :math:`(n - r)[h_i - (1/n)]/[(1 - h_i)(r - 1)]` follows a Fisher distribution with
    :math:`(r-1)` and :math:`(n-r)` degrees of freedom. While the factor matrix seldomly
    follows a normal distribution, :citeauthor:p:`belsley1980regression` still argues that
    this can be a good starting point for cut-off values of suspicious data points. They
    therefore say that :math:`2r/n` is a good cutoff in general and that :math:`3r/n`
    is a good cutoff when :math:`r < 6` and :math:`n-r > 12`.

    Fisher-distribution
    -------------------

    Another way to select ouliers is also based on the findings by :citeauthor:p:`belsley1980regression`.
    We can use the transformation into a Fisher distributed variable (assuming that the factor
    elements are drawn from a normal distribution), to compute cut-off values based on a p-value.
    The elements of the factor matrices are seldomly normally distributed, so this is
    also just a rule-of-thumb.

    Parameters
    ----------
    leverage_scores : np.ndarray or pd.DataFrame
    method : {"huber lower", "huber higher", "hw lower", "hw higher", "p-value"}
    p_value : float (optional, default=0.05)
        If ``method="p-value"``, then this is the p-value used for the cut-off.

    Returns
    -------
    threshold : float
        Threshold value, data points with a leverage score larger than the threshold are suspicious
        and may be outliers.
    """
    # TODO: Incorporate in plot
    # TODO: Unit tests
    num_samples = len(leverage_scores)
    num_components = np.sum(leverage_scores)

    method = method.lower()
    if method == "huber lower":
        return 0.2
    elif method == "huber higher":
        return 0.5
    elif method == "hw lower":
        return 2 * num_components / num_samples
    elif method == "hw higher":
        return 3 * num_components / num_samples
    elif method == "p-value":
        dofs1 = num_components - 1
        dofs2 = num_samples - num_components

        def func(h):
            F = dofs2 * (h - 1 / num_samples) / ((1 - h) * dofs1 + 1e-10)
            return stats.f.cdf(F, dofs1, dofs2) - (1 - p_value)

        if dofs1 <= 0:
            raise ValueError("Cannot use P-value when there is only one component.")
        if dofs2 <= 0:
            raise ValueError(
                "Cannot use P-value when there are fewer samples than components."
            )
        return brentq(func, 0, 1,)
    else:
        raise ValueError(
            f"Method must be one of 'huber lower', 'huber higher', 'hw lower' or 'hw higher', or 'p-value' not {method}"
        )


def get_slab_sse_outlier_threshold(slab_sse, method, p_value=0.05, dof=1):
    """Compute rule-of-thumb threshold values for suspicious residuals.

    One way to determine possible outliers is to examine how well the model describes
    the different data points. A standard way of measuring this, is by the slab-wise
    sum of squared errors (slabwise SSE), which is the sum of squared error for each
    data point.

    There is, unfortunately, no guaranteed way to detect outliers automatically based
    on the residuals. However, if the slabs we compute the SSE for are large, then we
    can use the central limit theorem to assume normally distributed slabwise SSE.
    Based on this, we can use the student t distribution to find an appropriate cut-off
    value.

    Another rule-of-thumb follows from :cite:p:`naes2002user` (p. 187), which states
    that two times the standard deviation of the slabwise SSE can be used for
    determining data points with a suspiciously high residual.

    Parameters
    ----------
    slab_sse : np.ndarray or pd.DataFrame
    method : {"two_sigma", "p_value"}
    p_value : float (optional, default=0.05)
        If ``method="p-value"``, then this is the p-value used for the cut-off.

    Returns
    -------
    threshold : float
        Threshold value, data points with a higher SSE than the threshold are suspicious
        and may be outliers.
    """
    # TODO: documentation example for get_slab_sse_outlier_threshold
    num_samples = len(slab_sse)
    std = np.std(slab_sse, dof=dof)
    mean = np.mean(slab_sse)
    if method == "two_sigma":
        return std * 2
    elif method == "p_value":
        return mean + std * stats.t.isf(p_value, num_samples - dof)
    else:
        raise ValueError(
            f"Method must be one of 'two_sigma' and 'p_value', not '{method}'."
        )

# -*- coding: utf-8 -*-

__author__ = "Marie Roald & Yngve Mardal Moe"

import numpy as np
import pandas as pd
import scipy.stats as stats

from ._module_utils import is_dataframe, is_iterable, is_xarray
from ._tl_utils import _handle_tensorly_backends_cp, _handle_tensorly_backends_dataset, _SINGLETON
from ._xarray_wrapper import is_labelled_dataset
from .utils import _alias_mode_axis, cp_to_tensor

_LEVERAGE_NAME = "Leverage score"
_SLABWISE_SSE_NAME = "Slabwise SSE"

__all__ = [
    "compute_slabwise_sse",
    "compute_leverage",
    "compute_outlier_info",
    "get_leverage_outlier_threshold",
    "get_slabwise_sse_outlier_threshold",
]


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


@_alias_mode_axis()
@_handle_tensorly_backends_dataset("true", _SINGLETON)
@_handle_tensorly_backends_dataset("estimated", None)
def compute_slabwise_sse(estimated, true, normalise=True, mode=0, axis=None):
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
    mode : int or iterable of ints
        Mode (or modes) that the SSE is computed across (i.e. these are not the ones summed over).
        The output will still have these axes.
    axis : int or iterable of ints (optional)
        Alias for mode. If this is set, then no value for mode can be given

    Returns
    -------
    slab_sse : xarray or numpy array
        The (normalised) slabwise SSE, if true tensor input is an xarray array,
        then the returned tensor is too.
    """
    # TODOC: example for compute_slabwise_sse
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
                raise ValueError(f"The dimension {dim} has different coordinates for the true and estimated tensor.")
    elif is_dataframe(estimated) and is_dataframe(true):
        if estimated.columns != true.columns:
            raise ValueError("Columns of true and estimated matrix must be equal")
        if estimated.index != true.index:
            raise ValueError("Index of true and estimated matrix must be equal")

    slab_sse = _compute_slabwise_sse(estimated, true, normalise=normalise, axis=mode)
    if is_labelled_dataset(true):
        slab_sse.name = _SLABWISE_SSE_NAME
    return slab_sse


@_handle_tensorly_backends_dataset("factor_matrix", _SINGLETON)
def compute_leverage(factor_matrix):
    r"""Compute the leverage score of the given factor matrix.

    The leverage score is a measure of how much "influence" a slab (often representing a sample)
    has on a tensor factorisation model. To compute the leverage score for the different slabs,
    we only need the factor matrix for the selected mode. If the selected mode is represented
    by :math:`\mathbf{A}`, then the leverage score is defined as

    .. math::

        h_i = \left[\mathbf{A} \left(\mathbf{A}^T \mathbf{A}\right)^{-1} \mathbf{A}^T\right]_{ii},

    that is, the :math:`i`-th diagonal entry of the matrix
    :math:`\mathbf{A} \left(\mathbf{A}^T \mathbf{A}\right)^{-1} \mathbf{A}^T`.
    If a given slab, :math:`i`, has a high leverage score, then it likely has a strong
    influence on the model. A good overview of the leverage score is :cite:p:`velleman1981efficient`.

    The leverage scores sums to the number of components for our model and is always between 0 and 1.
    Moreover, if a data point has a leverage score equal to 1, then one component is solely "devoted"
    to modelling that data point, and removing the corresponding row from :math:`A` will reduce the
    rank of :math:`A` by 1 :cite:p:`belsley1980regression`.

    A way of interpreting the leverage score is as a measure of how "similar" a data point
    is to the rest. If a row of :math:`A` is equal to the average row of :math:`A`, then its leverage
    score would be equal to :math:`\frac{1}{I}`. Likewise, if a data point has a leverage of 1, then
    no other data points have a similar model representation. If a data point has a leverage of 0.5,
    then there is one other data point (in some weighted sense) with a similar model representation,
    and a leverage of 0.2 means that there are five other data points with a similar model representation
    :cite:p:`huber2009robust`.

    If the factor matrix is a dataframe, then the output is also a dataframe with that index. Otherwise,
    the output is a NumPy array.

    Parameters
    ----------
    factor_matrix : DataFrame or numpy array
        The factor matrix whose leverage we compute

    Returns
    -------
    leverage : DataFrame or numpy array
        The leverage scores, if the input is a dataframe, then the index is preserved.

    .. note::

        The leverage score is related to the Hotelling T2-statistic (or D-statistic), which
        is equal to a scaled version of leverage computed based on centered factor matrices.

    Examples
    --------

    In this example, we compute the leverage of a random factor matrix

    >>> import numpy as np
    >>> from tlviz.outliers import compute_leverage
    >>> rng = np.random.default_rng(0)
    >>> A = rng.standard_normal(size=(5, 2))
    >>> leverage_scores = compute_leverage(A)
    >>> for index, leverage in enumerate(leverage_scores):
    ...     print(f"Sample {index} has leverage score {leverage:.2f}")
    Sample 0 has leverage score 0.04
    Sample 1 has leverage score 0.23
    Sample 2 has leverage score 0.50
    Sample 3 has leverage score 0.59
    Sample 4 has leverage score 0.64
    """
    leverage = _compute_leverage(factor_matrix)

    if is_dataframe(factor_matrix):
        return pd.DataFrame(leverage.reshape(-1, 1), columns=[_LEVERAGE_NAME], index=factor_matrix.index)
    else:
        return leverage


@_alias_mode_axis()
@_handle_tensorly_backends_dataset("true_tensor", None)
@_handle_tensorly_backends_cp("cp_tensor", None)
def compute_outlier_info(cp_tensor, true_tensor, normalise_sse=True, mode=0, axis=None):
    """Compute the leverage score and (normalised) slabwise SSE along one axis.

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
    mode : int
        The mode to compute the outlier info across.
    axis : int (optional)
        Alias for mode. If this is set, then no value for mode can be given.

    Returns
    -------
    DataFrame
        Dataframe with two columns, "Leverage score" and "Slabwise SSE".

    See Also
    --------
    compute_leverage : More information about the leverage score is given in this docstring
    compute_slabwise_sse : More information about the slabwise SSE is given in this docstring
    get_leverage_outlier_threshold : Cutoff for selecting potential outliers based on the leverage
    compute_slabwise_sse : Cutoff for selecting potential outliers based on the slabwise SSE
    """
    # TODOC: Example for compute_outlier_info
    # Add whether suspicious based on rule-of-thumb cutoffs as boolean columns
    leverage = compute_leverage(cp_tensor[1][mode])

    estimated_tensor = cp_to_tensor(cp_tensor)
    slab_sse = compute_slabwise_sse(estimated_tensor, true_tensor, normalise=normalise_sse, mode=mode)
    if is_xarray(slab_sse):
        slab_sse = pd.DataFrame(slab_sse.to_series())

    leverage_is_labelled = is_dataframe(leverage)
    sse_is_labelled = is_dataframe(slab_sse)
    if (leverage_is_labelled and not sse_is_labelled) or (not leverage_is_labelled and sse_is_labelled):
        raise ValueError(
            "If `cp_tensor` is labelled (factor matrices are dataframes), then"
            "`true_tensor` should be an xarray object and vice versa."
        )
    elif not leverage_is_labelled and not sse_is_labelled:
        return pd.DataFrame({_LEVERAGE_NAME: leverage, _SLABWISE_SSE_NAME: slab_sse})
    elif leverage_is_labelled and not all(slab_sse.index == leverage.index):
        raise ValueError("The indices of the labelled factor matrices does not match up with the xarray dataset")

    results = pd.concat([leverage, slab_sse], axis=1)
    results.columns = [_LEVERAGE_NAME, _SLABWISE_SSE_NAME]
    return results


@_handle_tensorly_backends_dataset("leverage_scores", None)
def get_leverage_outlier_threshold(leverage_scores, method="p_value", p_value=0.05):
    """Compute threshold for detecting possible outliers based on leverage.

    **Huber's heuristic for selecting outliers**

    In Robust Statistics, Huber :cite:p:`huber2009robust` shows that that if the leverage score,
    :math:`h_i`, of a sample is equal to :math:`1/r` and we duplicate that sample, then its leverage
    score will be equal to :math:`1/(1+r)`. We can therefore, think of of the reciprocal of the
    leverage score, :math:`1/h_i`, as the number of similar samples in the dataset. Following this
    logic, Huber recommends two thresholds for selecting outliers: 0.2 (which we name ``"huber low"``)
    and 0.5 (which we name ``"huber high"``).

    **Hoaglin and Welch's heuristic for selecting outliers**

    In :cite:p:`hoaglin1978hat`, :cite:authors:`hoaglin1978hat` state that :math:`2r/n` is a good cutoff
    for selecting samples that may be outliers. This choice is elaborated in :cite:p:`belsley1980regression`
    (page 17), where :cite:authors:`belsley1980regression` also propose :math:`3r/n` as a cutoff when
    :math:`r < 6` and :math:`n-r > 12`. They also defend thee cut-offs by proving that if the factor matrices
    are normally distributed, then :math:`(n - r)[h_i - (1/n)]/[(1 - h_i)(r - 1)]` follows a Fisher
    distribution with :math:`(r-1)` and :math:`(n-r)` degrees of freedom. While the factor matrix
    seldomly follows a normal distribution, :cite:authors:`belsley1980regression` still argues that this
    can be a good starting point for cut-off values of suspicious data points. Based on reasonable choices for
    :math:`n` and :math:`r`, they arive at the heuristics above.

    **Leverage p-value**

    Another way to select ouliers is also based on the findings by :cite:authors:`belsley1980regression`.
    We can use the transformation into a Fisher distributed variable (assuming that the factor elements
    are drawn from a normal distribution), to compute cut-off values based on a p-value. The elements of
    the factor matrices are seldomly normally distributed, so this is also just a rule-of-thumb.

    .. note::

        Note also that we, with bootstrap estimation, have found that this p-value is only valid for
        large number of components. For smaller number of components, the false positive rate will be higher
        than the specified p-value, even if the components follow a standard normal distribution (see example below).

    **Hotelling's T2 statistic**

    Yet another way to estimate a p-value is via Hotelling's T-squared statistic :cite:p:`jackson1980principal`
    (see also :cite:p:`nomikos1995multivariate`). The key here is to notice that if the factor matrices are
    normally distributed with zero mean, then the leverage is equivalent to a scaled version of the Hotelling's
    T-squared statistic. This is commonly used in PCA, where the data often is centered beforehand, which leads
    to components with zero mean (in the mode the data is centered across). Again, note that the elements of the
    factor matrices are seldomly normally distributed, so this is also just a rule-of-thumb.

    .. note::

        Note also that we, with bootstrap estimation, have found that this p-value is not valid for
        large numbers of components. In that case, the false positive rate will be higher than the specified
        p-value, even if the components follow a standard normal distribution (see example below).

    Parameters
    ----------
    leverage_scores : np.ndarray or pd.DataFrame
    method : {"huber lower", "huber higher", "hw lower", "hw higher", "p-value", "hotelling"}
    p_value : float (optional, default=0.05)
        If ``method="p-value"``, then this is the p-value used for the cut-off.

    Returns
    -------
    threshold : float
        Threshold value, data points with a leverage score larger than the threshold are suspicious
        and may be outliers.

    Examples
    --------

    **The leverage p-value is only accurate with many components:**
    Here, we use Monte-Carlo estimation to demonstrate that the p-value derived in :cite:p:`belsley1980regression`
    is valid only for large number of components.

    We start by importing some utilities

    >>> import numpy as np
    >>> from scipy.stats import bootstrap
    >>> from tlviz.outliers import compute_leverage, get_leverage_outlier_threshold

    Here, we create a function that computes the false positive rate

    >>> def compute_false_positive_rate(n, d, p_value):
    ...     X = np.random.standard_normal((n, d))
    ...
    ...     h = compute_leverage(X)
    ...     th = get_leverage_outlier_threshold(h, method="p-value", p_value=p_value)
    ...     return (h > th).mean()

    >>> np.random.seed(0)
    >>> n_samples = 2_000
    >>> leverages = [compute_false_positive_rate(10, 2, 0.05) for _ in range(n_samples)],
    >>> fpr_low, fpr_high = bootstrap(leverages, np.mean).confidence_interval
    >>> print(f"95% confidence interval for the false positive rate: [{fpr_low:.3f}, {fpr_high:.3f}]")
    95% confidence interval for the false positive rate: [0.083, 0.089]

    We see that the false positive rate is almost twice what we prescribe (0.05). However, if we increase
    the number of components, then the false positive rate improves

    >>> leverages = [compute_false_positive_rate(10, 9, 0.05) for _ in range(n_samples)],
    >>> fpr_low, fpr_high = bootstrap(leverages, np.mean).confidence_interval
    >>> print(f"95% confidence interval for the false positive rate: [{fpr_low:.3f}, {fpr_high:.3f}]")
    95% confidence interval for the false positive rate: [0.049, 0.056]

    This indicates that the false positive rate is most accurate when the number of components is equal
    to the number of samples - 1. We can increase the number of samples to assess this conjecture

    >>> leverages = [compute_false_positive_rate(100, 9, 0.05) for _ in range(n_samples)],
    >>> fpr_low, fpr_high = bootstrap(leverages, np.mean).confidence_interval
    >>> print(f"95% confidence interval for the false positive rate: [{fpr_low:.3f}, {fpr_high:.3f}]")
    95% confidence interval for the false positive rate: [0.055, 0.056]

    The increase in the false positive rate supports the conjecture that :cite:author:`belsley1980regression`'s
    method for computing the p-value is accurate only when the number of components is high. Still, it is
    important to remember that the original assumptions (normally distributed components) is seldomly satisfied
    also, so this method is only a rule-of-thumb and can still be useful.

    **Hotelling's T-squared statistic requires few components or many samples:**
    Here, we use Monte-Carlo estimation to demonstrate that the Hotelling T-squared statistic is only valid with
    many samples.

    >>> def compute_hotelling_false_positive_rate(n, d, p_value):
    ...     X = np.random.standard_normal((n, d))
    ...
    ...     h = compute_leverage(X)
    ...     th = get_leverage_outlier_threshold(h, method="hotelling", p_value=p_value)
    ...     return (h > th).mean()

    We set the simulation parameters and the seed

    >>> np.random.seed(0)
    >>> n_samples = 2_000
    >>> fprs = [compute_hotelling_false_positive_rate(10, 2, 0.05) for _ in range(n_samples)],
    >>> fpr_low, fpr_high = bootstrap(fprs, np.mean).confidence_interval
    >>> print(f"95% confidence interval for the false positive rate: [{fpr_low:.3f}, {fpr_high:.3f}]")
    95% confidence interval for the false positive rate: [0.052, 0.058]

    However, if we increase the number of components, then the false positive rate becomes to large

    >>> fprs = [compute_hotelling_false_positive_rate(10, 5, 0.05) for _ in range(n_samples)],
    >>> fpr_low, fpr_high = bootstrap(fprs, np.mean).confidence_interval
    >>> print(f"95% confidence interval for the false positive rate: [{fpr_low:.3f}, {fpr_high:.3f}]")
    95% confidence interval for the false positive rate: [0.078, 0.084]

    But if we increase the number of samples, then the estimate is good again

    >>> fprs = [compute_hotelling_false_positive_rate(100, 5, 0.05) for _ in range(n_samples)],
    >>> fpr_low, fpr_high = bootstrap(fprs, np.mean).confidence_interval
    >>> print(f"95% confidence interval for the false positive rate: [{fpr_low:.3f}, {fpr_high:.3f}]")
    95% confidence interval for the false positive rate: [0.049, 0.051]
    """
    num_samples = int(round(len(leverage_scores)))
    num_components = int(round(np.sum(leverage_scores)))

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
        n, p = num_samples, num_components

        if p <= 1:
            raise ValueError("Cannot use P-value when there is only one component.")
        if n <= p:
            raise ValueError("Cannot use P-value when there are fewer samples than components.")

        # TODO: Try with n - p - 1, and try maths too compare with hotelling
        F = stats.f.isf(p_value, p - 1, n - p)
        F_scale = F * (p - 1) / (n - p)
        # Solve the equation (h + (1/n)) / (1 - h) = F_scale:
        return (F_scale + (1 / n)) / (1 + F_scale)
    elif method == "hotelling":
        I, R = num_samples, num_components
        if I <= R + 1:
            raise ValueError("Cannot use Hotelling P-value when there are fewer samples than components minus one.")

        F = stats.f.isf(p_value, R, I - R - 1)
        B = F * (R / (I - R - 1)) / (1 + (R / (I - R - 1)) * F)
        # Remove the square compared to Nomikos & MacGregor since the leverage is:
        #  A(AtA)^-1 At,
        # not
        #  A(AtA / (I-1))^-1 At
        return B * (I - 1) / I
    elif method == "bonferroni p-value":
        return get_leverage_outlier_threshold(leverage_scores, method="p-value", p_value=p_value / len(leverage_scores))
    elif method == "bonferroni hotelling":
        return get_leverage_outlier_threshold(
            leverage_scores, method="hotelling", p_value=p_value / len(leverage_scores)
        )
    else:
        raise ValueError(
            "Method must be one of 'huber lower', 'huber higher', 'hw lower' or 'hw higher', "
            + f"'p-value', 'bonferroni p-value', 'hotelling' or 'bonferroni hotelling' not '{method}'"
        )


@_handle_tensorly_backends_dataset("slab_sse", None)
def get_slabwise_sse_outlier_threshold(slab_sse, method="p-value", p_value=0.05, ddof=1):
    r"""Compute rule-of-thumb threshold values for suspicious residuals.

    One way to determine possible outliers is to examine how well the model describes
    the different data points. A standard way of measuring this, is by the slab-wise
    sum of squared errors (slabwise SSE), which is the sum of squared error for each
    data point.

    There is, unfortunately, no guaranteed way to detect outliers automatically based
    on the residuals. However, if the noise is normally distributed, then the residuals
    follow a scaled chi-squared distribution. Specifically, we have that
    :math:`\text{SSE}_i^2 \sim g\chi^2_h`, where :math:`g = \frac{\sigma^2}{2\mu}`,
    :math:`h = \frac{\mu}{g} = \frac{2\mu^2}{\sigma^2}`, and :math:`\mu` is the
    average slabwise SSE and :math:`\sigma^2` is the variance of the slabwise
    SSE :cite:p:`box1954some`.

    Another rule-of-thumb follows from :cite:p:`naes2002user` (p. 187), which states
    that two times the standard deviation of the slabwise SSE can be used for
    determining data points with a suspiciously high residual.

    Parameters
    ----------
    slab_sse : np.ndarray or pd.DataFrame
    method : {"two_sigma", "p-value"}
    p_value : float (optional, default=0.05)
        If ``method="p-value"``, then this is the p-value used for the cut-off.

    Returns
    -------
    threshold : float
        Threshold value, data points with a higher SSE than the threshold are suspicious
        and may be outliers.

    Examples
    --------
    Here, we see that the p-value gives a good cutoff if the noise is normally distributed

    We start by importing the tools we'll need

    >>> import numpy as np
    >>> from scipy.stats import bootstrap
    >>> from tlviz.outliers import compute_slabwise_sse, get_slabwise_sse_outlier_threshold
    >>> from tlviz.utils import cp_to_tensor

    Then, we create a function to compute the false positive rate. This will be useful for our
    bootstrap estimate for the true false positive rate.

    >>> def compute_false_positive_rate(shape, num_components, p_value):
    ...     A = np.random.standard_normal((shape[0], num_components))
    ...     B = np.random.standard_normal((shape[1], num_components))
    ...     C = np.random.standard_normal((shape[2], num_components))
    ...
    ...     X = cp_to_tensor((None, [A, B, C]))
    ...     noisy_X = X + np.random.standard_normal(shape)*5
    ...
    ...
    ...
    ...     sse = compute_slabwise_sse(X, noisy_X)
    ...     th = get_slabwise_sse_outlier_threshold(sse, method="p-value", p_value=p_value)
    ...     return (sse > th).mean()

    Finally, we estimate the 95% confidence interval of the false positive rate to validate
    that it is approximately correct.

    >>> np.random.seed(0)
    >>> n_samples = 2_000
    >>> slab_sse = [compute_false_positive_rate((20, 20, 10), 5, 0.05) for _ in range(n_samples)],
    >>> fpr_low, fpr_high = bootstrap(slab_sse, np.mean).confidence_interval
    >>> print(f"95% confidence interval for the false positive rate: [{fpr_low:.3f}, {fpr_high:.3f}]")
    95% confidence interval for the false positive rate: [0.044, 0.047]

    We see that the 95% confidence interval lies just below our goal of 0.05! Let's also try
    with a false positive rate of 0.1

    >>> slab_sse = [compute_false_positive_rate((20, 20, 10), 5, 0.1) for _ in range(n_samples)],
    >>> fpr_low, fpr_high = bootstrap(slab_sse, np.mean).confidence_interval
    >>> print(f"95% confidence interval for the false positive rate: [{fpr_low:.3f}, {fpr_high:.3f}]")
    95% confidence interval for the false positive rate: [0.097, 0.100]

    Here we see that the false positive rate is sufficiently estimated. It may have been too low
    above since we either did not have enough samples in the first mode (which we compute) the
    false positive rate for). With only 20 samples, it will be difficult to correctly estimate a
    false positive rate of 0.05. If we increase the number of samples to 200 instead, we see that
    the false positive rate is within our expected bounds.

    >>> slab_sse = [compute_false_positive_rate((200, 20, 10), 5, 0.05) for _ in range(n_samples)],
    >>> fpr_low, fpr_high = bootstrap(slab_sse, np.mean).confidence_interval
    >>> print(f"95% confidence interval for the false positive rate: [{fpr_low:.3f}, {fpr_high:.3f}]")
    95% confidence interval for the false positive rate: [0.049, 0.050]
    """
    std = np.std(slab_sse, ddof=ddof)
    mean = np.mean(slab_sse)
    if method == "two sigma":
        return std * 2
    elif method == "p-value":
        g = std * std / (2 * mean)
        h = mean / g
        return stats.chi2.isf(p_value, h) * g
    elif method == "bonferroni p-value":
        return get_slabwise_sse_outlier_threshold(
            slab_sse, method="p-value", p_value=p_value / len(slab_sse), ddof=ddof
        )
    else:
        raise ValueError(f"Method must be one of 'two sigma', 'p-value', or 'bonferroni p-value', not '{method}'.")

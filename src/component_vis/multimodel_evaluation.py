"""
Utilities for comparing multiple decompositions against each other.
"""
import numpy as np

from . import model_evaluation
from .factor_tools import factor_match_score
from .utils import extract_singleton

__all__ = [
    "similarity_evaluation",
    "get_model_with_lowest_error",
    "sort_models_by_error",
]


def similarity_evaluation(cp_tensor, comparison_cp_tensors, similarity_metric=None, **kwargs):
    r"""Compute similarities between ``cp_tensor`` and all ``comparison_cp_tensors``.

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument
    comparison_cp_tensors : List[CPTensor or tuple]
        List of TensorLy-style CPTensors to compare with
    similarity_metric : Callable[CPTensor, CPTensor, \*\*kwargs] -> float
        Function that takes two CPTensors as input and returns their similarity
    **kwargs
        Extra keyword-arguments passed to ``similarity_metric``.

    Returns
    -------
    similarity : float
    """
    # TODOC: example for similarity_evaluation
    if similarity_metric is None:
        similarity_metric = factor_match_score

    return [
        similarity_metric(cp_tensor, comparison_cp_tensor, **kwargs) for comparison_cp_tensor in comparison_cp_tensors
    ]


def get_model_with_lowest_error(cp_tensors, X, error_function=None, return_index=False, return_errors=False):
    """Compute reconstruction error for all cp_tensors and return model with lowest error.

    This is useful to select the best initialisation if several random
    initialisations are used to fit the model. By default, the relative SSE
    is used, but another error function can be used too.

    Parameters
    ----------
    cp_tensors : list of CPTensors
        List of all CP tensors to compare
    X : ndarray
        Dataset modelled by the CP tensors
    error_function : Callable (optional)
        Callable with the signature ``error_function(cp_tensor, X)``,
        that should return a measure of the modelling error (e.g. SSE). Default
        is relative SSE.
    return_index : bool (optional, default=False)
        If True, then the index of the CP tensor with the lowest error is returned
    return_errors : bool (optional, defult=False)
        if True, then a list of errors for each CP tensor is returned.

    Returns
    -------
    CPTensor
        The CP tensor with the lowest error
    int
        The index of the selected CP tensor in ``cp_tensors``. Only returned
        if ``return_index=True``.
    list
        List of the error values for all CP tensors in ``cp_tensor`` (in the same
        order as ``cp_tensors``). only returned if ``return_errors=True``
    """
    # TODOC: example for get_model_with_lowest_error
    if error_function is None:
        error_function = model_evaluation.relative_sse

    selected_cp_tensor = None
    selected_index = None
    lowest_sse = np.inf
    all_sse = []
    for i, cp_tensor in enumerate(cp_tensors):
        sse = error_function(cp_tensor, X)
        all_sse.append(sse)
        if sse < lowest_sse:
            selected_cp_tensor = cp_tensor
            lowest_sse = sse
            selected_index = i

    returns = [selected_cp_tensor]
    if return_index:
        returns.append(selected_index)
    if return_errors:
        returns.append(all_sse)
    returns = tuple(returns)
    if len(returns) == 1:
        return returns[0]
    else:
        return returns


def sort_models_by_error(cp_tensors, X, error_function=None):
    """Sort the ``cp_tensors`` by their error so the model with the lowest error is first.

    Parameters
    ----------
    cp_tensors : list of CPTensors
        List of all CP tensors
    X : ndarray
        Dataset modelled by the CP tensors
    error_function : Callable (optional)
        Callable with the signature ``error_function(cp_tensor, X)``,
        that should return a measure of the modelling error (e.g. SSE).

    Returns
    -------
    list of CPTensors
        List of all CP tensors sorted so the CP tensor with the lowest error
        is first and highest error is last.
    list of floats
        List of error computed for each CP tensor (in sorted order)
    """
    errors = get_model_with_lowest_error(cp_tensors, X, error_function=error_function, return_errors=True)[1]
    sorted_errors = sorted(zip(errors, range(len(errors))))
    return (
        [cp_tensors[idx] for _error, idx in sorted_errors],
        [extract_singleton(error) for error, _cp_tensor in sorted_errors],
    )

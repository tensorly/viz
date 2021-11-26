from .factor_tools import factor_match_score
from . import model_evaluation
import numpy as np


# TODO: Tests for similarity_evaluation
# Set similarity metric to a function that only return ones to check that the argument is used
# Check comparison tensors equal to cp_tensor to check that we only get ones
# Test with CP tensors with known similarity
def similarity_evaluation(
    cp_tensor, comparison_cp_tensors, similarity_metric=None, **kwargs
):
    """Compute similarities between ``cp_tensor`` and all ``comparison_cp_tensors``.

    Parameters
    ----------
    cp_tensor : CPTensor or tuple
        TensorLy-style CPTensor object or tuple with weights as first
        argument and a tuple of components as second argument
    comparison_cp_tensors : List[CPTensor or tuple]
        List of TensorLy-style CPTensors to compare with
    similarity_metric : Callable[CPTensor, CPTensor, **kwargs] -> float
        Function that takes two CPTensors as input and returns their similarity
    **kwargs
        Extra keyword-arguments passed to ``similarity_metric``.

    Returns
    -------
    similarity : float 
    """
    # TODO: example for similarity_evaluation
    if similarity_metric is None:
        similarity_metric = factor_match_score

    return [
        similarity_metric(cp_tensor, comparison_cp_tensor, **kwargs)
        for comparison_cp_tensor in comparison_cp_tensors
    ]


def get_model_with_lowest_error(cp_tensors, X, error_function=None):
    # TODO: tests for get_model_with_lowest_error
    # TODO: documentation for get_model_with_lowest_error
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

    return selected_cp_tensor, selected_index, all_sse


def sort_models_by_error(cp_tensors, X, error_function=None):
    # TODO: documentation for sort_models_by_error
    # TODO: tests for sort_models_by_error
    errors = get_model_with_lowest_error(cp_tensors, X, error_function=error_function)[
        2
    ]
    sorted_tensors = sorted(zip(errors, cp_tensors))
    # We use np.asarray(error).item() because the error is an XArray object for X-array datasets
    return (
        [cp_tensor for error, cp_tensor in sorted_tensors],
        [np.asarray(error).item() for error, cp_tensor in sorted_tensors],
    )

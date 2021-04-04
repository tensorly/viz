from .factor_tools import factor_match_score


def similarity_evaluation(cp_tensor, comparison_cp_tensors, similarity_metric=None, **kwargs):
    """Compute similarities between ``cp_tensor`` and all ``comparison_cp_tensors``.

    Arguments:
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

    Returns:
    --------
    similarity : float
    """
    if similarity_metric is None:
        similarity_metric = factor_match_score
    
    return [
        similarity_metric(cp_tensor, comparison_cp_tensor, **kwargs)
        for comparison_cp_tensor in comparison_cp_tensors
    ]


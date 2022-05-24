# -*- coding: utf-8 -*-

__author__ = "Marie Roald & Yngve Mardal Moe"

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

    Examples
    --------
    In this example, we will fit several PARAFAC models to a simulated dataset and use ``similarity_evaluation`` to
    compute the similarities between the different fitted models and the model that obtained the lowest error.

    We start by importing the relevant functionality

    >>> from tlviz.multimodel_evaluation import sort_models_by_error, similarity_evaluation
    >>> from tlviz.data import simulated_random_cp_tensor
    >>> from tensorly.decomposition import parafac

    Then, we create a random simulated dataset and fit five parafac models to it.

    >>> cp_tensor, dataset = simulated_random_cp_tensor((10, 20, 30), 3, seed=0)
    >>> model_candidates = [
    ...     parafac(dataset, 3, init="random", random_state=i)
    ...     for i in range(5)
    ... ]

    Finally, we sort the models by their errors and compute the similarity between each model
    and the model that obtained the lowest error.

    >>> sorted_model_candidates, errors = sort_models_by_error(model_candidates, dataset)
    >>> similarities = similarity_evaluation(sorted_model_candidates[0], sorted_model_candidates[1:])
    >>> for i, s in enumerate(similarities):
    ...     print(f"Similarity between the model with the lowest loss and the model with the {i+2}. lowest loss: {s:.2}")
    Similarity between the model with the lowest loss and the model with the 2. lowest loss: 0.99
    Similarity between the model with the lowest loss and the model with the 3. lowest loss: 0.98
    Similarity between the model with the lowest loss and the model with the 4. lowest loss: 0.68
    Similarity between the model with the lowest loss and the model with the 5. lowest loss: 0.42

    We see that the three models with the lowest error were very similar, which indicates that the model is stable.
    """
    if similarity_metric is None:
        similarity_metric = factor_match_score

    return [
        similarity_metric(cp_tensor, comparison_cp_tensor, **kwargs) for comparison_cp_tensor in comparison_cp_tensors
    ]


def get_model_with_lowest_error(cp_tensors, dataset, error_function=None, return_index=False, return_errors=False):
    """Compute reconstruction error for all cp_tensors and return model with lowest error.

    This is useful to select the best initialisation if several random
    initialisations are used to fit the model. By default, the relative SSE
    is used, but another error function can be used too.

    Parameters
    ----------
    cp_tensors : list of CPTensors
        List of all CP tensors to compare
    dataset : ndarray
        Dataset modelled by the CP tensors
    error_function : Callable (optional)
        Callable with the signature ``error_function(cp_tensor, dataset)``,
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

    Examples
    --------
    Here, we illustrate how ``get_model_with_lowest_error`` can be used to get the selected model from a collection
    of model candidates, and how we can also get the errors for all model candidates and the index of the selected
    initialisation.

    We start by importing the relevant functionality

    >>> from tlviz.multimodel_evaluation import sort_models_by_error, get_model_with_lowest_error
    >>> from tlviz.model_evaluation import relative_sse
    >>> from tlviz.data import simulated_random_cp_tensor
    >>> from tlviz.factor_tools import check_cp_tensor_equal
    >>> from tensorly.decomposition import parafac

    Then, we create a simulated dataset and fit five model candidates using different random initialisations.

    >>> cp_tensor, dataset = simulated_random_cp_tensor((10, 20, 30), 3, noise_level=0.3, seed=0)
    >>> model_candidates = [
    ...     parafac(dataset, 3, init="random", random_state=i)
    ...     for i in range(5)
    ... ]

    Once we have the model candidates, we use ``get_model_with_lowest_error``. By default this function will only
    return the selected model, but in this case, we ask it to return the index of the selected model and the errors
    of all model candidates.

    >>> model, index, errors = get_model_with_lowest_error(model_candidates, dataset, return_index=True, return_errors=True)
    >>> print(f"Model {index} has lowest error")
    Model 3 has lowest error

    We can check that the selected model is the model with the init we got

    >>> check_cp_tensor_equal(model, model_candidates[index])
    True

    And that it is the model that has the lowest error

    >>> errors[index] == min(errors)
    True

    And finally that this error is equal to the relative SSE

    >>> errors[index] == relative_sse(model, dataset)
    True
    """
    if error_function is None:
        error_function = model_evaluation.relative_sse

    selected_cp_tensor = None
    selected_index = None
    lowest_sse = np.inf
    all_sse = []
    for i, cp_tensor in enumerate(cp_tensors):
        sse = error_function(cp_tensor, dataset)
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


def sort_models_by_error(cp_tensors, dataset, error_function=None):
    """Sort the ``cp_tensors`` by their error so the model with the lowest error is first.

    Parameters
    ----------
    cp_tensors : list of CPTensors
        List of all CP tensors
    dataset : ndarray
        Dataset modelled by the CP tensors
    error_function : Callable (optional)
        Callable with the signature ``error_function(cp_tensor, dataset)``,
        that should return a measure of the modelling error (e.g. SSE).

    Returns
    -------
    list of CPTensors
        List of all CP tensors sorted so the CP tensor with the lowest error
        is first and highest error is last.
    list of floats
        List of error computed for each CP tensor (in sorted order)

    Examples
    --------
    Here, we see how ``sort_models_by_error`` can be useful to get a collection of model candidates in a logical order.

    We start by importing the relevant functionality.

    >>> from tlviz.multimodel_evaluation import sort_models_by_error, get_model_with_lowest_error
    >>> from tlviz.data import simulated_random_cp_tensor
    >>> from tensorly.decomposition import parafac

    Then, we simulate a random dataset and fit five model candidates to it.

    >>> cp_tensor, dataset = simulated_random_cp_tensor((10, 20, 30), 3, noise_level=0.3, seed=0)
    >>> model_candidates = [
    ...     parafac(dataset, 3, init="random", random_state=0)
    ...     for i in range(5)
    ... ]

    Next, we sort the models by the error.

    >>> sorted_model_candidates, errors = sort_models_by_error(model_candidates, dataset)

    Now, the first element in sorted_model_candidates should be equal to the model with the lowest error.
    Let's double check by getting the model with the lowest error, and see which index it has.

    >>> lowest_error_model = get_model_with_lowest_error(model_candidates, dataset)
    >>> sorted_model_candidates.index(lowest_error_model)
    0

    Next, we can check if the errors are sorted

    >>> errors == sorted(errors)
    True
    """
    # TODOC: text example for sort_models_by_error
    errors = get_model_with_lowest_error(cp_tensors, dataset, error_function=error_function, return_errors=True)[1]
    sorted_errors = sorted(zip(errors, range(len(errors))))
    return (
        [cp_tensors[idx] for _error, idx in sorted_errors],
        [extract_singleton(error) for error, _cp_tensor in sorted_errors],
    )

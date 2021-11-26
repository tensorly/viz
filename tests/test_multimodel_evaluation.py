from pytest import approx

import component_vis.multimodel_evaluation as multimodel_evaluation
from component_vis.factor_tools import construct_cp_tensor, check_cp_tensors_equals


# TODO: NEXT
def test_get_model_with_lowest_error(rng):
    # Create n=5 random CP tensors in list comprehension
    # Iterate over each CP tensor, use it to construct a dense tensor
    # Use get_model_with_lowest_error store the selected CP tensor, lowest error and error list
    # Check that the correct CP tensor is chosen by checking that the decompositions are equal
    # Check that the selected index is i (for iteration i)
    # Check that the i-th error (for iteration i) is zero
    pass

    cp_tensors = [
        (None, (
            rng.standard_normal((10, 3)), 
            rng.standard_normal((20, 3)),
            rng.standard_normal((30, 3)),
            )
        )
        for i in range(5)]

    for i, cp_tensor in enumerate(cp_tensors):
        dense_tensor = construct_cp_tensor(cp_tensor)
        selected_cp_tensor, selected_index, all_sse = multimodel_evaluation.get_model_with_lowest_error(cp_tensors, dense_tensor)
        assert check_cp_tensors_equals(selected_cp_tensor, cp_tensor)
        assert i==selected_index
        assert all_sse[i] == approx(0)
        

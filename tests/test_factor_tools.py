import pytest
from pytest import approx
import numpy as np
from component_viz import factor_tools

def test_factor_match_score():
    A = np.random.standard_normal((30, 3))
    B = np.random.standard_normal((20, 3))

    assert factor_tools.factor_match_score((None, (A, B)), (None, (A, B))) == approx(1)
    assert factor_tools.factor_match_score((None, (A, B)), (None, (0.5*A, 0.5*B))) == approx(1)
    assert factor_tools.factor_match_score((None, (A, B)), (None, (0.5*A, 0.5*B)), consider_weights=False) == approx(1)
    assert factor_tools.factor_match_score((None, (A, B)), (None, (0.1*A, 0.1*B)), consider_weights=True) < 0.5

    # TODO: More tests for FMS -> Examples with known FMS
    # TODO: seed

def test_factor_match_score_permutation():
    num_components = 4
    A = np.random.standard_normal((30,num_components))
    B = np.random.standard_normal((20,num_components))
    permutation = np.random.permutation(num_components)

    A_permuted = A[:, permutation]
    B_permuted = B[:, permutation]

    fms, p = factor_tools.factor_match_score(
        (None, (A_permuted,B_permuted)), 
        (None, (A, B)), 
        return_permutation=True
        )
    assert fms == approx(1)
    assert np.allclose(permutation, p)

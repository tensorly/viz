import numpy as np
import scipy.linalg as sla
import pytest
from component_viz import factor_tools, model_evaluation


def _estimate_core_tensor(factors, X):
    lhs = factors[0]
    for factor in factors[1:]:
        lhs = np.kron(lhs, factor)

    rhs = X.reshape(-1, 1)
    return sla.lstsq(lhs, rhs)[0].ravel()


def test_estimate_core_tensor_against_reference(rng):
    """Test that the fast core estimation algorithm by Papalexakis and Faloutsos coincide with the reference
    """
    A = rng.standard_normal(size=(4, 3))
    B = rng.standard_normal(size=(5, 3))
    C = rng.standard_normal(size=(6, 3))
    core = rng.standard_normal(size=(3, 3, 3))
    tucker_tensor = (core, (A, B, C))
    X = factor_tools.construct_tucker_tensor(tucker_tensor)
    X += rng.standard_normal(X.shape)

    slow_estimate = _estimate_core_tensor((A, B, C), X)
    np.testing.assert_allclose(model_evaluation.estimate_core_tensor((A, B, C), X).ravel(), slow_estimate)

    D = rng.standard_normal(size=(7, 3))
    core = rng.standard_normal(size=(3, 3, 3, 3))
    tucker_tensor = (core, (A, B, C, D))
    X = factor_tools.construct_tucker_tensor(tucker_tensor)
    X += rng.standard_normal(X.shape)
    slow_estimate = _estimate_core_tensor((A, B, C, D), X)
    np.testing.assert_allclose(model_evaluation.estimate_core_tensor((A, B, C, D), X).ravel(), slow_estimate)
    

def test_estimate_core_tensor_known_tucker(rng):
    A = rng.standard_normal(size=(4, 3))
    B = rng.standard_normal(size=(5, 3))
    C = rng.standard_normal(size=(6, 3))
    core = rng.standard_normal(size=(3, 3, 3))
    tucker_tensor = (core, (A, B, C))
    X = factor_tools.construct_tucker_tensor(tucker_tensor)
    np.testing.assert_allclose(model_evaluation.estimate_core_tensor((A, B, C), X), core)

    D = rng.standard_normal(size=(7, 3))
    core = rng.standard_normal(size=(3, 3, 3, 3))
    tucker_tensor = (core, (A, B, C, D))
    X = factor_tools.construct_tucker_tensor(tucker_tensor)
    np.testing.assert_allclose(model_evaluation.estimate_core_tensor((A, B, C, D), X), core)


def test_core_consistency_cp_tensor(rng):
    A = rng.standard_normal(size=(4, 3))
    B = rng.standard_normal(size=(5, 3))
    C = rng.standard_normal(size=(6, 3))
    cp_decomposition = (None, (A, B, C))
    X = factor_tools.construct_cp_tensor(cp_decomposition)
    cc = model_evaluation.core_consistency(cp_decomposition, X)
    assert cc == pytest.approx(100)

    D = rng.standard_normal(size=(7, 3))
    cp_decomposition = (None, (A, B, C, D))
    X = factor_tools.construct_cp_tensor(cp_decomposition)
    cc = model_evaluation.core_consistency(cp_decomposition, X)
    assert cc == pytest.approx(100)
    

def test_core_consistency_with_known_tucker(rng):
    A = factor_tools.normalise(rng.standard_normal(size=(4, 3)))
    B = factor_tools.normalise(rng.standard_normal(size=(5, 3)))
    C = factor_tools.normalise(rng.standard_normal(size=(6, 3)))
    core = rng.standard_normal(size=(3, 3, 3))
    tucker_tensor = (core, (A, B, C))
    cp_tensor = (None, (A, B, C))
    X = factor_tools.construct_tucker_tensor(tucker_tensor)

    superdiagonal_ones = np.zeros((3, 3, 3))
    for i in range(3):
        superdiagonal_ones[i, i, i] = 1

    core_error = np.sum((core - superdiagonal_ones)**2)
    core_consistency = 100 - 100 * core_error / 3
    assert model_evaluation.core_consistency(cp_tensor, X, normalised=False) == pytest.approx(core_consistency)

    core_consistency = 100 - 100 * core_error / np.sum(core**2)
    assert model_evaluation.core_consistency(cp_tensor, X, normalised=True) == pytest.approx(core_consistency)
import numpy as np
import pytest
import scipy.linalg as sla
import tensorly as tl
from tensorly.random import random_cp

from component_vis import factor_tools, model_evaluation
from component_vis._utils import construct_cp_tensor, construct_tucker_tensor


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
    X = construct_tucker_tensor(tucker_tensor)
    X += rng.standard_normal(X.shape)

    slow_estimate = _estimate_core_tensor((A, B, C), X)
    np.testing.assert_allclose(model_evaluation.estimate_core_tensor((A, B, C), X).ravel(), slow_estimate)

    D = rng.standard_normal(size=(7, 3))
    core = rng.standard_normal(size=(3, 3, 3, 3))
    tucker_tensor = (core, (A, B, C, D))
    X = construct_tucker_tensor(tucker_tensor)
    X += rng.standard_normal(X.shape)
    slow_estimate = _estimate_core_tensor((A, B, C, D), X)
    np.testing.assert_allclose(model_evaluation.estimate_core_tensor((A, B, C, D), X).ravel(), slow_estimate)


def test_estimate_core_tensor_known_tucker(rng):
    A = rng.standard_normal(size=(4, 3))
    B = rng.standard_normal(size=(5, 3))
    C = rng.standard_normal(size=(6, 3))
    core = rng.standard_normal(size=(3, 3, 3))
    tucker_tensor = (core, (A, B, C))
    X = construct_tucker_tensor(tucker_tensor)
    np.testing.assert_allclose(model_evaluation.estimate_core_tensor((A, B, C), X), core)

    D = rng.standard_normal(size=(7, 3))
    core = rng.standard_normal(size=(3, 3, 3, 3))
    tucker_tensor = (core, (A, B, C, D))
    X = construct_tucker_tensor(tucker_tensor)
    np.testing.assert_allclose(model_evaluation.estimate_core_tensor((A, B, C, D), X), core)


def test_core_consistency_cp_tensor(rng):
    A = rng.standard_normal(size=(4, 3))
    B = rng.standard_normal(size=(5, 3))
    C = rng.standard_normal(size=(6, 3))
    cp_decomposition = (None, (A, B, C))
    X = construct_cp_tensor(cp_decomposition)
    cc = model_evaluation.core_consistency(cp_decomposition, X)
    assert cc == pytest.approx(100)

    D = rng.standard_normal(size=(7, 3))
    cp_decomposition = (None, (A, B, C, D))
    X = construct_cp_tensor(cp_decomposition)
    cc = model_evaluation.core_consistency(cp_decomposition, X)
    assert cc == pytest.approx(100)

    w = rng.standard_normal(size=(3))
    cp_decomposition = (w, (A, B, C, D))
    X = construct_cp_tensor(cp_decomposition)
    cc = model_evaluation.core_consistency(cp_decomposition, X)
    assert cc == pytest.approx(100)


def test_core_consistency_with_known_tucker(rng):
    A = factor_tools.normalise(rng.standard_normal(size=(4, 3)))
    B = factor_tools.normalise(rng.standard_normal(size=(5, 3)))
    C = factor_tools.normalise(rng.standard_normal(size=(6, 3)))
    core = rng.standard_normal(size=(3, 3, 3))
    tucker_tensor = (core, (A, B, C))
    cp_tensor = (None, (A, B, C))
    X = construct_tucker_tensor(tucker_tensor)

    superdiagonal_ones = np.zeros((3, 3, 3))
    for i in range(3):
        superdiagonal_ones[i, i, i] = 1

    core_error = np.sum((core - superdiagonal_ones) ** 2)
    core_consistency = 100 - 100 * core_error / 3
    assert model_evaluation.core_consistency(cp_tensor, X, normalised=False) == pytest.approx(core_consistency)

    core_consistency = 100 - 100 * core_error / np.sum(core ** 2)
    assert model_evaluation.core_consistency(cp_tensor, X, normalised=True) == pytest.approx(core_consistency)


def test_core_consistency_with_one_component(rng):
    """
    The core consistency of a one component model should be 100.

    The one component model is fitted with TensorLy.
    """
    X = np.array(
        [
            [
                [0.31147131, 0.52783545, 0.26642189, 0.64235561, 0.21453002, 0.52733376,],
                [0.90453298, 0.72577025, 0.42906596, 0.82763775, 0.80431794, 0.60144761,],
                [0.38229538, 0.62663986, 0.26112048, 0.68714129, 0.29639633, 0.57257494,],
                [0.58335707, 0.53130365, 0.20064246, 0.48084977, 0.55589144, 0.36965256,],
                [0.28321635, 0.62896352, 0.31016972, 0.7812717, 0.15945839, 0.6549802],
            ],
            [
                [0.16519647, 0.35396283, 0.06433226, 0.29446942, 0.15126687, 0.26900462,],
                [0.32653605, 0.59199309, 0.0887053, 0.45340927, 0.31944195, 0.41431456],
                [0.27286571, 0.58159856, 0.07688624, 0.4459044, 0.26453377, 0.4159955],
                [0.35707497, 0.7036691, 0.06581244, 0.49629388, 0.36419535, 0.4691295],
                [0.1729032, 0.39037184, 0.07657362, 0.33475942, 0.15348332, 0.30513679],
            ],
            [
                [0.30037468, 0.43326544, 0.15389418, 0.43000351, 0.25913448, 0.35877758,],
                [0.98137716, 0.80696452, 0.38246394, 0.80405414, 0.91380948, 0.59427858,],
                [0.41751219, 0.63848692, 0.16936067, 0.56614412, 0.38196243, 0.48741497,],
                [0.70775975, 0.77087099, 0.21936273, 0.64419824, 0.68493827, 0.52504628,],
                [0.24952121, 0.46833168, 0.15317907, 0.46932296, 0.20135204, 0.40443773,],
            ],
            [
                [0.25346554, 0.56232452, 0.22228408, 0.62677006, 0.17027687, 0.53685895,],
                [0.49892347, 0.7102087, 0.23699528, 0.6830915, 0.4398167, 0.57215165],
                [0.34165029, 0.72866965, 0.21686842, 0.7156926, 0.27110747, 0.62780439],
                [0.40707853, 0.64414682, 0.1028226, 0.48663565, 0.40321353, 0.43691713],
                [0.27148235, 0.67695523, 0.28095653, 0.78163621, 0.16013074, 0.6707594],
            ],
        ]
    )
    A = np.array([[2.88566669], [1.89412896], [2.84850993], [2.6549183]])
    B = np.array([[0.34164683], [0.57733088], [0.43787481], [0.45825688], [0.3850214]])
    C = np.array([[0.38033551], [0.52306521], [0.18119524], [0.5080392], [0.33523002], [0.42242025],])

    cc = model_evaluation.core_consistency((None, (A, B, C)), X, normalised=False)
    assert cc == pytest.approx(100)


def test_core_consistency_against_matlab(rng):
    """Test against the MATLAB implementation by Vagelis Papalexakis https://www.cs.ucr.edu/~epapalex/code.html

    The components are fitted using TensorLy.
    """
    X = np.array(
        [
            [
                [0.31147131, 0.52783545, 0.26642189, 0.64235561, 0.21453002, 0.52733376,],
                [0.90453298, 0.72577025, 0.42906596, 0.82763775, 0.80431794, 0.60144761,],
                [0.38229538, 0.62663986, 0.26112048, 0.68714129, 0.29639633, 0.57257494,],
                [0.58335707, 0.53130365, 0.20064246, 0.48084977, 0.55589144, 0.36965256,],
                [0.28321635, 0.62896352, 0.31016972, 0.7812717, 0.15945839, 0.6549802],
            ],
            [
                [0.16519647, 0.35396283, 0.06433226, 0.29446942, 0.15126687, 0.26900462,],
                [0.32653605, 0.59199309, 0.0887053, 0.45340927, 0.31944195, 0.41431456],
                [0.27286571, 0.58159856, 0.07688624, 0.4459044, 0.26453377, 0.4159955],
                [0.35707497, 0.7036691, 0.06581244, 0.49629388, 0.36419535, 0.4691295],
                [0.1729032, 0.39037184, 0.07657362, 0.33475942, 0.15348332, 0.30513679],
            ],
            [
                [0.30037468, 0.43326544, 0.15389418, 0.43000351, 0.25913448, 0.35877758,],
                [0.98137716, 0.80696452, 0.38246394, 0.80405414, 0.91380948, 0.59427858,],
                [0.41751219, 0.63848692, 0.16936067, 0.56614412, 0.38196243, 0.48741497,],
                [0.70775975, 0.77087099, 0.21936273, 0.64419824, 0.68493827, 0.52504628,],
                [0.24952121, 0.46833168, 0.15317907, 0.46932296, 0.20135204, 0.40443773,],
            ],
            [
                [0.25346554, 0.56232452, 0.22228408, 0.62677006, 0.17027687, 0.53685895,],
                [0.49892347, 0.7102087, 0.23699528, 0.6830915, 0.4398167, 0.57215165],
                [0.34165029, 0.72866965, 0.21686842, 0.7156926, 0.27110747, 0.62780439],
                [0.40707853, 0.64414682, 0.1028226, 0.48663565, 0.40321353, 0.43691713],
                [0.27148235, 0.67695523, 0.28095653, 0.78163621, 0.16013074, 0.6707594],
            ],
        ]
    )

    weights = np.array([1.0, 1.0])
    A = np.array(
        [[2.76226953, -0.53390772], [1.75876538, -0.34060455], [1.79354941, -0.71522305], [3.22462239, -0.33290826],]
    )
    B = np.array(
        [
            [-0.38495969, 0.26550081],
            [-0.28370532, 1.14039406],
            [-0.44012596, 0.4361716],
            [-0.22939878, 0.88900254],
            [-0.50484759, 0.16563351],
        ]
    )
    C = np.array(
        [
            [-0.12695248, -1.08474105],
            [-0.40821875, -0.84407922],
            [-0.14334649, -0.29662345],
            [-0.44798284, -0.68956002],
            [-0.06783491, -1.07348081],
            [-0.39525733, -0.50830426],
        ]
    )

    cc = model_evaluation.core_consistency((weights, (A, B, C)), X, normalised=False)
    assert cc == pytest.approx(99.830437445788107)


def test_sse(rng):
    cp = random_cp((4, 5, 6), 3, random_state=rng)
    tensor = cp.to_tensor()
    noise = rng.random_sample((4, 5, 6))
    sse = model_evaluation.sse(cp, tensor + noise)
    assert sse == pytest.approx(tl.sum(noise ** 2))


def test_relative_sse(rng):
    cp = random_cp((4, 5, 6), 3, random_state=rng)
    tensor = cp.to_tensor()
    noise = rng.random_sample((4, 5, 6))
    rel_sse = model_evaluation.relative_sse(cp, tensor + noise)
    assert rel_sse == pytest.approx(tl.sum(noise ** 2) / tl.sum((tensor + noise) ** 2))


def test_fit(rng):
    cp = random_cp((4, 5, 6), 3, random_state=rng)
    tensor = cp.to_tensor()
    noise = rng.random_sample((4, 5, 6))
    fit = model_evaluation.fit(cp, tensor + noise)
    assert fit == pytest.approx(1 - tl.sum(noise ** 2) / tl.sum((tensor + noise) ** 2))

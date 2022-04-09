import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture
def seed(pytestconfig):
    try:
        return pytestconfig.getoption("randomly_seed")
    except AttributeError:
        return 1


@pytest.fixture
def rng(seed):
    return np.random.RandomState(seed=seed)


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    yield
    plt.close("all")

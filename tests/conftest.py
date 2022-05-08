import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorly as tl


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


@pytest.fixture(autouse=True)
def set_numpy_backend():
    yield
    tl.set_backend("numpy")
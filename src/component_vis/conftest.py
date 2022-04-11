import matplotlib.pyplot as plt
import pytest

# TODO: Set matplotlib backend to pdfagg ? the name of the pdf backend at least
# TODO: Add fixture that closes all figures after each test


def pytest_configure(config):
    import matplotlib

    matplotlib.use("pdf")


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    yield
    plt.close("all")

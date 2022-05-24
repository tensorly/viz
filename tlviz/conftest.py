# -*- coding: utf-8 -*-

__author__ = "Marie Roald & Yngve Mardal Moe"

import matplotlib.pyplot as plt
import pytest


def pytest_configure(config):
    import matplotlib

    matplotlib.use("pdf")


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    yield
    plt.close("all")

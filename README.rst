=========================================================
ComponentVis — Visualising and analysing component models
=========================================================

.. image:: https://github.com/marieroald/componentvis/workflows/tests/badge.svg
    :target: https://github.com/MarieRoald/componentvis/actions/workflows/tests.yml
    :alt: Tests

.. image:: https://codecov.io/gh/MarieRoald/componentvis/branch/master/graph/badge.svg?token=BYEME3G8KG
    :target: https://codecov.io/gh/MarieRoald/componentvis
    :alt: Coverage

.. image:: https://readthedocs.org/projects/componentvis/badge/?version=latest
        :target: https://componentvis.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

ComponentVis is a Python package for visualising component-based decomposition models like PARAFAC and PCA.

Documentation
-------------

The documentation
is available on `ReadTheDocs <https://componentvis.readthedocs.io/en/latest/?badge=latest>`_ and includes

* A primer on tensors, tensor factorisations and the notation we use
* An example gallery
* The API reference
 

Dependencies
------------

ComponentVis supports Python 3.7 or above (it may also work with Python 3.6, though that is not officially supported).

Installation requires matplotlib, numpy, pandas, scipy, statsmodels and xarray. 

Installation
------------

To install the latest stable release of ComponentVis and its dependencies, run:

.. code:: raw

    pip install componentvis

There is also functionality to create improved QQ-plots with Pingoiun.
However, this is disabled by default due to the restrictive GPL lisence.
To enable this possibility, you must manually `install Pingoiun <https://pingouin-stats.org>`_.

To install the latest development version of ComponentVis, you can either clone
this repo or run

.. code:: raw

    pip install git+https://github.com/marieroald/componentvis.git


Example
-------

.. code:: python
    
    import component_vis
    import matplotlib.pyplot as plt
    from tensorly.decomposition import parafac

    def fit_parafac(dataset, num_components, num_inits):
        model_candidates = [
            parafac(dataset.data, num_components, init="random", random_state=i)
            for i in range(num_inits)
        ]
        model = component_vis.multimodel_evaluation.get_model_with_lowest_error(
            model_candidates, dataset
        )
        return component_vis.postprocessing.postprocess(model, dataset)

    data = component_vis.data.load_aminoacids()
    cp_tensor = fit_parafac(data, 3, num_inits=3)
    component_vis.visualisation.components_plot(cp_tensor)
    plt.show()

.. code:: raw

    Loading Aminoacids dataset from:
    Bro, R, PARAFAC: Tutorial and applications, Chemometrics and Intelligent Laboratory Systems, 1997, 38, 149-171

.. image:: docs/figures/readme_example.svg
    :width: 800
    :alt: An example figure showing the component vectors of a three component PARAFAC model fitted to a fluoresence spectroscopy dataset.

This example uses TensorLy to fit five three-component PARAFAC models to the data. Then it uses ComponentVis to:

1. Select the model that gave the lowest reconstruction error,
1. normalise the component vectors, storing their magnitude in a separate weight-vector,
1. permute the components in descending weight (i.e. signal strength) order,
1. flip the components so they point in a logical direction compared to the data,
1. convert the factor matrices into Pandas DataFrames with logical indices,
1. and plot the components using matplotlib.

All these steps are described with references to the literature.

Testing
-------

The test suite requires an additional set of dependencies. To install these, run

.. code:: raw

    pip install componentvis[test]

or

.. code:: raw

    pip install -e .[test]

inside your local copy of the ComponentVis repository.

The tests can be run by calling ``pytest`` with no additional arguments.
All doctests are ran by default and a coverage summary will be printed on the screen.
To generate a coverage report, run ``coverage html``.

Contributing
------------

Contributions are welcome to ComponentVis, see the `contribution guidelines <https://componentvis.readthedocs.io/en/latest/contributing.html>`_.
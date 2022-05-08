=========================================================
TLVis — Visualising and analysing component models
=========================================================

.. image:: https://github.com/marieroald/tlvis/workflows/tests/badge.svg
    :target: https://github.com/MarieRoald/tlvis/actions/workflows/tests.yml
    :alt: Tests

.. image:: https://codecov.io/gh/MarieRoald/tlvis/branch/master/graph/badge.svg?token=BYEME3G8KG
    :target: https://codecov.io/gh/MarieRoald/tlvis
    :alt: Coverage

.. image:: https://readthedocs.org/projects/tlvis/badge/?version=latest
        :target: https://tlvis.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

TLVis is a Python package for visualising component-based decomposition models like PARAFAC and PCA.

Documentation
-------------

The documentation
is available on `ReadTheDocs <https://tlvis.readthedocs.io/en/latest/?badge=latest>`_ and includes

* A `primer on tensors <https://tlvis.readthedocs.io/en/latest/about_tensors.html#what-are-tensors-and-tensor-decompositions>`_, `tensor factorisations <https://tlvis.readthedocs.io/en/latest/about_tensors.html#what-are-tensor-factorisations>`_ and the `notation we use <https://tlvis.readthedocs.io/en/latest/about_tensors.html#notation>`_
* `An example gallery <https://tlvis.readthedocs.io/en/latest/auto_examples/index.html>`_
* `The API reference <https://tlvis.readthedocs.io/en/latest/api.html>`_
 

Dependencies
------------

TLVis supports Python 3.7 or above (it may also work with Python 3.6, though that is not officially supported).

Installation requires matplotlib, numpy, pandas, scipy, statsmodels and xarray. 

Installation
------------

To install the latest stable release of TLVis and its dependencies, run:

.. code:: raw

    pip install tlvis

There is also functionality to create improved QQ-plots with Pingoiun.
However, this is disabled by default due to the restrictive GPL lisence.
To enable this possibility, you must manually `install Pingoiun <https://pingouin-stats.org>`_.

To install the latest development version of TLVis, you can either clone
this repo or run

.. code:: raw

    pip install git+https://github.com/marieroald/tlvis.git


Example
-------

.. code:: python
    
    import tlvis
    import matplotlib.pyplot as plt
    from tensorly.decomposition import parafac

    def fit_parafac(dataset, num_components, num_inits):
        model_candidates = [
            parafac(dataset.data, num_components, init="random", random_state=i)
            for i in range(num_inits)
        ]
        model = tlvis.multimodel_evaluation.get_model_with_lowest_error(
            model_candidates, dataset
        )
        return tlvis.postprocessing.postprocess(model, dataset)

    data = tlvis.data.load_aminoacids()
    cp_tensor = fit_parafac(data, 3, num_inits=3)
    tlvis.visualisation.components_plot(cp_tensor)
    plt.show()

.. code:: raw

    Loading Aminoacids dataset from:
    Bro, R, PARAFAC: Tutorial and applications, Chemometrics and Intelligent Laboratory Systems, 1997, 38, 149-171

.. image:: docs/figures/readme_example.svg
    :width: 800
    :alt: An example figure showing the component vectors of a three component PARAFAC model fitted to a fluoresence spectroscopy dataset.

This example uses TensorLy to fit five three-component PARAFAC models to the data. Then it uses TLVis to:

#. Select the model that gave the lowest reconstruction error,
#. normalise the component vectors, storing their magnitude in a separate weight-vector,
#. permute the components in descending weight (i.e. signal strength) order,
#. flip the components so they point in a logical direction compared to the data,
#. convert the factor matrices into Pandas DataFrames with logical indices,
#. and plot the components using matplotlib.

All these steps are described in the `API documentation <https://tlvis.readthedocs.io/en/latest/api.html>`_ with references to the literature.

Testing
-------

The test suite requires an additional set of dependencies. To install these, run

.. code:: raw

    pip install tlvis[test]

or

.. code:: raw

    pip install -e .[test]

inside your local copy of the TLVis repository.

The tests can be run by calling ``pytest`` with no additional arguments.
All doctests are ran by default and a coverage summary will be printed on the screen.
To generate a coverage report, run ``coverage html``.

Contributing
------------

Contributions are welcome to TLVis, see the `contribution guidelines <https://tlvis.readthedocs.io/en/latest/contributing.html>`_.
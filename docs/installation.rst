.. highlight:: shell

============
Installation
============


Stable release
--------------

To install TLViz, run this command in your terminal:

.. code-block:: console

    pip install tensorly-viz

Alternatively, to get the latest development version of TLViz, run:

.. code-block:: console

    pip install git+https://github.com/tensorly/viz


Interoperability with TensorLy
------------------------------
TLViz doesn't strictly depend on TensorLy.
However, we recommend installing both if you want to analyse multi-way data with tensor decomposition.

Optional dependencies
---------------------
By default, TLViz uses SciPy to generate QQ-plots.
However, the GPL lisenced `Pingouin <https://pingouin-stats.org/>`_ package can be used instead if you want better looking QQ-plots with confidence intervals.
To install the optional (GPL lisenced) dependencies run

.. code:: bash

    pip install tensorly-viz[all]


Running the examples
--------------------
The examples depend on a variety of extra packages, such as TensorLy and PlotLy. 
To install these as well, run

.. code:: bash

    pip install tensorly-viz[docs]


Running the test suite
----------------------
The test suite have some extra dependencies, such as pytest and pytorch.
To install these, run

.. code:: bash

    pip install tensorly-viz[test]

Installing multiple extra dependencies
--------------------------------------
If you want to install multiple extra dependencies at once (e.g. ``docs`` and ``test``), consider running

.. code:: bash

    pip install tensorly-viz[docs,test]
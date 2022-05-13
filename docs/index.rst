TLVis — Visualising and analysing component models
==================================================

.. figure:: figures/overview.png
   :alt: Illustration of figures made with TLVis

TLVis, short for TensorLy-Vis, is a Python package that helps you inspect your component models with insightful analysis and beautiful visualisations!
It is designed around the TensorLy API for PARAFAC/CP decompositions to fit seamlessly into your data analysis workflow.
With TLVis, you can combine metadata xarray DataArrays and Pandas DataFrames with your extracted factor matrices, turning them into rich DataFrames with meaningful indices.
To get started, take a look at the :ref:`examples` or the :ref:`API`.

.. figure:: figures/cp_tensor.svg
   :alt: Illustration of a CP tensor
   :width: 90 %

   Illustration of a PARAFAC/CP model and how such models are represented by TensorLy and TLVis. See :ref:`about-tensors` for more information.


.. toctree::
   :maxdepth: 2

   about_tensors
   installation
   auto_examples/index
   api
   tensorly_backends
   contributing
   references

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

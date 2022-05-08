.. _tensorly-backends:

Working with TensorLy
=====================

TensorLy supports a variety of different backends, which makes it possible to fit tensor decompositions
on the GPU. However, we generally do not need the compute power of deep learning frameworks when analysing and visualising tensor decompositions.
For simplicity, TLVis, therefore, officially only supports the NumPy
backend, but it also has (mostly) untested experimental support for all other backends.
If you encounter any bugs with the non-NumPy backends, then we encourage you to submit an
`issue detailing the problem <https://github.com/MarieRoald/tlvis/issues/new/choose>`_.

.. _about-tensors:

What are tensors and tensor decompositions?
===========================================

Tensors are higher-order generalisations of a matrix.
A matrix is a square of numbers and is, therefore, a second-order tensor.
However, we can also have a cube of numbers, which would be a third-order tensor.
So, a third-order tensor is a three-dimensional cube of numbers.
However, since *dimensions* often denote the number of entries in a vector, we do not use the word
dimension for tensors. Instead, we use the word *mode* :cite:p:`kolda2009tensor`.

Tensors in Python are often represented by NumPy arrays, and NumPy uses the word *axis* for the different modes.
Therefore, in TLViz, we alias ``mode`` and ``axis`` in all the function calls.
However, the word ``mode`` is generally used in the documentation.


Notation
--------


.. figure:: figures/cp_tensor.svg
   :alt: Illustration of a CP tensor
   :width: 90 %

Above, we see an illustration of the notation used in TLViz.
We represent dense tensors either as NumPy arrays,
Pandas DataFrames or xarray DataArrays with the name ``dataset``.
PARAFAC (or CP or CPD) models are represented by a tuple, where the first element is a vector of weights,
one for each component, and the second element is a list of factor matrices, one for each mode.
This form is compatible with TensorLy, and we have also taken care to give variables
on this form the name ``cp_tensor`` to match TensorLy.

In TLViz, we also use the terms *labelled* and *unlabelled* dataset and decompositions.
A labelled dataset is either a Pandas DataFrame or an xarray DataArray.
By using Pandas and xarray objects, we keep the metadata together with the dataset,
making it easier to produce rich visualisations later.
Similarly, a labelled decomposition is a decomposition where the factor matrices are stored as Pandas DataFrames
with an index equal to the coordinates along the corresponding mode of the xarray DataArray.
TLViz can seamlessly work with both labelled and unlabelled data and decompositions,
but we recommend using the labelled variant whenever possible.


What are tensor factorisations?
-------------------------------
Similarly to matrix factorisation methods, tensor factorisation methods decompose a tensor into low-rank components.
These components can be very informative and give insight into the patterns in the data.
One of the most commonly used tensor factorisation methods is PARAFAC (also known as CP and CPD).
PARAFAC can be considered a generalisation of two-way methods such as principal component analysis (PCA) and nonnegative matrix factorisation (NMF) to higher-order data.
Let us here see how PARAFAC decomposition relates to matrix decomposition.
With matrix factorisation, we describe a matrix, :math:`\mathbf{X}`, as the outer product of two-factor matrices:

.. math::

    \mathbf{X} \approx \mathbf{A} \mathbf{B}^\mathsf{T},

where :math:`\mathbf{A}` and :math:`\mathbf{B}` are *factor matrices* that contain the patterns along the rows and
columns of :math:`\mathbf{X}`. The first component is represented by the first columns of :math:`\mathbf{A}` and :math:`\mathbf{B}`,
the second component is represented by the second columns and so forth.

We can look at an example to better understand these factor matrices. If :math:`\mathbf{X}` is a matrix
of movie scores given by various users, then each component could represent a genre and the :math:`i`-th row of
:math:`\mathbf{A}` could contain information about which movie genres the :math:`i`-th user likes. Likewise, the
:math:`j`-th row of :math:`\mathbf{B}` could contain information about how "strongly" each movie belonged to each
genre.

The next step is to generalise this for more dimensions. We may for example have time. In that case, we have a
tensor, :math:`\mathcal{X}`, which represents how much different people like different movies at different time
points. To see how we can matrix factorisation for such a case, we first rewrite the equation above so we consider
each entry, :math:`x_{ij}`, of :math:`\mathbf{X}` instead:

.. math::

    x_{ij} \approx \sum_{r=1}^R a_{ir} b_{jr}.

If we now introduce a third mode (represented by a new index, :math:`k`), we see an obvious way to extend this

.. math::

    x_{ijk} \approx \sum_{r=1}^R a_{ir} b_{jr} c_{kr}.

Here, we have three sets of factor matrices, :math:`\mathbf{A}`, :math:`\mathbf{B}` and :math:`\mathbf{C}`.
In the movie example, the first two factor matrices still represent the same. The third-factor matrix,
:math:`\mathbf{C}` represents how popular the different genres are at different time points.

The model we just described is called the PARAFAC, CP or CPD model.
However, it is also common to introduce *weights* to the components.
These weights represent each component's "signal strength" (similar to a singular value if you are familiar with the SVD).
If we include the weights in the equation
above, we get

.. math::

    x_{ijk} \approx \sum_{r=1}^R w_r a_{ir} b_{jr} c_{kr}.


This overview was only a very brief introduction to tensor factorisations.
For a more thorough introduction, we recommend :cite:p:`kolda2009tensor` (a thorough introduction to tensors)
and :cite:p:`bro1997parafac` (a thorough introduction to PARAFAC).
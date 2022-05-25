"""
Working with xarray
-------------------

TLViz recommends storing your datasets as
`xarray DataArrays <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_, which
supports labelled multi-way datasets, so the metadata is stored together with the dataset in
one object.

This example shows how you can create and work with xarray DataArrays.
For this, we will create a simulated dataset where the modes represent time-of-day, day-of-week and month and
the entries represent some count value.
"""

###############################################################################
# Imports and utilities
# ^^^^^^^^^^^^^^^^^^^^^
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

rng = np.random.default_rng(0)

###############################################################################
# Creating the simulated dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
numpy_data = rng.poisson(10, size=(24, 7, 12))
hour_label = range(1, 25)
weekday_label = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
month_label = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

###############################################################################
# Storing this dataset as an xarray data array
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
dataset = xr.DataArray(
    data=numpy_data,
    coords={
        "month": month_label,
        "day-of-week": weekday_label,
        "hour": hour_label,
    },
    dims=["hour", "day-of-week", "month"]
)
dataset

###############################################################################
# Slicing DataArrays
# ^^^^^^^^^^^^^^^^^^
#
# There are two common ways to slice xarray DataArrays, either by numerical index or by coordinate.

dataset[0, 0, 0]

###############################################################################
dataset.loc[1, "Mon", "Jan"]

###############################################################################
dataset.loc[{"month": "Jan", "hour": 1, "day-of-week": "Mon"}]

###############################################################################
# Arithmetic on DataArrays
# ^^^^^^^^^^^^^^^^^^^^^^^^
# xarray includes functionality that makes it very easy to perform reduction operations, such as averages
# and standard deviations across the different modes of a dataset. Below, we compute the average across
# the hour mode.

dataset.mean("hour")

###############################################################################
# Plotting DataArrays
# ^^^^^^^^^^^^^^^^^^^
#
# DataArrays also provide powerful plotting functionality. You can, for example, easily create both
# heatmaps and histograms. For more examples of the xarray functionality, see the
# `xarray example gallery <https://docs.xarray.dev/en/stable/gallery.html>`_.

dataset.plot()
plt.show()

###############################################################################
dataset.mean(dim="hour").plot()
plt.show()

###############################################################################
# Accessing the underlying dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# TensorLy does not support DataArrays, so to fit tensor decomposition models,
# you need to use the ``data``-attribute of the DataArray
# to access the NumPy array that xarray stores the data in behind the scenes.

dataset.data
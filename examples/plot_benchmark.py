"""
Benchmarking the TensorLy PARAFAC methods
=================================================

This example demonstrates how to benchmark Tensorly methods, using dataset available in Tensorly.
The user may also compare built-in methods with any external method. 

.. note::
    Our goal is to provide more generalized benchmark functions to Tensorly users in the future versions of Tensorly.
"""

##############################################################################
# Introduction
# -----------------------
# As of Autumn 2022, Tensorly includes a few dataset:
#
# 1. IL2data
#        * Mutein treatment responses
# 2. Covid19
#        * Three-mode tensor of samples, antigens, and receptors
# 3. Indian pines
#        * Hyperspectral image
# 4. Kinetic
#        * 4-way quantified measurement data
#
# Each dataset includes meta-information such as a scholar reference, the name of the modes,
# a suggested data mining task and a description. This information is used as labels when visualizing the results.

import tensorly as tl
from tensorly.metrics import RMSE
from tensorly.datasets import load_IL2data
import numpy as np

##########################################################################
# First let us load the IL2data dataset. The syntax is inspired from scikit-learn.
# For simplicity here we replaced missing values in the data with zeroes.
dataset = load_IL2data()
tensor = dataset.tensor
tensor[np.isnan(tensor)] = 0
dims = dataset.dims
print(dims)


##############################################################################
# How to compare several methods
# --------------------------------------------
# To compare several methods fairly, we suggest to use same initial factors and errors
# for all of them. Here we use `initialize_cp` as an example. You can also define
# your initial CPTensor.

from tensorly.decomposition._nn_cp import initialize_cp
rank = 5

initial_tensor = initialize_cp(tensor, rank=rank, init="random", non_negative=True)
first_error = tl.norm(tensor - tl.cp_to_tensor(initial_tensor), 2) / tl.norm(tensor, 2)

##########################################################################
# Let us import some decomposition methods from Tensorly. We are going to compare three built-in algorithms:
# - non_negative_parafac (using multiplicative updates)
# - non_negative_parafac_hals (using block-coordinate updates)
# - parafac (an implementation of alternating least squares, without nonnegativity)
# Then, we put them in a list below. You can also append your method to this list to compare your own method
# with them.

from tensorly.decomposition import non_negative_parafac_hals, non_negative_parafac, parafac
method = [non_negative_parafac, non_negative_parafac_hals, parafac]

##########################################################################
# If you have a custom algorithm `my_alg` to compare with, simply add the function name as follows
# method = [non_negative_parafac, non_negative_parafac_hals, parafac, my_alg]
# Note that it should have a signature including the following inputs
# my_alg(tensor, rank, n_iter_max=, init=, tol=, return_errors=)


###########################################################################
# Here we define some variables to compare selected methods.

import time

# A few storage arrays
cp_tensors = []
errors = []
rmse = []
proc_time = []
# Algorithms hyperparameters
n_iter_max = 100
tol = 1-16

##########################################################################
# We can run each method in a loop and save the results. Then, we print
# root-mean-square error and processing time for each method.

for i in range(len(method)):
    tic = time.time()
    cp_tensor_res, error = method[i](tensor, rank, n_iter_max=n_iter_max, init=initial_tensor.cp_copy(), tol=tol,
                                     return_errors=True)
    proc_time.append(time.time() - tic)
    error.insert(0, first_error)  # assuming the methods do not compute initial error, as in tensorly.
    cp_tensors.append(cp_tensor_res.cp_copy())
    errors.append(error)
    rmse.append(RMSE(tensor, tl.cp_to_tensor(cp_tensor_res)))
    print("RMSE with" + ' ' + str(method[i].__name__) + ":" + str("{:.2f}".format(rmse[i])))

for i in range(len(proc_time)):
    print("Processing time of" + ' ' + str(method[i].__name__) + ":", str("{:.2f}".format(proc_time[i])))

##########################################################################
# We can plot error per iteration now. Since we inserted same first error for
# each method, error will start to decrease from the same point.

import matplotlib.pyplot as plt

plt.figure()
plt.title("Error for each iteration")
for i in range(len(errors)):
    plt.plot(errors[i])
plt.legend([str(method[i].__name__) for i in range(len(errors))])
plt.show()

##########################################################################
# We saved factors from all the methods in cp_tensors. Now, we will plot
# them by using a suitable function from tlviz. First, in order to benefit
# from label information, we will convert tensorly dataset dictionary to xarray.

import xarray as xr
from tlviz.visualisation import component_comparison_plot
from tlviz.postprocessing import postprocess

dataset = xr.DataArray(dataset.tensor, coords=dataset.ticks,
                       dims=dataset.dims)
fig, axes = component_comparison_plot({method[i].__name__: postprocess(cp_tensors[i], dataset) for i in range(len(method))})
plt.show()

###########################################################################
# It is also possible to use your own ndarray as a tensor. However, figures
# won't have labels.

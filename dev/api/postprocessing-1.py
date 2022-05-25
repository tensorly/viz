import tlviz
import numpy as np
import matplotlib.pyplot as plt
from tensorly.decomposition import parafac
dataset = tlviz.data.load_aminoacids()
# Expected:
## Loading Aminoacids dataset from:
## Bro, R, PARAFAC: Tutorial and applications, Chemometrics and Intelligent Laboratory Systems, 1997, 38, 149-171
#
# The dataset is an xarray DataArray and it contains relevant side information
#
print(type(dataset))
# Expected:
## <class 'xarray.core.dataarray.DataArray'>
#
# We see that after postprocessing, the cp_tensor contains pandas DataFrames
#
cp_tensor = parafac(dataset.data, 3, init="random", random_state=0)
cp_tensor_postprocessed = tlviz.postprocessing.postprocess(cp_tensor, dataset)
print(type(cp_tensor[1][0]))
# Expected:
## <class 'numpy.ndarray'>
print(type(cp_tensor_postprocessed[1][0]))
# Expected:
## <class 'pandas.core.frame.DataFrame'>
#
# We see that after postprocessing, the factor matrix has unit norm
#
print(np.linalg.norm(cp_tensor[1][0], axis=0))
# Expected:
## [160.82985402 182.37338941 125.3689186 ]
print(np.linalg.norm(cp_tensor_postprocessed[1][0], axis=0))
# Expected:
## [1. 1. 1.]
#
# When we construct a dense tensor from a postprocessed cp_tensor it is constructed
# as an xarray DataArray
#
print(type(tlviz.utils.cp_to_tensor(cp_tensor)))
# Expected:
## <class 'numpy.ndarray'>
print(type(tlviz.utils.cp_to_tensor(cp_tensor_postprocessed)))
# Expected:
## <class 'xarray.core.dataarray.DataArray'>
#
# The visualisation of the postprocessed cp_tensor shows that the scaling and sign indeterminacy
# is taken care of and x-xaxis has correct labels and ticks
#
fig, ax = tlviz.visualisation.components_plot(cp_tensor)
plt.show()
#
fig, ax = tlviz.visualisation.components_plot(cp_tensor_postprocessed)
plt.show()

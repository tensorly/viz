import matplotlib.pyplot as plt
from tensorly.decomposition import parafac
from tlviz.data import simulated_random_cp_tensor
from tlviz.visualisation import histogram_of_residuals
true_cp, X = simulated_random_cp_tensor((10, 20, 30), 3, seed=0)
est_cp = parafac(X, 3)
histogram_of_residuals(est_cp, X)
# Expected:
## <AxesSubplot:title={'center':'Histogram of residuals'}, xlabel='Standardised residuals', ylabel='Frequency'>
plt.show()

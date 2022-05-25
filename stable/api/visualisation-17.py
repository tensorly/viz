import matplotlib.pyplot as plt
from tensorly.decomposition import parafac
from tlviz.data import simulated_random_cp_tensor
from tlviz.visualisation import residual_qq
true_cp, X = simulated_random_cp_tensor((10, 20, 30), 3, seed=0)
est_cp = parafac(X, 3)
residual_qq(est_cp, X)
# Expected:
## <AxesSubplot:title={'center':'QQ-plot of residuals'}, xlabel='Theoretical Quantiles', ylabel='Sample Quantiles'>
plt.show()

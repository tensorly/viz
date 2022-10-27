import matplotlib.pyplot as plt
from tensorly.decomposition import parafac
from tlviz.data import simulated_random_cp_tensor
from tlviz.visualisation import core_element_plot
true_cp, X = simulated_random_cp_tensor((10, 20, 30), 3, seed=42)
est_cp = parafac(X, 3)
core_element_plot(est_cp, X)
# Expected:
## <AxesSubplot: title={'center': 'Core consistency: 99.8'}, xlabel='Core element', ylabel='Value'>
plt.show()

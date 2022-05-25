from tlviz.visualisation import core_element_heatmap
from tlviz.data import simulated_random_cp_tensor
import matplotlib.pyplot as plt
cp_tensor, dataset = simulated_random_cp_tensor((20, 30, 40), 3, seed=0)
fig, axes = core_element_heatmap(cp_tensor, dataset)
plt.show()

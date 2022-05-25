from tlviz.data import simulated_random_cp_tensor
from tlviz.visualisation import scree_plot
import matplotlib.pyplot as plt
from tensorly.decomposition import parafac
dataset = simulated_random_cp_tensor((10, 20, 30), rank=3, noise_level=0.2, seed=42)[1]
cp_tensors = {}
for rank in range(1, 5):
    cp_tensors[f"{rank} components"] = parafac(dataset, rank, random_state=1)
ax = scree_plot(cp_tensors, dataset)
plt.show()

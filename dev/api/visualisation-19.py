from tlviz.data import simulated_random_cp_tensor
from tlviz.visualisation import scree_plot
import matplotlib.pyplot as plt
from tensorly.decomposition import parafac
dataset = simulated_random_cp_tensor((10, 20, 30), rank=3, noise_level=0.2, seed=42)[1]
cp_tensors = {}
for rank in range(1, 5):
    cp_tensors[rank] = parafac(dataset, rank, random_state=1)
fig, axes = plt.subplots(1, 2, figsize=(8, 2), tight_layout=True)
ax = scree_plot(cp_tensors, dataset, ax=axes[0])
ax = scree_plot(cp_tensors, dataset, metric="Core consistency", ax=axes[1])
for ax in axes:
    xlabel = ax.set_xlabel("Number of components")
    xticks = ax.set_xticks(list(cp_tensors.keys()))
limits = axes[1].set_ylim((0, 105))
plt.show()

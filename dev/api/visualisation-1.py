import matplotlib.pyplot as plt
from tensorly.decomposition import parafac, non_negative_parafac_hals
from tlviz.data import simulated_random_cp_tensor
from tlviz.visualisation import component_comparison_plot
from tlviz.postprocessing import postprocess
true_cp, X = simulated_random_cp_tensor((10, 20, 30), 3, noise_level=0.5, seed=42)
cp_tensors = {
    "True": true_cp,
    "CP": parafac(X, 3),
    "NN CP": non_negative_parafac_hals(X, 3),
}
fig, axes = component_comparison_plot(cp_tensors, row="component")
plt.show()

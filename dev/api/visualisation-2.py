import matplotlib.pyplot as plt
from tlviz.data import simulated_random_cp_tensor
from tlviz.factor_tools import permute_cp_tensor
from tlviz.postprocessing import postprocess
from tlviz.visualisation import component_comparison_plot
four_components = simulated_random_cp_tensor((5, 6, 7), 4, noise_level=0.5, seed=42)[0]
three_components = permute_cp_tensor(four_components, permutation=[0, 1, 2])
two_components = permute_cp_tensor(four_components, permutation=[0, 2])
cp_tensors = {
    "True": three_components,  # Reference decomposition
    "subset": two_components,  # Only component 0 and 2
    "superset": four_components,  # All components in reference plus one additional
}
fig, axes = component_comparison_plot(cp_tensors, row="model")
plt.show()

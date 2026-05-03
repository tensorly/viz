import numpy as np
import matplotlib.pyplot as plt
from tensorly.random import random_cp
from tensorly.decomposition import parafac
from tlviz.visualisation import optimisation_diagnostic_plots
rng = np.random.RandomState(1)
cp_tensor = random_cp((5, 6, 7), 2, random_state=rng)
dataset = cp_tensor.to_tensor() + rng.standard_normal((5, 6, 7))
errs = []
for i in range(10):
    errs.append(parafac(dataset, 3, n_iter_max=500, return_errors=True, init="random", random_state=rng)[1])
fig, axes = optimisation_diagnostic_plots(errs, 500)
plt.show()

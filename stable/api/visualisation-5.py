from tensorly.random import random_cp
from tlviz.visualisation import components_plot
import matplotlib.pyplot as plt
cp_tensor = random_cp(shape=(5,10,15), rank=3)
fig, axes = components_plot(cp_tensor)
plt.show()

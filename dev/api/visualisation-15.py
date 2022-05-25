from tlviz.visualisation import percentage_variation_plot
from tlviz.data import simulated_random_cp_tensor
import matplotlib.pyplot as plt
cp_tensor, dataset = simulated_random_cp_tensor(shape=(5,10,15), rank=3, noise_level=0.5, seed=0)
percentage_variation_plot(cp_tensor, dataset, method="data")
# Expected:
## <AxesSubplot:xlabel='Component number', ylabel='Percentage variation explained [%]'>
plt.show()

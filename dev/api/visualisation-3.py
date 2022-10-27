from tensorly.random import random_cp
from tlviz.visualisation import component_scatterplot
import matplotlib.pyplot as plt
cp_tensor = random_cp(shape=(5,10,15), rank=2)
component_scatterplot(cp_tensor, mode=0)
# Expected:
## <AxesSubplot: title={'center': 'Component plot'}, xlabel='Component 0', ylabel='Component 1'>
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from tlviz.postprocessing import label_cp_tensor
from tlviz.visualisation import component_scatterplot
stocks = px.data.stocks().set_index("date").stack()
stocks.index.names = ["Date", "Stock"]
stocks = stocks.to_xarray()
stocks -= stocks.mean(axis=0)
U, s, Vh = np.linalg.svd(stocks, full_matrices=False)
num_components = 2
cp_tensor = s[:num_components], (U[:, :num_components], Vh.T[:, :num_components])
cp_tensor = label_cp_tensor(cp_tensor, stocks)
component_scatterplot(cp_tensor, mode=1)
# Expected:
## <AxesSubplot:title={'center':'Component plot'}, xlabel='Component 0', ylabel='Component 1'>
plt.show()

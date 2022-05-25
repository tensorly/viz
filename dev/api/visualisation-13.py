import matplotlib.pyplot as plt
from tensorly.decomposition import non_negative_parafac_hals
from tlviz.data import load_oslo_city_bike
from tlviz.postprocessing import postprocess
from tlviz.visualisation import outlier_plot
data = load_oslo_city_bike()
X = data.data
cp = non_negative_parafac_hals(X, 3, init="random")
cp = postprocess(cp, dataset=data, )
outlier_plot(
    cp, data, leverage_rules_of_thumb=['huber lower', 'hw higher'], residual_rules_of_thumb='two sigma'
)
# Expected:
## <AxesSubplot:title={'center':'Outlier plot for End station name'}, xlabel='Leverage score', ylabel='Slabwise SSE'>
plt.show()

"""
ComponentVis + PlotLy for interactive visualisations
----------------------------------------------------

In this example, we'll see how ComponentVis can be used together with PlotLy Express for easily making thorough interactive visualisations.
"""

###############################################################################
# Imports and utilities
# ^^^^^^^^^^^^^^^^^^^^^

import plotly.express as px
from tensorly.decomposition import non_negative_parafac_hals

import component_vis

###############################################################################
# To fit CP models, we need to solve a non-convex optimization problem, possibly with local minima. It is therefore useful
# to fit several models with the same number of components using many different random initialisations.


def fit_many_nn_parafac(X, num_components, num_inits=5):
    return [non_negative_parafac_hals(X, num_components, n_iter_max=500, init="random",) for i in range(num_inits)]


###############################################################################
# Loading the data
# ^^^^^^^^^^^^^^^^

bike_data = component_vis.data.load_oslo_city_bike()
bike_data


###############################################################################
# We see that there are two metadata columns in the ``End station name`` axis: ``lat`` and ``lon``.
# These contain the coordinates for each bike station.

###############################################################################
# Fitting the model
# ^^^^^^^^^^^^^^^^^
#
# We know from the split-half analysis example **TODO** that 3 components are a good choice for this data.
# Therefore, we fit five model candidates with three components and select the one with the lowest error.

model_candidates = fit_many_nn_parafac(bike_data.data, 3, num_inits=5)
selected_cp = component_vis.multimodel_evaluation.get_model_with_lowest_error(model_candidates, bike_data)

###############################################################################
# Postprocessing
# ^^^^^^^^^^^^^^
#
# If we just postprocess, then we get a labelled CP tensor. That is, the factor matrices are transformed into Pandas
# DataFrames with the same index as the main coordinates of the xarray DataArray. However, in this case, we also want
# to include the additional coordinates (i.e. the latitude and longitude) from this DataArray as metadata-columns.
# To include these as well, we set the ``include_metadata`` flag in ``postprocess`` to ``True``.

cp_with_metadata = component_vis.postprocessing.postprocess(selected_cp, bike_data, include_metadata=True)

weights, (end_station, year, month, day_of_week, hour) = cp_with_metadata

###############################################################################
# Converting the data to a tidy format
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# PlotLy assumes that the data is *tidy*, but factor matrices are not. We have one column per factor matrix, which
# makes it cumbersome to use with plotly. We therefore have a ``factor_matrix_to_tidy``-function in ``component_vis``
# which simply converts a factor matrix (with potential metadata) into a tidy format. See
# ``component_vis.postprocessing.factor_matrix_to_tidy`` for more info.

tidy_end_station_data = component_vis.postprocessing.factor_matrix_to_tidy(end_station, value_name="Popularity")

###############################################################################
# Density map of the end station components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now, we use the ``density_mapbox`` function in PlotLy Express to create a density map of the End station components.
# We can use the animation frame to get a nice slider for the different components.

px.density_mapbox(
    component_vis.postprocessing.factor_matrix_to_tidy(end_station, value_name="Popularity"),
    lat="lat",
    lon="lon",
    z="Popularity",
    animation_frame="Component",
    hover_data=["End station name"],
    zoom=10.5,
    opacity=0.5,
    mapbox_style="carto-positron",
)


###############################################################################
# We see that there are three distinct patterns. Component 0 is spread across most of the city, while component 1
# is focused on central areas. Component 2 is also spread widely, but has a strong signal at leisure-places like the
# beach at Bygdøy.

###############################################################################
# Plotting the various time-components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

###############################################################################
# Time of day
# ~~~~~~~~~~~
#
# First, we look at the time-of-day components to see if these reveal what kind of patterns the different components
# represent.

px.line(
    component_vis.postprocessing.factor_matrix_to_tidy(hour, value_name="Popularity"),
    x="Hour",
    y="Popularity",
    color="Component",
)

###############################################################################
# We see that the first component shows a strong signal after 16:00, which is when a normal working day in Norway ends.
# Likewise, the second component shows the strongest signal at 08:00, which is when a normal working day in Norway
# starts. This indicates that the first two components represent getting to and from work. The third component shows
# activity during midday, which combined with the map-plot above indicates that it represents leisurely activities.

###############################################################################
# Weekday
# ~~~~~~~
# Next, we look at the weekday components.

tidy_day_of_week = component_vis.postprocessing.factor_matrix_to_tidy(day_of_week, value_name="Popularity")

px.line(
    tidy_day_of_week, x="Day of week", y="Popularity", color="Component",
)

###############################################################################
# Here, we see that the first two components are the most active on weekdays, which is when people mostly work.
# The morning component barely shows any signal in the weekends at all and the leisure component has the strongest
# signal on Saturdays.

###############################################################################
# Month plots
# ~~~~~~~~~~~

tidy_month = component_vis.postprocessing.factor_matrix_to_tidy(month, value_name="Popularity")

px.line(
    tidy_month, x="Month", y="Popularity", color="Component",
)

###############################################################################
# The month-mode components show reasonable patterns. People bike the most during summer, but the work-components
# get a dip during July, when most people are on holidays

###############################################################################
# Year
# ~~~~

px.line(
    component_vis.postprocessing.factor_matrix_to_tidy(year, value_name="Popularity"),
    x="Year",
    y="Popularity",
    color="Component",
)

###############################################################################
# We see that there is  overall less biking in 2021 than in 2020. Perhaps people biked more early in the pandemic
# when most public transport was closed?

###############################################################################
# Important about including factor metadata
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# By including factor metadata, we add additional columns to the factor matrices. As a consequence, some of the
# functions in ``component_vis`` will no longer work properly on CP tensors with metadata on the factor matrices.
# It can therefore be beneficial to not include the metadata in the beginning (this is the default behaviour of
# ``postprocess``) and only add the metadata in the end. Alternatively, it can be useful to have a normally
# postprocessed CP tensor as well as a CP tensor with metadata.

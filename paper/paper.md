---
title: 'ComponentVis: Visualising and analysing tensor decomposition models with Python'
tags:
  - Python
  - tensor decompositions
  - data mining
  - data visualisation
authors:
  - name: Marie Roald^[Co-first author, corresponding author] # note this makes a footnote saying 'Co-first author'
    orcid: 0000-0002-9571-8829
    affiliation: "1,2" # (Multiple affiliations must be quoted)
  - name: Yngve Mardal Moe^[Co-first author] # note this makes a footnote saying 'Co-first author'
    orcid: 0000-0002-5159-9012
    affiliation: 3
affiliations:
 - name: Simula Metropolitan Center for Digital Engineering, Norway
   index: 1
 - name: Oslo Metropolitan University, Norway
   index: 2
 - name: Independent Researcher, Norway
   index: 3
date: 24. April 2022
bibliography: paper.bib
---

# Introduction

Multi-way data, also known as tensor data or data cubes occur in many applications, such as text mining [@bader2008discussion], neuroscience [@andersen2004structure] and chemical analysis [@bro1997parafac]. Uncovering the meaningful patterns within such data can provide crucial insights about the data source and tensor decompositions have proven an effective tool for this task. In particular, the PARAFAC model (also known as CANDECOMP/PARAFAC, or CP, and the canonical polyadic decomposition, or CPD), has shown great promise for extracting interpretable components. PARAFAC has, for example, extracted topics from an email corpus [@bader2008discussion] and chemical spectra from fluorescence spectroscopy data [@bro1997parafac]. For a thorough introduction to tensor methods we refer the reader to [@kolda2009tensor] and [@bro1997parafac]. The goal of ComponentVis is to provide utilities for analysing, visualising and working with tensor decompositions for data analysis in Python.

# Statement of need

Python has become a language of choice for data science for both industrial and academic research. Open source tools, such as scikit-learn [@pedregosa2011scikit] and Pandas [@mckinney-proc-scipy-2010] have made a variety of machine learning methods accessible within Python. Recently, TensorLy has also made tensor methods available in Python [@kossaifi2016tensorly], providing seamless integration of multi-way data mining methods within the python scientific environment. However, while TensorLy is an open-source community-driven package for calculating tensor decompositions, it does not include tools for analysing or visualising the tensor decomposition models. Because tensor decompositions provide powerful tools to extract insight from multi-way data, effective visualisations are crucial as they are needed to communicate this insight. Furthermore, visualisation and evaluation are essential steps in the multi-way analysis pipeline — without tools for these steps, we cannot find suitable models.

There is, to our knowledge, no free open source software (FOSS) that facilitates all these steps. For MATLAB, some tools cover part of this scope, such as Tensor Toolbox (which provides some model evaluation) [@osti_1230898] or the N-Way toolbox [@andersson2000n]. PLSToolbox covers most of our scope, but it is a closed source commercial software. There is, therefore, a growing need for FOSS tools for the visualisation and evaluation of tensor decompositions.

# Example

The PARAFAC model is straightforward, but there are several pitfalls to consider. Two main pitfalls are the scaling and permutation indeterminacy [@bro1997parafac]. The order of the components does not matter, and the magnitude of one factor matrix can be scaled arbitrarily so long as another factor matrix is inversely scaled (an even number of components may even change sign!). Therefore, it can be time-consuming and cumbersome to go from having fitted a PARAFAC model to visualising it. ComponentVis takes care of these hurdles in a transparent way. The code below shows how easy we can use ComponentVis and TensorLy to analyse a fluorescence spectroscopy dataset.

```python
import component_vis
import matplotlib.pyplot as plt
from tensorly.decomposition import parafac

def fit_parafac(dataset, num_components, num_inits):
    model_candidates = [
        parafac(dataset.data, num_components, init="random", random_state=i)
        for i in range(num_inits)
    ]
    model = component_vis.multimodel_evaluation.get_model_with_lowest_error(
        model_candidates, dataset
    )
    return component_vis.postprocessing.postprocess(model, dataset)

data = component_vis.data.load_aminoacids()
cp_tensor = fit_parafac(data, 3, num_inits=3)
component_vis.visualisation.components_plot(cp_tensor)
plt.show()
```

```raw
Loading Aminoacids dataset from:
Bro, R, PARAFAC: Tutorial and applications, Chemometrics and Intelligent
   ↩ Laboratory Systems, 1997, 38, 149-171
```

![An example figure showing the component vectors of a three component PARAFAC model fitted to a fluoresence spectroscopy dataset](paper_demo.pdf)

The above code uses TensorLy to fit five three-component PARAFAC models to the data. Then it uses ComponentVis to:

 1. Select the model that gave the lowest reconstruction error,
 1. normalise the component vectors, storing their magnitude in a separate weight-vector,
 1. permute the components in descending weight (i.e. signal strength) order,
 1. flip the components so they point in a logical direction compared to the data,
 1. convert the factor matrices into Pandas DataFrames with logical indices,
 1. and plot the components using matplotlib.

All these steps are well documented with references to the literature. This makes it easy for new practitioners to analyse multi-way data without falling for known pitfalls.

# Overview

ComponentVis follows the procedural paradigm, and all of ComponentVis’s functionality lies in functions separated over 8 public modules:

 1. `component_vis.data` - various open datasets
 1. `component_vis.factor_tools` - transforms and compares PARAFAC models without using reference data
 1. `component_vis.model_evaluation` - evaluates a PARAFAC model
 1. `component_vis.multimodel_evaluation` - compares and evaluates multiple models at once
 1. `component_vis.outliers` - finds data points that may be outliers
 1. `component_vis.postprocessing` - post-processes PARAFAC models, usually used before visualising
 1. `component_vis.utils` - general utilities that can be useful (e.g. forming dense tensor from decompositions)
 1. `component_vis.visualisation` - visualising component models

A core design choice behind ComponentVis is how to store metadata. Consider the example above. It is necessary to know the values along the x-axis to interpret these components. Therefore, we use xarray DataArrays to store data tensors [@hoyer2017xarray], keeping the correct indices for each tensor mode (i.e. axis), and Pandas DataFrames to store factor matrices. However, TensorLy works with NumPy arrays. ComponentVis, therefore, provides useful tools to add the coordinates from an xarray DataArray onto the factor matrices obtained with TensorLy. Furthermore, all functions of ComponentVis support both labelled and unlabelled decompositions (i.e. DataFrames and NumPy arrays) and will use the labels whenever possible.

The visualisation module uses matplotlib to create the plots, and the goal of this module is to facilitate fast prototyping and exploratory analysis. However, ComponentVis can also seamlessly convert factor matrices into tidy tables, which are better suited for visualisation libraries such as Seaborn [@Waskom2021] and PlotLy Express, thus making it painless to combine tensor decomposition with the plotting library that best suits the user’s specific needs.

To be easy to use, scientific software should have thorough and accurate documentation. For ComponentVis, this means two things: Explaining what the code does and why this is important. Therefore, we have taken care to review the literature for all methods, citing original sources wherever possible. By gathering the references together with the API documentation and examples, we make it straightforward for researchers new to the field to discover suitable references for their analysis.

The gallery of examples provided with ComponentVis explains the tools included in the package and how to use them. The gallery contains, among others, examples that explain how to select the number of components in PARAFAC models, how to detect outliers and how to combine ComponentVis with PlotLy to get interactive visualisations. All examples include the relevant references, making it easy for new practitioners to get started.

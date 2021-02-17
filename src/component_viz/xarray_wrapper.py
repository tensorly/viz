import xarray as xr
import pandas as pd


def label_factor_matrices(factor_matrices, dataset):
    factor_matrices = [
        pd.DataFrame(factor_matrix, index=dataset.coords[dim_name])
        for factor_matrix, dim_name in zip(factor_matrices, dataset.coords)
    ]
    for factor_matrix, dim_name in zip(factor_matrices, dataset.coords):
        factor_matrix.index.name = dim_name
    return factor_matrices


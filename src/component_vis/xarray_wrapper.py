import pandas as pd
import xarray as xr


def label_factor_matrices(factor_matrices, dataset):
    factor_matrices = [
        pd.DataFrame(factor_matrix, index=dataset.coords[dim_name])
        for factor_matrix, dim_name in zip(factor_matrices, dataset.dims)
    ]
    for factor_matrix, dim_name in zip(factor_matrices, dataset.dims):
        factor_matrix.index.name = dim_name
    return factor_matrices


def label_cp_tensor(cp_tensor, dataset):
    return (cp_tensor[0], label_factor_matrices(cp_tensor[1], dataset))


def is_xarray(x):
    # TODO: Is this how we want to check?
    return isinstance(x, xr.DataArray) or isinstance(x, xr.Dataset)


# TODO: is_dataframe too
def get_data(x):
    # TODO: extract data array from xarray/dataframe in a safe manner
    pass

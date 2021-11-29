import pandas as pd
import xarray as xr


def label_factor_matrices(factor_matrices, dataset):
    if is_xarray(dataset):
        factor_matrices = [
            pd.DataFrame(factor_matrix, index=dataset.coords[dim_name])
            for factor_matrix, dim_name in zip(factor_matrices, dataset.dims)
        ]
        for factor_matrix, dim_name in zip(factor_matrices, dataset.dims):
            factor_matrix.index.name = dim_name
    elif is_dataframe(dataset) and len(factor_matrices) == 2:
        factor_matrices = [
            pd.DataFrame(factor_matrices[0], index=dataset.index),
            pd.DataFrame(factor_matrices[1], index=dataset.columns),
        ]
    else:
        raise ValueError(
            "``dataset`` must be xarray.DataArray or, pandas.DataFrame "
            "(only possible if ``len(factor_matrices) == 2``)"
        )
    return factor_matrices


def label_cp_tensor(cp_tensor, dataset):
    """Label the CP tensor by converting the factor matrices into data frames with a sensible index.

    Convert the factor matrices into Pandas data frames where the data frame indices
    are given by the coordinate names of an xarray DataArray. If the dataset has only
    two modes, then it can also be a pandas data frame.

    Parameters
    ----------
    cp_tensor : CPTensor
        CP Tensor whose factor matrices should be labelled
    dataset : xarray.DataArray of pandas.DataFrame
        Dataset used to label the factor matrices

    Returns
    -------
    CPTensor
        Tuple on the CPTensor format, except that the factor matrices are data frames.
    """
    # TODO: Unit test for label_cp_tensor
    #   - Create CPTensor and DataArray. Check that labelling it works
    return (cp_tensor[0], label_factor_matrices(cp_tensor[1], dataset))


def is_xarray(x):
    # TODO: Is this how we want to check?
    return isinstance(x, xr.DataArray)


def is_dataframe(x):
    return isinstance(x, pd.DataFrame)


def get_data(x):
    # TODO: extract data array from xarray/dataframe in a safe manner
    pass

import numpy as np
import xarray

from .xarray_wrapper import _handle_labelled_dataset


def is_iterable(x):
    """Check if variable is iterable

    Arguments
    ---------
    x
        Variable to check if is iterable
    
    Returns
    -------
    bool
        Whether ``x`` is iterable or not.
    """
    try:
        iter(x)
    except TypeError:
        return False
    else:
        return True


@_handle_labelled_dataset("tensor", None)
def unfold_tensor(tensor, mode):
    """Unfolds (matricises) a potentially labelled data tensor into a numpy array along given mode.

    Arguments
    ---------
    tensor : np.ndarray or xarray.DataArray
        Dataset to unfold
    mode : int
        Which mode (axis) to unfold the dataset along.
    
    Returns
    -------
    np.ndarray
        The unfolded dataset as a numpy array.
    """
    dataset = np.asarray(tensor)
    return np.moveaxis(dataset, mode, 0).reshape(dataset.shape[mode], -1)


def extract_singleton(x):
    """Extracts a singleton from an array.

    This is useful whenever XArray or Pandas is used, since many NumPy functions that
    return a number may return a singleton array instead.

    Parameters
    ----------
    x : float, numpy.ndarray, xarray.DataArray or pandas.DataFrame
        Singleton array to extract value from.

    Returns
    -------
    float
        Singleton value extracted from ``x``.
    """
    return np.asarray(x).reshape(-1).item()

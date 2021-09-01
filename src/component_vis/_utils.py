import numpy as np
import xarray


def is_iterable(x):
    """Check if variable is iterable
    """
    try:
        iter(x)
    except TypeError:
        return False
    else:
        return True


def unfold_tensor(tensor, mode):
    # TODO: Docstring for unfold_tensor
    dataset = np.asarray(tensor)
    return np.moveaxis(dataset, mode, 0).reshape(dataset.shape[mode], -1)


def extract_singleton(x):
    """Extracts a singleton from an array.

    This is useful whenever XArray is used, since many NumPy functions that
    return a number will return an XArray singleton.

    Parameters:
    -----------
    x : numpy.ndarray or xarray.DataArray
    """
    # TODO: Change code so this utility is used
    return np.asarray(x).item()
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
    dataset = np.asarray(tensor)
    return np.moveaxis(dataset, mode, 0).reshape(dataset.shape[mode], -1)


def extract_singleton(x):
    return np.asarray(x).item()
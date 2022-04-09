import pandas as pd
import xarray as xr


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


def is_xarray(x):
    """Check if ``x`` is an xarray data array.
    
    Arguments
    ---------
    x
        Object to check
    
    Returns
    -------
    bool
        ``True`` if x is an xarray data array, ``False`` otherwise.
    """
    # TODO: Is this how we want to check?
    return isinstance(x, xr.DataArray)


def is_dataframe(x):
    """Check if ``x`` is a data frame.
    
    Arguments
    ---------
    x
        Object to check
    
    Returns
    -------
    bool
        ``True`` if x is a data frame, ``False`` otherwise.
    """
    return isinstance(x, pd.DataFrame)

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import component_vis._utils as utils


@pytest.mark.parametrize(
    "iterable",
    [
        "a",
        [],
        (1,),
        set(),
        {},
        np.zeros(2),
        xr.DataArray(np.zeros((2, 2, 2))),
        pd.DataFrame(np.zeros((2, 2))),
        range(4),
        (i for i in range(2)),
    ],
)
def test_is_iterable_is_true_for_iterables(iterable):
    assert utils.is_iterable(iterable)


@pytest.mark.parametrize("non_iterable", [1, 1.0, object(), lambda x: x, np.float64(1), np.int64(1)])
def test_is_iterable_is_false_for_noniterables(non_iterable):
    assert not utils.is_iterable(non_iterable)


from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from squidpy.experimental.im._stain._background import estimate_background_intensity
from squidpy.experimental.im._stain._validation import StainFittingError


def _da(values: np.ndarray, *, chunked: bool) -> xr.DataArray:
    data = da.from_array(values, chunks=(3, 8, 8)) if chunked else values
    return xr.DataArray(data, dims=("c", "y", "x"))


@pytest.mark.parametrize("chunked", [False, True])
def test_recovers_white_point(chunked: bool) -> None:
    rng = np.random.default_rng(0)
    # mostly bright background near (240, 245, 250), a darker tissue blob
    values = np.empty((3, 32, 32))
    values[0] = 240.0
    values[1] = 245.0
    values[2] = 250.0
    values[:, :8, :8] = rng.uniform(20.0, 60.0, size=(3, 8, 8))  # tissue
    bg = estimate_background_intensity(_da(values, chunked=chunked))
    assert bg.shape == (3,)
    np.testing.assert_allclose(bg, [240.0, 245.0, 250.0], atol=1.0)


def test_blank_image_raises() -> None:
    black = np.zeros((3, 16, 16))
    with pytest.raises(StainFittingError, match="non-positive"):
        estimate_background_intensity(_da(black, chunked=False))


def test_bad_percentile_raises() -> None:
    with pytest.raises(ValueError, match="percentile"):
        estimate_background_intensity(_da(np.ones((3, 8, 8)), chunked=False), percentile=0.0)

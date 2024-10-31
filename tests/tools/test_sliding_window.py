from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pytest
from anndata import AnnData

from squidpy.tl import _calculate_window_corners, sliding_window


class TestSlidingWindow:
    @pytest.mark.parametrize("overlap", [0, 20])
    def test_sliding_window_several_slices(
        self,
        adata_mibitof: AnnData,
        overlap: int,
        sliding_window_key: str = "sliding_window_key",
        window_size: int = 200,
        library_key: str = "library_id",
    ):
        df = sliding_window(
            adata_mibitof,
            library_key=library_key,
            window_size=window_size,
            overlap=overlap,
            coord_columns=("globalX", "globalY"),
            sliding_window_key=sliding_window_key,
            copy=True,
        )

        assert len(df) == adata_mibitof.n_obs  # correct amount of rows

        if overlap == 0:
            sliding_window_columns = [col for col in df.columns if sliding_window_key in col]
            assert len(sliding_window_columns) == 1  # only one sliding window
            assert df[sliding_window_key].isnull().sum() == 0  # no unassigned cells
        else:
            for i in range(36):  # we expect 36 windows
                assert (
                    f"{sliding_window_key}_window_{i}" in df.columns
                )  # correct number of columns; multiple sliding windows

    @pytest.mark.parametrize("overlap", [0, 2])
    def test_sliding_window_square_grid(
        self,
        adata_squaregrid: AnnData,
        overlap: int,
        sliding_window_key: str = "sliding_window_key",
        window_size: int = 5,
    ):
        df = sliding_window(
            adata_squaregrid,
            window_size=window_size,
            overlap=overlap,
            coord_columns=("globalX", "globalY"),
            sliding_window_key=sliding_window_key,
            copy=True,
        )

        assert len(df) == adata_squaregrid.n_obs  # correct amount of rows

        if overlap == 0:
            sliding_window_columns = [col for col in df.columns if sliding_window_key in col]
            assert len(sliding_window_columns) == 1  # only one sliding window
            assert df[sliding_window_key].isnull().sum() == 0  # no unassigned cells
        else:
            for i in range(9):  # we expect 9 windows
                assert (
                    f"{sliding_window_key}_window_{i}" in df.columns
                )  # correct number of columns; multiple sliding windows

    def test_sliding_window_invalid_window_size(
        self,
        adata_squaregrid: AnnData,
    ):
        with pytest.raises(ValueError, match="Window size must be larger than 0."):
            sliding_window(
                adata_squaregrid,
                window_size=-10,
                overlap=0,
                coord_columns=("globalX", "globalY"),
                sliding_window_key="sliding_window",
                copy=True,
            )

        with pytest.raises(ValueError, match="Overlap must be non-negative."):
            sliding_window(
                adata_squaregrid,
                window_size=10,
                overlap=-10,
                coord_columns=("globalX", "globalY"),
                sliding_window_key="sliding_window",
                copy=True,
            )

    def test_calculate_window_corners_overlap(self):
        min_x = 0
        max_x = 200
        min_y = 0
        max_y = 200
        window_size = 100
        overlap = 20

        windows = _calculate_window_corners(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            window_size=window_size,
            overlap=overlap,
            drop_partial_windows=False,
        )

        assert windows.shape == (9, 4)
        assert windows.iloc[0].values.tolist() == [0, 100, 0, 100]
        assert windows.iloc[-1].values.tolist() == [160, 200, 160, 200]

    def test_calculate_window_corners_no_overlap(self):
        min_x = 0
        max_x = 200
        min_y = 0
        max_y = 200
        window_size = 100
        overlap = 0

        windows = _calculate_window_corners(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            window_size=window_size,
            overlap=overlap,
            drop_partial_windows=False,
        )

        assert windows.shape == (4, 4)
        assert windows.iloc[0].values.tolist() == [0, 100, 0, 100]
        assert windows.iloc[-1].values.tolist() == [100, 200, 100, 200]

    def test_calculate_window_corners_drop_partial_windows(self):
        min_x = 0
        max_x = 200
        min_y = 0
        max_y = 200
        window_size = 100
        overlap = 20

        windows = _calculate_window_corners(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            window_size=window_size,
            overlap=overlap,
            drop_partial_windows=True,
        )

        assert windows.shape == (4, 4)
        assert windows.iloc[0].values.tolist() == [0, 100, 0, 100]
        assert windows.iloc[-1].values.tolist() == [80, 180, 80, 180]

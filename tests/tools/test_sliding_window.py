from __future__ import annotations

from typing import Any, Literal

import pytest
from anndata import AnnData

from squidpy.tl import sliding_window
from squidpy.tl._sliding_window import _sliding_window_stats


class TestSlidingWindow:
    @pytest.mark.parametrize("library_key", ["point"])
    @pytest.mark.parametrize("window_size", [-10, 0, 10])
    @pytest.mark.parametrize("overlap", [-1, 0, 2])
    @pytest.mark.parametrize("sliding_window_key", ["sliding_window", "sliding_window_v2"])
    def test_sliding_window_several_slices(
        self,
        adata_mibitof: AnnData,
        library_key: str,
        window_size: int,
        overlap: int,
        sliding_window_key: str,
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
            grid_len = window_size // overlap
            num_grid_systems = grid_len**2
            for i in range(num_grid_systems):
                assert (
                    f"{sliding_window_key}_{i+1}" in df.columns
                )  # correct number of columns; multiple sliding windows

            for col in df.columns:
                lib_windows = []
                for lib in adata_mibitof.obs[library_key].unique():
                    lib_windows.append(set(df[adata_mibitof.obs[library_key] == lib][col]))
                assert (
                    len(set.intersection(*lib_windows)) == 0
                )  # no intersection of sliding windows across library_keys

    @pytest.mark.parametrize("window_size", [-10, 0, 10])
    @pytest.mark.parametrize("overlap", [-1, 0, 2])
    def test_sliding_window_square_grid(
        self,
        adata_squaregrid: AnnData,
        window_size: int,
        overlap: int,
    ):
        df = sliding_window(
            adata_squaregrid,
            window_size=window_size,
            overlap=overlap,
            coord_columns=("globalX", "globalY"),
            sliding_window_key="sliding_window",
            copy=True,
        )

        assert len(df) == adata_squaregrid.n_obs  # correct amount of rows

        if overlap == 0:
            sliding_window_columns = [col for col in df.columns if "sliding_window" in col]
            assert len(sliding_window_columns) == 1  # only one sliding window
            assert df["sliding_window"].isnull().sum() == 0  # no unassigned cells
        else:
            grid_len = window_size // overlap
            num_grid_systems = grid_len**2
            for i in range(num_grid_systems):
                assert f"sliding_window_{i+1}" in df.columns  # correct number of columns; multiple sliding windows

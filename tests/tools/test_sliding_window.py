from __future__ import annotations

from typing import List

import pytest
from anndata import AnnData
from squidpy.tl import sliding_window
from squidpy.tl._sliding_window import _sliding_window_stats


class TestSlidingWindow:

    def test_design_matrix_several_slices(
        self,
        adata: AnnData,
        library_key: str,
        window_size: int,
        overlap: int,
    ):
        df = sliding_window(
                adata,
                library_key=library_key,
                window_size=window_size,
                overlap=overlap,
                coord_columns=("globalX", "globalY"),
                sliding_window_key="sliding_window",
                copy=True
            )

        assert len(df) == adata.n_obs  # correct amount of rows

        if overlap == 0:
            sliding_window_columns = [col for col in df.columns if 'sliding_window' in col]
            assert len(sliding_window_columns) == 1 # only one sliding window
            assert df['sliding_window'].isnull().sum() == 0 # no unassigned cells
        else:
            grid_len = window_size // overlap
            num_grid_systems = grid_len**2
            for i in range(num_grid_systems):
                assert f'sliding_window_{i+1}' in df.columns # correct number of columns; multiple sliding windows

            for col in df.columns:
                lib_windows = []
                for lib in adata.obs['fov'].unique():
                    lib_windows.append(set(df[adata.obs['fov'] == lib][col]))
                    print(lib_windows)
                assert len(set.intersection(*lib_windows)) == 0 # no intersection of sliding windows across library_keys


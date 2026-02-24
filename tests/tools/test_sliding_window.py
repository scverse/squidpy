from __future__ import annotations

import pandas as pd
import pytest
from anndata import AnnData

from squidpy.tl import _calculate_window_corners, sliding_window


class TestSlidingWindow:
    @pytest.mark.parametrize(
        "window_size, overlap, partial_windows",
        [
            (300, 0, None),
            (300, 50, None),
            (300, 50, "drop"),
        ],
    )
    def test_sliding_window_several_slices(
        self,
        adata_mibitof: AnnData,
        window_size: int,
        overlap: int,
        partial_windows: str | None,
        sliding_window_key: str = "sliding_window_key",
        library_key: str = "library_id",
    ):
        def count_total_assignments(df: pd.DataFrame) -> int:
            total = 0
            for lib_key in ["point8", "point16", "point23"]:
                cols = df.columns[df.columns.str.contains(lib_key)]
                for col in cols:
                    total += df[col].sum()
            return total

        df = sliding_window(
            adata_mibitof,
            library_key=library_key,
            window_size=window_size,
            overlap=overlap,
            coord_columns=("globalX", "globalY"),
            sliding_window_key=sliding_window_key,
            partial_windows=partial_windows,
            copy=True,
        )

        assert len(df) == adata_mibitof.n_obs

        if overlap == 0:
            # single categorical assignment
            assert sliding_window_key in df.columns
            assert df[sliding_window_key].notnull().all()
        else:
            sliding_window_cols = df.columns[df.columns.str.contains(sliding_window_key)]

            if partial_windows == "drop":
                assert len(sliding_window_cols) == 27
                assert count_total_assignments(df) == 2536
            else:
                assert len(sliding_window_cols) == 70
                assert count_total_assignments(df) == 4569

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
            partial_windows=None,
            copy=True,
        )

        assert len(df) == adata_squaregrid.n_obs

        if overlap == 0:
            assert sliding_window_key in df.columns
            assert df[sliding_window_key].notnull().all()
        else:
            for i in range(9):  # 3x3 grid
                assert f"{sliding_window_key}_window_{i}" in df.columns

    def test_sliding_window_invalid_arguments(self, adata_squaregrid: AnnData):
        with pytest.raises(ValueError, match="Window size must be larger than 0"):
            sliding_window(
                adata_squaregrid,
                window_size=-1,
                overlap=0,
                coord_columns=("globalX", "globalY"),
                copy=True,
            )

        with pytest.raises(ValueError, match="Overlap must be non-negative"):
            sliding_window(
                adata_squaregrid,
                window_size=10,
                overlap=-1,
                coord_columns=("globalX", "globalY"),
                copy=True,
            )

        with pytest.raises(ValueError, match="max_nr_cells"):
            sliding_window(
                adata_squaregrid,
                window_size=None,
                overlap=0,
                partial_windows="split",
                coord_columns=("globalX", "globalY"),
                copy=True,
            )

    def test_sliding_window_adaptive_assigns_all_cells(
        self,
        adata_squaregrid: AnnData,
        sliding_window_key: str = "sliding_window_key",
    ):
        df = sliding_window(
            adata_squaregrid,
            window_size=5,
            overlap=0,
            coord_columns=("globalX", "globalY"),
            sliding_window_key=sliding_window_key,
            partial_windows="adaptive",
            copy=True,
        )

        assert sliding_window_key in df.columns
        assert df[sliding_window_key].notnull().all()
        assert len(df) == adata_squaregrid.n_obs

    def test_sliding_window_split_nr_cells(
        self,
        adata_mibitof: AnnData,
        sliding_window_key: str = "sliding_window_key",
        library_key: str = "library_id",
    ):
        """
        Test that when using 'split', each window contains at most max_nr_cells
        and at least max_nr_cells // 2 cells,
        unless the total number of cells is smaller than max_nr_cells // 2.
        """
        max_nr_cells = 100
        total_cells = adata_mibitof.n_obs

        df = sliding_window(
            adata_mibitof,
            library_key=library_key,
            window_size=None,  # ignored in split mode
            overlap=0,
            coord_columns=("globalX", "globalY"),
            sliding_window_key=sliding_window_key,
            partial_windows="split",
            max_nr_cells=max_nr_cells,
            copy=True,
        )

        counts = df[sliding_window_key].value_counts()

        # all windows respect the upper bound
        assert counts.max() <= max_nr_cells

        # determine strict lower bound
        lower_bound = max_nr_cells // 2
        if total_cells < lower_bound:
            # if total cells are too few, just one window is allowed smaller
            assert counts.max() == total_cells
        else:
            # otherwise, every window must satisfy the lower bound
            assert (counts >= lower_bound).all()


class TestCalculateWindowCorners:
    def test_overlap(self):
        windows = _calculate_window_corners(
            min_x=0,
            max_x=200,
            min_y=0,
            max_y=200,
            window_size=100,
            overlap=20,
            partial_windows=None,
        )

        assert windows.shape == (9, 4)
        assert windows.iloc[0].tolist() == [0, 100, 0, 100]
        assert windows.iloc[-1].tolist() == [160, 200, 160, 200]

    def test_no_overlap(self):
        windows = _calculate_window_corners(
            min_x=0,
            max_x=200,
            min_y=0,
            max_y=200,
            window_size=100,
            overlap=0,
            partial_windows=None,
        )

        assert windows.shape == (4, 4)
        assert windows.iloc[-1].tolist() == [100, 200, 100, 200]

    def test_drop_partial_windows(self):
        windows = _calculate_window_corners(
            min_x=0,
            max_x=200,
            min_y=0,
            max_y=200,
            window_size=100,
            overlap=20,
            partial_windows="drop",
        )

        assert windows.shape == (4, 4)
        assert windows.iloc[-1].tolist() == [80, 180, 80, 180]

    def test_adaptive_windows_cover_extent(self):
        windows = _calculate_window_corners(
            min_x=0,
            max_x=200,
            min_y=0,
            max_y=200,
            window_size=90,
            overlap=0,
            partial_windows="adaptive",
        )

        assert windows["x_start"].min() == 0
        assert windows["y_start"].min() == 0
        assert windows["x_end"].max() == 200
        assert windows["y_end"].max() == 200

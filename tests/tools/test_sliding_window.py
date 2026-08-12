from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from squidpy.tl import _calculate_window_corners, sliding_window
from squidpy.tl._sliding_window import _split_cells


def _grid_adata(n_per_side: int = 30, extent: float = 300.0, seed: int = 0, library_key: str | None = None) -> AnnData:
    """A uniform point cloud in an ``extent`` x ``extent`` square (optionally split across libraries)."""
    rng = np.random.default_rng(seed)
    n = n_per_side * n_per_side
    xy = rng.uniform(0, extent, size=(n, 2))
    obs = pd.DataFrame({"globalX": xy[:, 0], "globalY": xy[:, 1]}, index=[f"c{i}" for i in range(n)])
    if library_key is not None:
        obs[library_key] = rng.choice(["a", "b"], size=n)
    return AnnData(X=np.zeros((n, 1), dtype=np.float32), obs=obs)


class TestSlidingWindow:
    @pytest.mark.parametrize(
        "windowsize_overlap_partial",
        [
            (300, 0, "keep"),
            (300, 50, "keep"),
            (300, 50, "drop"),
            (300, 0, "adaptive"),
            (300, 50, "adaptive"),
        ],
    )
    def test_sliding_window_several_slices(
        self,
        adata_mibitof: AnnData,
        windowsize_overlap_partial: tuple[int, int, str],
        sliding_window_key: str = "sliding_window_key",
        library_key: str = "library_id",
    ):
        def _count_total_assignments():
            total_cells = 0
            for lib_key in ["point8", "point16", "point23"]:
                cols_in_lib = df.columns[df.columns.str.contains(lib_key)]
                for col in cols_in_lib:
                    total_cells += df[col].sum()
            return total_cells

        window_size, overlap, partial_windows = windowsize_overlap_partial
        df = sliding_window(
            adata_mibitof,
            library_key=library_key,
            window_size=window_size,
            overlap=overlap,
            coord_columns=("globalX", "globalY"),
            sliding_window_key=sliding_window_key,
            copy=True,
            partial_windows=partial_windows,
        )

        if overlap == 0:
            sliding_window_columns = [col for col in df.columns if sliding_window_key in col]
            assert len(sliding_window_columns) == 1  # only one sliding window
            assert df[sliding_window_key].isnull().sum() == 0  # no NaN (unassigned is an explicit category)
            assert len(df) == adata_mibitof.n_obs  # correct amount of rows
        else:
            sliding_window_cols = df.columns[df.columns.str.contains("sliding_window")]

            if partial_windows == "drop":
                assert len(sliding_window_cols) == 27
                assert _count_total_assignments() == 2536
            elif partial_windows == "adaptive":
                assert len(sliding_window_cols) == 48
                assert _count_total_assignments() == 4411
            else:
                assert len(sliding_window_cols) == 70
                assert _count_total_assignments() == 4569

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
            sliding_window(adata_squaregrid, window_size=-10, overlap=0, coord_columns=("globalX", "globalY"), copy=True)

        with pytest.raises(ValueError, match="Overlap must be non-negative."):
            sliding_window(adata_squaregrid, window_size=10, overlap=-10, coord_columns=("globalX", "globalY"), copy=True)

        with pytest.raises(ValueError, match="`max_nr_cells` must be set when method='split'."):
            sliding_window(adata_squaregrid, method="split", coord_columns=("globalX", "globalY"), copy=True)

    def test_sliding_window_method_validation(self):
        adata = _grid_adata()
        with pytest.raises(ValueError, match="must be 'grid' or 'split'"):
            sliding_window(adata, method="nope", copy=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=">= 1"):
            sliding_window(adata, method="split", max_nr_cells=0, copy=True)
        with pytest.raises(ValueError, match="only used with method='split'"):
            sliding_window(adata, method="grid", max_nr_cells=100, copy=True)
        with pytest.raises(ValueError, match="not used with method='split'"):
            sliding_window(adata, method="split", max_nr_cells=100, overlap=5, copy=True)
        with pytest.raises(ValueError, match="not used with method='split'"):
            sliding_window(adata, method="split", max_nr_cells=100, window_size=50, copy=True)

    def test_sliding_window_drop_overlap0_no_crash(self):
        """Regression: drop + overlap==0 used to raise in the categorical sort; unassigned cells are labelled."""
        adata = _grid_adata()
        df = sliding_window(adata, window_size=100, overlap=0, partial_windows="drop", copy=True)
        col = df["sliding_window_assignment"]
        assert col.isnull().sum() == 0  # no NaN
        assert "unassigned" in list(col.cat.categories)
        assert col.cat.categories[-1] == "unassigned"  # ordered last
        assert (col == "unassigned").sum() > 0  # drop genuinely strands some edge cells here

    def test_sliding_window_deprecated_drop_partial_windows(self):
        adata = _grid_adata()
        with pytest.warns(FutureWarning, match="drop_partial_windows"):
            deprecated = sliding_window(adata, window_size=100, overlap=0, drop_partial_windows=True, copy=True)
        current = sliding_window(adata, window_size=100, overlap=0, partial_windows="drop", copy=True)
        assert deprecated["sliding_window_assignment"].astype(str).equals(current["sliding_window_assignment"].astype(str))
        with pytest.raises(ValueError, match="not both"):
            sliding_window(adata, drop_partial_windows=True, partial_windows="drop", copy=True)

    def test_sliding_window_adaptive_small_library(self):
        """Regression (B5): a library smaller than one window must not divide-by-zero / drop its only window."""
        rng = np.random.default_rng(1)
        big = rng.uniform(0, 1000, size=(500, 2))
        tiny = rng.uniform(0, 10, size=(20, 2))  # span (~10) <= overlap (50)
        xy = np.vstack([big, tiny])
        obs = pd.DataFrame(
            {"globalX": xy[:, 0], "globalY": xy[:, 1], "library_id": ["big"] * 500 + ["tiny"] * 20},
            index=[f"c{i}" for i in range(520)],
        )
        adata = AnnData(X=np.zeros((520, 1), dtype=np.float32), obs=obs)
        df = sliding_window(
            adata, library_key="library_id", window_size=200, overlap=50, partial_windows="adaptive", copy=True
        )
        tiny_cols = [c for c in df.columns if "tiny_" in c]
        assert tiny_cols  # the tiny library still gets a window
        assert (df.iloc[500:][tiny_cols].sum(axis=1) > 0).all()  # every tiny cell is covered

    def test_sliding_window_split_nr_cells(
        self,
        adata_mibitof: AnnData,
        sliding_window_key: str = "sliding_window_key",
        library_key: str = "library_id",
    ):
        """Each window holds <= max_nr_cells and >= max_nr_cells // 2 cells (per library)."""
        max_nr_cells = 100
        df = sliding_window(
            adata_mibitof,
            library_key=library_key,
            sliding_window_key=sliding_window_key,
            method="split",
            max_nr_cells=max_nr_cells,
            copy=True,
        )
        counts = df[sliding_window_key].value_counts()
        assert df[sliding_window_key].isnull().sum() == 0  # split covers every cell
        assert counts.max() <= max_nr_cells
        assert (counts >= max_nr_cells // 2).all()

    def test_split_cells_partition_and_bounds(self):
        """_split_cells partitions the cells (each in exactly one window) and respects the bounds, even with ties."""
        rng = np.random.default_rng(2)
        coords = pd.DataFrame({"globalX": rng.uniform(0, 100, 1000), "globalY": rng.uniform(0, 100, 1000)})
        labels = _split_cells(coords, ("globalX", "globalY"), max_cells=50)
        assert len(labels) == len(coords)  # one label per cell -> a partition (non-overlapping, full cover)
        counts = pd.Series(labels).value_counts()
        assert counts.max() <= 50
        assert (counts >= 25).all()

        # degenerate: all cells share a coordinate -> must still terminate and stay bounded
        same = pd.DataFrame({"globalX": np.zeros(200), "globalY": np.zeros(200)})
        labels = _split_cells(same, ("globalX", "globalY"), max_cells=50)
        assert pd.Series(labels).value_counts().max() <= 50

    def test_calculate_window_corners_overlap(self):
        windows = _calculate_window_corners(
            min_x=0, max_x=200, min_y=0, max_y=200, window_size=100, overlap=20, partial_windows="keep"
        )
        assert windows.shape == (9, 4)
        assert windows.iloc[0].values.tolist() == [0, 100, 0, 100]
        assert windows.iloc[-1].values.tolist() == [160, 200, 160, 200]

    def test_calculate_window_corners_no_overlap(self):
        windows = _calculate_window_corners(
            min_x=0, max_x=200, min_y=0, max_y=200, window_size=100, overlap=0, partial_windows="keep"
        )
        assert windows.shape == (4, 4)
        assert windows.iloc[0].values.tolist() == [0, 100, 0, 100]
        assert windows.iloc[-1].values.tolist() == [100, 200, 100, 200]

    def test_calculate_window_corners_drop_partial_windows(self):
        windows = _calculate_window_corners(
            min_x=0, max_x=200, min_y=0, max_y=200, window_size=100, overlap=20, partial_windows="drop"
        )
        assert windows.shape == (4, 4)
        assert windows.iloc[0].values.tolist() == [0, 100, 0, 100]
        assert windows.iloc[-1].values.tolist() == [80, 180, 80, 180]

    def test_calculate_window_corners_adaptive_partial_windows(self):
        windows = _calculate_window_corners(
            min_x=0, max_x=200, min_y=0, max_y=200, window_size=100, overlap=20, partial_windows="adaptive"
        )
        assert windows.shape == (9, 4)
        assert windows.iloc[0].values.tolist() == [0, 80, 0, 80]
        assert windows.iloc[-1].values.tolist() == [120, 200, 120, 200]

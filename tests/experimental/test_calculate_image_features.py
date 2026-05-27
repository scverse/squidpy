"""Tests for calculate_image_features.

Uses a small synthetic SpatialData (200×200 image, ~20 cells) so tests
run in seconds without downloading real data.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel

import squidpy as sq


@pytest.fixture()
def sdata_synthetic():
    """Synthetic SpatialData with a small 3-channel image and ~20 rectangular cells."""
    rng = np.random.default_rng(42)
    H, W, C = 200, 200, 3

    image_data = rng.integers(0, 255, (C, H, W), dtype=np.uint8)
    image_xr = xr.DataArray(
        image_data,
        dims=["c", "y", "x"],
        coords={"c": ["R", "G", "B"]},
    )

    # Place ~20 rectangular cells in a grid (non-overlapping, 30×30 each)
    labels_data = np.zeros((H, W), dtype=np.int32)
    cell_id = 0
    for y in range(10, H - 30, 40):
        for x in range(10, W - 30, 40):
            cell_id += 1
            labels_data[y : y + 30, x : x + 30] = cell_id

    labels_xr = xr.DataArray(labels_data, dims=["y", "x"])

    return SpatialData(
        images={"test_img": Image2DModel.parse(image_xr)},
        labels={"test_labels": Labels2DModel.parse(labels_xr)},
    )


class TestCalculateImageFeatures:
    """Tests for calculate_image_features function."""

    # --- Basic functionality ---

    def test_skimage_label_inplace(self, sdata_synthetic):
        """Inplace stores AnnData in sdata.tables."""
        sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            features=["skimage:label"],
            adata_key_added="morphology",
            inplace=True,
        )

        assert "morphology" in sdata_synthetic.tables
        adata = sdata_synthetic.tables["morphology"]
        assert adata.n_obs > 0
        assert adata.n_vars > 0
        assert "spatialdata_attrs" in adata.uns
        assert adata.uns["spatialdata_attrs"]["region"] == "test_labels"
        assert "region" in adata.obs
        assert "label_id" in adata.obs

    def test_not_inplace_returns_anndata(self, sdata_synthetic):
        result = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            features=["skimage:label"],
            inplace=False,
        )
        assert isinstance(result, ad.AnnData)
        assert result.n_obs > 0
        assert result.n_vars > 0

    # --- Feature sources ---

    def test_skimage_label_properties(self, sdata_synthetic):
        """skimage:label produces mask-only morphological features."""
        result = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            features=["skimage:label"],
            inplace=False,
        )
        assert "area" in result.var_names

    def test_skimage_label_single_property(self, sdata_synthetic):
        """Fine-grained: skimage:label:area → only area column."""
        result = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            features=["skimage:label:area"],
            inplace=False,
        )
        assert list(result.var_names) == ["area"]

    def test_skimage_intensity(self, sdata_synthetic):
        """skimage:label+image produces per-channel intensity features."""
        result = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            features=["skimage:label+image"],
            inplace=False,
        )
        assert result.n_vars > 0
        assert any("_" in col for col in result.var_names)

    def test_skimage_intensity_single_property(self, sdata_synthetic):
        """Fine-grained: only intensity_mean per channel."""
        result = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            features=["skimage:label+image:intensity_mean"],
            inplace=False,
        )
        assert all(col.startswith("intensity_mean_") for col in result.var_names)
        assert not any(col.startswith("intensity_max") for col in result.var_names)

    def test_cpmeasure_sizeshape(self, sdata_synthetic):
        result = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            features=["cpmeasure:sizeshape"],
            inplace=False,
        )
        assert isinstance(result, ad.AnnData)
        assert result.n_obs > 0
        assert any("Area" in col for col in result.var_names)

    def test_cpmeasure_intensity(self, sdata_synthetic):
        result = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            features=["cpmeasure:intensity"],
            inplace=False,
        )
        assert isinstance(result, ad.AnnData)
        assert result.n_obs > 0
        # intensity features are per-channel, so column names contain "__"
        assert any("__" in col for col in result.var_names)

    def test_cpmeasure_correlation(self, sdata_synthetic):
        result = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            features=["cpmeasure:correlation"],
            inplace=False,
        )
        assert isinstance(result, ad.AnnData)
        assert result.n_obs > 0

    def test_cpmeasure_default_all(self, sdata_synthetic):
        """features=None enables all cp_measure features."""
        result = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            inplace=False,
        )
        assert isinstance(result, ad.AnnData)
        assert result.n_obs > 0
        assert result.n_vars > 50

    def test_squidpy_summary(self, sdata_synthetic):
        result = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            features=["squidpy:summary"],
            inplace=False,
        )
        assert isinstance(result, ad.AnnData)
        assert result.n_obs > 0
        assert any(col.startswith("summary_mean") for col in result.var_names)

    def test_squidpy_texture(self, sdata_synthetic):
        result = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            features=["squidpy:texture"],
            inplace=False,
        )
        assert isinstance(result, ad.AnnData)
        assert result.n_obs > 0
        assert any(col.startswith("texture_contrast") for col in result.var_names)

    def test_squidpy_color_hist(self, sdata_synthetic):
        result = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            features=["squidpy:color_hist"],
            inplace=False,
        )
        assert isinstance(result, ad.AnnData)
        assert result.n_obs > 0
        assert any(col.startswith("color_hist_bin") for col in result.var_names)

    # --- Mixed sources ---

    def test_mixed_cpmeasure_and_skimage(self, sdata_synthetic):
        result = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            features=["cpmeasure:sizeshape", "skimage:label:area"],
            inplace=False,
        )
        assert "area" in result.var_names
        assert any("Area" in col for col in result.var_names)

    # --- Validation errors ---

    def test_invalid_image_key(self, sdata_synthetic):
        with pytest.raises(ValueError, match="Image key 'nonexistent' not found"):
            sq.experimental.im.calculate_image_features(
                sdata_synthetic,
                image_key="nonexistent",
                labels_key="test_labels",
                features=["skimage:label"],
            )

    def test_invalid_labels_key(self, sdata_synthetic):
        with pytest.raises(ValueError, match="Labels key 'nonexistent' not found"):
            sq.experimental.im.calculate_image_features(
                sdata_synthetic,
                image_key="test_img",
                labels_key="nonexistent",
                features=["skimage:label"],
            )

    def test_both_labels_and_shapes_error(self, sdata_synthetic):
        with pytest.raises(ValueError, match="Use either"):
            sq.experimental.im.calculate_image_features(
                sdata_synthetic,
                image_key="test_img",
                labels_key="test_labels",
                shapes_key="fake",
                features=["skimage:label"],
            )

    def test_missing_labels_and_shapes(self, sdata_synthetic):
        with pytest.raises(ValueError, match="Provide either"):
            sq.experimental.im.calculate_image_features(
                sdata_synthetic,
                image_key="test_img",
                features=["skimage:label"],
            )

    def test_invalid_feature(self, sdata_synthetic):
        with pytest.raises(ValueError, match="Unknown feature"):
            sq.experimental.im.calculate_image_features(
                sdata_synthetic,
                image_key="test_img",
                labels_key="test_labels",
                features=["nonexistent:measurement"],
            )

    def test_no_valid_features(self, sdata_synthetic):
        with pytest.raises(ValueError, match="No valid features requested"):
            sq.experimental.im.calculate_image_features(
                sdata_synthetic,
                image_key="test_img",
                labels_key="test_labels",
                features=[],
                inplace=False,
            )

    def test_dimension_mismatch_identity_succeeds_with_overlap(self, capsys):
        """Image and labels have different dims but identity transforms;
        we crop to the overlap rectangle and process the cells that fall inside.
        Regression for the PR #982 'fails despite alignment transform' note.
        """
        rng = np.random.default_rng(42)
        image_xr = xr.DataArray(
            rng.integers(0, 255, (3, 200, 200), dtype=np.uint8),
            dims=["c", "y", "x"],
            coords={"c": ["R", "G", "B"]},
        )
        labels_arr = np.zeros((100, 100), dtype=np.int32)
        labels_arr[10:40, 10:40] = 1
        labels_arr[60:90, 60:90] = 2
        labels_xr = xr.DataArray(labels_arr, dims=["y", "x"])
        sdata = SpatialData(
            images={"img": Image2DModel.parse(image_xr)},
            labels={"lbl": Labels2DModel.parse(labels_xr)},
        )

        adata = sq.experimental.im.calculate_image_features(
            sdata,
            image_key="img",
            labels_key="lbl",
            features=["skimage:label"],
            inplace=False,
        )
        assert adata is not None and adata.n_obs == 2
        captured = capsys.readouterr()
        # No cells dropped: labels (100, 100) sits fully inside image (200, 200) at origin
        assert "Cell drop report" in captured.out

    # --- Channel selection ---

    def test_channel_selection_by_name(self, sdata_synthetic):
        """Selecting a single channel reduces feature columns."""
        result_all = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            features=["skimage:label+image:intensity_mean"],
            inplace=False,
        )
        # Image2DModel.parse converts channel coords to integers [0,1,2]
        result_one = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            channels=["0"],
            features=["skimage:label+image:intensity_mean"],
            inplace=False,
        )
        # All channels → 3 columns; one channel → 1 column
        assert result_all.n_vars == 3
        assert result_one.n_vars == 1
        assert "intensity_mean_0" in result_one.var_names

    def test_channel_selection_rejects_int(self, sdata_synthetic):
        """Integer channel indices are no longer accepted -- names only."""
        with pytest.raises(TypeError, match="channels must contain strings"):
            sq.experimental.im.calculate_image_features(
                sdata_synthetic,
                image_key="test_img",
                labels_key="test_labels",
                channels=[0],  # int, not str -- should fail validation
                features=["squidpy:summary"],
                inplace=False,
            )

    def test_channel_selection_invalid(self, sdata_synthetic):
        with pytest.raises(ValueError, match="Channel 'DAPI' not found"):
            sq.experimental.im.calculate_image_features(
                sdata_synthetic,
                image_key="test_img",
                labels_key="test_labels",
                channels=["DAPI"],
                features=["skimage:label"],
            )

    # --- Tiled vs non-tiled equivalence ---

    def test_tiled_vs_single_tile_equivalence(self, sdata_synthetic):
        """Tile-invariant features should be identical whether we tile or not.

        Position-dependent features (centroid, perimeter_crofton) are expected
        to differ across tile boundaries, so we test with ``area`` and
        ``squidpy:summary`` which depend only on the cell's pixel values.
        """
        kw = {
            "image_key": "test_img",
            "labels_key": "test_labels",
            "features": ["skimage:label:area", "squidpy:summary"],
            "inplace": False,
            "invalid_as_zero": True,
        }
        # Single tile (tile_size >= image → no tiling)
        result_single = sq.experimental.im.calculate_image_features(sdata_synthetic, tile_size=1000, **kw)
        # Multiple tiles (tile_size=100 → 4 tiles on 200×200)
        result_tiled = sq.experimental.im.calculate_image_features(sdata_synthetic, tile_size=100, **kw)

        # Same cells, same features
        assert result_single.n_obs == result_tiled.n_obs
        assert set(result_single.var_names) == set(result_tiled.var_names)

        # Align columns and rows for comparison
        common_cols = list(result_single.var_names)
        df_single = pd.DataFrame(result_single.X, index=result_single.obs["label_id"].values, columns=common_cols)
        df_tiled = pd.DataFrame(
            result_tiled[:, common_cols].X, index=result_tiled.obs["label_id"].values, columns=common_cols
        )
        df_single = df_single.sort_index()
        df_tiled = df_tiled.sort_index()

        np.testing.assert_array_equal(df_single.index, df_tiled.index)
        np.testing.assert_allclose(df_single.values, df_tiled.values, rtol=1e-5, atol=1e-5)

    # --- Parallelization ---

    def test_n_jobs_produces_same_result(self, sdata_synthetic):
        """n_jobs>1 produces the same result as n_jobs=1."""
        kw = {
            "image_key": "test_img",
            "labels_key": "test_labels",
            "features": ["skimage:label:area"],
            "inplace": False,
        }
        result_seq = sq.experimental.im.calculate_image_features(sdata_synthetic, n_jobs=1, **kw)
        result_par = sq.experimental.im.calculate_image_features(sdata_synthetic, n_jobs=2, **kw)

        assert result_seq.n_obs == result_par.n_obs
        np.testing.assert_array_equal(
            result_seq.X[np.argsort(result_seq.obs["label_id"].values)],
            result_par.X[np.argsort(result_par.obs["label_id"].values)],
        )


# ---------------------------------------------------------------------------
# Per-PR-#982-concern regression tests
# ---------------------------------------------------------------------------


def _toy_sdata(
    image_shape: tuple[int, int] = (200, 200),
    n_channels: int = 3,
    channel_names: list[str] | None = None,
    labels_shape: tuple[int, int] | None = None,
    labels_translation: tuple[float, float] | None = None,
    labels_scale: tuple[float, float] | None = None,
    label_ids: list[int] | None = None,
) -> SpatialData:
    """Build a synthetic SpatialData with controllable label/image transforms."""
    from spatialdata.transformations import Scale, Translation, set_transformation

    rng = np.random.default_rng(0)
    H, W = image_shape
    image_data = rng.integers(0, 255, (n_channels, H, W), dtype=np.uint8)
    image_xr = xr.DataArray(image_data, dims=["c", "y", "x"])

    LH, LW = labels_shape if labels_shape is not None else image_shape
    labels_data = np.zeros((LH, LW), dtype=np.int32)
    ids = label_ids if label_ids is not None else list(range(1, 6))
    cell_h, cell_w = max(LH // 8, 4), max(LW // 8, 4)
    for i, lid in enumerate(ids):
        row = i // 3
        col = i % 3
        y0 = 10 + row * (cell_h + 6)
        x0 = 10 + col * (cell_w + 6)
        if y0 + cell_h > LH or x0 + cell_w > LW:
            continue
        labels_data[y0 : y0 + cell_h, x0 : x0 + cell_w] = lid
    labels_xr = xr.DataArray(labels_data, dims=["y", "x"])

    img_el = (
        Image2DModel.parse(image_xr, c_coords=channel_names)
        if channel_names is not None
        else Image2DModel.parse(image_xr)
    )
    lbl_el = Labels2DModel.parse(labels_xr)

    if labels_translation is not None:
        ty, tx = labels_translation
        set_transformation(lbl_el, Translation([tx, ty], axes=("x", "y")), "global")
    if labels_scale is not None:
        sy, sx = labels_scale
        set_transformation(lbl_el, Scale([sx, sy], axes=("x", "y")), "global")

    return SpatialData(images={"img": img_el}, labels={"lbl": lbl_el})


class TestPR982Concerns:
    """Regression tests for the six open concerns on PR #982."""

    # -- Concern 1: channel names are str-typed in output columns --

    def test_concern1_channel_str_names_in_columns(self):
        sdata = _toy_sdata(channel_names=["DAPI", "CD3", "CD8"])
        adata = sq.experimental.im.calculate_image_features(
            sdata,
            image_key="img",
            labels_key="lbl",
            features=["squidpy:summary"],
            inplace=False,
        )
        cols = list(adata.var_names)
        assert any("_DAPI" in c for c in cols)
        assert any("_CD3" in c for c in cols)
        assert any("_CD8" in c for c in cols)
        # Make sure the numeric-fallback names did not slip in:
        assert not any(c.endswith("_0") or c.endswith("_1") or c.endswith("_2") for c in cols)

    # -- Concern 2: progress logs are emitted --

    def test_concern2_progress_log_emitted(self, capsys):
        sdata = _toy_sdata()
        sq.experimental.im.calculate_image_features(
            sdata,
            image_key="img",
            labels_key="lbl",
            features=["skimage:label:area"],
            tile_size=80,  # forces >1 tile on 200x200
            inplace=False,
        )
        captured = capsys.readouterr()
        import re

        # spatialdata's logger renders via rich and injects ANSI escapes
        # between tokens, so the digits in "Tile 1/9" are wrapped.
        ansi_re = re.compile(r"\x1b\[[0-9;]*m")
        plain = ansi_re.sub("", captured.out)
        assert re.search(r"Tile \d+/\d+", plain), f"no progress log in:\n{plain}"

    # -- Concern 3 (a): identity transforms, dim mismatch -> overlap path --

    def test_concern3_identity_dim_mismatch_uses_overlap(self, capsys):
        # labels (100, 100) sits inside image (200, 200) at the origin under Identity.
        sdata = _toy_sdata(image_shape=(200, 200), labels_shape=(100, 100))
        adata = sq.experimental.im.calculate_image_features(
            sdata,
            image_key="img",
            labels_key="lbl",
            features=["skimage:label:area"],
            inplace=False,
        )
        captured = capsys.readouterr()
        assert "Cell drop report" in captured.out
        # All cells fit inside the (100, 100) labels image, so none are dropped:
        assert adata.n_obs > 0

    # -- Concern 3 (b): integer translation -> overlap is offset --

    def test_concern3_integer_translation(self):
        sdata = _toy_sdata(
            image_shape=(200, 200),
            labels_shape=(100, 100),
            labels_translation=(50.0, 0.0),
        )
        adata = sq.experimental.im.calculate_image_features(
            sdata,
            image_key="img",
            labels_key="lbl",
            features=["skimage:label:area"],
            inplace=False,
        )
        assert adata.n_obs > 0

    # -- Concern 3 (c): non-integer scale, strict -> raises with hint --

    def test_concern3_strict_raises_on_non_integer_scale(self):
        sdata = _toy_sdata(
            image_shape=(200, 200),
            labels_shape=(100, 100),
            labels_scale=(1.7, 1.7),
        )
        with pytest.raises(ValueError, match="spatialdata.rasterize"):
            sq.experimental.im.calculate_image_features(
                sdata,
                image_key="img",
                labels_key="lbl",
                features=["skimage:label:area"],
                align_mode="strict",
                inplace=False,
            )

    # -- Concern 3 (d): align_mode='rasterize' resamples and proceeds --

    def test_concern3_rasterize_path_succeeds(self, capsys):
        sdata = _toy_sdata(
            image_shape=(200, 200),
            labels_shape=(100, 100),
            labels_scale=(1.7, 1.7),
        )
        adata = sq.experimental.im.calculate_image_features(
            sdata,
            image_key="img",
            labels_key="lbl",
            features=["skimage:label:area"],
            align_mode="rasterize",
            inplace=False,
        )
        assert adata.n_obs > 0
        captured = capsys.readouterr()
        # rasterize path emits a warning about materialization
        assert "rasterize" in captured.out.lower() or "rasterize" in captured.err.lower()

    # -- Concern 4: channel subset selection --

    def test_concern4_channel_subset_by_name(self):
        sdata = _toy_sdata(n_channels=4, channel_names=["c0", "c1", "c2", "c3"])
        adata = sq.experimental.im.calculate_image_features(
            sdata,
            image_key="img",
            labels_key="lbl",
            features=["squidpy:summary"],
            channels=["c0", "c2"],
            inplace=False,
        )
        cols = list(adata.var_names)
        assert any("_c0" in c for c in cols)
        assert any("_c2" in c for c in cols)
        assert not any("_c1" in c for c in cols)
        assert not any("_c3" in c for c in cols)

    # -- Concern 5: spatialdata_attrs on output table --

    def test_concern5_spatialdata_attrs_present(self):
        sdata = _toy_sdata()
        sq.experimental.im.calculate_image_features(
            sdata,
            image_key="img",
            labels_key="lbl",
            features=["skimage:label:area"],
            inplace=True,
            adata_key_added="morphology",
        )
        attrs = sdata.tables["morphology"].uns["spatialdata_attrs"]
        assert "region" in attrs
        assert "region_key" in attrs
        assert "instance_key" in attrs
        assert attrs["region"] == "lbl"

    # -- Concern 6: non-contiguous label IDs survive cp_measure roundtrip --

    def test_concern6_non_contiguous_label_ids(self):
        sdata = _toy_sdata(label_ids=[1, 37, 82])
        adata = sq.experimental.im.calculate_image_features(
            sdata,
            image_key="img",
            labels_key="lbl",
            features=["skimage:label:area"],
            inplace=False,
        )
        observed = set(adata.obs["label_id"].astype(int).tolist())
        assert {1, 37, 82}.issubset(observed)

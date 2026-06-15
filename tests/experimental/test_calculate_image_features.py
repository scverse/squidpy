"""Tests for calculate_image_features.

Uses a small synthetic SpatialData (200x200 image, ~20 cells) so tests
run in seconds without downloading real data.
"""

from __future__ import annotations

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely import Polygon
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel, ShapesModel
from spatialdata.transformations import Scale, Translation, set_transformation

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

    # Place ~20 rectangular cells in a grid (non-overlapping, 30x30 each)
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

    def test_skimage_morphology_inplace(self, sdata_synthetic):
        """Inplace stores AnnData in sdata.tables."""
        sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            features=["skimage:morphology"],
            key_added="morphology",
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
            features=["skimage:morphology"],
            inplace=False,
        )
        assert isinstance(result, ad.AnnData)
        assert result.n_obs > 0
        assert result.n_vars > 0
        assert "area" in result.var_names

    # --- Feature sources ---
    # (the single-property `== ["area"]` contract is covered by the bare-string and
    # morphology-only tests below.)

    def test_skimage_intensity(self, sdata_synthetic):
        """skimage:intensity produces per-channel intensity features."""
        result = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            features=["skimage:intensity"],
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
            features=["skimage:intensity:intensity_mean"],
            inplace=False,
        )
        assert all(col.startswith("intensity_mean_") for col in result.var_names)
        assert not any(col.startswith("intensity_max") for col in result.var_names)

    def test_cpmeasure_sizeshape(self, sdata_synthetic):
        """cpmeasure:sizeshape yields cp_measure shape columns (CellProfiler names)."""
        result = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            features=["cpmeasure:sizeshape"],
            inplace=False,
        )
        assert isinstance(result, ad.AnnData)
        assert result.n_obs > 0
        assert "Area" in result.var_names

    def test_cpmeasure_intensity(self, sdata_synthetic):
        """cpmeasure:intensity yields per-channel cp_measure intensity columns."""
        result = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            features=["cpmeasure:intensity"],
            inplace=False,
        )
        # cp_measure suffixes intensity columns with the positional channel index.
        assert any("Intensity_MeanIntensity__0" in c for c in result.var_names)

    def test_features_none_runs_all_backends(self, sdata_synthetic):
        """features=None computes cp_measure + skimage + squidpy together."""
        result = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            features=None,
            inplace=False,
        )
        cols = list(result.var_names)
        assert "Area" in cols  # cp_measure sizeshape
        assert "feret_diameter_max" in cols  # skimage-only morphology (kept)
        assert any(c.startswith("summary_mean") for c in cols)  # squidpy
        # cp:sizeshape covers skimage morphology, so the redundant skimage copy is deduped.
        assert "area" not in cols

    def test_features_none_without_image_raises(self, sdata_synthetic):
        """features=None needs pixels; without image_key it raises a clear error."""
        with pytest.raises(ValueError, match="require pixel data"):
            sq.experimental.im.calculate_image_features(
                sdata_synthetic,
                labels_key="test_labels",
                features=None,
                inplace=False,
            )

    def test_skimage_morphology_deduped_against_cp_sizeshape(self, sdata_synthetic):
        """A skimage morphology prop that cp:sizeshape reproduces is dropped (cp wins)."""
        result = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            features=["cpmeasure:sizeshape", "skimage:morphology:area"],
            inplace=False,
        )
        cols = list(result.var_names)
        assert "Area" in cols  # cp_measure keeps it
        assert "area" not in cols  # redundant skimage copy dropped
        assert len(cols) == len(set(cols))

    def test_skimage_only_morphology_prop_survives_dedup(self, sdata_synthetic):
        """skimage-only props (no cp:sizeshape twin) are kept alongside cp."""
        result = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            features=["cpmeasure:sizeshape", "skimage:morphology:feret_diameter_max"],
            inplace=False,
        )
        assert "feret_diameter_max" in result.var_names  # cp:sizeshape has no identical twin

    def test_no_dedup_without_cp_sizeshape(self, sdata_synthetic):
        """cp without sizeshape does not trigger morphology dedup."""
        result = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            features=["cpmeasure:intensity", "skimage:morphology:area"],
            inplace=False,
        )
        assert "area" in result.var_names  # cp:intensity does not compute shape

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

    # --- Validation errors ---

    def test_invalid_image_key(self, sdata_synthetic):
        with pytest.raises(ValueError, match="Image key 'nonexistent' not found"):
            sq.experimental.im.calculate_image_features(
                sdata_synthetic,
                image_key="nonexistent",
                labels_key="test_labels",
                features=["skimage:morphology"],
            )

    def test_invalid_labels_key(self, sdata_synthetic):
        with pytest.raises(ValueError, match="Labels key 'nonexistent' not found"):
            sq.experimental.im.calculate_image_features(
                sdata_synthetic,
                image_key="test_img",
                labels_key="nonexistent",
                features=["skimage:morphology"],
            )

    def test_both_labels_and_shapes_error(self, sdata_synthetic):
        with pytest.raises(ValueError, match="Use either"):
            sq.experimental.im.calculate_image_features(
                sdata_synthetic,
                image_key="test_img",
                labels_key="test_labels",
                shapes_key="fake",
                features=["skimage:morphology"],
            )

    def test_missing_labels_and_shapes(self, sdata_synthetic):
        with pytest.raises(ValueError, match="Provide either"):
            sq.experimental.im.calculate_image_features(
                sdata_synthetic,
                image_key="test_img",
                features=["skimage:morphology"],
            )

    def test_invalid_feature(self, sdata_synthetic):
        with pytest.raises(ValueError, match="Unknown feature") as excinfo:
            sq.experimental.im.calculate_image_features(
                sdata_synthetic,
                image_key="test_img",
                labels_key="test_labels",
                features=["nonexistent:measurement"],
            )
        # All real feature groups are advertised in the unknown-feature error.
        assert "cpmeasure:intensity" in str(excinfo.value)
        assert "squidpy:summary" in str(excinfo.value)

    def test_mixed_group_and_fine_grained_raises(self, sdata_synthetic):
        """Mixing 'skimage:morphology' (all props) with 'skimage:morphology:area' (one prop)
        is ambiguous; raise rather than silently take one or the other."""
        with pytest.raises(ValueError, match="ambiguous"):
            sq.experimental.im.calculate_image_features(
                sdata_synthetic,
                image_key="test_img",
                labels_key="test_labels",
                features=["skimage:morphology", "skimage:morphology:area"],
                inplace=False,
            )

    def test_no_valid_features(self, sdata_synthetic):
        with pytest.raises(ValueError, match="No features requested"):
            sq.experimental.im.calculate_image_features(
                sdata_synthetic,
                image_key="test_img",
                labels_key="test_labels",
                features=[],
                inplace=False,
            )

    def test_non_integer_transform_strict_raises(self):
        """A non-pixel-aligned (non-integer) transform must raise under 'strict'."""
        rng = np.random.default_rng(42)
        image_xr = xr.DataArray(
            rng.integers(0, 255, (3, 200, 200), dtype=np.uint8),
            dims=["c", "y", "x"],
            coords={"c": ["R", "G", "B"]},
        )
        labels_arr = np.zeros((200, 200), dtype=np.int32)
        labels_arr[10:40, 10:40] = 1
        sdata = SpatialData(
            images={"img": Image2DModel.parse(image_xr)},
            labels={"lbl": Labels2DModel.parse(xr.DataArray(labels_arr, dims=["y", "x"]))},
        )
        # Non-integer scale -> not pixel-aligned -> strict raises with a hint.
        set_transformation(sdata.labels["lbl"], Scale([1.5, 1.5], axes=("x", "y")), to_coordinate_system="global")

        with pytest.raises(ValueError, match="different pixel grids"):
            sq.experimental.im.calculate_image_features(
                sdata,
                image_key="img",
                labels_key="lbl",
                features=["skimage:morphology"],
                align_mode="strict",
                inplace=False,
            )

    def test_identity_size_mismatch_uses_overlap(self):
        """Identity transform + different sizes is not an error: features run on the overlap."""
        rng = np.random.default_rng(42)
        image_xr = xr.DataArray(
            rng.integers(0, 255, (3, 200, 200), dtype=np.uint8),
            dims=["c", "y", "x"],
            coords={"c": ["R", "G", "B"]},
        )
        labels_arr = np.zeros((100, 100), dtype=np.int32)
        labels_arr[10:40, 10:40] = 1
        sdata = SpatialData(
            images={"img": Image2DModel.parse(image_xr)},
            labels={"lbl": Labels2DModel.parse(xr.DataArray(labels_arr, dims=["y", "x"]))},
        )
        result = sq.experimental.im.calculate_image_features(
            sdata,
            image_key="img",
            labels_key="lbl",
            features=["skimage:morphology"],
            inplace=False,
        )
        assert result.n_obs == 1

    # --- Channel selection ---

    def test_channel_selection_by_name(self, sdata_synthetic):
        """Selecting a single channel reduces feature columns."""
        result_all = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            features=["skimage:intensity:intensity_mean"],
            inplace=False,
        )
        # Image2DModel.parse converts channel coords to integers [0,1,2]
        result_one = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            channels=["0"],
            features=["skimage:intensity:intensity_mean"],
            inplace=False,
        )
        # All channels -> 3 columns; one channel -> 1 column
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
                features=["skimage:morphology"],
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
            "features": ["skimage:morphology:area", "squidpy:summary"],
            "inplace": False,
            "invalid_as_zero": True,
        }
        # Single tile (tile_size >= image -> no tiling)
        result_single = sq.experimental.im.calculate_image_features(sdata_synthetic, tile_size=1000, **kw)
        # Multiple tiles (tile_size=100 -> 4 tiles on 200x200)
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
            "features": ["skimage:morphology:area"],
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
# Behavioural regression tests
# ---------------------------------------------------------------------------


def _toy_sdata(
    image_shape: tuple[int, int] = (200, 200),
    n_channels: int = 3,
    channel_names: list[str] | None = None,
    labels_shape: tuple[int, int] | None = None,
    labels_translation: tuple[float, float] | None = None,
    labels_scale: tuple[float, float] | None = None,
    label_ids: list[int] | None = None,
    multiscale: bool = False,
) -> SpatialData:
    """Build a synthetic SpatialData with controllable label/image transforms."""
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
    lbl_el = Labels2DModel.parse(labels_xr, scale_factors=[2] if multiscale else None)

    if labels_translation is not None:
        ty, tx = labels_translation
        set_transformation(lbl_el, Translation([tx, ty], axes=("x", "y")), "global")
    if labels_scale is not None:
        sy, sx = labels_scale
        set_transformation(lbl_el, Scale([sx, sy], axes=("x", "y")), "global")

    return SpatialData(images={"img": img_el}, labels={"lbl": lbl_el})


class TestBehaviouralRegressions:
    """Regression tests for previously-reported issues."""

    # -- channel names are str-typed in output columns --

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

    # -- progress logs are emitted --

    def test_concern2_progress_log_emitted(self, capsys):
        sdata = _toy_sdata()
        sq.experimental.im.calculate_image_features(
            sdata,
            image_key="img",
            labels_key="lbl",
            features=["skimage:morphology:area"],
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

    # -- channel subset selection --

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

    # -- spatialdata_attrs on output table --

    def test_concern5_spatialdata_attrs_present(self):
        sdata = _toy_sdata()
        sq.experimental.im.calculate_image_features(
            sdata,
            image_key="img",
            labels_key="lbl",
            features=["skimage:morphology:area"],
            inplace=True,
            key_added="morphology",
        )
        attrs = sdata.tables["morphology"].uns["spatialdata_attrs"]
        assert "region" in attrs
        assert "region_key" in attrs
        assert "instance_key" in attrs
        assert attrs["region"] == "lbl"

    # -- non-contiguous label IDs survive the roundtrip --

    def test_concern6_non_contiguous_label_ids(self):
        sdata = _toy_sdata(label_ids=[1, 37, 82])
        adata = sq.experimental.im.calculate_image_features(
            sdata,
            image_key="img",
            labels_key="lbl",
            features=["skimage:morphology:area"],
            inplace=False,
        )
        observed = set(adata.obs["label_id"].astype(int).tolist())
        assert {1, 37, 82}.issubset(observed)


# ---------------------------------------------------------------------------
# Feature-string parsing: accepted scalar form + contract error messages
# ---------------------------------------------------------------------------


class TestFeatureParsing:
    """Parsing of the ``features`` argument and its error contract."""

    def test_features_as_bare_string(self, sdata_synthetic):
        """A single feature may be passed as a string, not just a list."""
        result = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            features="skimage:morphology:area",
            inplace=False,
        )
        assert list(result.var_names) == ["area"]

    @pytest.mark.parametrize(
        ("features", "match"),
        [
            # group flag after a fine-grained prop of the same group.
            (["skimage:morphology:area", "skimage:morphology"], "ambiguous"),
            (["skimage:intensity:intensity_mean", "skimage:intensity"], "ambiguous"),
            # unknown fine-grained property names, per group.
            (["skimage:morphology:bogus"], "Unknown skimage morphology property"),
            (["skimage:intensity:bogus"], "Unknown skimage intensity property"),
        ],
    )
    def test_parse_errors(self, sdata_synthetic, features, match):
        with pytest.raises(ValueError, match=match):
            sq.experimental.im.calculate_image_features(
                sdata_synthetic,
                image_key="test_img",
                labels_key="test_labels",
                features=features,
                inplace=False,
            )


# ---------------------------------------------------------------------------
# Shapes input: rasterized to labels internally
# ---------------------------------------------------------------------------


def _sdata_with_shapes() -> tuple[SpatialData, dict[int, float]]:
    """3-channel image plus four square polygons of *distinct* sizes.

    Each polygon has a unique edge length (hence a unique rasterized area) and a
    non-default index, so the label_id<->cell correspondence can be checked
    instead of trivially comparing a default RangeIndex against itself.

    Returns the SpatialData and the expected ``{label_id: area}`` mapping.
    """
    rng = np.random.default_rng(7)
    image_xr = xr.DataArray(
        rng.integers(0, 255, (3, 200, 200), dtype=np.uint8),
        dims=["c", "y", "x"],
        coords={"c": ["R", "G", "B"]},
    )
    centers = [(50, 50), (150, 50), (50, 150), (150, 150)]
    edges = [20, 30, 40, 50]
    index = [10, 20, 30, 40]
    polys = []
    for (cx, cy), e in zip(centers, edges, strict=True):
        h = e / 2.0
        polys.append(Polygon([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]))
    shapes = ShapesModel.parse(gpd.GeoDataFrame(geometry=polys, index=index))
    expected_area = {idx: float(e * e) for idx, e in zip(index, edges, strict=True)}
    sdata = SpatialData(images={"test_img": Image2DModel.parse(image_xr)}, shapes={"cells": shapes})
    return sdata, expected_area


class TestShapesInput:
    """The ``shapes_key`` path rasterizes polygons to labels internally."""

    def test_shapes_input_featurized(self):
        sdata, expected_area = _sdata_with_shapes()
        adata = sq.experimental.im.calculate_image_features(
            sdata,
            image_key="test_img",
            shapes_key="cells",
            features=["skimage:morphology:area"],
            inplace=False,
        )
        # One row per polygon; region attr points at the shapes element.
        assert adata.n_obs == 4
        assert adata.uns["spatialdata_attrs"]["region"] == "cells"
        # label_id carries the (non-default) shapes index; the per-polygon distinct
        # area proves each index maps to the *correct* cell, not just that the index
        # equals itself.
        observed_area = {
            int(lid): float(ar)
            for lid, ar in zip(adata.obs["label_id"].values, adata[:, "area"].X.ravel(), strict=True)
        }
        assert observed_area == expected_area

    def test_invalid_shapes_key(self, sdata_synthetic):
        with pytest.raises(ValueError, match="Shapes key 'nope' not found"):
            sq.experimental.im.calculate_image_features(
                sdata_synthetic,
                image_key="test_img",
                shapes_key="nope",
                features=["skimage:morphology"],
                inplace=False,
            )


# ---------------------------------------------------------------------------
# All-zero labels
# ---------------------------------------------------------------------------


def test_all_zero_labels_raises(sdata_synthetic):
    """Labels with no foreground cells must raise a clear error."""
    zero_labels = xr.DataArray(np.zeros((200, 200), dtype=np.int32), dims=["y", "x"])
    sdata_synthetic.labels["empty"] = Labels2DModel.parse(zero_labels)
    with pytest.raises(ValueError, match="No cells found in labels"):
        sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="empty",
            features=["skimage:morphology:area"],
            inplace=False,
        )


# ---------------------------------------------------------------------------
# GLCM texture on a flat (constant-intensity) channel
# ---------------------------------------------------------------------------


def test_texture_on_constant_channel():
    """A flat cell hits the degenerate GLCM branch and yields its forced values.

    With zero intensity variation, GLCM contrast and dissimilarity must be 0 and
    homogeneity 1 (and nothing NaN) -- asserting the values, not just that texture
    columns exist, locks the degenerate-branch behaviour against regressions.
    """
    image_xr = xr.DataArray(
        np.full((1, 100, 100), 100, dtype=np.uint8),
        dims=["c", "y", "x"],
        coords={"c": ["flat"]},
    )
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[20:50, 20:50] = 1
    labels_xr = xr.DataArray(labels, dims=["y", "x"])
    sdata = SpatialData(
        images={"img": Image2DModel.parse(image_xr)},
        labels={"lbl": Labels2DModel.parse(labels_xr)},
    )
    adata = sq.experimental.im.calculate_image_features(
        sdata,
        image_key="img",
        labels_key="lbl",
        features=["squidpy:texture"],
        inplace=False,
    )
    assert adata.n_obs == 1
    vals = {c: float(adata[:, c].X[0, 0]) for c in adata.var_names}
    assert not np.isnan(list(vals.values())).any()
    assert next(v for c, v in vals.items() if c.startswith("texture_contrast_")) == 0.0
    assert next(v for c, v in vals.items() if c.startswith("texture_dissimilarity_")) == 0.0
    assert next(v for c, v in vals.items() if c.startswith("texture_homogeneity_")) == 1.0


# ---------------------------------------------------------------------------
# Multiscale (DataTree) image / labels
# ---------------------------------------------------------------------------


def _multiscale_sdata(multiscale_image: bool = True, multiscale_labels: bool = True) -> SpatialData:
    """SpatialData whose image and/or labels are multiscale (DataTree-backed)."""
    rng = np.random.default_rng(3)
    image_xr = xr.DataArray(
        rng.integers(0, 255, (3, 256, 256), dtype=np.uint8),
        dims=["c", "y", "x"],
        coords={"c": ["R", "G", "B"]},
    )
    labels = np.zeros((256, 256), dtype=np.int32)
    cid = 0
    for y in range(20, 220, 60):
        for x in range(20, 220, 60):
            cid += 1
            labels[y : y + 30, x : x + 30] = cid
    labels_xr = xr.DataArray(labels, dims=["y", "x"])

    img = Image2DModel.parse(image_xr, scale_factors=[2] if multiscale_image else None)
    lbl = Labels2DModel.parse(labels_xr, scale_factors=[2] if multiscale_labels else None)
    return SpatialData(images={"img": img}, labels={"lbl": lbl})


class TestMultiscale:
    """Multi-scale (DataTree) inputs require an explicit ``scale`` and then work."""

    def test_multiscale_image_requires_scale(self):
        sdata = _multiscale_sdata(multiscale_image=True, multiscale_labels=False)
        with pytest.raises(ValueError, match="multi-scale images"):
            sq.experimental.im.calculate_image_features(
                sdata,
                image_key="img",
                labels_key="lbl",
                features=["skimage:morphology:area"],
                inplace=False,
            )

    def test_multiscale_labels_requires_scale(self):
        sdata = _multiscale_sdata(multiscale_image=False, multiscale_labels=True)
        with pytest.raises(ValueError, match="multi-scale labels"):
            sq.experimental.im.calculate_image_features(
                sdata,
                image_key="img",
                labels_key="lbl",
                features=["skimage:morphology:area"],
                inplace=False,
            )

    def test_multiscale_featurized_with_scale(self):
        sdata = _multiscale_sdata(multiscale_image=True, multiscale_labels=True)
        adata = sq.experimental.im.calculate_image_features(
            sdata,
            image_key="img",
            labels_key="lbl",
            scale="scale0",
            features=["skimage:morphology:area"],
            inplace=False,
        )
        # The fixture places 16 cells of 30x30=900 px at full resolution. Asserting
        # the exact count, label IDs, and area proves scale0 was read (scale1 would
        # give area ~225) and that no cell was silently dropped.
        assert adata.n_obs == 16
        assert set(adata.obs["label_id"].astype(int)) == set(range(1, 17))
        np.testing.assert_array_equal(adata[:, "area"].X.ravel(), np.full(16, 900.0))

    def test_invalid_scale_name(self):
        sdata = _multiscale_sdata(multiscale_image=True, multiscale_labels=True)
        with pytest.raises(ValueError, match="Scale 'scale9' not found"):
            sq.experimental.im.calculate_image_features(
                sdata,
                image_key="img",
                labels_key="lbl",
                scale="scale9",
                features=["skimage:morphology:area"],
                inplace=False,
            )


# ---------------------------------------------------------------------------
# Shapes that fail to rasterize
# ---------------------------------------------------------------------------


def test_shapes_rasterize_failure_raises():
    """Empty geometries raise a clear, actionable error during rasterization."""
    image_xr = xr.DataArray(
        np.random.default_rng(5).integers(0, 255, (3, 100, 100), dtype=np.uint8),
        dims=["c", "y", "x"],
        coords={"c": ["R", "G", "B"]},
    )
    # An empty polygon is unsupported by rasterize; the function should wrap the
    # failure in an actionable error rather than let the raw one surface.
    degenerate = ShapesModel.parse(gpd.GeoDataFrame(geometry=[Polygon()]))
    sdata = SpatialData(images={"img": Image2DModel.parse(image_xr)}, shapes={"cells": degenerate})
    with pytest.raises(ValueError, match="Failed to rasterize shapes"):
        sq.experimental.im.calculate_image_features(
            sdata,
            image_key="img",
            shapes_key="cells",
            features=["skimage:morphology:area"],
            inplace=False,
        )


# ---------------------------------------------------------------------------
# Optional image_key: morphology needs only the labels
# ---------------------------------------------------------------------------


class TestOptionalImage:
    """`image_key` is required only for intensity / squidpy features."""

    def test_morphology_only_without_image(self, sdata_synthetic):
        """skimage:morphology runs from the labels alone, no image_key."""
        result = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            labels_key="test_labels",
            features=["skimage:morphology:area"],
            inplace=False,
        )
        assert isinstance(result, ad.AnnData)
        assert result.n_obs > 0
        assert list(result.var_names) == ["area"]

    def test_morphology_only_without_image_parallel(self, sdata_synthetic):
        """The no-image path also works under threaded tile dispatch."""
        result = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            labels_key="test_labels",
            features=["skimage:morphology:area"],
            tile_size=100,  # >1 tile on the 200x200 grid
            n_jobs=2,
            inplace=False,
        )
        assert result.n_obs > 0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"features": ["skimage:intensity"]}, r"require pixel data.*image_key"),
            # mixed request: the error must name the offending (intensity) flag
            ({"features": ["skimage:morphology", "skimage:intensity"]}, "skimage:intensity"),
            ({"features": ["squidpy:summary"]}, "squidpy:summary"),
            ({"features": ["skimage:morphology:area"], "channels": ["R"]}, "`channels` selection requires `image_key`"),
        ],
    )
    def test_requires_image_key_raises(self, sdata_synthetic, kwargs, match):
        """Intensity / squidpy features and channel selection need image_key."""
        with pytest.raises(ValueError, match=match):
            sq.experimental.im.calculate_image_features(
                sdata_synthetic, labels_key="test_labels", inplace=False, **kwargs
            )

    def test_shapes_without_image_raises(self):
        rng = np.random.default_rng(7)
        shapes = ShapesModel.parse(gpd.GeoDataFrame(geometry=[Polygon([(40, 40), (70, 40), (70, 70), (40, 70)])]))
        # An image exists in the object, but we deliberately do not pass image_key.
        image_xr = xr.DataArray(
            rng.integers(0, 255, (1, 200, 200), dtype=np.uint8), dims=["c", "y", "x"], coords={"c": ["x"]}
        )
        sdata = SpatialData(images={"img": Image2DModel.parse(image_xr)}, shapes={"cells": shapes})
        with pytest.raises(ValueError, match="`shapes_key` requires `image_key`"):
            sq.experimental.im.calculate_image_features(
                sdata,
                shapes_key="cells",
                features=["skimage:morphology:area"],
                inplace=False,
            )


class TestAlignment:
    """Coordinate-system alignment and the rasterize fallback."""

    @pytest.mark.parametrize("multiscale", [False, True])
    def test_integer_translation_strict_succeeds(self, multiscale):
        sdata = _toy_sdata(labels_translation=(10, 10), multiscale=multiscale)
        result = sq.experimental.im.calculate_image_features(
            sdata,
            image_key="img",
            labels_key="lbl",
            scale="scale0" if multiscale else None,
            features=["skimage:morphology"],
            inplace=False,
        )
        assert result.n_obs > 0

    def test_rasterize_resamples_and_warns(self):
        sdata = _toy_sdata(labels_scale=(1.3, 1.3))
        with pytest.warns(UserWarning, match="Materializing labels onto the image grid"):
            result = sq.experimental.im.calculate_image_features(
                sdata,
                image_key="img",
                labels_key="lbl",
                features=["skimage:morphology"],
                align_mode="rasterize",
                inplace=False,
            )
        assert result.n_obs > 0

    def test_translation_drops_outside_and_partial_cells(self):
        # Shift labels so some cells straddle / fall outside the overlap edge.
        sdata = _toy_sdata(labels_translation=(120, 120))
        n_cells = int((np.unique(sdata.labels["lbl"].values) != 0).sum())
        with pytest.warns(UserWarning, match="Dropping"):
            result = sq.experimental.im.calculate_image_features(
                sdata, image_key="img", labels_key="lbl", features=["skimage:morphology:area"], inplace=False
            )
        # Dropped cells (outside + partial) leave the output; every survivor is
        # fully inside the overlap, so its area is untruncated (cells are 25x25).
        assert 0 < result.n_obs < n_cells
        np.testing.assert_array_equal(result[:, "area"].X.ravel(), 625.0)

    def test_multiscale_rasterize_raises(self):
        sdata = _toy_sdata(labels_scale=(1.3, 1.3), multiscale=True)
        with pytest.raises(ValueError, match="not supported for multiscale"):
            sq.experimental.im.calculate_image_features(
                sdata,
                image_key="img",
                labels_key="lbl",
                scale="scale0",
                features=["skimage:morphology"],
                align_mode="rasterize",
                inplace=False,
            )


class TestCpMeasure:
    """cp_measure feature paths."""

    def test_cpmeasure_correlation(self, sdata_synthetic):
        result = sq.experimental.im.calculate_image_features(
            sdata_synthetic,
            image_key="test_img",
            labels_key="test_labels",
            features=["cpmeasure:correlation"],
            inplace=False,
        )
        assert any("Correlation" in c for c in result.var_names)

    def test_cpmeasure_correlation_single_channel_raises(self):
        """Correlation-only with one channel fails fast instead of crashing cp_measure."""
        img = xr.DataArray(
            np.random.default_rng(0).integers(0, 255, (1, 100, 100), dtype=np.uint8),
            dims=["c", "y", "x"],
            coords={"c": ["DAPI"]},
        )
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[10:40, 10:40] = 1
        sdata = SpatialData(
            images={"img": Image2DModel.parse(img)},
            labels={"lbl": Labels2DModel.parse(xr.DataArray(labels, dims=["y", "x"]))},
        )
        with pytest.raises(ValueError, match="require >=2 channels"):
            sq.experimental.im.calculate_image_features(
                sdata, image_key="img", labels_key="lbl", features=["cpmeasure:correlation"], inplace=False
            )

    def test_cpmeasure_n_jobs_equivalence(self, sdata_synthetic):
        """cp_measure under threads (n_jobs>1) matches serial output."""
        kw = {
            "image_key": "test_img",
            "labels_key": "test_labels",
            "features": ["cpmeasure:sizeshape"],
            "inplace": False,
        }
        r1 = sq.experimental.im.calculate_image_features(sdata_synthetic, n_jobs=1, **kw)
        r4 = sq.experimental.im.calculate_image_features(sdata_synthetic, n_jobs=4, **kw)
        r1 = r1[r1.obs_names.sort_values()]
        r4 = r4[r4.obs_names.sort_values()]
        assert list(r1.var_names) == list(r4.var_names)
        np.testing.assert_allclose(r1.X, r4[:, r1.var_names].X, rtol=1e-5, atol=1e-6)

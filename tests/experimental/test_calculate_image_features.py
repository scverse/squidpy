from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel

import squidpy as sq


@pytest.fixture()
def sdata_hne_small(sdata_hne):
    """Small subset of sdata_hne for faster tests (aim for 10–100 spots)."""

    if "spots" not in sdata_hne.shapes:
        return sdata_hne

    spots = sdata_hne.shapes["spots"]
    try:
        spots = spots.loc[~spots.geometry.is_empty]  # type: ignore[attr-defined]
    except AttributeError:
        pass

    # Take the first ~100 spots to keep rasterization fast and non-empty
    spots_subset = spots.iloc[:100] if len(spots) > 100 else spots
    if len(spots_subset) == 0:
        return sdata_hne

    return SpatialData(
        images=sdata_hne.images,
        labels=sdata_hne.labels,
        shapes={"spots": spots_subset},
        tables=sdata_hne.tables,
    )


class TestCalculateImageFeatures:
    """Tests for calculate_image_features function."""

    def test_calculate_features_with_shapes(self, sdata_hne_small):
        """Test basic feature calculation with shapes."""
        # Use minimal measurements to keep test fast
        sq.experimental.im.calculate_image_features(
            sdata_hne_small,
            image_key="hne",
            shapes_key="spots",
            scale="scale0",
            measurements=["skimage:label"],
            adata_key_added="morphology",
            n_jobs=1,
            inplace=True,
        )

        # Check that the table was added
        assert "morphology" in sdata_hne_small.tables
        adata = sdata_hne_small.tables["morphology"]

        # Check basic structure
        assert adata.n_obs > 0
        assert adata.n_vars > 0

        # Check that spatialdata_attrs is set
        assert "spatialdata_attrs" in adata.uns
        assert adata.uns["spatialdata_attrs"]["region"] == "spots"
        assert adata.uns["spatialdata_attrs"]["region_key"] == "region"
        assert adata.uns["spatialdata_attrs"]["instance_key"] == "label_id"

        # Check that region and label_id are in obs
        assert "region" in adata.obs
        assert "label_id" in adata.obs

    def test_calculate_features_copy(self, sdata_hne_small):
        """Test that copy=False returns DataFrame."""
        result = sq.experimental.im.calculate_image_features(
            sdata_hne_small,
            image_key="hne",
            shapes_key="spots",
            scale="scale0",
            measurements=["skimage:label"],
            n_jobs=1,
            inplace=False,
        )

        # Should return DataFrame when inplace=False
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] > 0
        assert result.shape[1] > 0

    def test_invalid_image_key(self, sdata_hne_small):
        """Test error when image key doesn't exist."""
        with pytest.raises(ValueError, match="Image key 'nonexistent' not found"):
            sq.experimental.im.calculate_image_features(
                sdata_hne_small,
                image_key="nonexistent",
                shapes_key="spots",
                measurements=["skimage:label"],
            )

    def test_invalid_shapes_key(self, sdata_hne_small):
        """Test error when shapes key doesn't exist."""
        with pytest.raises(ValueError, match="Shapes key 'nonexistent' not found"):
            sq.experimental.im.calculate_image_features(
                sdata_hne_small,
                image_key="hne",
                shapes_key="nonexistent",
                measurements=["skimage:label"],
            )

    def test_both_labels_and_shapes_error(self, sdata_hne_small):
        """Test error when both labels_key and shapes_key are provided."""
        with pytest.raises(ValueError, match="Use either `labels_key` or `shapes_key`, not both"):
            sq.experimental.im.calculate_image_features(
                sdata_hne_small,
                image_key="hne",
                labels_key="fake_labels",
                shapes_key="spots",
                measurements=["skimage:label"],
            )

    def test_missing_labels_and_shapes(self, sdata_hne_small):
        """Test error when neither labels_key nor shapes_key is provided."""
        with pytest.raises(ValueError, match="Provide either `labels_key` or `shapes_key`."):
            sq.experimental.im.calculate_image_features(
                sdata_hne_small,
                image_key="hne",
                measurements=["skimage:label"],
            )

    def test_invalid_measurement(self, sdata_hne_small):
        """Test error with invalid measurement type."""
        with pytest.raises(ValueError, match="Invalid measurement"):
            sq.experimental.im.calculate_image_features(
                sdata_hne_small,
                image_key="hne",
                shapes_key="spots",
                scale="scale0",
                measurements=["nonexistent:measurement"],
            )

    def test_no_valid_measurements(self, sdata_hne_small):
        """Test error when no valid measurements are requested."""
        with pytest.raises(ValueError, match="No valid measurements requested"):
            sq.experimental.im.calculate_image_features(
                sdata_hne_small,
                image_key="hne",
                shapes_key="spots",
                scale="scale0",
                measurements=[],
                n_jobs=1,
                inplace=False,
            )

    def test_with_intensity_features(self, sdata_hne_small):
        """Test intensity-based features with multi-channel image."""
        result = sq.experimental.im.calculate_image_features(
            sdata_hne_small,
            image_key="hne",
            shapes_key="spots",
            scale="scale0",
            measurements=["skimage:label+image"],
            n_jobs=1,
            inplace=False,
        )

        assert result.shape[0] > 0
        assert result.shape[1] > 0
        # Column names should include channel information
        assert any("_" in col for col in result.columns)

    def test_dimension_mismatch(self):
        """Test error when image and labels have mismatched dimensions."""
        rng = np.random.default_rng(42)

        # Create image: 100x100, 3 channels
        image_data = rng.integers(0, 255, (3, 100, 100), dtype=np.uint8)
        image_xr = xr.DataArray(
            image_data,
            dims=["c", "y", "x"],
            coords={"c": ["R", "G", "B"]},
        )

        # Create labels: 80x80 (different dimensions)
        labels_data = rng.integers(1, 10, (80, 80), dtype=np.uint32)
        labels_xr = xr.DataArray(labels_data, dims=["y", "x"])

        sdata = SpatialData(
            images={"test_img": Image2DModel.parse(image_xr)},
            labels={"test_labels": Labels2DModel.parse(labels_xr)},
        )

        with pytest.raises(ValueError, match="do not match"):
            sq.experimental.im.calculate_image_features(
                sdata,
                image_key="test_img",
                labels_key="test_labels",
                measurements=["skimage:label"],
                n_jobs=1,
            )

    def test_with_progress_bar(self, sdata_hne_small):
        """Test that progress bar can be enabled."""
        result = sq.experimental.im.calculate_image_features(
            sdata_hne_small,
            image_key="hne",
            shapes_key="spots",
            scale="scale0",
            measurements=["skimage:label"],
            show_progress_bar=True,
            n_jobs=1,
            inplace=False,
        )

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] > 0

    def test_single_mask_property(self, sdata_hne_small):
        """Test selecting a single skimage mask property (area) only."""
        result = sq.experimental.im.calculate_image_features(
            sdata_hne_small,
            image_key="hne",
            shapes_key="spots",
            scale="scale0",
            measurements=["skimage:label:area"],
            inplace=False,
            n_jobs=1,
        )

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] > 0
        assert list(result.columns) == ["area"]

    def test_single_intensity_property(self, sdata_hne_small):
        """Test selecting a single intensity property (mean) per channel."""
        result = sq.experimental.im.calculate_image_features(
            sdata_hne_small,
            image_key="hne",
            shapes_key="spots",
            scale="scale0",
            measurements=["skimage:label+image:intensity_mean"],
            inplace=False,
            n_jobs=1,
        )

        # Expect one column per channel
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] > 0
        assert all(col.endswith(("_0", "_1", "_2")) or "_" in col for col in result.columns)
        # Should not contain other intensity props
        assert not any(col.startswith("intensity_max") for col in result.columns)

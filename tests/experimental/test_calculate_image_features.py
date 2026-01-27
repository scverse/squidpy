from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel

import squidpy as sq


class TestCalculateImageFeatures:
    """Tests for calculate_image_features function."""

    def test_calculate_features_with_shapes(self, sdata_hne):
        """Test basic feature calculation with shapes."""
        # Use minimal measurements to keep test fast
        sq.experimental.im.calculate_image_features(
            sdata_hne,
            image_key="hne",
            shapes_key="spots",
            scale="scale0",
            measurements=["skimage:label"],
            adata_key_added="morphology",
            n_jobs=1,
            inplace=True,
        )

        # Check that the table was added
        assert "morphology" in sdata_hne.tables
        adata = sdata_hne.tables["morphology"]

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

    def test_calculate_features_copy(self, sdata_hne):
        """Test that copy=False returns DataFrame."""
        result = sq.experimental.im.calculate_image_features(
            sdata_hne,
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

    def test_invalid_image_key(self, sdata_hne):
        """Test error when image key doesn't exist."""
        with pytest.raises(ValueError, match="Image key 'nonexistent' not found"):
            sq.experimental.im.calculate_image_features(
                sdata_hne,
                image_key="nonexistent",
                shapes_key="spots",
                measurements=["skimage:label"],
            )

    def test_invalid_shapes_key(self, sdata_hne):
        """Test error when shapes key doesn't exist."""
        with pytest.raises(ValueError, match="Shapes key 'nonexistent' not found"):
            sq.experimental.im.calculate_image_features(
                sdata_hne,
                image_key="hne",
                shapes_key="nonexistent",
                measurements=["skimage:label"],
            )

    def test_both_labels_and_shapes_error(self, sdata_hne):
        """Test error when both labels_key and shapes_key are provided."""
        with pytest.raises(ValueError, match="Use either `labels_key` or `shapes_key`, not both"):
            sq.experimental.im.calculate_image_features(
                sdata_hne,
                image_key="hne",
                labels_key="fake_labels",
                shapes_key="spots",
                measurements=["skimage:label"],
            )

    def test_invalid_measurement(self, sdata_hne):
        """Test error with invalid measurement type."""
        with pytest.raises(ValueError, match="Invalid measurement"):
            sq.experimental.im.calculate_image_features(
                sdata_hne,
                image_key="hne",
                shapes_key="spots",
                scale="scale0",
                measurements=["nonexistent:measurement"],
            )

    def test_with_intensity_features(self, sdata_hne):
        """Test intensity-based features with multi-channel image."""
        result = sq.experimental.im.calculate_image_features(
            sdata_hne,
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

    def test_with_progress_bar(self, sdata_hne):
        """Test that progress bar can be enabled."""
        result = sq.experimental.im.calculate_image_features(
            sdata_hne,
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

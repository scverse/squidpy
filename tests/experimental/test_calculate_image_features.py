from __future__ import annotations

import pandas as pd
import pytest

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

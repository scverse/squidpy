"""Tests for the unified dataset registry."""

from __future__ import annotations

import pytest

from squidpy.datasets._registry import (
    DatasetEntry,
    DatasetRegistry,
    DatasetType,
    FileEntry,
    get_registry,
)


class TestFileEntry:
    """Tests for FileEntry dataclass."""

    def test_entry_creation(self):
        entry = FileEntry(
            name="test.h5ad",
            s3_key="figshare/test.h5ad",
            sha256="abc123",
        )
        assert entry.name == "test.h5ad"
        assert entry.sha256 == "abc123"

    def test_get_urls_with_s3(self):
        entry = FileEntry(
            name="test.h5ad",
            s3_key="figshare/test.h5ad",
        )
        urls = entry.get_urls("https://s3.example.com")
        assert len(urls) == 1
        assert urls[0] == "https://s3.example.com/figshare/test.h5ad"


class TestDatasetEntry:
    """Tests for DatasetEntry dataclass."""

    def test_single_file_dataset(self):
        entry = DatasetEntry(
            name="test",
            type=DatasetType.ANNDATA,
            files=[
                FileEntry(
                    name="test.h5ad",
                    s3_key="test.h5ad",
                )
            ],
            shape=(100, 50),
        )
        assert len(entry.files) == 1
        assert entry.shape == (100, 50)

    def test_visium_10x_dataset(self):
        entry = DatasetEntry(
            name="V1_Test",
            type=DatasetType.VISIUM_10X,
            files=[
                FileEntry(
                    name="filtered_feature_bc_matrix.h5",
                    s3_key="test.h5",
                ),
                FileEntry(name="spatial.tar.gz", s3_key="test.tar.gz"),
                FileEntry(name="image.tif", s3_key="test.tif"),
            ],
        )
        assert len(entry.files) == 3
        assert entry.type == DatasetType.VISIUM_10X
        assert entry.get_file_by_name_prefix("image.") is not None

    def test_get_file(self):
        entry = DatasetEntry(
            name="test",
            type=DatasetType.VISIUM_10X,
            files=[
                FileEntry(
                    name="filtered_feature_bc_matrix.h5",
                    s3_key="test.h5",
                ),
                FileEntry(name="spatial.tar.gz", s3_key="test.tar.gz"),
            ],
        )
        f = entry.get_file("spatial.tar.gz")
        assert f is not None
        assert f.name == "spatial.tar.gz"

        assert entry.get_file("nonexistent") is None


class TestDatasetRegistry:
    """Tests for DatasetRegistry class."""

    def test_from_yaml_loads_config(self):
        registry = DatasetRegistry.from_yaml()
        assert registry is not None
        assert len(registry.datasets) > 0

    def test_anndata_datasets_loaded(self):
        registry = DatasetRegistry.from_yaml()
        assert "four_i" in registry
        assert "imc" in registry
        assert "seqfish" in registry
        assert "visium_hne_adata" in registry

    def test_anndata_dataset_fields(self):
        registry = DatasetRegistry.from_yaml()
        four_i = registry["four_i"]
        assert four_i.type == DatasetType.ANNDATA
        assert four_i.shape == (270876, 43)
        assert len(four_i.files) == 1

    def test_image_datasets_loaded(self):
        registry = DatasetRegistry.from_yaml()
        assert "visium_hne_image" in registry
        assert "visium_hne_image_crop" in registry
        assert "visium_fluo_image_crop" in registry

    def test_image_has_library_id(self):
        registry = DatasetRegistry.from_yaml()
        img = registry["visium_hne_image"]
        assert img.library_id == "V1_Adult_Mouse_Brain"

    def test_spatialdata_loaded(self):
        registry = DatasetRegistry.from_yaml()
        assert "visium_hne_sdata" in registry
        sdata = registry["visium_hne_sdata"]
        assert sdata.type == DatasetType.SPATIALDATA

    def test_visium_10x_datasets_loaded(self):
        registry = DatasetRegistry.from_yaml()
        # Check samples from different versions
        assert "V1_Adult_Mouse_Brain" in registry
        assert "Parent_Visium_Human_Cerebellum" in registry
        assert "Visium_FFPE_Mouse_Brain" in registry

    def test_visium_10x_dataset_structure(self):
        registry = DatasetRegistry.from_yaml()
        v1_sample = registry["V1_Adult_Mouse_Brain"]
        assert v1_sample.type == DatasetType.VISIUM_10X
        assert len(v1_sample.files) == 3  # matrix, spatial, image
        assert v1_sample.get_file_by_name_prefix("image.") is not None

    def test_visium_10x_has_jpg(self):
        """Test that Visium_FFPE_Human_Normal_Prostate has jpg image."""
        registry = DatasetRegistry.from_yaml()
        sample = registry["Visium_FFPE_Human_Normal_Prostate"]
        assert sample.type == DatasetType.VISIUM_10X
        # Check it's a jpg
        img_file = sample.get_file_by_name_prefix("image.")
        assert img_file is not None
        assert img_file.name == "image.jpg"

    def test_get_dataset(self):
        registry = DatasetRegistry.from_yaml()
        entry = registry.get("four_i")
        assert entry is not None
        assert entry.name == "four_i"

        assert registry.get("nonexistent") is None

    def test_getitem(self):
        registry = DatasetRegistry.from_yaml()
        entry = registry["four_i"]
        assert entry.name == "four_i"

        with pytest.raises(KeyError):
            _ = registry["nonexistent"]

    def test_contains(self):
        registry = DatasetRegistry.from_yaml()
        assert "four_i" in registry
        assert "nonexistent" not in registry

    def test_iter_by_type(self):
        registry = DatasetRegistry.from_yaml()
        anndata_entries = list(registry.iter_by_type(DatasetType.ANNDATA))
        assert len(anndata_entries) == 11  # 11 h5ad datasets

        visium_10x_entries = list(registry.iter_by_type(DatasetType.VISIUM_10X))
        assert len(visium_10x_entries) == 35  # 35 Visium samples

    def test_property_lists(self):
        registry = DatasetRegistry.from_yaml()
        assert len(registry.anndata_datasets) == 11
        assert len(registry.image_datasets) == 3
        assert len(registry.spatialdata_datasets) == 1
        assert len(registry.visium_datasets) == 35

    def test_all_names(self):
        registry = DatasetRegistry.from_yaml()
        names = registry.all_names
        assert "four_i" in names
        assert "visium_hne_image" in names
        assert "V1_Adult_Mouse_Brain" in names
        # Total: 11 + 3 + 1 + 35 = 50
        assert len(names) == 50


class TestGetRegistry:
    """Tests for get_registry singleton function."""

    def test_returns_registry(self):
        registry = get_registry()
        assert isinstance(registry, DatasetRegistry)

    def test_returns_same_instance(self):
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2

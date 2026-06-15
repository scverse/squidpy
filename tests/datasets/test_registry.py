"""Tests for squidpy's dataset registry (built on scverse_misc.datasets)."""

from __future__ import annotations

import pytest
from scverse_misc.datasets import DatasetRegistry

from squidpy.datasets._registry import dataset_names, get_registry


class TestGetRegistry:
    def test_returns_registry(self):
        assert isinstance(get_registry(), DatasetRegistry)

    def test_returns_same_instance(self):
        assert get_registry() is get_registry()

    def test_base_url(self):
        assert get_registry().base_url == "https://exampledata.scverse.org/squidpy/"


class TestRegistryContents:
    def test_anndata_entry(self):
        four_i = get_registry()["four_i"]
        assert four_i.type == "anndata"
        assert four_i.metadata["shape"] == (270876, 43)
        assert four_i.file(suffix=".h5ad").s3_key == "four_i.h5ad"

    def test_image_entry_has_library_id(self):
        img = get_registry()["visium_hne_image"]
        assert img.type == "image"
        assert img.metadata["library_id"] == "V1_Adult_Mouse_Brain"

    def test_spatialdata_entries(self):
        reg = get_registry()
        assert reg["visium_hne_sdata"].type == "spatialdata"
        assert reg["cells"].type == "spatialdata"
        assert reg["cells"].file(suffix=".zip").name == "cells.zip"

    def test_visium_10x_entry(self):
        sample = get_registry()["V1_Adult_Mouse_Brain"]
        assert sample.type == "visium_10x"
        assert len(sample.files) == 3
        assert any(f.name.startswith("image.") for f in sample.files)

    def test_unknown_dataset_raises(self):
        with pytest.raises(KeyError):
            _ = get_registry()["nonexistent"]
        assert "nonexistent" not in get_registry()


class TestDatasetNames:
    def test_counts_by_type(self):
        assert len(dataset_names("anndata")) == 11
        assert len(dataset_names("image")) == 3
        assert len(dataset_names("spatialdata")) == 2
        assert len(dataset_names("visium_10x")) == 35

    def test_all_names(self):
        names = dataset_names()
        assert {"four_i", "visium_hne_image", "V1_Adult_Mouse_Brain", "cells"} <= set(names)
        assert len(names) == 51

"""Tests for the unified dataset downloader."""

from __future__ import annotations

from pathlib import Path

import pytest
from scanpy import settings

from squidpy.datasets._downloader import (
    DatasetDownloader,
    download,
    get_downloader,
)
from squidpy.datasets._registry import get_registry


class TestDatasetDownloader:
    """Tests for DatasetDownloader class."""

    def test_init_default_cache_dir(self):
        downloader = DatasetDownloader(registry=get_registry())
        assert downloader.cache_dir == Path(settings.datasetdir)

    def test_init_custom_cache_dir(self, tmp_path: Path):
        downloader = DatasetDownloader(registry=get_registry(), cache_dir=tmp_path / "custom_cache")
        assert downloader.cache_dir == tmp_path / "custom_cache"
        assert downloader.cache_dir.exists()

    def test_init_custom_s3_url(self):
        s3_url = "https://my-bucket.s3.amazonaws.com"
        downloader = DatasetDownloader(registry=get_registry(), s3_base_url=s3_url)
        assert downloader._s3_base_url == s3_url

    def test_registry_loaded(self):
        downloader = DatasetDownloader(registry=get_registry())
        assert downloader.registry is not None
        assert len(downloader.registry.datasets) > 0

    def test_download_unknown_dataset(self, tmp_path: Path):
        downloader = DatasetDownloader(registry=get_registry(), cache_dir=tmp_path)
        with pytest.raises(ValueError, match="Unknown dataset"):
            downloader.download("nonexistent_dataset")


class TestGetDownloader:
    """Tests for get_downloader singleton function."""

    def test_returns_downloader(self):
        downloader = get_downloader()
        assert isinstance(downloader, DatasetDownloader)

    def test_returns_same_instance(self):
        # lru_cache ensures singleton behavior
        downloader1 = get_downloader()
        downloader2 = get_downloader()
        assert downloader1 is downloader2


class TestDownloadFunction:
    """Tests for download convenience function."""

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            download("nonexistent_dataset")


class TestDownloaderIntegration:
    """Integration tests that require network access."""

    @pytest.mark.timeout(120)
    @pytest.mark.internet()
    def test_download_imc_dataset(self):
        """Test downloading a small AnnData dataset."""
        from anndata import AnnData

        # Use scanpy.settings.datasetdir to match what download_data.py uses
        downloader = DatasetDownloader(registry=get_registry(), cache_dir=settings.datasetdir)
        adata = downloader.download("imc")

        assert isinstance(adata, AnnData)
        assert adata.shape == (4668, 34)

    @pytest.mark.timeout(120)
    @pytest.mark.internet()
    def test_download_caches_file(self):
        """Test that downloaded files are cached."""
        cache_dir = Path(settings.datasetdir)
        downloader = DatasetDownloader(registry=get_registry(), cache_dir=cache_dir)

        # First download
        adata1 = downloader.download("imc")

        # Check file exists in cache
        cache_files = list((cache_dir / "anndata").glob("*.h5ad"))
        # At least one file (may have more from other tests)
        assert len(cache_files) >= 1

        # Second download should use cache (no network)
        adata2 = downloader.download("imc")
        assert adata1.shape == adata2.shape

    @pytest.mark.timeout(180)
    @pytest.mark.internet()
    def test_download_visium_sample(self):
        """Test downloading a Visium sample."""
        from anndata import AnnData

        downloader = DatasetDownloader(registry=get_registry(), cache_dir=settings.datasetdir)
        adata = downloader.download("V1_Mouse_Kidney", include_hires_tiff=False)

        assert isinstance(adata, AnnData)
        assert "spatial" in adata.uns

    @pytest.mark.timeout(300)
    @pytest.mark.internet()
    def test_include_hires_tiff_caching_behavior(self):
        """Test include_hires_tiff: cached files persist, return varies.

        On CI, V1_Mouse_Kidney is pre-cached via .scripts/ci/download_data.py
        with include_hires_tiff=True, so this tests return behavior.
        """
        sample_id = "V1_Mouse_Kidney"
        cache_dir = Path(settings.datasetdir)
        hires_image_path = cache_dir / "visium" / sample_id / "image.tif"
        downloader = DatasetDownloader(registry=get_registry(), cache_dir=cache_dir)

        # include_hires_tiff=False: no source_image_path in metadata
        adata = downloader.download(sample_id, include_hires_tiff=False)
        metadata = adata.uns["spatial"][sample_id].get("metadata", {})
        assert "source_image_path" not in metadata

        # include_hires_tiff=True: source_image_path in metadata, file cached
        adata = downloader.download(sample_id, include_hires_tiff=True)
        metadata = adata.uns["spatial"][sample_id].get("metadata", {})
        assert "source_image_path" in metadata
        assert Path(metadata["source_image_path"]).exists()
        assert hires_image_path.exists()

        # include_hires_tiff=False again: cached file persists, not in metadata
        adata = downloader.download(sample_id, include_hires_tiff=False)
        metadata = adata.uns["spatial"][sample_id].get("metadata", {})
        assert "source_image_path" not in metadata
        assert hires_image_path.exists()  # file still cached

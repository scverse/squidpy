"""Tests for the unified dataset downloader."""

from __future__ import annotations

from pathlib import Path

import pytest

from squidpy.datasets._downloader import (
    DEFAULT_CACHE_DIR,
    DatasetDownloader,
    download,
    get_downloader,
)
from squidpy.datasets._registry import get_registry


class TestDatasetDownloader:
    """Tests for DatasetDownloader class."""

    def test_init_default_cache_dir(self):
        downloader = DatasetDownloader()
        assert downloader.cache_dir == DEFAULT_CACHE_DIR

    def test_init_custom_cache_dir(self, tmp_path: Path):
        downloader = DatasetDownloader(cache_dir=tmp_path / "custom_cache")
        assert downloader.cache_dir == tmp_path / "custom_cache"
        assert downloader.cache_dir.exists()

    def test_init_custom_s3_url(self):
        downloader = DatasetDownloader(s3_base_url="https://my-bucket.s3.amazonaws.com")
        assert downloader._s3_base_url == "https://my-bucket.s3.amazonaws.com"

    def test_registry_loaded(self):
        downloader = DatasetDownloader()
        assert downloader._registry is not None
        assert len(downloader._registry.datasets) > 0

    def test_download_unknown_dataset(self, tmp_path: Path):
        downloader = DatasetDownloader(cache_dir=tmp_path)
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
    def test_download_imc_dataset(self, tmp_path: Path):
        """Test downloading a small AnnData dataset."""
        from anndata import AnnData

        downloader = DatasetDownloader(cache_dir=tmp_path)
        adata = downloader.download("imc")

        assert isinstance(adata, AnnData)
        assert adata.shape == (4668, 34)

    @pytest.mark.timeout(120)
    @pytest.mark.internet()
    def test_download_caches_file(self, tmp_path: Path):
        """Test that downloaded files are cached."""
        downloader = DatasetDownloader(cache_dir=tmp_path)

        # First download
        adata1 = downloader.download("imc")
        
        # Check file exists in cache
        cache_files = list((tmp_path / "anndata").glob("*.h5ad"))
        assert len(cache_files) == 1

        # Second download should use cache (no network)
        adata2 = downloader.download("imc")
        assert adata1.shape == adata2.shape

    @pytest.mark.timeout(180)
    @pytest.mark.internet()
    def test_download_visium_sample(self, tmp_path: Path):
        """Test downloading a Visium sample."""
        from anndata import AnnData

        downloader = DatasetDownloader(cache_dir=tmp_path)
        adata = downloader.download("V1_Mouse_Kidney", include_hires_tiff=False)

        assert isinstance(adata, AnnData)
        assert "spatial" in adata.uns


class TestFallbackUrls:
    """Tests for fallback URL behavior."""

    def test_primary_url_is_s3_when_configured(self, tmp_path: Path):
        """Test that S3 URL is tried first when configured."""
        downloader = DatasetDownloader(
            cache_dir=tmp_path,
            s3_base_url="https://my-bucket.s3.amazonaws.com",
        )

        registry = get_registry()
        entry = registry["imc"]
        file_entry = entry.files[0]
        urls = file_entry.get_urls(downloader._s3_base_url)

        # S3 should be first
        assert urls[0].startswith("https://my-bucket.s3.amazonaws.com")
        # Fallback should be second
        assert "figshare.com" in urls[1]

    def test_fallback_only_when_no_s3(self, tmp_path: Path):
        """Test that only fallback is used when S3 not configured."""
        downloader = DatasetDownloader(
            cache_dir=tmp_path,
            s3_base_url="",  # No S3
        )

        registry = get_registry()
        entry = registry["imc"]
        file_entry = entry.files[0]
        urls = file_entry.get_urls(downloader._s3_base_url)

        # Only fallback URL
        assert len(urls) == 1
        assert "figshare.com" in urls[0]

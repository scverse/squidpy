"""Tests for squidpy's dataset loaders + downloader (built on scverse_misc.datasets)."""

from __future__ import annotations

from pathlib import Path

import pytest
from scanpy import settings
from scverse_misc.datasets import available_loaders

from squidpy.datasets._downloader import (
    DatasetDownloader,
    download,
    get_downloader,
)
from squidpy.datasets._registry import get_registry


class TestLoaderRegistration:
    def test_squidpy_loaders_registered(self):
        # importing the downloader module registers squidpy's domain loaders
        loaders = available_loaders()
        assert {"anndata", "image", "spatialdata", "visium_10x"} <= set(loaders)


class TestDatasetDownloader:
    def test_default_cache_dir(self):
        assert DatasetDownloader(registry=get_registry()).cache_dir == Path(settings.datasetdir)

    def test_custom_cache_dir(self, tmp_path: Path):
        assert DatasetDownloader(registry=get_registry(), cache_dir=tmp_path).cache_dir == tmp_path

    def test_unknown_dataset_raises(self, tmp_path: Path):
        dl = DatasetDownloader(registry=get_registry(), cache_dir=tmp_path)
        with pytest.raises(ValueError, match="Unknown dataset"):
            dl.download("nonexistent_dataset")


class TestGetDownloader:
    def test_returns_downloader(self):
        assert isinstance(get_downloader(), DatasetDownloader)

    def test_singleton(self):
        assert get_downloader() is get_downloader()


class TestDownloadFunction:
    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            download("nonexistent_dataset")


class TestDownloaderIntegration:
    """Integration tests that require network access."""

    @pytest.mark.timeout(120)
    @pytest.mark.internet()
    def test_download_anndata(self):
        from anndata import AnnData

        adata = DatasetDownloader(registry=get_registry(), cache_dir=settings.datasetdir).download("imc")
        assert isinstance(adata, AnnData)
        assert adata.shape == (4668, 34)

    @pytest.mark.timeout(120)
    @pytest.mark.internet()
    def test_anndata_is_cached(self):
        cache_dir = Path(settings.datasetdir)
        dl = DatasetDownloader(registry=get_registry(), cache_dir=cache_dir)
        dl.download("imc")
        assert list((cache_dir / "anndata").glob("*.h5ad"))

    @pytest.mark.timeout(180)
    @pytest.mark.internet()
    def test_download_visium_sample(self):
        from anndata import AnnData

        adata = DatasetDownloader(registry=get_registry(), cache_dir=settings.datasetdir).download(
            "V1_Mouse_Kidney", include_hires_tiff=False
        )
        assert isinstance(adata, AnnData)
        assert "spatial" in adata.uns

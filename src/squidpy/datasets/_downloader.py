"""Unified dataset downloader using pooch."""

from __future__ import annotations

import shutil
import tarfile
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pooch
from scanpy import logging as logg

from squidpy.datasets._registry import DatasetEntry, DatasetType, FileEntry, get_registry

if TYPE_CHECKING:
    from anndata import AnnData

    from squidpy.im import ImageContainer

__all__ = ["DatasetDownloader", "download", "get_downloader"]

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "squidpy"


class DatasetDownloader:
    """Unified downloader for all squidpy datasets.

    Parameters
    ----------
    cache_dir
        Directory to cache downloaded files. Defaults to ~/.cache/squidpy.
    s3_base_url
        Base URL for S3 bucket. If None, uses the value from datasets.yaml.
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        s3_base_url: str | None = None,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._registry = get_registry()
        self._s3_base_url = s3_base_url if s3_base_url is not None else self._registry.s3_base_url

    def _download_file(
        self,
        file_entry: FileEntry,
        target_dir: Path,
        target_name: str | None = None,
    ) -> Path:
        """Download a single file."""
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = target_name or file_entry.name
        local_path = target_dir / filename

        # Return if already cached
        if local_path.exists():
            logg.debug(f"Using cached file: {local_path}")
            return local_path

        urls = file_entry.get_urls(self._s3_base_url)
        last_error = None

        for url in urls:
            try:
                logg.info(f"Downloading {filename} from {url}")
                downloaded = pooch.retrieve(
                    url=url,
                    known_hash=f"sha256:{file_entry.sha256}" if file_entry.sha256 else None,
                    fname=filename,
                    path=str(target_dir),
                    progressbar=True,
                )
                return Path(downloaded)
            except (OSError, ValueError, RuntimeError) as e:
                last_error = e
                logg.warning(f"Failed to download from {url}: {e}")
                continue

        raise RuntimeError(f"Failed to download {filename} from all sources. Last error: {last_error}")

    def download(self, name: str, path: Path | str | None = None, **kwargs: Any) -> Any:
        """Download a dataset by name and return the appropriate object.

        Parameters
        ----------
        name
            Dataset name from the registry.
        path
            Optional custom path for download.
        **kwargs
            Additional arguments passed to the loader.

        Returns
        -------
        Loaded dataset (AnnData, ImageContainer, SpatialData, or AnnData for Visium).
        """
        if name not in self._registry:
            raise ValueError(f"Unknown dataset: {name}. Available: {self._registry.all_names}")

        entry = self._registry[name]

        if entry.type == DatasetType.ANNDATA:
            return self._load_anndata(entry, path, **kwargs)
        elif entry.type == DatasetType.IMAGE:
            return self._load_image(entry, path, **kwargs)
        elif entry.type == DatasetType.SPATIALDATA:
            return self._load_spatialdata(entry, path)
        elif entry.type == DatasetType.ADATA_WITH_IMAGE:
            include_hires_tiff = kwargs.pop("include_hires_tiff", False)
            return self._load_adata_with_image(entry, path, include_hires_tiff=include_hires_tiff)
        else:
            raise ValueError(f"Unknown dataset type: {entry.type}")

    def _load_anndata(
        self,
        entry: DatasetEntry,
        path: Path | str | None = None,
        **kwargs: Any,
    ) -> AnnData:
        """Download and load an AnnData dataset."""
        import anndata

        if not entry.files:
            raise ValueError(f"Dataset {entry.name} has no files")

        file_entry = entry.files[0]
        if path is not None:
            path = Path(path)
            target_dir = path.parent
            # Ensure proper extension is used
            expected_suffix = Path(file_entry.name).suffix
            target_name = path.name if path.suffix else f"{path.name}{expected_suffix}"
        else:
            target_dir = self.cache_dir / "anndata"
            target_name = file_entry.name

        local_path = self._download_file(file_entry, target_dir, target_name)
        adata = anndata.read_h5ad(local_path, **kwargs)

        # Validate shape if specified
        if entry.shape is not None and adata.shape != entry.shape:
            logg.warning(f"Expected shape {entry.shape}, got {adata.shape}")

        return adata

    def _load_image(
        self,
        entry: DatasetEntry,
        path: Path | str | None = None,
        **kwargs: Any,
    ) -> ImageContainer:
        """Download and load an image dataset."""
        from squidpy.im import ImageContainer

        if not entry.files:
            raise ValueError(f"Dataset {entry.name} has no files")

        file_entry = entry.files[0]
        if path is not None:
            path = Path(path)
            target_dir = path.parent
            # Ensure proper extension is used (e.g., .tiff, .png)
            expected_suffix = Path(file_entry.name).suffix
            target_name = path.name if path.suffix else f"{path.name}{expected_suffix}"
        else:
            target_dir = self.cache_dir / "images"
            target_name = file_entry.name

        local_path = self._download_file(file_entry, target_dir, target_name)

        img = ImageContainer()
        img.add_img(local_path, layer="image", library_id=entry.library_id, **kwargs)

        return img

    def _load_spatialdata(
        self,
        entry: DatasetEntry,
        path: Path | str | None = None,
    ) -> Any:  # Returns SpatialData
        """Download and load a SpatialData dataset."""
        import spatialdata as sd

        if not entry.files:
            raise ValueError(f"Dataset {entry.name} has no files")

        file_entry = entry.files[0]
        folder = Path(path) if path is not None else self.cache_dir / "spatialdata"
        folder.mkdir(parents=True, exist_ok=True)

        zarr_path = folder / f"{entry.name}.zarr"

        # Return if already extracted
        if zarr_path.exists():
            logg.info(f"Loading existing dataset from {zarr_path}")
            return sd.read_zarr(zarr_path)

        # Download zip
        zip_path = self._download_file(file_entry, folder)

        # Extract
        logg.info(f"Extracting {zip_path} to {folder}")
        shutil.unpack_archive(str(zip_path), folder)

        if not zarr_path.exists():
            raise RuntimeError(f"Expected extracted data at {zarr_path}, but not found")

        return sd.read_zarr(zarr_path)

    def _load_adata_with_image(
        self,
        entry: DatasetEntry,
        path: Path | str | None = None,
        include_hires_tiff: bool = False,
    ) -> AnnData:
        """Download and load an AnnData with image dataset (e.g., 10x Visium)."""
        from squidpy.read._read import visium as read_visium

        # Set up directories
        if path is not None:
            base_dir = Path(path)
        else:
            base_dir = self.cache_dir / "visium"
        sample_dir = base_dir / entry.name
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Download feature matrix
        matrix_file = entry.get_file("filtered_feature_bc_matrix.h5")
        if matrix_file is None:
            raise ValueError(f"Dataset {entry.name} missing filtered_feature_bc_matrix.h5")

        matrix_path = self._download_file(matrix_file, sample_dir)
        # Copy to expected name for reader
        expected_matrix = sample_dir / "filtered_feature_bc_matrix.h5"
        if not expected_matrix.exists() and matrix_path != expected_matrix:
            shutil.copy(matrix_path, expected_matrix)

        # Download and extract spatial data
        spatial_file = entry.get_file("spatial.tar.gz")
        if spatial_file is None:
            raise ValueError(f"Dataset {entry.name} missing spatial.tar.gz")

        spatial_path = self._download_file(spatial_file, sample_dir)
        with tarfile.open(spatial_path) as f:
            for member in f:
                target = sample_dir / member.name
                if not target.exists():
                    f.extract(member, sample_dir)

        # Optionally download high-res image (can be tif, jpg, etc.)
        source_image_path = None
        if include_hires_tiff:
            # Look for any image file (image.tif, image.jpg, etc.)
            image_file = entry.get_file_by_prefix("image.")
            if image_file is None:
                logg.warning(f"High-res image not available for {entry.name}")
            else:
                try:
                    image_path = self._download_file(image_file, sample_dir)
                    # Use the actual image filename (preserves extension)
                    source_image_path = sample_dir / image_file.name
                    if not source_image_path.exists() and image_path != source_image_path:
                        shutil.copy(image_path, source_image_path)
                except (OSError, ValueError, RuntimeError) as e:
                    logg.warning(f"Failed to download high-res image: {e}")

        # Read using squidpy reader
        if source_image_path and source_image_path.exists():
            return read_visium(sample_dir, source_image_path=source_image_path)
        return read_visium(sample_dir)


@lru_cache(maxsize=1)
def get_downloader() -> DatasetDownloader:
    """Get the singleton downloader instance.

    Uses lru_cache to ensure a single instance without mutable global state.
    """
    return DatasetDownloader()


def download(name: str, path: Path | str | None = None, **kwargs: Any) -> Any:
    """Download a dataset by name.

    This is a convenience function that automatically determines the dataset type.

    Parameters
    ----------
    name
        Dataset name.
    path
        Optional custom path.
    **kwargs
        Additional arguments passed to the loader.

    Returns
    -------
    Loaded dataset (AnnData, ImageContainer, SpatialData, or AnnData for Visium).
    """
    return get_downloader().download(name, path, **kwargs)

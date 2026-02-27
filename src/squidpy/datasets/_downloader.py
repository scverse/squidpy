"""Unified dataset downloader using pooch."""

from __future__ import annotations

import shutil
import tarfile
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pooch
from scanpy import settings
from spatialdata._logging import logger as logg

from squidpy.datasets._registry import (
    DatasetEntry,
    DatasetRegistry,
    DatasetType,
    FileEntry,
    get_registry,
)

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData

    from squidpy.im import ImageContainer

__all__ = [
    "DatasetDownloader",
    "download",
    "get_downloader",
]


class DatasetDownloader:
    """Unified downloader for all squidpy datasets.

    Parameters
    ----------
    cache_dir
        Directory to cache downloaded files. Defaults to :attr:`scanpy.settings.datasetdir`.
    s3_base_url
        Base URL for S3 bucket. If None, uses the value from datasets.yaml.
    """

    def __init__(
        self,
        registry: DatasetRegistry,
        cache_dir: Path | str | None = None,
        s3_base_url: str | None = None,
    ):
        self.cache_dir = Path(cache_dir or settings.datasetdir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.registry = registry
        self._s3_base_url = s3_base_url or self.registry.s3_base_url

    def _resolve_path(
        self,
        path: Path | str | None,
        file_entry: FileEntry,
        default_subdir: str,
    ) -> tuple[Path, str]:
        """Resolve target directory and filename from path argument."""
        if path is not None:
            path = Path(path)
            target_dir = path.parent
            suffix = Path(file_entry.name).suffix
            target_name = path.name if path.suffix else f"{path.name}{suffix}"
        else:
            target_dir = self.cache_dir / default_subdir
            target_name = file_entry.name
        return target_dir, target_name

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

        if local_path.exists():
            logg.debug(f"Using cached file: {local_path}")
            return local_path

        urls = file_entry.get_urls(self._s3_base_url)
        errors: list[Exception] = []

        for url in urls:
            try:
                logg.info(f"Downloading {filename} from {url}")
                downloaded = pooch.retrieve(
                    url=url,
                    known_hash=(f"sha256:{file_entry.sha256}" if file_entry.sha256 else None),
                    fname=filename,
                    path=str(target_dir),
                    progressbar=True,
                )
                return Path(downloaded)
            except (OSError, ValueError, RuntimeError) as e:
                errors.append(e)
                logg.warning(f"Failed to download from {url}: {e}")

        msg = f"Failed to download {filename}"
        raise ExceptionGroup(msg, errors)

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
        Loaded dataset.
        """
        if name not in self.registry:
            raise ValueError(f"Unknown dataset: {name}. Available: {self.registry.all_names}")

        entry = self.registry[name]
        loaders = {
            DatasetType.ANNDATA: lambda: self._load_anndata(entry, path, **kwargs),
            DatasetType.IMAGE: lambda: self._load_image(entry, path, **kwargs),
            DatasetType.SPATIALDATA: lambda: self._load_spatialdata(entry, path),
            DatasetType.VISIUM_10X: lambda: self._load_visium_10x(
                entry,
                path,
                include_hires_tiff=kwargs.pop("include_hires_tiff", False),
            ),
        }

        loader = loaders.get(entry.type)
        if loader is None:
            raise ValueError(f"Unknown dataset type: {entry.type}")
        return loader()

    def _load_anndata(
        self,
        entry: DatasetEntry,
        path: Path | str | None = None,
        **kwargs: Any,
    ) -> AnnData:
        """Download and load an AnnData dataset."""
        import anndata

        file_entry = entry.get_file_by_suffix(".h5ad")
        if file_entry is None:
            raise ValueError(f"Dataset {entry.name} has no .h5ad file")
        target_dir, target_name = self._resolve_path(path, file_entry, "anndata")

        local_path = self._download_file(file_entry, target_dir, target_name)
        adata = anndata.read_h5ad(local_path, **kwargs)

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

        file_entry = entry.get_file_by_suffix(".tiff")
        if file_entry is None:
            raise ValueError(f"Dataset {entry.name} has no .tiff file")
        target_dir, target_name = self._resolve_path(path, file_entry, "images")

        local_path = self._download_file(file_entry, target_dir, target_name)

        img = ImageContainer()
        img.add_img(local_path, layer="image", library_id=entry.library_id, **kwargs)
        return img

    def _load_spatialdata(
        self,
        entry: DatasetEntry,
        path: Path | str | None = None,
    ) -> SpatialData:
        """Download and load a SpatialData dataset."""
        import spatialdata as sd

        file_entry = entry.get_file_by_suffix(".zip")
        if file_entry is None:
            raise ValueError(f"Dataset {entry.name} has no .zip file")
        folder = Path(path or self.cache_dir / "spatialdata")
        folder.mkdir(parents=True, exist_ok=True)

        zarr_path = folder / f"{entry.name}.zarr"

        if zarr_path.exists():
            logg.info(f"Loading existing dataset from {zarr_path}")
            return sd.read_zarr(zarr_path)

        zip_path = self._download_file(file_entry, folder)
        logg.info(f"Extracting {zip_path} to {folder}")
        shutil.unpack_archive(str(zip_path), folder)

        if not zarr_path.exists():
            raise RuntimeError(f"Expected extracted data at {zarr_path}, but not found")

        return sd.read_zarr(zarr_path)

    def _load_visium_10x(
        self,
        entry: DatasetEntry,
        path: Path | str | None = None,
        include_hires_tiff: bool = False,
    ) -> AnnData:
        """Download and load a 10x Genomics Visium dataset."""
        from squidpy.read._read import visium as read_visium

        base_dir = Path(path or self.cache_dir / "visium")
        sample_dir = base_dir / entry.name
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Download feature matrix
        matrix_file = entry.get_file("filtered_feature_bc_matrix.h5")
        if matrix_file is None:
            raise ValueError(f"Dataset {entry.name} missing filtered_feature_bc_matrix.h5")
        self._download_file(matrix_file, sample_dir)

        # Download and extract spatial data
        spatial_file = entry.get_file("spatial.tar.gz")
        if spatial_file is None:
            raise ValueError(f"Dataset {entry.name} missing spatial.tar.gz")

        spatial_path = self._download_file(spatial_file, sample_dir)
        with tarfile.open(spatial_path) as f:
            for member in f:
                if not (sample_dir / member.name).exists():
                    f.extract(member, sample_dir)

        # Optionally download high-res image
        source_image_path = None
        if include_hires_tiff:
            image_file = entry.get_file_by_name_prefix("image.")
            if image_file is None:
                logg.warning(f"High-res image not available for {entry.name}")
            else:
                try:
                    self._download_file(image_file, sample_dir)
                    source_image_path = sample_dir / image_file.name
                except (OSError, ValueError, RuntimeError) as e:
                    logg.warning(f"Failed to download high-res image: {e}")

        if source_image_path and source_image_path.exists():
            return read_visium(sample_dir, source_image_path=source_image_path)
        return read_visium(sample_dir)


@lru_cache(maxsize=1)
def get_downloader() -> DatasetDownloader:
    """Get the singleton downloader instance."""
    return DatasetDownloader(registry=get_registry())


def download(name: str, path: Path | str | None = None, **kwargs: Any) -> Any:
    """Download a dataset by name.

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
    Loaded dataset.
    """
    return get_downloader().download(name, path, **kwargs)

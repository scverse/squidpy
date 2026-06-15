"""squidpy's dataset loaders, registered against :mod:`scverse_misc.datasets`.

The generic download/extract/verify machinery lives in ``scverse-misc``. squidpy only
registers loaders for its domain-specific dataset types (``image`` ->
:class:`~squidpy.im.ImageContainer`, ``visium_10x`` -> :func:`squidpy.read.visium`,
``spatialdata`` -> :func:`spatialdata.read_zarr`) and overrides the built-in ``anndata``
loader to emit squidpy's shape warning.
"""

from __future__ import annotations

import tarfile
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scanpy import settings
from scverse_misc.datasets import Fetcher, register_loader
from spatialdata._logging import logger as logg

from squidpy.datasets._registry import get_registry

if TYPE_CHECKING:
    from anndata import AnnData
    from scverse_misc.datasets import DatasetRegistry, FetchContext

__all__ = ["DatasetDownloader", "download", "get_downloader"]


@register_loader("anndata")
def _load_anndata(ctx: FetchContext, /, **kwargs: Any) -> AnnData:
    import anndata

    path = ctx.download(ctx.entry.file(suffix=".h5ad"))
    adata = anndata.read_h5ad(path, **kwargs)
    shape = ctx.entry.metadata.get("shape")
    if shape is not None and tuple(adata.shape) != tuple(shape):
        logg.warning(f"Expected shape {tuple(shape)}, got {adata.shape}")
    return adata


@register_loader("image")
def _load_image(ctx: FetchContext, /, **kwargs: Any) -> Any:
    from squidpy.im import ImageContainer

    path = ctx.download(ctx.entry.file(suffix=".tiff"))
    img = ImageContainer()
    img.add_img(path, layer="image", library_id=ctx.entry.metadata.get("library_id"), **kwargs)
    return img


@register_loader("spatialdata")
def _load_spatialdata(ctx: FetchContext, /, **kwargs: Any) -> Any:
    import spatialdata as sd

    zarr_path = ctx.target_dir / f"{ctx.entry.name}.zarr"
    if zarr_path.exists():
        logg.info(f"Loading existing dataset from {zarr_path}")
        return sd.read_zarr(zarr_path)

    zip_path = ctx.download(ctx.entry.file(suffix=".zip"))
    ctx.extract_archive(zip_path)
    if not zarr_path.exists():
        raise RuntimeError(f"Expected extracted data at {zarr_path}, but not found")
    return sd.read_zarr(zarr_path)


@register_loader("visium_10x")
def _load_visium_10x(ctx: FetchContext, /, *, include_hires_tiff: bool = False, **kwargs: Any) -> AnnData:
    from squidpy.read._read import visium as read_visium

    sample_dir = ctx.target_dir / ctx.entry.name

    ctx.download(ctx.entry.file(name="filtered_feature_bc_matrix.h5"), dest=sample_dir)

    spatial_path = ctx.download(ctx.entry.file(name="spatial.tar.gz"), dest=sample_dir)
    with tarfile.open(spatial_path) as f:
        for member in f:
            if not (sample_dir / member.name).exists():
                f.extract(member, sample_dir)

    source_image_path = None
    if include_hires_tiff:
        image_file = next((fe for fe in ctx.entry.files if fe.name.startswith("image.")), None)
        if image_file is None:
            logg.warning(f"High-res image not available for {ctx.entry.name}")
        else:
            try:
                source_image_path = ctx.download(image_file, dest=sample_dir)
            except (OSError, ValueError, RuntimeError) as e:
                logg.warning(f"Failed to download high-res image: {e}")

    if source_image_path is not None and source_image_path.exists():
        return read_visium(sample_dir, source_image_path=source_image_path)
    return read_visium(sample_dir)


class DatasetDownloader:
    """Thin squidpy wrapper over :class:`scverse_misc.datasets.Fetcher`.

    Parameters
    ----------
    registry
        The dataset registry to download from.
    cache_dir
        Base download directory. Defaults to :attr:`scanpy.settings.datasetdir`.
    """

    def __init__(self, registry: DatasetRegistry, cache_dir: Path | str | None = None) -> None:
        self.registry = registry
        self.cache_dir = Path(cache_dir or settings.datasetdir)

    def download(self, name: str, path: Path | str | None = None, **kwargs: Any) -> Any:
        """Download and load a dataset by name, optionally into a custom ``path``."""
        if name not in self.registry:
            raise ValueError(f"Unknown dataset: {name}. Available: {sorted(self.registry.datasets)}")
        cache_dir = Path(path) if path is not None else self.cache_dir
        return Fetcher(self.registry, cache_dir=cache_dir).fetch(name, **kwargs)


@lru_cache(maxsize=1)
def get_downloader() -> DatasetDownloader:
    """Get the singleton downloader instance."""
    return DatasetDownloader(registry=get_registry())


def download(name: str, path: Path | str | None = None, **kwargs: Any) -> Any:
    """Download a dataset by name (convenience wrapper around :func:`get_downloader`)."""
    return get_downloader().download(name, path, **kwargs)

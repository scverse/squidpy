"""squidpy's dataset loaders, registered against :mod:`scverse_misc.datasets`.

The generic download/extract/verify machinery and the ``anndata``/``spatialdata`` loaders
live in ``scverse-misc``. squidpy registers loaders for its domain-specific types
(``image`` -> :class:`~squidpy.im.ImageContainer`, ``visium_10x`` -> :func:`squidpy.read.visium`)
and overrides the built-in ``anndata`` loader to emit squidpy's shape warning.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from scanpy import settings
from scverse_misc.datasets import fetch, register_loader
from spatialdata._logging import logger as logg

from squidpy.datasets._registry import get_base_url, get_registry

if TYPE_CHECKING:
    from collections.abc import Callable

    from anndata import AnnData
    from scverse_misc.datasets import DatasetEntry

__all__ = ["download"]


# NB: ``register_loader`` mutates scverse-misc's process-global loader registry at import
# time. Overriding the built-in ``anndata`` loader here is intentional (it adds squidpy's
# shape warning) and applies process-wide to any scverse-misc consumer in the same process.
@register_loader("anndata")
def _load_anndata(entry: DatasetEntry, target: Path, download: Callable[..., Any], **kwargs: Any) -> AnnData:
    import anndata

    adata = anndata.read_h5ad(download(entry.file(suffix=".h5ad")), **kwargs)
    shape = entry.metadata.get("shape")
    if shape is not None and tuple(adata.shape) != tuple(shape):
        logg.warning(f"Expected shape {tuple(shape)}, got {adata.shape}")
    return adata


@register_loader("image")
def _load_image(entry: DatasetEntry, target: Path, download: Callable[..., Any], **kwargs: Any) -> Any:
    from squidpy.im import ImageContainer

    img = ImageContainer()
    img.add_img(
        download(entry.file(suffix=".tiff")), layer="image", library_id=entry.metadata.get("library_id"), **kwargs
    )
    return img


@register_loader("visium_10x")
def _load_visium_10x(
    entry: DatasetEntry, target: Path, download: Callable[..., Any], *, include_hires_tiff: bool = False, **kwargs: Any
) -> AnnData:
    import pooch

    from squidpy.read._read import visium as read_visium

    sample_dir = target / entry.name
    download(entry.file(name="filtered_feature_bc_matrix.h5"), dest=sample_dir)
    download(entry.file(name="spatial.tar.gz"), dest=sample_dir, processor=pooch.Untar(extract_dir="."))

    source_image_path = None
    if include_hires_tiff:
        image_file = next((f for f in entry.files if f.name.startswith("image.")), None)
        if image_file is None:
            logg.warning(f"High-res image not available for {entry.name}")
        else:
            try:
                source_image_path = download(image_file, dest=sample_dir)
            except (OSError, ValueError, RuntimeError) as e:
                logg.warning(f"Failed to download high-res image: {e}")

    if source_image_path is not None:
        return read_visium(sample_dir, source_image_path=source_image_path)
    return read_visium(sample_dir)


def download(name: str, path: Path | str | None = None, **kwargs: Any) -> Any:
    """Download and load a dataset by name via :func:`scverse_misc.datasets.fetch`.

    Files are cached under ``path / <type>`` (or ``settings.datasetdir / <type>`` if ``path`` is None).
    """
    registry = get_registry()
    if name not in registry:
        raise ValueError(f"Unknown dataset: {name}. Available: {sorted(registry)}")
    cache_dir = Path(path) if path is not None else Path(settings.datasetdir)
    return fetch(registry[name], cache_dir, base_url=get_base_url(), **kwargs)

"""squidpy's dataset registry, built on top of :mod:`scverse_misc.datasets`.

The generic registry/downloader lives in ``scverse-misc``; squidpy only provides
its own ``datasets.yaml`` and folds squidpy-specific metadata (``shape``,
``library_id``) into the generic :class:`~scverse_misc.datasets.DatasetEntry`
``metadata`` mapping.
"""

from __future__ import annotations

import importlib.resources
from functools import lru_cache
from typing import TYPE_CHECKING

import yaml
from scverse_misc.datasets import DatasetEntry, DatasetRegistry, FileEntry

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

__all__ = ["get_registry", "dataset_names"]


def _config_traversable() -> Traversable:
    return importlib.resources.files("squidpy.datasets").joinpath("datasets.yaml")


@lru_cache(maxsize=1)
def get_registry() -> DatasetRegistry:
    """Load squidpy's dataset registry (cached)."""
    with _config_traversable().open() as f:
        config = yaml.safe_load(f) or {}

    datasets: dict[str, DatasetEntry] = {}
    for name, data in (config.get("datasets") or {}).items():
        files = tuple(
            FileEntry(
                name=fd["name"],
                url=fd.get("url"),
                s3_key=fd.get("s3_key"),
                sha256=fd.get("sha256"),
            )
            for fd in data.get("files", [])
        )
        metadata: dict[str, object] = {}
        if "shape" in data:
            metadata["shape"] = tuple(data["shape"])
        if "library_id" in data:
            metadata["library_id"] = data["library_id"]
        datasets[name] = DatasetEntry(
            name=name,
            type=data["type"],
            files=files,
            doc_header=data.get("doc_header"),
            metadata=metadata,
        )

    base_url = config.get("s3_base_url") or config.get("base_url")
    return DatasetRegistry(base_url=base_url, datasets=datasets)


def dataset_names(dataset_type: str | None = None) -> list[str]:
    """Return dataset names, optionally filtered by ``type`` (e.g. ``"visium_10x"``)."""
    registry = get_registry()
    return [name for name, entry in registry.datasets.items() if dataset_type is None or entry.type == dataset_type]

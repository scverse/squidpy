"""squidpy's dataset registry: ``datasets.yaml`` parsed via :func:`scverse_misc.datasets.parse_registry`.

squidpy-specific fields (``shape``, ``library_id``, ``doc_header``) land in each entry's
``metadata`` mapping automatically.
"""

from __future__ import annotations

import importlib.resources
from functools import lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING

from scverse_misc.datasets import parse_registry

if TYPE_CHECKING:
    from collections.abc import Mapping

    from scverse_misc.datasets import DatasetEntry

__all__ = ["get_registry", "get_base_url", "dataset_names"]


@lru_cache(maxsize=1)
def _load() -> tuple[str | None, Mapping[str, DatasetEntry]]:
    """Parse ``datasets.yaml`` once, returning ``(base_url, read-only mapping)``."""
    path = importlib.resources.files("squidpy.datasets").joinpath("datasets.yaml")
    base_url, datasets = parse_registry(str(path))
    return base_url, MappingProxyType(datasets)


def get_registry() -> Mapping[str, DatasetEntry]:
    """Return squidpy's datasets as a read-only ``{name: DatasetEntry}`` mapping (cached)."""
    return _load()[1]


def get_base_url() -> str | None:
    """Return the registry's base URL."""
    return _load()[0]


def dataset_names(dataset_type: str | None = None) -> list[str]:
    """Return dataset names, optionally filtered by ``type`` (e.g. ``"visium_10x"``)."""
    return [name for name, entry in get_registry().items() if dataset_type is None or entry.type == dataset_type]

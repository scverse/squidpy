"""squidpy's dataset registry: ``datasets.yaml`` parsed via :func:`scverse_misc.datasets.parse_registry`.

squidpy-specific fields (``shape``, ``library_id``, ``doc_header``) land in each entry's
``metadata`` mapping automatically.
"""

from __future__ import annotations

import importlib.resources
from functools import lru_cache
from typing import TYPE_CHECKING

from scverse_misc.datasets import parse_registry

if TYPE_CHECKING:
    from scverse_misc.datasets import DatasetEntry

__all__ = ["get_registry", "get_base_url", "dataset_names"]


@lru_cache(maxsize=1)
def _parsed() -> tuple[str | None, dict[str, DatasetEntry]]:
    path = importlib.resources.files("squidpy.datasets").joinpath("datasets.yaml")
    return parse_registry(str(path))


def get_registry() -> dict[str, DatasetEntry]:
    """Return squidpy's datasets as ``{name: DatasetEntry}`` (cached)."""
    return _parsed()[1]


def get_base_url() -> str | None:
    """Return the registry's base URL."""
    return _parsed()[0]


def dataset_names(dataset_type: str | None = None) -> list[str]:
    """Return dataset names, optionally filtered by ``type`` (e.g. ``"visium_10x"``)."""
    return [name for name, entry in get_registry().items() if dataset_type is None or entry.type == dataset_type]

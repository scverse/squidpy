"""Unified dataset registry loaded from YAML configuration."""

from __future__ import annotations

import importlib.resources
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from collections.abc import Iterator
    from importlib.resources.abc import Traversable

    from squidpy.read._utils import PathLike

__all__ = ["DatasetType", "FileEntry", "DatasetEntry", "DatasetRegistry", "get_registry"]


def _get_config_traversable() -> Traversable:
    """Get the file-like object to datasets.yaml using importlib.resources for robustness."""
    # Using importlib.resources for robust path resolution across different installation methods
    # (editable installs, zip imports, etc.)
    return importlib.resources.files("squidpy.datasets").joinpath("datasets.yaml")


class DatasetType(Enum):
    """Types of datasets."""

    ANNDATA = "anndata"
    IMAGE = "image"
    SPATIALDATA = "spatialdata"
    VISIUM_10X = "visium_10x"


@dataclass(frozen=True)
class FileEntry:
    """Metadata for a single file within a dataset."""

    name: str
    s3_key: str
    sha256: str | None = None

    def get_urls(self, s3_base_url: str) -> list[str]:
        """Return list of URLs to try, primary (S3) first, then fallback."""
        urls = []
        if s3_base_url and self.s3_key:
            urls.append(f"{s3_base_url.rstrip('/')}/{self.s3_key}")
        return urls


@dataclass
class DatasetEntry:
    """Metadata for a dataset (can have one or multiple files)."""

    name: str
    type: DatasetType
    files: list[FileEntry]
    shape: tuple[int, ...] | None = None
    doc_header: str | None = None
    library_id: str | None = None

    def get_file(self, name: str) -> FileEntry | None:
        """Get a specific file by name."""
        for f in self.files:
            if f.name == name:
                return f
        return None

    def get_file_by_suffix(self, suffix: str) -> FileEntry | None:
        """Get a file by suffix (e.g., 'filtered_feature_bc_matrix.h5')."""
        for f in self.files:
            if f.name.endswith(suffix):
                return f
        return None

    def get_file_by_name_prefix(self, prefix: str) -> FileEntry | None:
        """Get a file by prefix of its name (e.g., 'image.' to find image.tif or image.jpg)."""
        for f in self.files:
            if f.name.startswith(prefix):
                return f
        return None


@dataclass
class DatasetRegistry:
    """Central registry for all squidpy datasets."""

    s3_base_url: str = ""
    datasets: dict[str, DatasetEntry] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config_path: PathLike | None = None) -> DatasetRegistry:
        """Load registry from YAML configuration file."""
        # This case should be always true
        # only for testing and tinkering config_path should be provided
        if config_path is None:
            with _get_config_traversable().open() as f:
                config = yaml.safe_load(f)
        else:
            with open(config_path) as f:
                config = yaml.safe_load(f)

        registry = cls(s3_base_url=config.get("s3_base_url", ""))

        # Load all datasets
        for name, data in config.get("datasets", {}).items():
            # Parse files
            files = []
            for file_data in data.get("files", []):
                files.append(
                    FileEntry(
                        name=file_data["name"],
                        s3_key=file_data["s3_key"],
                        sha256=file_data.get("sha256"),
                    )
                )

            # Parse shape
            shape = None
            if "shape" in data:
                shape_data = data["shape"]
                if isinstance(shape_data, list):
                    shape = tuple(shape_data)
                else:
                    shape = shape_data

            registry.datasets[name] = DatasetEntry(
                name=name,
                type=DatasetType(data["type"]),
                files=files,
                shape=shape,
                doc_header=data.get("doc_header"),
                library_id=data.get("library_id"),
            )

        return registry

    def get(self, name: str) -> DatasetEntry | None:
        """Get a dataset by name."""
        return self.datasets.get(name)

    def __getitem__(self, name: str) -> DatasetEntry:
        """Get a dataset by name, raises KeyError if not found."""
        if name not in self.datasets:
            raise KeyError(f"Unknown dataset: {name}. Available: {list(self.datasets.keys())}")
        return self.datasets[name]

    def __contains__(self, name: str) -> bool:
        """Check if dataset exists."""
        return name in self.datasets

    def iter_by_type(self, dataset_type: DatasetType) -> Iterator[DatasetEntry]:
        """Iterate over datasets of a specific type."""
        for entry in self.datasets.values():
            if entry.type == dataset_type:
                yield entry

    @property
    def anndata_datasets(self) -> list[str]:
        """Return names of all AnnData datasets."""
        return [name for name, entry in self.datasets.items() if entry.type == DatasetType.ANNDATA]

    @property
    def image_datasets(self) -> list[str]:
        """Return names of all image datasets."""
        return [name for name, entry in self.datasets.items() if entry.type == DatasetType.IMAGE]

    @property
    def spatialdata_datasets(self) -> list[str]:
        """Return names of all SpatialData datasets."""
        return [name for name, entry in self.datasets.items() if entry.type == DatasetType.SPATIALDATA]

    @property
    def visium_10x_datasets(self) -> list[str]:
        """Return names of all 10x Genomics Visium datasets."""
        return [name for name, entry in self.datasets.items() if entry.type == DatasetType.VISIUM_10X]

    @property
    def visium_datasets(self) -> list[str]:
        """Return names of all Visium datasets (alias for visium_10x_datasets)."""
        return self.visium_10x_datasets

    @property
    def all_names(self) -> list[str]:
        """Return all dataset names."""
        return list(self.datasets.keys())


@lru_cache(maxsize=1)
def get_registry() -> DatasetRegistry:
    """Get the singleton dataset registry instance.

    Uses lru_cache to ensure a single instance without mutable global state.
    """
    return DatasetRegistry.from_yaml()

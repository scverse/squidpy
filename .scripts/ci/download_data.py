#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

_CNT = 0  # increment this when you want to rebuild the CI cache


def _print_message(
    func_name: str, path: Path, *, dry_run: bool = False
) -> None:
    prefix = "[DRY RUN]" if dry_run else ""
    if path.exists():
        print(f"{prefix}[Cached]      {func_name:>35} <- {str(path):>40}")
    else:
        msg = f"{prefix}[Downloading] {func_name:>35} -> {str(path):>40}"
        print(msg)


def _maybe_download_data(func_name: str) -> Any:
    """Download data using default cache (DEFAULT_CACHE_DIR)."""
    import squidpy as sq

    try:
        return getattr(sq.datasets, func_name)()
    except Exception as e:  # noqa: BLE001
        print(f"Error downloading {func_name}: {e}")
        raise


def main(args: argparse.Namespace) -> None:
    from anndata import AnnData

    import squidpy as sq
    from squidpy.datasets._downloader import DEFAULT_CACHE_DIR
    from squidpy.datasets._registry import get_registry

    registry = get_registry()

    # AnnData datasets (single .h5ad files) - registry returns list[str]
    anndata_datasets = registry.anndata_datasets

    # Image datasets (single .tiff files) - registry returns list[str]
    image_datasets = registry.image_datasets

    # SpatialData datasets (.zarr directories) - registry returns list[str]
    spatialdata_datasets = registry.spatialdata_datasets

    # Visium samples tested in CI
    # Add any sample here that's used in tests to ensure it's cached
    visium_samples_to_cache = [
        "V1_Mouse_Kidney",
        "Targeted_Visium_Human_SpinalCord_Neuroscience",
        "Visium_FFPE_Human_Breast_Cancer",
    ]
    # Note: Only samples listed here will be cached. If you add a test
    # using a different Visium sample, add it to this list.

    if args.dry_run:
        print(f"Cache directory: {DEFAULT_CACHE_DIR}")
        print("\nAnnData datasets:")
        for name in anndata_datasets:
            path = DEFAULT_CACHE_DIR / "anndata" / f"{name}.h5ad"
            _print_message(name, path, dry_run=True)

        print("\nImage datasets:")
        for name in image_datasets:
            entry = registry[name]
            file_name = entry.files[0].name
            path = DEFAULT_CACHE_DIR / "images" / file_name
            _print_message(name, path, dry_run=True)

        print("\nSpatialData datasets:")
        for name in spatialdata_datasets:
            path = DEFAULT_CACHE_DIR / "spatialdata" / name
            _print_message(name, path, dry_run=True)

        print("\nVisium samples:")
        for name in visium_samples_to_cache:
            path = DEFAULT_CACHE_DIR / "visium" / name
            _print_message(name, path, dry_run=True)

        return

    print(f"Cache directory: {DEFAULT_CACHE_DIR}")

    # Download AnnData datasets
    print("\nDownloading AnnData datasets...")
    for name in anndata_datasets:
        path = DEFAULT_CACHE_DIR / "anndata" / f"{name}.h5ad"
        _print_message(name, path)
        obj = _maybe_download_data(name)
        assert isinstance(obj, AnnData), f"Expected AnnData, got {type(obj)}"
        assert path.is_file(), f"Expected file at {path}"

    # Download image datasets
    print("\nDownloading image datasets...")
    for name in image_datasets:
        entry = registry[name]
        file_name = entry.files[0].name
        path = DEFAULT_CACHE_DIR / "images" / file_name
        _print_message(name, path)
        obj = _maybe_download_data(name)
        assert isinstance(obj, sq.im.ImageContainer), (
            f"Expected ImageContainer, got {type(obj)}"
        )
        assert path.is_file(), f"Expected file at {path}"

    # Download SpatialData datasets
    print("\nDownloading SpatialData datasets...")
    for name in spatialdata_datasets:
        path = DEFAULT_CACHE_DIR / "spatialdata" / name
        _print_message(name, path)
        obj = _maybe_download_data(name)
        # Don't import spatialdata just for type check
        assert path.is_dir(), f"Expected directory at {path}"

    # Download Visium samples (these are needed for tests)
    # visium() now defaults to DEFAULT_CACHE_DIR/visium
    print("\nDownloading Visium samples...")
    for sample in visium_samples_to_cache:
        sample_dir = DEFAULT_CACHE_DIR / "visium" / sample
        matrix_file = sample_dir / "filtered_feature_bc_matrix.h5"
        _print_message(sample, sample_dir)

        obj = sq.datasets.visium(sample)
        assert isinstance(obj, AnnData), f"Expected AnnData, got {type(obj)}"
        assert matrix_file.is_file(), f"Expected file at {matrix_file}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download data used for tutorials/examples."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not download, just print what would be downloaded.",
    )

    main(parser.parse_args())

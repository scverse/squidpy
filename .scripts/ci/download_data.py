#!/usr/bin/env python3
"""Download datasets to populate CI cache.

This script downloads all datasets that tests might need.
The downloader handles caching to DEFAULT_CACHE_DIR (~/.cache/squidpy).
"""

from __future__ import annotations

import argparse

_CNT = 0  # increment this when you want to rebuild the CI cache


def main(args: argparse.Namespace) -> None:
    from anndata import AnnData

    import squidpy as sq
    from squidpy.datasets._downloader import DEFAULT_CACHE_DIR
    from squidpy.datasets._registry import get_registry

    registry = get_registry()
    print(f"Cache directory: {DEFAULT_CACHE_DIR}")

    # Visium samples tested in CI
    # Add any sample here that's used in tests to ensure it's cached
    visium_samples_to_cache = [
        "V1_Mouse_Kidney",
        "Targeted_Visium_Human_SpinalCord_Neuroscience",
        "Visium_FFPE_Human_Breast_Cancer",
    ]

    if args.dry_run:
        print("\nWould download:")
        print(f"  - {len(registry.anndata_datasets)} AnnData datasets")
        print(f"  - {len(registry.image_datasets)} Image datasets")
        print(f"  - {len(registry.spatialdata_datasets)} SpatialData datasets")
        print(f"  - {len(visium_samples_to_cache)} Visium samples")
        return

    # Download AnnData datasets - just call the function, it handles caching
    print("\nDownloading AnnData datasets...")
    for name in registry.anndata_datasets:
        print(f"  {name}")
        obj = getattr(sq.datasets, name)()
        assert isinstance(obj, AnnData), f"Expected AnnData, got {type(obj)}"

    # Download image datasets
    print("\nDownloading image datasets...")
    for name in registry.image_datasets:
        print(f"  {name}")
        obj = getattr(sq.datasets, name)()
        assert isinstance(obj, sq.im.ImageContainer)

    # Download SpatialData datasets
    print("\nDownloading SpatialData datasets...")
    for name in registry.spatialdata_datasets:
        print(f"  {name}")
        obj = getattr(sq.datasets, name)()
        # Returns SpatialData object

    # Download Visium samples (needed for tests)
    # Include high-res images since tests use include_hires_tiff=True
    print("\nDownloading Visium samples (with high-res images)...")
    for sample in visium_samples_to_cache:
        print(f"  {sample}")
        obj = sq.datasets.visium(sample, include_hires_tiff=True)
        assert isinstance(obj, AnnData), f"Expected AnnData, got {type(obj)}"

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download datasets to populate CI cache."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not download, just print what would be downloaded.",
    )

    main(parser.parse_args())

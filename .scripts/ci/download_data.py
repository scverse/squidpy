#!/usr/bin/env python3
"""Download datasets to populate CI cache.

This script downloads all datasets that tests might need.
The downloader handles caching to scanpy.settings.datasetdir.
"""

from __future__ import annotations

import argparse

from scanpy import settings
from spatialdata._logging import logger

_CNT = 0  # increment this when you want to rebuild the CI cache


def main(args: argparse.Namespace) -> None:
    from anndata import AnnData

    import squidpy as sq
    from squidpy.datasets._registry import dataset_names

    # Visium samples tested in CI
    visium_samples_to_cache = [
        "V1_Mouse_Kidney",
        "Targeted_Visium_Human_SpinalCord_Neuroscience",
        "Visium_FFPE_Human_Breast_Cancer",
    ]

    if args.dry_run:
        logger.info("Cache: %s", settings.datasetdir)
        logger.info(
            "Would download: %d AnnData, %d images, %d SpatialData, %d Visium",
            len(dataset_names("anndata")),
            len(dataset_names("image")),
            len(dataset_names("spatialdata")),
            len(visium_samples_to_cache),
        )
        return

    # Download all datasets - the downloader handles caching
    for name in dataset_names("anndata"):
        obj = getattr(sq.datasets, name)()
        assert isinstance(obj, AnnData)

    for name in dataset_names("image"):
        obj = getattr(sq.datasets, name)()
        assert isinstance(obj, sq.im.ImageContainer)

    for name in dataset_names("spatialdata"):
        getattr(sq.datasets, name)()

    for sample in visium_samples_to_cache:
        obj = sq.datasets.visium(sample, include_hires_tiff=True)
        assert isinstance(obj, AnnData)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets to populate CI cache.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not download, just print what would be downloaded.",
    )

    main(parser.parse_args())

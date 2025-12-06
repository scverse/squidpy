from __future__ import annotations

from squidpy.datasets._datasets import (
    # Type aliases for dataset names
    AnnDataDatasets,
    ImageDatasets,
    SpatialDataDatasets,
    VisiumDatasets,
    # AnnData datasets
    four_i,
    imc,
    merfish,
    mibitof,
    sc_mouse_cortex,
    seqfish,
    slideseqv2,
    # 10x Genomics Visium
    visium,
    visium_fluo_adata,
    visium_fluo_adata_crop,
    # Image datasets
    visium_fluo_image_crop,
    visium_hne_adata,
    visium_hne_adata_crop,
    visium_hne_image,
    visium_hne_image_crop,
    visium_hne_sdata,
)

__all__ = [
    # Type aliases
    "VisiumDatasets",
    "AnnDataDatasets",
    "ImageDatasets",
    "SpatialDataDatasets",
    # 10x Genomics
    "visium",
    "visium_hne_sdata",
    # AnnData datasets
    "four_i",
    "imc",
    "seqfish",
    "visium_hne_adata",
    "visium_hne_adata_crop",
    "visium_fluo_adata",
    "visium_fluo_adata_crop",
    "sc_mouse_cortex",
    "mibitof",
    "merfish",
    "slideseqv2",
    # Image datasets
    "visium_fluo_image_crop",
    "visium_hne_image_crop",
    "visium_hne_image",
]

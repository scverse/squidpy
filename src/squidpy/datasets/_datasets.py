"""Public dataset interface functions using hardcoded dataset names.

This module provides the public API for downloading squidpy datasets.
All functions fetch datasets by their known names from the registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from scanpy import settings

from squidpy.datasets._downloader import get_downloader
from squidpy.datasets._registry import DatasetType, get_registry
from squidpy.read._utils import PathLike

if TYPE_CHECKING:
    import spatialdata as sd
    from anndata import AnnData


# =============================================================================
# Hardcoded dataset name types
# =============================================================================

# 10x Genomics Visium datasets (visium_10x type)
VisiumDatasets = Literal[
    # spaceranger version 1.1.0 datasets
    "V1_Breast_Cancer_Block_A_Section_1",
    "V1_Breast_Cancer_Block_A_Section_2",
    "V1_Human_Heart",
    "V1_Human_Lymph_Node",
    "V1_Mouse_Kidney",
    "V1_Adult_Mouse_Brain",
    "V1_Mouse_Brain_Sagittal_Posterior",
    "V1_Mouse_Brain_Sagittal_Posterior_Section_2",
    "V1_Mouse_Brain_Sagittal_Anterior",
    "V1_Mouse_Brain_Sagittal_Anterior_Section_2",
    "V1_Human_Brain_Section_1",
    "V1_Human_Brain_Section_2",
    "V1_Adult_Mouse_Brain_Coronal_Section_1",
    "V1_Adult_Mouse_Brain_Coronal_Section_2",
    # spaceranger version 1.2.0 datasets
    "Targeted_Visium_Human_Cerebellum_Neuroscience",
    "Parent_Visium_Human_Cerebellum",
    "Targeted_Visium_Human_SpinalCord_Neuroscience",
    "Parent_Visium_Human_SpinalCord",
    "Targeted_Visium_Human_Glioblastoma_Pan_Cancer",
    "Parent_Visium_Human_Glioblastoma",
    "Targeted_Visium_Human_BreastCancer_Immunology",
    "Parent_Visium_Human_BreastCancer",
    "Targeted_Visium_Human_OvarianCancer_Pan_Cancer",
    "Targeted_Visium_Human_OvarianCancer_Immunology",
    "Parent_Visium_Human_OvarianCancer",
    "Targeted_Visium_Human_ColorectalCancer_GeneSignature",
    "Parent_Visium_Human_ColorectalCancer",
    # spaceranger version 1.3.0 datasets
    "Visium_FFPE_Mouse_Brain",
    "Visium_FFPE_Mouse_Brain_IF",
    "Visium_FFPE_Mouse_Kidney",
    "Visium_FFPE_Human_Breast_Cancer",
    "Visium_FFPE_Human_Prostate_Acinar_Cell_Carcinoma",
    "Visium_FFPE_Human_Prostate_Cancer",
    "Visium_FFPE_Human_Prostate_IF",
    "Visium_FFPE_Human_Normal_Prostate",
]

# AnnData datasets (.h5ad)
AnnDataDatasets = Literal[
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
]

# Image datasets (.tiff)
ImageDatasets = Literal[
    "visium_fluo_image_crop",
    "visium_hne_image_crop",
    "visium_hne_image",
]

# SpatialData datasets (.zarr)
SpatialDataDatasets = Literal["visium_hne_sdata",]


# =============================================================================
# 10x Genomics Visium functions
# =============================================================================


def visium(
    sample_id: VisiumDatasets,
    *,
    include_hires_tiff: bool = False,
    base_dir: PathLike | None = None,
) -> AnnData:
    """
    Download Visium `datasets <https://support.10xgenomics.com/spatial-gene-expression/datasets>`_ from *10x Genomics*.

    Parameters
    ----------
    sample_id
        Name of the Visium dataset.
    include_hires_tiff
        Whether to download the high-resolution tissue section into
        :attr:`anndata.AnnData.uns` ``['spatial']['{sample_id}']['metadata']['source_image_path']``.
    base_dir
        Directory where to download the data. If `None`, uses :attr:`scanpy.settings.datasetdir`.

    Returns
    -------
    :class:`anndata.AnnData`
        Spatial AnnData object.
    """
    # Validate sample_id against known names
    downloader = get_downloader()

    if sample_id not in downloader.registry:
        msg = f"Unknown Visium sample: {sample_id}. "
        msg += f"Available samples: {downloader.registry.visium_datasets}"
        raise ValueError(msg)

    # Use scanpy.settings.datasetdir/visium if base_dir not specified
    if base_dir is None:
        base_dir = Path(settings.datasetdir) / "visium"

    return downloader.download(sample_id, base_dir, include_hires_tiff=include_hires_tiff)


def visium_hne_sdata(folderpath: Path | str | None = None) -> sd.SpatialData:
    """
    Download a Visium H&E dataset as a SpatialData object.

    Parameters
    ----------
    folderpath
        A folder path where the dataset will be downloaded and extracted.
        If `None`, uses :attr:`scanpy.settings.datasetdir`.

    Returns
    -------
    :class:`spatialdata.SpatialData`
        The downloaded and extracted Visium H&E dataset.
    """
    downloader = get_downloader()
    return downloader.download("visium_hne_sdata", folderpath)


# =============================================================================
# Dataset loader factory
# =============================================================================


@dataclass(frozen=True)
class _DocParts:
    """Documentation parts for dataset loader functions."""

    shape_prefix: str
    path_desc: str
    kwargs_desc: str
    return_type: str


_ANNDATA_DOC = _DocParts(
    shape_prefix="The shape of this :class:`anndata.AnnData` object",
    path_desc="Path where to save the dataset.",
    kwargs_desc="Keyword arguments for ``anndata.read_h5ad``.",
    return_type=":class:`anndata.AnnData`\n        The dataset.",
)

_IMAGE_DOC = _DocParts(
    shape_prefix="The shape of this image is",
    path_desc="Path where to save the .tiff image.",
    kwargs_desc="Keyword arguments for :meth:`squidpy.im.ImageContainer.add_img`.",
    return_type=":class:`squidpy.im.ImageContainer`\n        The image data.",
)

_DOC_PARTS_BY_TYPE: dict[DatasetType, _DocParts] = {
    DatasetType.ANNDATA: _ANNDATA_DOC,
    DatasetType.IMAGE: _IMAGE_DOC,
}


def _make_loader(dataset_name: str):
    """Factory function to create dataset loader functions.

    Automatically derives documentation from the registry based on dataset type.
    """
    entry = get_registry().get(dataset_name)

    if entry is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    doc_parts = _DOC_PARTS_BY_TYPE.get(entry.type)
    if doc_parts is None:
        raise ValueError(f"Unsupported type for loader factory: {entry.type}")

    def loader(path: PathLike | None = None, **kwargs: Any):
        return get_downloader().download(dataset_name, path, **kwargs)

    loader.__doc__ = f"""
    {entry.doc_header}

    {doc_parts.shape_prefix} ``{entry.shape}``.

    Parameters
    ----------
    path
        {doc_parts.path_desc}
    kwargs
        {doc_parts.kwargs_desc}

    Returns
    -------
    {doc_parts.return_type}
    """
    loader.__name__ = dataset_name
    return loader


# AnnData datasets
four_i = _make_loader("four_i")
imc = _make_loader("imc")
seqfish = _make_loader("seqfish")
visium_hne_adata = _make_loader("visium_hne_adata")
visium_fluo_adata = _make_loader("visium_fluo_adata")
visium_hne_adata_crop = _make_loader("visium_hne_adata_crop")
visium_fluo_adata_crop = _make_loader("visium_fluo_adata_crop")
sc_mouse_cortex = _make_loader("sc_mouse_cortex")
mibitof = _make_loader("mibitof")
merfish = _make_loader("merfish")
slideseqv2 = _make_loader("slideseqv2")
# Image datasets
visium_fluo_image_crop = _make_loader("visium_fluo_image_crop")
visium_hne_image_crop = _make_loader("visium_hne_image_crop")
visium_hne_image = _make_loader("visium_hne_image")

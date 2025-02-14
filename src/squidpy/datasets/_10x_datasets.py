from __future__ import annotations

import os
import shutil
import tarfile
from pathlib import Path
from typing import (
    Literal,
    NamedTuple,
    Union,  # noqa: F401
)

import spatialdata as sd
from anndata import AnnData
from scanpy import _utils
from scanpy import logging as logg
from scanpy._settings import settings
from scanpy._utils import check_presence_download
from spatialdata import SpatialData

from squidpy._constants._constants import TenxVersions
from squidpy.datasets._utils import PathLike

__all__ = ["visium"]


class VisiumFiles(NamedTuple):
    feature_matrix: str
    spatial_attrs: str
    tif_image: str


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
        Directory where to download the data. If `None`, use :attr:`scanpy._settings.ScanpyConfig.datasetdir`.

    Returns
    -------
    Spatial :class:`anndata.AnnData`.
    """
    from squidpy.read._read import visium as read_visium

    if sample_id.startswith("V1_"):
        spaceranger_version = TenxVersions.V1
    elif sample_id.startswith("Targeted_") or sample_id.startswith("Parent_"):
        spaceranger_version = TenxVersions.V2
    else:
        spaceranger_version = TenxVersions.V3

    if base_dir is None:
        base_dir = settings.datasetdir
    base_dir = Path(base_dir)
    sample_dir = base_dir / sample_id
    sample_dir.mkdir(exist_ok=True, parents=True)

    url_prefix = f"https://cf.10xgenomics.com/samples/spatial-exp/{spaceranger_version}/{sample_id}/"
    visium_files = VisiumFiles(
        f"{sample_id}_filtered_feature_bc_matrix.h5",
        f"{sample_id}_spatial.tar.gz",
        f"{sample_id}_image.tif",
    )

    # download spatial data
    tar_pth = sample_dir / visium_files.spatial_attrs
    _utils.check_presence_download(filename=tar_pth, backup_url=url_prefix + visium_files.spatial_attrs)
    with tarfile.open(tar_pth) as f:
        for el in f:
            if not (sample_dir / el.name).exists():
                f.extract(el, sample_dir)

    # download counts
    _utils.check_presence_download(
        filename=sample_dir / "filtered_feature_bc_matrix.h5",
        backup_url=url_prefix + visium_files.feature_matrix,
    )

    if include_hires_tiff:  # download image
        _utils.check_presence_download(
            filename=sample_dir / "image.tif",
            backup_url=url_prefix + visium_files.tif_image,
        )
        return read_visium(
            base_dir / sample_id,
            source_image_path=base_dir / sample_id / "image.tif",
        )

    return read_visium(base_dir / sample_id)


def visium_hne_sdata(filename: Path | str | None = None) -> SpatialData:
    """
    Downloads a Visium H&E dataset and provides it as a `SpatialData` object.

    This function combines the outputs from `squidpy.datasets.visium_hne_adata()`
    and `squidpy.datasets.visium_hne_image()` into a `SpatialData` object containing:
      - A multi-scale representation of the H&E image.
      - The spots as a shapes layer.
      - Gene expression data as an AnnData object.

    If no filename is provided, it defaults to `~/.cache/squidpy/visium_hne_sdata.zip`.

    Parameters
    ----------
    filename : Path | str | None, optional
        Path to save the dataset. If a directory is provided, the default filename
        `visium_hne_sdata.zip` will be used inside that directory. If a `.zarr` filename
        is provided, it will be converted to `.zip` for downloading.

    Returns
    -------
    SpatialData
        The downloaded and extracted Visium H&E dataset as a `SpatialData` object.
    """

    FIGSHARE_ID = "52370645"

    if filename is None:
        filename = Path.home() / ".cache/squidpy/visium_hne_sdata.zip"
    else:
        if not isinstance(filename, Path | str):
            raise TypeError(f"Expected `filename` to be of type `Path` or `str`, found `{type(filename).__name__}`.")

        filename = Path(filename).expanduser()

        if filename.is_dir():
            filename = filename / "visium_hne_sdata.zip"
        elif filename.suffix not in {".zip", ".zarr"}:
            raise ValueError(f"Expected `filename` to have suffix `.zip` or `.zarr`, found `{filename.suffix}`.")
        elif filename.suffix == ".zarr":
            filename = filename.with_suffix(".zip")  # Ensure zip download

    filename = filename.absolute()
    extracted_path = filename.with_suffix(".zarr")

    if not extracted_path.exists():
        try:
            logg.info(f"Downloading Visium H&E SpatialData to {filename}")
            check_presence_download(
                filename=filename,
                backup_url=f"https://ndownloader.figshare.com/files/{FIGSHARE_ID}",
            )
            logg.info(f"Extracting dataset from {filename} to {extracted_path}")
            shutil.unpack_archive(str(filename), str(extracted_path))
        except Exception as e:
            logg.error(f"Failed to download or extract dataset: {e}")
            raise

    return sd.read_zarr(extracted_path)

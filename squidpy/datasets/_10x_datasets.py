from typing import Literal, Optional
from pathlib import Path

from scanpy import _utils
from anndata import AnnData
from scanpy._settings import settings
from scanpy.readwrite import read_visium


def _download_visium_dataset(
    sample_id: str, spaceranger_version: str, base_dir: Optional[Path] = None, download_image: bool = False
) -> None:
    """Download Visium dataset.

    Params
    ------
    sample_id
        String name of example visium dataset.
    base_dir
        Where to download the dataset to.
    download_image
        Whether to download the high-resolution tissue section.
    """
    import tarfile

    if base_dir is None:
        base_dir = settings.datasetdir

    url_prefix = f"https://cf.10xgenomics.com/samples/spatial-exp/{spaceranger_version}/{sample_id}/"

    sample_dir = base_dir / sample_id
    sample_dir.mkdir(exist_ok=True)

    # Download spatial data
    tar_filename = f"{sample_id}_spatial.tar.gz"
    tar_pth = sample_dir / tar_filename
    _utils.check_presence_download(filename=tar_pth, backup_url=url_prefix + tar_filename)
    with tarfile.open(tar_pth) as f:
        for el in f:
            if not (sample_dir / el.name).exists():
                f.extract(el, sample_dir)

    # Download counts
    _utils.check_presence_download(
        filename=sample_dir / "filtered_feature_bc_matrix.h5",
        backup_url=url_prefix + f"{sample_id}_filtered_feature_bc_matrix.h5",
    )

    # Download image
    if download_image:
        _utils.check_presence_download(
            filename=sample_dir / "image.tif",
            backup_url=url_prefix + f"{sample_id}_image.tif",
        )


def visium_sge(
    sample_id: Literal[
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
        # spaceranger version 1.2.0
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
    ] = "V1_Breast_Cancer_Block_A_Section_1",
    *,
    include_hires_tiff: bool = False,
) -> AnnData:
    """Process Visium Spatial Gene Expression data from 10x Genomics.

    Database: https://support.10xgenomics.com/spatial-gene-expression/datasets
    Parameters
    ----------
    sample_id
        The ID of the data sample in 10x’s spatial database.
    include_hires_tiff
        Download and include the high-resolution tissue image (tiff)
        in `adata.uns["spatial"][sample_id]["metadata"]["source_image_path"]`.
    Returns
    -------
    Annotated data matrix.
    """
    if "V1_" in sample_id:
        spaceranger_version = "1.1.0"
    else:
        spaceranger_version = "1.2.0"
    _download_visium_dataset(sample_id, spaceranger_version, download_image=include_hires_tiff)
    if include_hires_tiff:
        adata = read_visium(
            settings.datasetdir / sample_id,
            source_image_path=settings.datasetdir / sample_id / "image.tif",
        )
    else:
        adata = read_visium(settings.datasetdir / sample_id)
    return adata
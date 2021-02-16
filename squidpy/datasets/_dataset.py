from copy import copy

from squidpy.datasets._utils import AMetadata

_4i = AMetadata(
    name="four_i",
    doc_header="Pre-processed subset 4i dataset from `Gut et al <https://doi.org/10.1126/science.aar7042>`_.",
    shape=(270876, 43),
    url="https://ndownloader.figshare.com/files/26254294",
)
_imc = AMetadata(
    name="imc",
    doc_header="Pre-processed subset IMC dataset from `Jackson et al "
    "<https://www.nature.com/articles/s41586-019-1876-x>`_.",
    shape=(4668, 34),
    url="https://ndownloader.figshare.com/files/26098406",
)
_seqfish = AMetadata(
    name="seqfish",
    doc_header="Pre-processed subset seqFISH dataset from `Lohoff et al "
    "<https://www.biorxiv.org/content/10.1101/2020.11.20.391896v1>`_.",
    shape=(19416, 351),
    url="https://ndownloader.figshare.com/files/26098403",
)
_vha = AMetadata(
    name="visium_hne_adata",
    doc_header="Pre-processed `10x Genomics Visium H&E dataset "
    "<https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Adult_Mouse_Brain>`_.",
    shape=(2688, 18078),
    url="https://ndownloader.figshare.com/files/26098397",
)
_vfa = AMetadata(
    name="visium_fluo_adata",
    doc_header="Pre-processed `10x Genomics Visium Fluorecent dataset "
    "<https://support.10xgenomics.com/spatial-gene-expression/datasets/"
    "1.1.0/V1_Adult_Mouse_Brain_Coronal_Section_2>`_.",
    shape=(2800, 16562),
    url="https://ndownloader.figshare.com/files/26098391",
)
_vhac = AMetadata(
    name="visium_hne_adata_crop",
    doc_header="Pre-processed subset `10x Genomics Visium H&E dataset "
    "<https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Adult_Mouse_Brain>`_.",
    shape=(684, 18078),
    url="https://ndownloader.figshare.com/files/26098382",
)
_vfac = AMetadata(
    name="visium_fluo_adata_crop",
    doc_header="Pre-processed subset `10x Genomics Visium Fluorescent dataset "
    "<https://support.10xgenomics.com/spatial-gene-expression/datasets/"
    "1.1.0/V1_Adult_Mouse_Brain_Coronal_Section_2>`_.",
    shape=(704, 16562),
    url="https://ndownloader.figshare.com/files/26098376",
)
_smc = AMetadata(
    name="sc_mouse_cortex",
    doc_header="Pre-processed `scRNA-seq mouse cortex " "<https://doi.org/10.1038/s41586-018-0654-5>`_.",
    shape=(21697, 36826),
    url="https://ndownloader.figshare.com/files/26404781",
)

for name, var in copy(locals()).items():
    if isinstance(var, AMetadata):
        var._create_function(name, globals())


__all__ = [  # noqa: F822
    "four_i",
    "imc",
    "seqfish",
    "visium_hne_adata",
    "visium_hne_adata_crop",
    "visium_fluo_adata",
    "visium_fluo_adata_crop",
    "sc_mouse_cortex",
]

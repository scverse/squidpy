from __future__ import annotations

from copy import copy
from typing import Any, Protocol, Union

from squidpy.datasets._utils import ImgMetadata, PathLike
from squidpy.im._container import ImageContainer


class ImageDataset(Protocol):
    def __call__(self, path: PathLike | None = ..., **kwargs: Any) -> ImageContainer: ...


_vfic = ImgMetadata(
    name="visium_fluo_image_crop",
    doc_header="Cropped Fluorescent image from `10x Genomics Visium dataset "
    "<https://support.10xgenomics.com/spatial-gene-expression/datasets"
    "/1.1.0/V1_Adult_Mouse_Brain_Coronal_Section_2>`__.",
    shape=(7272, 7272),
    library_id="V1_Adult_Mouse_Brain_Coronal_Section_2",
    url="https://ndownloader.figshare.com/files/26098364",
)
_vhic = ImgMetadata(
    name="visium_hne_image_crop",
    doc_header="Cropped H&E image from `10x Genomics Visium dataset "
    "<https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Adult_Mouse_Brain>`__.",
    shape=(3527, 3527),
    library_id="V1_Adult_Mouse_Brain",
    url="https://ndownloader.figshare.com/files/26098328",
)
_vhn = ImgMetadata(
    name="visium_hne_image",
    doc_header="H&E image from `10x Genomics Visium dataset "
    "<https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Adult_Mouse_Brain>`__.",
    shape=(11757, 11291),
    library_id="V1_Adult_Mouse_Brain",
    url="https://ndownloader.figshare.com/files/26098124",
)


for name, var in copy(locals()).items():
    if isinstance(var, ImgMetadata):
        var._create_function(name, glob_ns=globals())


__all__ = [  # noqa: F822
    "visium_fluo_image_crop",
    "visium_hne_image_crop",
    "visium_hne_image",
]

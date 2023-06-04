from typing import Any, Protocol, Union

from squidpy.datasets._utils import PathLike
from squidpy.im._container import ImageContainer

class ImageDataset(Protocol):
    def __call__(self, path: PathLike | None = ..., **kwargs: Any) -> ImageContainer: ...

visium_fluo_image_crop: ImageDataset
visium_hne_image_crop: ImageDataset
visium_hne_image: ImageDataset

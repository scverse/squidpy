from typing import Any, Union, Protocol

from squidpy.im._container import ImageContainer
from squidpy.datasets._utils import PathLike

class ImageDataset(Protocol):
    def __call__(self, path: Union[PathLike, None] = ..., **kwargs: Any) -> ImageContainer: ...

visium_fluo_image_crop: ImageDataset
visium_hne_image_crop: ImageDataset
visium_hne_image: ImageDataset

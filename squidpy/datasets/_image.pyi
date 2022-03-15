from typing import Any, Callable

from squidpy.im._container import ImageContainer
from squidpy.datasets._utils import PathLike

visium_fluo_image_crop: Callable[[PathLike, Any], ImageContainer]
visium_hne_image_crop: Callable[[PathLike, Any], ImageContainer]
visium_hne_image: Callable[[PathLike, Any], ImageContainer]

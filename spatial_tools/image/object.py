import dask
import dask.array as da
import imageio
import numpy as np
from typing import Union
import xarray as xr


class ImageContainer:
    data: xr.Dataset

    def __init__(self, img: Union[str, np.ndarray], lazy: bool = True, dtype="float32"):
        """
        Processes image as in memory numpy array or sets up lazy loading from disk via wrapping dask array
        if image is a file path.

        An instance of this class is given to all image processing functions, along with and anndata instance
        if necessary.

        :param img:
        """
        if isinstance(img, np.ndarray):
            pass
        elif isinstance(img, str):
            if lazy:
                shape = None # TODO how can we get access to image shape without loading? in worst case, load once
                # and immidiately delete again.
                lazy_img = dask.delayed(imageio.imread)(img)
                lazy_img = da.from_delayed(lazy_img, shape=shape, dtype=dtype)
                img = lazy_img
            else:
                img = imageio.imread
        else:
            raise ValueError(img)

        self.data = xr.Dataset(
            {
                "image": (["x", "y", "channels"], img),
            },
            coords={
                "xpos_pixel": (["x"], np.arange(0, img.shape[0])),
                "ypos_pixel": (["y"], np.arange(0, img.shape[1])),
                "xpos": (["x"], np.arange(0, img.shape[0])),  # TODO maybe add actual scaling of image here?
                "ypos": (["y"], np.arange(0, img.shape[1])),
            },
        )

    def add_layer(self, x: np.ndarray, image_id: str, channel_id: str = "1"):
        """
        Add layer assuming first two coords are the image coords.
        :param x:
        :param id:
        :return:
        """
        self.data[image_id] = (["x", "y", "channel_id"], x)

    def crop(self):
        """
        Yield image crop.

        TODO copy over cropping code.
        :return:
        """

import numpy as np
from typing import Union, List
import xarray as xr
from ._utils import _num_pages 
from .crop import crop_img

class ImageContainer:
    """
    Container for in memory / tif images. Allows for lazy and chunked reading via rasterio and dask
    An instance of this class is given to all image processing functions, along with an anndata instance
    if necessary.
    """
    data: xr.Dataset
        
    def __init__(self, img: Union[str, np.ndarray], img_id: Union[str, List[str]] = None, lazy: bool = True, chunks: int = None, ):
        """
        Processes image as in memory numpy array or uses xarrays rasterio reading functions to load from disk 
        (with caching) if image is a file path.
        If chunks are specified, the xarray is wrapped in a dask lazy dask array using the chunk size.

        An instance of this class is given to all image processing functions, along with an anndata instance
        if necessary.

        :param img:
        :param lazy: use rasterio/dask to lazily load image
        :param chunks: chunk size for dask
        :param img_id: name for img. For multi-page tiffs should be a list. 
            If not specified, DataArrays will be named "image_{i}"
        """
        if chunks is not None:
            chunks = {'x': chunks, 'y': chunks}
        self.chunks = chunks
        self.lazy = lazy
        self.data = xr.Dataset()
        if img is not None:
            self.add_img(img, img_id)
        
    @classmethod
    def open(cls, fname: str, lazy: bool = True, chunks: int = None):
        """
        initialize using a previously saved netcdf file
        """
        self = cls(img=None, lazy=lazy, chunks=chunks)
        self.data = xr.open_dataset(fname, chunks=self.chunks)
        if not self.lazy:
            self.data.load()
        return self
    
    def save(self, fname: str):
        """saves dataset as netcdf file"""
        self.data.to_netcdf(fname, mode='a')
        
        
    def add_img(self, img: Union[str, np.ndarray], img_id: Union[str, List[str]] = None):
        """
        Add layer(s) from numpy image / tiff file. 
        For numpy arrays, assume that dims are: channels, y, x
        
        :param img:
        :param img_id:
        :return: None
        """
        imgs = self._load_img(img)
        if img_id is None:
            img_id = 'image'
        if isinstance(img_id, str):
            if len(imgs) > 1:
                img_ids = [f"{img_id}_{i}" for i in range(len(imgs))]
            else:
                img_ids = [img_id]
        elif isinstance(img_id, list):
            img_ids = img_id
        else:
            raise ValueError(img_id)
        assert len(img_ids) == len(imgs), f"Have {len(imgs)} images, but {len(img_ids)} image ids"
        # add to data
        for img, img_id in zip(imgs, img_ids):
            self.data[img_id] = img
        if not self.lazy:
            # load in memory
            self.data.load()
    
    def _load_img(self, img: Union[str, np.ndarray]):
        """
        Load img as xarray. Supports numpy arrays and (multi-page) tiff files.
        For numpy arrays, assume that dims are: channels, y, x
        
        :returns: list of DataArrays
        """
        imgs = []
        if isinstance(img, np.ndarray):
            if len(img.shape) == 2:
                # add empty channel dimension
                img = img[np.newaxis, :, :]
            xr_img = xr.DataArray(img, dims=['channels', 'y', 'x'])
            imgs.append(xr_img)
        elif isinstance(img, str):
            # get the number of pages in the file
            num_pages = _num_pages(img)
            # read all pages using rasterio
            for i in range(1, num_pages+1):
                data = xr.open_rasterio(f"GTIFF_DIR:{i}:{img}", chunks=self.chunks, parse_coordinates=False)
                data = data.rename({'band': 'channels'})
                imgs.append(data)
        else:
            raise ValueError(img)
        return imgs

    def crop(self, x, y, s=100, img_id: Union[str, List[str]] = None, **kwargs):
        """
        extract a centered at `x` and `y`. 
        Attrs:
            img_ids (list): list of images that should be used to obtain the crop
            x (int): x coord of crop in `img`
            y (int): y coord of crop in `img`
            s (int): width and heigh of the crop in pixels
            scale (float): resolution of the crop (smaller -> smaller image)
            mask_circle (bool): mask crop to a circle
            cval (float): the value outside image boundaries or the mask
            dtype (str): optional, type to which the output should be (safely cast)
            
        Returns:
            np.ndarray with dimentions: y, x, channels (concatenated over all images)
        """
        if img_id is None:
            img_ids = list(self.data.keys())
        elif isinstance(img_id, str):
            img_ids = [img_id]
        else:
            img_ids = img_id
        
        crops = []
        for img_id in img_ids:
            img = self.data[img_id]
            crops.append(crop_img(img, x, y, s, **kwargs))
        return np.concatenate(crops, axis=-1)
    

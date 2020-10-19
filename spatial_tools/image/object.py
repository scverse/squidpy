import numpy as np
from typing import Union, List, Tuple, Optional
import xarray as xr
from ._utils import _num_pages


class ImageContainer:
    """\
    Container for in memory or on-disk tiff images. 
    
    Allows for lazy and chunked reading via rasterio and dask.
    An instance of this class is given to all image processing functions, along with an anndata instance
    if necessary.
    
    Attributes
    ----------
    data
        Xarray dataset containing the image data
    
    Methods
    -------
    add_img(img, img_id)
        Add layers from numpy / tiff file to `data` with key `img_id`.
        
    crop(x, y)
        Crop image centered around coordinates (x,y) from `data`.
    """

    data: xr.Dataset

    def __init__(
        self,
        img: Union[str, np.ndarray],
        img_id: Optional[Union[str, List[str]]] = None,
        lazy: bool = True,
        chunks: Optional[int] = None,
    ):
        """\
        Set up ImageContainer from numpy array or on-disk tiff.
        
        Processes image as in memory numpy array or uses xarrays rasterio reading functions to load from disk 
        (with caching) if image is a file path.
        If chunks are specified, the xarray is wrapped in a dask lazy dask array using the chunk size.

        Params
        ------
        img
            Numpy array or path to tiff file.
        img_id
            Key (name) to be used for img. For multi-page tiffs this should be a list.
            If not specified, DataArrays will be named "image_{i}".
        lazy
            Use rasterio/dask to lazily load image.
        chunks
            Chunk size for dask.
        """
        if chunks is not None:
            chunks = {"x": chunks, "y": chunks}
        self._chunks = chunks
        self._lazy = lazy
        self.data = xr.Dataset()
        if img is not None:
            self.add_img(img, img_id)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.data.dims["x"], self.data.dims["y"])

    @property
    def nchannels(self) -> int:
        return self.data.dims["channels"]

    @classmethod
    def open(cls, fname: str, lazy: bool = True, chunks: Optional[int] = None):
        """\
        Initialize using a previously saved netcdf file.
        
        Params
        ------
        fname
            Path to the saved .nc file.
        lazy
            Use dask to lazily load image.
        chunks
            Chunk size for dask.
        """
        self = cls(img=None, lazy=lazy, chunks=chunks)
        self.data = xr.open_dataset(fname, chunks=self._chunks)
        if not self._lazy:
            self.data.load()
        return self

    def save(self, fname: str):
        """Save dataset as netcdf file.

        Params
        ------
        fname
            Path to the saved .nc file.
        """
        self.data.to_netcdf(fname, mode="a")

    def add_img(
        self, img: Union[str, np.ndarray], img_id: Union[str, List[str]] = None, channel_id: str = "channels"
    ):
        """
        Add layer(s) from numpy image / tiff file.
        For numpy arrays, assume that dims are: channels, y, x

        The added image has to have the same number of channels as the original image, or no channels.

        Params
        ------
        img
            Numpy array or path to tiff file.
        img_id
            Key (name) to be used for img. For multi-page tiffs this should be a list.
            If not specified, DataArrays will be named "image_{i}".

        Returns
        -------
        None

        Raises
        ------
        ValueError
            if img_id is neither a string nor a list
        """
        imgs = self._load_img(img=img, channel_id=channel_id)
        if img_id is None:
            img_id = "image"
        if isinstance(img_id, str):
            if len(imgs) > 1 and isinstance(imgs, list):
                img_ids = [f"{img_id}_{i}" for i in range(len(imgs))]
            else:
                img_ids = [img_id]
        elif isinstance(img_id, list):
            img_ids = img_id
        else:
            raise ValueError(img_id)
        assert len(img_ids) == len(
            imgs
        ), f"Have {len(imgs)} images, but {len(img_ids)} image ids"
        # add to data
        print("adding %s into object" % img_id)
        for img, img_id in zip(imgs, img_ids):
            self.data[img_id] = img
        if not self._lazy:
            # load in memory
            self.data.load()

    def _load_img(self, img: Union[str, np.ndarray], channel_id: str = "channels") -> List[xr.DataArray]:
        """\
        Load img as xarray. 
        
        Supports numpy arrays and (multi-page) tiff files.
        For numpy arrays, assume that dims are: `'channels, y, x'`
        
        Params
        ------
        img
            Numpy array or path to tiff file.
            
        Returns
        -------
        List of DataArrays containing loaded images.
        
        Raises
        ------
        ValueError:
            if img is a np.ndarray and has more than 3 dimensions
        """
        imgs = []
        if isinstance(img, np.ndarray):
            if len(img.shape) > 3:
                raise ValueError(
                    f"img has more than 3 dimensions. img.shape is {img.shape}"
                )
            dims = [channel_id, "y", "x"]
            if len(img.shape) == 2:
                dims = ["y", "x"]
            xr_img = xr.DataArray(img, dims=dims)
            imgs.append(xr_img)
        elif isinstance(img, xr.DataArray):
            assert "x" in img.dims
            assert "y" in img.dims
            imgs.append(img)
        elif isinstance(img, str):
            # get the number of pages in the file
            num_pages = _num_pages(img)
            # read all pages using rasterio
            for i in range(1, num_pages + 1):
                data = xr.open_rasterio(
                    f"GTIFF_DIR:{i}:{img}", chunks=self._chunks, parse_coordinates=False
                )
                data = data.rename({"band": channel_id})
                imgs.append(data)
        else:
            raise ValueError(img)
        return imgs

    def crop(
        self,
        x: int,
        y: int,
        xs: int = 100,
        ys: int = 100,
        img_id: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> xr.DataArray:
        """\
        Extract a crop centered at `x` and `y`. 
        
        Params
        ------
        x: int
            X coord of crop (in pixel space).
        y: int
            Y coord of crop (in pixel space).
        xs: int
            Width of the crop in pixels.
        ys: int
            Geigh of the crop in pixels.
        scale: float
            Default is 1.0.
            Resolution of the crop (smaller -> smaller image).
        mask_circle: bool
            Default is False.
            Mask crop to a circle.
        cval: float
            Default is 0
            The value outside image boundaries or the mask.
        dtype: str
            Optional, type to which the output should be (safely) cast. 
            Currently supported dtypes: 'uint8'.
            
        Returns
        -------
        np.ndarray with dimentions: y, x, channels (concatenated over all images)
        """
        from .crop import crop_img

        if img_id is None:
            assert False

        img = self.data.data_vars[img_id]
        return crop_img(img=img, x=x, y=y, xs=xs, ys=ys, **kwargs)

    def crop_equally(
        self,
        xs: Union[int, None] = None,
        ys: Union[int, None] = None,
        img_id: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Tuple[List[xr.DataArray], np.ndarray, np.ndarray]:
        """\
        Decompose image into equally sized crops

        Params
        ------
        xs: int
            Width of the crops in pixels. Defaults to image size if None.
        ys: int
            Height of the crops in pixels. Defaults to image size if None.
            # TODO add support as soon as crop supports this
        cval: float
            Default is 0
            The value outside image boundaries or the mask.
        dtype: str
            Optional, type to which the output should be (safely) cast.
            Currently supported dtypes: 'uint8'.

        Returns
        -------
        Tuple:
            List[xr.DataArray with dimentions: y, x, channels (concatenated over all images)]: crops
            np.ndarray: length number of crops: x positions of crops
            np.ndarray: length number of crops: y positions of crops
        """
        if xs is None:
            xs = self.shape[0]
        if ys is None:
            ys = self.shape[1]
        unique_xcoord = np.arange(
            start=0, stop=(self.shape[0] // xs) * xs, step=xs
        )
        unique_ycoord = np.arange(
            start=0, stop=(self.shape[0] // ys) * ys, step=ys
        )
        xcoords = np.repeat(unique_xcoord, len(unique_ycoord))
        ycoords = np.tile(unique_xcoord, len(unique_ycoord))
        crops = [
            self.crop(x=x, y=y, xs=xs, ys=ys, img_id=img_id)
            for x, y in zip(xcoords, ycoords)
        ]
        return crops, xcoords, ycoords

import numpy as np
from typing import Union, List, Tuple, Optional
import xarray as xr
from ._utils import _num_pages
from imageio import imread


class ImageContainer:
    """\
    Container for in memory or on-disk tiff or jpg images.
    
    Allows for lazy and chunked reading via rasterio and dask (if input is a tiff image).
    An instance of this class is given to all image processing functions, along with an anndata instance
    if necessary.
    
    Attributes
    ----------
    data
        Xarray dataset containing the image data
    
    Methods
    -------
    add_img(img, img_id)
        Add layers from numpy / image file to `data` with key `img_id`.
        
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
        Set up ImageContainer from numpy array or on-disk tiff / jpg.
        
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
        self,
        img: Union[str, np.ndarray, xr.DataArray],
        img_id: Union[str, List[str]] = None,
        channel_id: str = "channels",
    ):
        """\
        Add layer from numpy image / tiff file.
        For numpy arrays, assume that dims are: channels, y, x

        The added image has to have the same number of channels as the original image, or no channels.

        Params
        ------
        img
            Numpy array or path to image file.
        img_id
            Key (name) to be used for img. For multi-page tiffs this should be a list.
            If not specified, DataArrays will be named "image".

        Returns
        -------
        None

        Raises
        ------
        ValueError
            if img_id is neither a string nor a list
        """
        img = self._load_img(img=img, channel_id=channel_id)
        if img_id is None:
            img_id = "image"
        # add to data
        print("adding %s into object" % img_id)
        self.data[img_id] = img
        if not self._lazy:
            # load in memory
            self.data.load()

    def _load_img(
        self, img: Union[str, np.ndarray], channel_id: str = "channels"
    ) -> xr.DataArray:
        """\
        Load img as xarray. 
        
        Supports numpy arrays and (multi-page) tiff files, and jpg files
        For numpy arrays, assume that dims are: `'channels, y, x'`
        
        NOTE: lazy loading via dask is currently not supported for on-disk jpg files.
        They will be loaded in memory.

        Params
        ------
        img
            Numpy array or path to image file.
            
        Returns
        -------
        DataArray containing loaded image.
        
        Raises
        ------
        ValueError:
            if img is a np.ndarray and has more than 3 dimensions
        """
        if isinstance(img, np.ndarray):
            if len(img.shape) > 3:
                raise ValueError(
                    f"img has more than 3 dimensions. img.shape is {img.shape}"
                )
            dims = [channel_id, "y", "x"]
            if len(img.shape) == 2:
                dims = ["y", "x"]
            xr_img = xr.DataArray(img, dims=dims)
        elif isinstance(img, xr.DataArray):
            assert "x" in img.dims
            assert "y" in img.dims
        elif isinstance(img, str):
            ext = img.split(".")[-1]
            if ext in ("tif", "tiff"):
                # get the number of pages in the file
                num_pages = _num_pages(img)
                # read all pages using rasterio
                xr_img_byband = []
                for i in range(1, num_pages + 1):
                    data = xr.open_rasterio(
                        f"GTIFF_DIR:{i}:{img}", chunks=self._chunks, parse_coordinates=False
                    )
                    data = data.rename({"band": channel_id})
                    xr_img_byband.append(data)
                xr_img = xr.concat(xr_img_byband, dim=channel_id)
            elif ext in ("jpg", "jpeg"):
                img = imread(img)
                # jpeg has channels as last dim - transpose
                img = img.transpose(2, 0, 1)
                dims = [channel_id, "y", "x"]
                xr_img = xr.DataArray(img, dims=dims)
            else:
                raise NotImplementedError(f"Files with extension {ext}")
        else:
            raise ValueError(img)
        return xr_img

    def crop(
        self,
        x: float,
        y: float,
        xs: int = 100,
        ys: int = 100,
        img_id: Optional[str] = None,
        centred: bool = True,
        **kwargs,
    ) -> xr.DataArray:
        """\
        Extract a crop based on coordiantes `x` and `y` of `img_id`.

        Centred on x, y if centred is True, else right and down from x, y.
        
        Params
        ------
        x: float
            X coord of crop (in pixel space). Can be float (ie. int+0.5) if model is centered and if x+xs/2 is integer.
        y: float
            Y coord of crop (in pixel space). Can be float (ie. int+0.5) if model is centered and if y+ys/2 is integer.
        xs: int
            Width of the crop in pixels.
        ys: int
            Height of the crop in pixels.
        img_id: str
            id of the image layer to be cropped.
        scale: float
            Default is 1.0.
            Resolution of the crop (smaller -> smaller image).
            TODO: when scaling, will return a numpy array instead of an xarray
        mask_circle: bool
            Default is False.
            Mask crop to a circle.
        cval: float
            Default is 0
            The value outside image boundaries or the mask.
        centred: bool
            Whether the crop coordinates are centred.
        dtype: str
            Optional, type to which the output should be (safely) cast. 
            Currently supported dtypes: 'uint8'.
            TODO: currenty, using this argument will return a numpy array instead of an xarray

        TODO: enable cropping of several channels at once?

        Returns
        -------
        xr.DataArray with dimentions: channels, y, x
        """
        from .crop import crop_img

        if img_id is None:
            img_id = list(self.data.keys())[0]

        img = self.data.data_vars[img_id]
        if centred:
            assert (
                xs / 2.0 + x
            ) % 1 == 0, "x and xs/2 have to add up to an integer to use centred model"
            assert (
                ys / 2.0 + y
            ) % 1 == 0, "y and ys/2 have to add up to an integer to use centred model"
            x = int(x - xs // 2)  # move from centre to corner
            y = int(y - ys // 2)  # move from centre to corner
        else:
            assert x % 1.0 == 0, "x needs to be integer"
            assert y % 1.0 == 0, "x needs to be integer"
            x = int(x)
            y = int(y)
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
            List[xr.DataArray with dimentions: channels, y, x: crops
            np.ndarray: length number of crops: x positions of crops
            np.ndarray: length number of crops: y positions of crops
        """
        if xs is None:
            xs = self.shape[0]
        if ys is None:
            ys = self.shape[1]
        unique_xcoord = np.arange(start=0, stop=(self.shape[0] // xs) * xs, step=xs)
        unique_ycoord = np.arange(start=0, stop=(self.shape[1] // ys) * ys, step=ys)
        xcoords = np.repeat(unique_xcoord, len(unique_ycoord))
        ycoords = np.tile(unique_xcoord, len(unique_ycoord))
        crops = [
            self.crop(x=x, y=y, xs=xs, ys=ys, img_id=img_id, centred=False)
            for x, y in zip(xcoords, ycoords)
        ]
        return crops, xcoords, ycoords

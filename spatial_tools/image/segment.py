import abc
import anndata
import numpy as np
import skimage
from typing import List, Union

from .crop import uncrop_img
from .object import ImageContainer

"""
Functions exposed: segment(), evaluate_nuclei_segmentation() 
"""


def evaluate_nuclei_segmentation(
        adata,
        copy: bool = False,
        **kwargs
) -> Union[anndata.AnnData, None]:
    """
    Perform basic nuclei segmentation evaluation.

    Metrics on H&E signal in segments vs outside.

    Attrs:
        adata:
        copy:
        kwargs:
    """
    pass


class SegmentationModel:
    """
    Base class for segmentation models.

    Contains core shared functions related contained to cell and nuclei segmentation.
    Specific seegmentation models can be implemented by inheriting from this class.
    This class is not instantiated by user but used in the background by the functional API.
    """

    def __init__(
            self,
            model,
    ):
        self.model = model

    def segment(self, arr: np.ndarray, **kwargs) -> np.ndarray:
        """

        Params
        ------
        arr: np.ndarray
            High-resolution image.

        Yields
        -----
        (x, y, 1)
        Segmentation mask for high-resolution image.
        """
        return self._segment(arr, **kwargs)

    @abc.abstractmethod
    def _segment(self, arr, **kwargs) -> np.ndarray:
        pass


class SegmentationModelBlob(SegmentationModel):

    def _segment(self, arr, invert: bool = True, **kwargs) -> np.ndarray:
        """

        Params
        ------
        arr: np.ndarray
            High-resolution image.
        kwargs: dicct
            Model arguments

        Yields
        -----
        (x, y, 1)
        Segmentation mask for high-resolution image.
        """
        if invert:
            arr = 0.-arr

        if self.model == "log":
            y = skimage.feature.blob_log(
                image=arr,
                **kwargs
            )
        elif self.model == "dog":
            y = skimage.feature.blob_dog(
                image=arr,
                **kwargs
            )
        elif self.model == "doh":
            y = skimage.feature.blob_doh(
                image=arr,
                **kwargs
            )
        else:
            raise ValueError("did not recognize self.model %s" % self.model)
        return y


class SegmentationModelWatershed(SegmentationModel):

    def _segment(self, arr, thresh=0.5, geq: bool = True, **kwargs) -> np.ndarray:
        """

        Params
        ------
        arr: np.ndarray
            High-resolution image.
        thresh: float
             Threshold for discretisation of image scale to define areas to segment.
        geq:
            Treat thres as uppper or lower (greater-equal = geq) bound for defining state to segement.
        kwargs: dicct
            Model arguments

        Yields
        -----
        (x, y, 1)
        Segmentation mask for high-resolution image.
        """
        from skimage.filters import threshold_otsu
        from scipy import ndimage as ndi

        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max

        # get binarized image
        if geq:
            mask = arr[:, :, 0] >= thresh
        else:
            mask = arr[:, :, 0] < thresh

        # calculate markers as maximal distanced points from background (locally)
        distance = ndi.distance_transform_edt(1 - mask)
        local_maxi = peak_local_max(
            distance,
            indices=False,
            footprint=np.ones((5, 5)),
            labels=1 - mask
        )
        markers = ndi.label(local_maxi)[0]
        y = watershed(255-arr[:, :, 0], markers, mask=1-mask)
        return y


class SegmentationModelPretrainedTensorflow(SegmentationModel):

    def __init__(
            self,
            model,
            **kwargs
    ):
        import tensorflow as tf
        assert isinstance(model, tf.keras.model.Model), "model should be a tf keras model instance"
        super(SegmentationModelPretrainedTensorflow, self).__init__(
            model=model
        )

    def _segment(self, arr, **kwargs) -> np.ndarray:
        """

        Params
        ------
        arr: np.ndarray
            High-resolution image.
        kwargs: dicct
            Model arguments

        Yields
        -----
        (x, y, 1)
        Segmentation mask for high-resolution image.
        """
        # Uses callable tensorflow keras model.
        return self.model(arr, **kwargs)


def segment(
        img: ImageContainer,
        img_id: str,
        model_group: Union[str],
        model_instance: Union[None, str, SegmentationModel] = None,
        model_kwargs: dict = {},
        xs=None,
        ys=None,
        key_added: Union[str, None] = None
) -> Union[anndata.AnnData, None]:
    """
    Segments image.

    Params
    ------
    img: ImageContainer
        High-resolution image.
    img_id: str
        Key of image object to segment.
    model_group: str
        Name segmentation method to use. Available are:
            - skimage_blob: Blob extraction with skimage
            - tensorflow: tensorflow executable model
    model_instance: float
        Instance of executable segmentation model or name of specific method within model_group.
    model_kwargs: Optional [dict]
        Key word arguments to segmentation method.
    xs: int
        Width of the crops in pixels.
    ys: int
        Height of the crops in pixels.  # TODO add support as soon as crop supports this
    key_added: str
        Key of new image sized array to add into img object. Defaults to "segmentation_$model_group"

    Yields
    -----
    """
    if model_group == "skimage_blob":
        segmentation_model = SegmentationModelBlob(model=model_instance)
    elif model_group == "watershed":
        segmentation_model = SegmentationModelWatershed(model=model_instance)
    elif model_group == "tensorflow":
        segmentation_model = SegmentationModelPretrainedTensorflow(model=model_instance)
    else:
        raise ValueError("did not recognize model instance %s" % model_group)

    crops, xcoord, ycoord = img.crop_equally(xs=xs, ys=ys, img_id=img_id)
    crops = [
        segmentation_model.segment(
            arr=x,
            **model_kwargs
        ) for x in crops
    ]
    img_segmented = uncrop_img(
        crops=crops,
        x=xcoord,
        y=ycoord,
        shape=img.shape,
    )

    img_id = "segmented_" + model_group.lower() if key_added is None else key_added
    img.add_img(img=img_segmented, img_id=img_id)

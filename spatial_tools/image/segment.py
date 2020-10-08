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


def segment(
        img: ImageContainer,
        img_id: str,
        model_group: Union[str],
        model_instance: Union[None, SegmentationModel] = None,
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
        model_group = SegmentationModelBlob(data=img, model=model_group)
    elif model_group == "tensorflow":
        model_group = SegmentationModelPretrainedTensorflow(data=img, model=model_instance)
    else:
        raise ValueError("did not recognize model instance %s" % model_group)

    crops, xcoord, ycoord = img.crop_equally(xs=xs, ys=ys, img_id=img_id)
    img_segmented, x, y = [
        model_group.segment(
            arr=x,
            **model_kwargs
        ) for x in crops
    ]
    img_segmented = uncrop_img(
        crops=img_segmented,
        x=xcoord,
        y=ycoord,
        shape=img.shape,
    )

    img_id = "segmented_" + model_group.lower() if key_added is None else key_added
    img.add_img(img=img_segmented, img_id=img_id)


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
    crop_positions: Union[np.ndarray, None]

    def __init__(
            self,
            model,
    ):
        self.model = model

    def segment(self, arr) -> np.ndarray:
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
        return self._segment(arr)

    @abc.abstractmethod
    def _segment(self, arr) -> List[np.ndarray]:
        pass


class SegmentationModelBlob(SegmentationModel):

    def _segment(self, arr, invert: bool = True, **kwargs) -> List[np.ndarray]:
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
            arr = [1.-x for x in arr]

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

    def _segment(self, arr, **kwargs) -> List[np.ndarray]:
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
        return self.model(arr)

import abc
import anndata
import numpy as np
import skimage
from typing import List, Union

from .manipulate import crop_img, uncrop_img
from ._utils import _write_img_in_adata, _access_img_in_adata

"""
Functions exposed: smooth(), segment(), evaluate_nuclei_segmentation() 
"""


def smooth(
        adata: anndata.AnnData,
        img_key: str,
        method: str,
        new_img: Union[str, None] = None,
        copy: bool = False,
        **kwargs
):
    img = _access_img_in_adata(adata=adata, img_key=img_key)
    if method == "gaussian":
        img_smoothed = skimage.filters.gaussian(
            X=img,
            **kwargs
        )
    else:
        raise ValueError("did not recognize method %s" % method)
    if copy:
        adata = adata.copy()

    if new_img is None:
        new_img = img_key + "_smoothed"
    _write_img_in_adata(adata=adata, img_key=new_img, img=img_smoothed)
    if copy:
        return adata


def segment(
        adata: anndata.AnnData,
        img_key: str,
        model_group: Union[str],
        model_instance: Union[None, SegmentationModel] = None,
        model_kwargs: dict = {},
        crop_kwargs: dict = {},
        new_img: Union[str, None] = None,
        copy: bool = False,
        **kwargs
) -> Union[anndata.AnnData, None]:
    if model_group == "skimage_blob":
        model_group = SegmentationModelBlob(data=adata, model=model_group)
    elif model_group == "tensorflow":
        model_group = SegmentationModelPretrainedTensorflow(data=adata, model=model_instance)
    else:
        raise ValueError("did not recognize model instance %s" % model_group)

    model_group.crop(**crop_kwargs)
    img_segmented = model_group.segment(**model_kwargs)
    if copy:
        adata = adata.copy()
    if new_img is None:
        new_img = img_key + "_segmented"
    _write_img_in_adata(adata=adata, img_key=new_img, img=img_segmented)
    if copy:
        return adata


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
    data: Union[np.ndarray]
    crop_positions: Union[np.ndarray, None]

    def __init__(
            self,
            adata: anndata.AnnData,
            img_key: str,
            model,
    ):
        self.data = _access_img_in_adata(adata=adata, img_key=img_key)
        self.model = model

    def crop(self, x=None, y=None, **kwargs):
        """
        Extract image as list of np.ndarray from input data on which segmentation is run.

        Attrs:
            adata: Anndata instance containing image.
        """
        self.crop_positions = [x, y]
        if x is not None:
            self.data_cropped = [
                crop_img(
                    img=self.data,
                    x=xi,
                    y=yi,
                    **kwargs
                ) for xi, yi in zip(x, y)
            ]
        else:
            self.data_cropped = [self.data]

    def segment(self) -> np.ndarray:
        segments = self._segment()
        return uncrop_img(
            crops=segments,
            x=self.crop_positions[0],
            y=self.crop_positions[1],
            shape=self.data.shape
        )

    @abc.abstractmethod
    def _segment(self) -> List[np.ndarray]:
        pass


class SegmentationModelBlob(SegmentationModel):

    def _segment(self, invert: bool = True, **kwargs) -> List[np.ndarray]:
        """

        Attrs:
            kwargs: Model arguments
        """
        data = self.data_cropped
        if invert:
            data = [1.-x for x in data]

        channel_id = 0
        segmentation = []
        for x in data:
            if self.model == "log":
                y = skimage.feature.blob_log(
                    image=x,
                    **kwargs
                )
            elif self.model == "dog":
                y = skimage.feature.blob_dog(
                    image=x,
                    **kwargs
                )
            elif self.model == "doh":
                y = skimage.feature.blob_doh(
                    image=x,
                    **kwargs
                )
            else:
                raise ValueError("did not recognize self.model %s" % self.model)
            segmentation.append(y)
        return segmentation


class SegmentationModelPretrainedTensorflow(SegmentationModel):

    def __init__(
            self,
            adata: anndata.AnnData,
            img_key: str,
            model,
            **kwargs
    ):
        import tensorflow as tf
        assert isinstance(model, tf.keras.model.Model), "model should be a tf keras model instance"
        super(SegmentationModelPretrainedTensorflow, self).__init__(
            adata=adata,
            img_key=img_key,
            model=model
        )

    def _segment(self, **kwargs) -> List[np.ndarray]:
        # Uses callable tensorflow keras model.
        segmentation = []
        for x in self.data_cropped:
            segmentation.append(self.model(x))
        return segmentation

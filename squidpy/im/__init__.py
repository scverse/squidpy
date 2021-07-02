"""The image module."""
from squidpy.im._feature import calculate_image_features
from squidpy.im._process import process
from squidpy.im._segment import (
    segment,
    SegmentationModel,
    SegmentationCustom,
    SegmentationWatershed,
)
from squidpy.im._container import ImageContainer

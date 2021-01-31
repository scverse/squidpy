"""The image module."""
from squidpy.im.object import ImageContainer
from squidpy.im.segment import (
    segment_img,
    SegmentationBlob,
    SegmentationModel,
    SegmentationGeneric,
    SegmentationWatershed,
)
from squidpy.im.features import calculate_image_features
from squidpy.im.processing import process_img

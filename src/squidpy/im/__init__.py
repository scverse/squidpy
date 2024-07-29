"""The image module."""

from __future__ import annotations

from squidpy.im._container import ImageContainer
from squidpy.im._feature import calculate_image_features
from squidpy.im._process import process
from squidpy.im._segment import (
    SegmentationCustom,
    SegmentationModel,
    SegmentationWatershed,
    segment,
)

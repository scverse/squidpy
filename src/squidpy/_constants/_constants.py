"""Constants that user deals with."""

from __future__ import annotations

from enum import unique

from squidpy._constants._utils import ModeEnum


@unique
class ImageFeature(ModeEnum):
    TEXTURE = "texture"  # doc: This would be a docstring.
    SUMMARY = "summary"
    COLOR_HIST = "histogram"
    SEGMENTATION = "segmentation"
    CUSTOM = "custom"


# _ligrec.py
@unique
class CorrAxis(ModeEnum):
    INTERACTIONS = "interactions"
    CLUSTERS = "clusters"


@unique
class ComplexPolicy(ModeEnum):
    MIN = "min"
    ALL = "all"


@unique
class Processing(ModeEnum):
    SMOOTH = "smooth"
    GRAY = "gray"


@unique
class SegmentationBackend(ModeEnum):
    LOG = "log"
    DOG = "dog"
    DOH = "doh"
    WATERSHED = "watershed"
    CUSTOM = "custom"  # callable function


@unique
class InferDimensions(ModeEnum):
    DEFAULT = "default"
    CHANNELS_LAST = "channels_last"
    Z_LAST = "z_last"


@unique
class ScatterShape(ModeEnum):
    CIRCLE = "circle"
    SQUARE = "square"
    HEX = "hex"

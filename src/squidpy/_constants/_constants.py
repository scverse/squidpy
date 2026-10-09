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

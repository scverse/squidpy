"""Constants that can exposed to the user."""
from squidpy.constants._utils import ModeEnum


# an example, can even document the values using enum_tools:
# from enum_tools import document_enum
# @document_enum
class ImageFeature(ModeEnum):  # noqa: D101
    HOG = "hog"  # doc: This would be a docstring.
    TEXTURE = "texture"
    SUMMARY = "summary"
    COLOR_HIS = "color_hist"

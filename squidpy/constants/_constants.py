"""Constants that can exposed to the user."""
from enum import unique

from squidpy.constants._utils import ModeEnum


# an example, can even document the values using enum_tools:
# from enum_tools import document_enum
# @document_enum
@unique
class ImageFeature(ModeEnum):  # noqa: D101
    HOG = "hog"  # doc: This would be a docstring.
    TEXTURE = "texture"
    SUMMARY = "summary"
    COLOR_HIST = "color_hist"


# _ligrec.py
@unique
class FdrAxis(ModeEnum):  # noqa: D101
    INTERACTIONS = "interactions"
    CLUSTERS = "clusters"


@unique
class ComplexPolicy(ModeEnum):  # noqa: D101
    MIN = "min"
    ALL = "all"


@unique
class Transform(ModeEnum):  # noqa: D101
    SPECTRAL = "spectral"
    COSINE = "cosine"
    NONE = None


@unique
class CoordType(ModeEnum):  # noqa: D101
    VISIUM = "visium"
    NONE = None


@unique
class Processing(ModeEnum):  # noqa: D101
    SMOOTH = "smooth"


@unique
class SegmentationBackend(ModeEnum):  # noqa: D101
    BLOB = "blob"
    WATERSHED = "watershed"
    TENSORFLOW = "tensorflow"


@unique
class BlobModel(ModeEnum):  # noqa: D101
    LOG = "log"
    DOG = "dog"
    DOH = "doh"


@unique
class Dataset(ModeEnum):  # noqa: D101
    OB = "ob"
    SVZ = "svz"

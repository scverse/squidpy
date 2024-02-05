"""Constants that user deals with."""

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
class Transform(ModeEnum):
    SPECTRAL = "spectral"
    COSINE = "cosine"
    NONE = None


@unique
class CoordType(ModeEnum):
    GRID = "grid"
    GENERIC = "generic"


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
class BlobModel(ModeEnum):
    LOG = "log"
    DOG = "dog"
    DOH = "doh"


@unique
class Dataset(ModeEnum):
    OB = "ob"
    SVZ = "svz"


@unique
class Centrality(ModeEnum):
    DEGREE = "degree_centrality"
    CLUSTERING = "average_clustering"
    CLOSENESS = "closeness_centrality"


@unique
class DendrogramAxis(ModeEnum):
    INTERACTING_MOLS = "interacting_molecules"
    INTERACTING_CLUSTERS = "interacting_clusters"
    BOTH = "both"


@unique
class Symbol(ModeEnum):
    DISC = "disc"
    SQUARE = "square"


@unique
class SpatialAutocorr(ModeEnum):
    MORAN = "moran"
    GEARY = "geary"


@unique
class InferDimensions(ModeEnum):
    DEFAULT = "default"
    CHANNELS_LAST = "channels_last"
    Z_LAST = "z_last"


@unique
class RipleyStat(ModeEnum):
    F = "F"
    G = "G"
    L = "L"


@unique
class ScatterShape(str, ModeEnum):
    CIRCLE = "circle"
    SQUARE = "square"
    HEX = "hex"


@unique
class TenxVersions(str, ModeEnum):
    # Version numbers as class objects of TenxVersions
    V1 = "1.1.0"
    V2 = "1.2.0"
    V3 = "1.3.0"

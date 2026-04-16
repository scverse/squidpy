"""Dataclasses for the alignment skeleton.

These types are the contract between the public ``align_*`` functions, the
input/output helpers in :mod:`squidpy.experimental.tl._align._io`, and the
backend implementations in :mod:`squidpy.experimental.tl._align._backends`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import xarray as xr
    from anndata import AnnData
    from spatialdata import SpatialData
    from spatialdata.transformations import Affine


@dataclass(frozen=True)
class AlignPair:
    """Canonical pair of aligned-or-to-be-aligned elements.

    Returned by every resolver in :mod:`._io`.  ``ref``/``query`` carry the
    actual data to fit on; ``*_container``/``*_element_key`` remember where
    they came from so the writeback step can register a transformation on the
    correct element of the correct :class:`spatialdata.SpatialData`.
    """

    # ``ref``/``query`` are ``None`` for landmark-only flows where the fit
    # operates on user-provided coordinates and the resolver only needs to
    # locate the target containers + coordinate systems.
    ref: AnnData | xr.DataArray | None
    query: AnnData | xr.DataArray | None
    ref_container: SpatialData | None = None
    query_container: SpatialData | None = None
    ref_element_key: str | None = None
    query_element_key: str | None = None
    ref_cs: str | None = None
    query_cs: str | None = None


@dataclass(frozen=True)
class AffineTransform:
    """A ``(3, 3)`` homogeneous affine in ``(x, y)`` convention.

    This matches the coordinate axis order spatialdata uses for points -
    ``spatialdata.transformations.get_transformation_between_landmarks``
    asserts ``axes == ("x", "y")`` - and the order squidpy / scanpy use for
    ``adata.obsm["spatial"]``. Image elements are stored ``(c, y, x)`` in
    spatialdata, so when registering an ``AffineTransform`` on an *image*
    element you may need a separate matrix; this skeleton currently only
    deals with point coordinates.
    """

    matrix: np.ndarray
    source_cs: str | None = None
    target_cs: str | None = None

    def __post_init__(self) -> None:
        if self.matrix.shape != (3, 3):
            raise ValueError(f"Expected a (3, 3) homogeneous matrix, got shape {self.matrix.shape}.")

    def to_spatialdata(self) -> Affine:
        """Build a :class:`spatialdata.transformations.Affine` for writeback."""
        from spatialdata.transformations import Affine

        return Affine(
            self.matrix,
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        )

    def apply(self, coords: np.ndarray) -> np.ndarray:
        """Apply the affine to an ``(N, 2)`` ``(x, y)`` coordinate array."""
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"Expected an (N, 2) coordinate array, got shape {coords.shape}.")
        return coords @ self.matrix[:2, :2].T + self.matrix[:2, 2]


@dataclass(frozen=True)
class ObsDisplacement:
    """Per-obs ``(N, 2)`` ``(x, y)`` displacement field.

    Used by non-affine fits (e.g. LDDMM) where a single matrix cannot
    represent the deformation.  Displacements are added to the source
    observation coordinates (also ``(x, y)``) to obtain the aligned
    coordinates.
    """

    deltas: np.ndarray
    source_cs: str | None = None
    target_cs: str | None = None

    def __post_init__(self) -> None:
        if self.deltas.ndim != 2 or self.deltas.shape[1] != 2:
            raise ValueError(f"Expected an (N, 2) deltas array, got shape {self.deltas.shape}.")

    def apply(self, coords: np.ndarray) -> np.ndarray:
        """Bake the displacement into an ``(N, 2)`` obs coordinate array."""
        if coords.shape != self.deltas.shape:
            raise ValueError(f"Coord shape {coords.shape} does not match displacement shape {self.deltas.shape}.")
        return coords + self.deltas


Transform = AffineTransform | ObsDisplacement


@dataclass(frozen=True)
class AlignResult:
    """The output of an alignment backend call."""

    transform: Transform
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_affine(self) -> bool:
        return isinstance(self.transform, AffineTransform)

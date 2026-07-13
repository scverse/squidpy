"""Closed-form landmark alignment estimators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from squidpy._utils import NDArrayA
from squidpy.experimental.methods.registry import ALIGN_LANDMARKS


@dataclass
class AffineFitResult:
    """A fitted ``(3, 3)`` homogeneous affine mapping query onto ref, in ``(x, y)``."""

    matrix: np.ndarray
    source_cs: str | None = None
    target_cs: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.matrix.shape != (3, 3):
            raise ValueError(f"Expected a (3, 3) homogeneous matrix, found shape {self.matrix.shape}.")

    def transform(self, x: npt.ArrayLike) -> NDArrayA:
        """Apply the affine to an ``(N, 2)`` ``(x, y)`` coordinate array."""
        coords = np.asarray(x, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"Expected an (N, 2) coordinate array, found shape {coords.shape}.")
        return coords @ self.matrix[:2, :2].T + self.matrix[:2, 2]


def _fit_landmark_relation(
    ref: np.ndarray,
    query: np.ndarray,
    *,
    method: str,
    solve_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    source_cs: str | None = None,
    target_cs: str | None = None,
) -> AffineFitResult:
    ref = _validate_landmarks(ref, name="ref")
    query = _validate_landmarks(query, name="query")
    if ref.shape != query.shape:
        raise ValueError(f"`ref` and `query` must have the same shape; got {ref.shape} and {query.shape}.")
    if ref.shape[0] < 3:
        raise ValueError(f"`{method}` needs at least 3 landmark pairs, got {ref.shape[0]}.")

    matrix = solve_fn(ref, query)
    return AffineFitResult(
        matrix=matrix,
        source_cs=source_cs,
        target_cs=target_cs,
        metadata={"method": method},
    )


@ALIGN_LANDMARKS.register("similarity")
def fit_similarity(
    ref: np.ndarray,
    query: np.ndarray,
    *,
    source_cs: str | None = None,
    target_cs: str | None = None,
) -> AffineFitResult:
    """4-DOF similarity fit (rotation + uniform scale + translation), via spatialdata.

    Parameters
    ----------
    ref, query
        Pre-paired ``(N, 2)`` ``(x, y)`` landmark arrays (``N >= 3``).
    source_cs, target_cs
        Optional coordinate-system labels stamped onto the result for
        traceability; they do not affect the fit.
    """
    return _fit_landmark_relation(
        ref,
        query,
        method="similarity",
        solve_fn=_fit_similarity,
        source_cs=source_cs,
        target_cs=target_cs,
    )


@ALIGN_LANDMARKS.register("affine")
def fit_affine(
    ref: np.ndarray,
    query: np.ndarray,
    *,
    source_cs: str | None = None,
    target_cs: str | None = None,
) -> AffineFitResult:
    """6-DOF affine fit (rotation + non-uniform scale + shear + translation), via skimage.

    Parameters
    ----------
    ref, query
        Pre-paired ``(N, 2)`` ``(x, y)`` landmark arrays (``N >= 3``).
    source_cs, target_cs
        Optional coordinate-system labels stamped onto the result for
        traceability; they do not affect the fit.
    """
    return _fit_landmark_relation(
        ref,
        query,
        method="affine",
        solve_fn=_fit_affine,
        source_cs=source_cs,
        target_cs=target_cs,
    )


def _validate_landmarks(points: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"`{name}` must be a sequence of (x, y) pairs, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"`{name}` must contain only finite values.")
    return arr


def _fit_similarity(ref_xy: np.ndarray, query_xy: np.ndarray) -> np.ndarray:
    """4-DOF similarity fit, delegated to spatialdata."""
    from spatialdata.models import PointsModel
    from spatialdata.transformations import get_transformation_between_landmarks

    refs_pts = PointsModel.parse(ref_xy)
    moving_pts = PointsModel.parse(query_xy)
    sd_transform = get_transformation_between_landmarks(refs_pts, moving_pts)
    return _extract_affine_matrix(sd_transform)


def _fit_affine(ref_xy: np.ndarray, query_xy: np.ndarray) -> np.ndarray:
    """Full 6-DOF affine fit, delegated to skimage's least-squares estimator."""
    from skimage.transform import estimate_transform

    model_obj = estimate_transform("affine", src=query_xy, dst=ref_xy)
    return np.asarray(model_obj.params)


def _extract_affine_matrix(sd_transform: object) -> np.ndarray:
    """Pull a ``(3, 3)`` homogeneous matrix out of a spatialdata transformation."""
    from spatialdata.transformations import Affine as SDAffine
    from spatialdata.transformations import Sequence as SDSequence

    if isinstance(sd_transform, SDAffine):
        return np.asarray(sd_transform.matrix)
    if isinstance(sd_transform, SDSequence):
        return np.asarray(sd_transform.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")))
    raise TypeError(f"Unexpected transformation type from spatialdata: {type(sd_transform).__name__}.")

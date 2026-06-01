"""Closed-form landmark alignment estimators.

Ported from scverse/squidpy#1162 (``fit_landmark_affine``). Two pure models, no
JAX:

- ``"similarity"`` (4 DOF: rotation + uniform scale + translation) via
  :func:`spatialdata.transformations.get_transformation_between_landmarks`.
- ``"affine"`` (6 DOF) via :func:`skimage.transform.estimate_transform`.

Both consume **pre-paired** ``(N, 2)`` ``(x, y)`` landmark arrays (row ``i`` of
the query matches row ``i`` of the reference); ``N`` must be at least 3. No
automatic correspondence matching is performed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np

from squidpy.experimental._fit._estimator import Estimator
from squidpy.experimental._fit._methods._families import LANDMARK
from squidpy.experimental._fit._result import FitResult


@dataclass
class AffineFitResult(FitResult):
    """A fitted ``(3, 3)`` homogeneous affine mapping query onto ref, in ``(x, y)``."""

    matrix: np.ndarray
    source_cs: str | None = None
    target_cs: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.matrix.shape != (3, 3):
            raise ValueError(f"Expected a (3, 3) homogeneous matrix, found shape {self.matrix.shape}.")

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the affine to an ``(N, 2)`` ``(x, y)`` coordinate array."""
        coords = np.asarray(x, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"Expected an (N, 2) coordinate array, found shape {coords.shape}.")
        return coords @ self.matrix[:2, :2].T + self.matrix[:2, 2]


class _LandmarkEstimator(Estimator):
    """Base for closed-form landmark fits; subclasses pin :attr:`model`."""

    model: ClassVar[Literal["similarity", "affine"]]

    def fit(
        self,
        landmarks_ref: np.ndarray,
        landmarks_query: np.ndarray,
        *,
        source_cs: str | None = None,
        target_cs: str | None = None,
    ) -> AffineFitResult:
        """Fit an affine mapping ``landmarks_query`` onto ``landmarks_ref``.

        Parameters
        ----------
        landmarks_ref, landmarks_query
            Pre-paired ``(N, 2)`` ``(x, y)`` landmark arrays (``N >= 3``).
        source_cs, target_cs
            Optional coordinate-system labels stamped onto the result for
            traceability; they do not affect the fit.
        """
        ref = _validate_landmarks(landmarks_ref, name="landmarks_ref")
        query = _validate_landmarks(landmarks_query, name="landmarks_query")
        if ref.shape != query.shape:
            raise ValueError(
                f"`landmarks_ref` and `landmarks_query` must have the same shape; got {ref.shape} and {query.shape}."
            )
        if ref.shape[0] < 3:
            raise ValueError(f"`model={self.model!r}` needs at least 3 landmark pairs, got {ref.shape[0]}.")

        matrix = _fit_similarity(ref, query) if self.model == "similarity" else _fit_affine(ref, query)
        return AffineFitResult(
            matrix=matrix,
            source_cs=source_cs,
            target_cs=target_cs,
            metadata={"method": self.model},
        )


@LANDMARK.register("similarity")
class LandmarkSimilarityEstimator(_LandmarkEstimator):
    """4-DOF similarity fit (rotation + uniform scale + translation)."""

    name = "similarity"
    model = "similarity"


@LANDMARK.register("affine")
class LandmarkAffineEstimator(_LandmarkEstimator):
    """6-DOF affine fit (rotation + non-uniform scale + shear + translation)."""

    name = "affine"
    model = "affine"


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

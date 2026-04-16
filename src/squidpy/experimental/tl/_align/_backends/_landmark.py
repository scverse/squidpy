"""Closed-form landmark fit.

Two models, both pure NumPy / no JAX:

- ``"similarity"`` (4 DOF: rotation + uniform scale + translation, plus an
  optional reflection check) - delegated to
  :func:`spatialdata.transformations.get_transformation_between_landmarks`.
- ``"affine"`` (6 DOF: rotation + non-uniform scale + shear + translation) -
  delegated to :func:`skimage.transform.estimate_transform`, the same
  least-squares solver spatialdata uses internally.

Useful as a one-shot alignment when you already have corresponding landmarks,
and as a sanity-check baseline for the much heavier STalign LDDMM solver.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from squidpy.experimental.tl._align._types import AffineTransform


def fit_landmark_affine(
    landmarks_ref: np.ndarray,
    landmarks_query: np.ndarray,
    *,
    model: Literal["similarity", "affine"] = "similarity",
    source_cs: str | None = None,
    target_cs: str | None = None,
) -> AffineTransform:
    """Fit a 2D affine that maps ``landmarks_query`` onto ``landmarks_ref``.

    Both inputs are ``(N, 2)`` ``(x, y)`` arrays of corresponding landmarks
    (the ``i``-th row of ``landmarks_query`` matches the ``i``-th row of
    ``landmarks_ref``).  ``N`` must be at least 3.

    Parameters
    ----------
    landmarks_ref, landmarks_query
        Corresponding landmark coordinates in ``(x, y)`` convention.
    model
        ``"similarity"`` (4 DOF, via spatialdata) or ``"affine"`` (6 DOF,
        via skimage).
    source_cs, target_cs
        Optional coordinate-system labels stamped onto the returned
        :class:`AffineTransform` for traceability.
    """
    from squidpy.experimental.tl._align._types import AffineTransform

    ref = np.asarray(landmarks_ref, dtype=float)
    query = np.asarray(landmarks_query, dtype=float)

    if model == "similarity":
        matrix = _fit_similarity_via_spatialdata(ref, query)
    elif model == "affine":
        matrix = _fit_affine_via_skimage(ref, query)
    else:
        raise ValueError(f"Unknown landmark `model={model!r}`; expected 'similarity' or 'affine'.")

    return AffineTransform(matrix=matrix, source_cs=source_cs, target_cs=target_cs)


def _fit_similarity_via_spatialdata(ref_xy: np.ndarray, query_xy: np.ndarray) -> np.ndarray:
    """4-DOF similarity fit, delegated to spatialdata."""
    from spatialdata.models import PointsModel
    from spatialdata.transformations import get_transformation_between_landmarks

    refs_pts = PointsModel.parse(ref_xy)
    moving_pts = PointsModel.parse(query_xy)
    sd_transform = get_transformation_between_landmarks(refs_pts, moving_pts)
    return _extract_affine_matrix(sd_transform)


def _fit_affine_via_skimage(ref_xy: np.ndarray, query_xy: np.ndarray) -> np.ndarray:
    """Full 6-DOF affine fit, delegated to skimage's least-squares solver.

    This is what :func:`spatialdata.transformations.get_transformation_between_landmarks`
    uses under the hood before collapsing to a similarity; for the affine
    model we keep the raw matrix instead.
    """
    from skimage.transform import estimate_transform

    model_obj = estimate_transform("affine", src=query_xy, dst=ref_xy)
    return np.asarray(model_obj.params)


def _extract_affine_matrix(sd_transform: object) -> np.ndarray:
    """Pull a ``(3, 3)`` homogeneous matrix out of a spatialdata transformation.

    :func:`get_transformation_between_landmarks` may return either a single
    :class:`spatialdata.transformations.Affine` or a
    :class:`spatialdata.transformations.Sequence` of two affines (when a
    reflection is detected and rolled into the fit).  Use
    ``to_affine_matrix`` to collapse either back to a single 3x3.
    """
    from spatialdata.transformations import Affine as SDAffine
    from spatialdata.transformations import Sequence as SDSequence

    if isinstance(sd_transform, SDAffine):
        return np.asarray(sd_transform.matrix)
    if isinstance(sd_transform, SDSequence):
        return np.asarray(sd_transform.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")))
    raise TypeError(f"Unexpected transformation type from spatialdata: {type(sd_transform).__name__}.")

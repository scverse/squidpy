"""Closed-form landmark alignment estimators."""

from __future__ import annotations

from typing import Literal

import numpy as np

from squidpy._utils import NDArrayA

from ._common import validate_xy


def _fit(ref: np.ndarray, query: np.ndarray, *, method: Literal["similarity", "affine"]) -> NDArrayA:
    ref = validate_xy(ref, name="ref")
    query = validate_xy(query, name="query")
    if ref.shape != query.shape:
        raise ValueError(f"`ref` and `query` must have the same shape; got {ref.shape} and {query.shape}.")
    if ref.shape[0] < 3:
        raise ValueError(f"`{method}` needs at least 3 landmark pairs, got {ref.shape[0]}.")

    if method == "similarity":
        # spatialdata solves the 4-DOF case; skimage's "similarity" would do too, but
        # this is the transform napari-spatialdata registers, so it matches interactively.
        from spatialdata.models import PointsModel
        from spatialdata.transformations import get_transformation_between_landmarks

        matrix = _extract_affine_matrix(
            get_transformation_between_landmarks(PointsModel.parse(ref), PointsModel.parse(query))
        )
    else:
        from skimage.transform import estimate_transform

        matrix = np.asarray(estimate_transform("affine", src=query, dst=ref).params)

    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a (3, 3) homogeneous matrix, found shape {matrix.shape}.")
    return matrix


def apply_affine(matrix: np.ndarray, points: np.ndarray) -> NDArrayA:
    """Apply a homogeneous ``(3, 3)`` ``(x, y)`` affine to an ``(N, 2)`` coordinate array."""
    coords = np.asarray(points, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected an (N, 2) coordinate array, found shape {coords.shape}.")
    return coords @ matrix[:2, :2].T + matrix[:2, 2]


def fit_similarity(ref: np.ndarray, query: np.ndarray) -> NDArrayA:
    """4-DOF similarity fit (rotation + uniform scale + translation), via spatialdata.

    Parameters
    ----------
    ref, query
        Pre-paired ``(N, 2)`` ``(x, y)`` landmark arrays (``N >= 3``).

    Returns
    -------
    The homogeneous ``(3, 3)`` affine mapping query onto ref, in ``(x, y)``.
    """
    return _fit(ref, query, method="similarity")


def fit_affine(ref: np.ndarray, query: np.ndarray) -> NDArrayA:
    """6-DOF affine fit (rotation + non-uniform scale + shear + translation), via skimage.

    Parameters
    ----------
    ref, query
        Pre-paired ``(N, 2)`` ``(x, y)`` landmark arrays (``N >= 3``).

    Returns
    -------
    The homogeneous ``(3, 3)`` affine mapping query onto ref, in ``(x, y)``.
    """
    return _fit(ref, query, method="affine")


def _extract_affine_matrix(sd_transform: object) -> np.ndarray:
    """Pull a ``(3, 3)`` homogeneous matrix out of a spatialdata transformation."""
    from spatialdata.transformations import Affine as SDAffine
    from spatialdata.transformations import Sequence as SDSequence

    if isinstance(sd_transform, SDAffine):
        return np.asarray(sd_transform.matrix)
    if isinstance(sd_transform, SDSequence):
        return np.asarray(sd_transform.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")))
    raise TypeError(f"Unexpected transformation type from spatialdata: {type(sd_transform).__name__}.")

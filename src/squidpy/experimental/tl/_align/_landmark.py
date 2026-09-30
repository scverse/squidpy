"""Closed-form landmark alignment estimators."""

from __future__ import annotations

from typing import Literal

import numpy as np

from squidpy._utils import NDArrayA
from squidpy._validators import validate_xy


def check_spans_plane(points: np.ndarray, *, name: str, method: str) -> None:
    """Raise unless ``points`` span a plane: a line (or a point) cannot fix a 2D affine."""
    if np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-8) < 2:
        raise ValueError(
            f"{method} needs {name} landmarks spanning a plane, but they lie on a line "
            f"(or a single point). Pick landmarks that are not collinear."
        )


def _fit(ref: np.ndarray, query: np.ndarray, *, method: Literal["similarity", "affine"]) -> NDArrayA:
    ref = validate_xy(ref, name="ref")
    query = validate_xy(query, name="query")
    if ref.shape != query.shape:
        raise ValueError(f"`ref` and `query` must have the same shape; got {ref.shape} and {query.shape}.")
    # 2 distinct pairs fix the 4 DOF of a similarity; the 6-DOF affine needs 3 off a line,
    # since on a line it is exact *on* the landmarks and arbitrary off them, with no residual
    # to reveal it.
    need = 2 if method == "similarity" else 3
    if ref.shape[0] < need:
        raise ValueError(f"`{method}` needs at least {need} landmark pairs, got {ref.shape[0]}.")
    for name, points in (("ref", ref), ("query", query)):
        if method == "affine":
            check_spans_plane(points, name=f"`{name}`", method="`affine`")
        elif np.ptp(points, axis=0).max() <= 1e-8:
            raise ValueError(f"`similarity` needs `{name}` landmarks at two distinct places at least.")

    from skimage.transform import estimate_transform

    # skimage's similarity is Umeyama restricted to proper rotations: it never mirrors, where
    # spatialdata's (and so napari-spatialdata's) picks the reflection sign from an affine fit
    # and flips on (near-)collinear landmarks.
    fitted = estimate_transform(method, src=query, dst=ref)
    # skimage >= 0.26 returns a falsy FailedEstimation; 0.25 returns NaN params
    if not fitted or not np.isfinite(fitted.params).all():
        raise ValueError(f"`{method}` fit failed: {fitted}")
    return np.asarray(fitted.params)


def apply_affine(matrix: np.ndarray, points: np.ndarray) -> NDArrayA:
    """Apply a homogeneous ``(3, 3)`` ``(x, y)`` affine to an ``(N, 2)`` coordinate array.

    Parameters
    ----------
    matrix
        Homogeneous ``(3, 3)`` affine in ``(x, y)``, as
        :func:`~squidpy.experimental.tl.align_landmarks` returns.
    points
        ``(N, 2)`` ``(x, y)`` coordinates to map.

    Returns
    -------
    The mapped ``(N, 2)`` coordinates.
    """
    coords = np.asarray(points, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected an (N, 2) coordinate array, found shape {coords.shape}.")
    return coords @ matrix[:2, :2].T + matrix[:2, 2]


def fit_similarity(ref: np.ndarray, query: np.ndarray) -> NDArrayA:
    """4-DOF similarity fit (rotation + uniform scale + translation), via skimage.

    Never a reflection: a mirrored query is fitted by the best rotation instead. This differs
    from napari-spatialdata, whose similarity may reflect, so the two can disagree when the
    landmarks are mirrored (or nearly collinear).

    Parameters
    ----------
    ref, query
        Pre-paired ``(N, 2)`` ``(x, y)`` landmark arrays (``N >= 2``), at two distinct
        places at least: a line determines a similarity.

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
        Pre-paired ``(N, 2)`` ``(x, y)`` landmark arrays (``N >= 3``), not collinear:
        a line leaves the 6 degrees of freedom underdetermined.

    Returns
    -------
    The homogeneous ``(3, 3)`` affine mapping query onto ref, in ``(x, y)``.
    """
    return _fit(ref, query, method="affine")

"""Validation helpers shared by the public align_* functions.

These wrap the generic checks in :mod:`squidpy._validators` with messages
tailored to alignment.  The goal is to fail fast and tell the user *why* a
combination of arguments is wrong, not just that it is.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from squidpy._validators import assert_one_of

# Flavour identifiers - shared between dispatch, validation, and the public
# function defaults so a typo lights up everywhere it's used.
STALIGN = "stalign"
MOSCOT = "moscot"

ALLOWED_FLAVOURS_OBS = (STALIGN, MOSCOT)
ALLOWED_FLAVOURS_IMAGES = (STALIGN,)
ALLOWED_OUTPUT_MODES_OBS = ("affine", "obs", "return")
ALLOWED_OUTPUT_MODES_NONOBS = ("affine", "return")
ALLOWED_LANDMARK_MODELS = ("similarity", "affine")


def validate_flavour(flavour: str, *, allowed: Sequence[str], op: str) -> None:
    assert_one_of(flavour, allowed, name=f"{op}.flavour")


def validate_output_mode(output_mode: str, *, allowed: Sequence[str], op: str) -> None:
    assert_one_of(output_mode, allowed, name=f"{op}.output_mode")


def validate_key_added(key_added: str | None, output_mode: str) -> None:
    """``key_added`` only makes sense in the obs-materialisation path."""
    if key_added is not None and output_mode != "obs":
        raise ValueError(
            f"`key_added={key_added!r}` is only meaningful when `output_mode='obs'`. "
            f"Got `output_mode={output_mode!r}`. The other modes either register a "
            f"transformation on the existing element ('affine') or return the raw "
            f"result ('return'), so there is nothing for `key_added` to name."
        )


def validate_landmark_model(model: str) -> None:
    assert_one_of(model, ALLOWED_LANDMARK_MODELS, name="align_by_landmarks.model")


def validate_landmarks(
    landmarks_ref: Sequence[tuple[float, float]],
    landmarks_query: Sequence[tuple[float, float]],
    *,
    model: str,
    cs_ref_extent: tuple[float, float, float, float] | None = None,
    cs_query_extent: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate landmark sequences and return them as ``(N, 2)`` arrays.

    Parameters
    ----------
    landmarks_ref, landmarks_query
        Sequences of ``(x, y)`` tuples.  Must have the same length and at
        least 3 entries (the closed-form solvers under both ``model``
        choices need at least 3 corresponding points).
    model
        ``"similarity"`` (4 DOF, via spatialdata) or ``"affine"`` (6 DOF,
        via skimage's least-squares estimator).
    cs_ref_extent, cs_query_extent
        Optional ``(x_min, y_min, x_max, y_max)`` bounds of the named
        coordinate system at the requested scale.  When provided, every
        landmark must fall inside.  Catches the "I extracted these from
        scale0 but asked for scale2" footgun.
    """
    ref = np.asarray(landmarks_ref, dtype=float)
    query = np.asarray(landmarks_query, dtype=float)

    if ref.ndim != 2 or ref.shape[1] != 2:
        raise ValueError(f"`landmarks_ref` must be a sequence of (x, y) pairs, got shape {ref.shape}.")
    if query.ndim != 2 or query.shape[1] != 2:
        raise ValueError(f"`landmarks_query` must be a sequence of (x, y) pairs, got shape {query.shape}.")
    if len(ref) != len(query):
        raise ValueError(
            f"`landmarks_ref` and `landmarks_query` must have the same length; got {len(ref)} and {len(query)}."
        )

    if len(ref) < 3:
        raise ValueError(
            f"`model={model!r}` needs at least 3 landmark pairs (spatialdata requirement), got {len(ref)}."
        )

    if cs_ref_extent is not None:
        _check_in_extent(ref, cs_ref_extent, name="landmarks_ref")
    if cs_query_extent is not None:
        _check_in_extent(query, cs_query_extent, name="landmarks_query")

    return ref, query


def _check_in_extent(
    points: np.ndarray,
    extent: tuple[float, float, float, float],
    *,
    name: str,
) -> None:
    x_min, y_min, x_max, y_max = extent
    out_of_bounds = (points[:, 0] < x_min) | (points[:, 0] > x_max) | (points[:, 1] < y_min) | (points[:, 1] > y_max)
    if out_of_bounds.any():
        bad = points[out_of_bounds]
        raise ValueError(
            f"{name}: {int(out_of_bounds.sum())} landmark(s) fall outside the coordinate-system "
            f"extent (x in [{x_min}, {x_max}], y in [{y_min}, {y_max}]). "
            f"This usually means the landmarks were extracted at a different scale than the "
            f"one requested. First out-of-bounds point: {tuple(bad[0])}."
        )


def validate_unexpected(
    *,
    name: str,
    value: Any,
    when: str,
    hint: str = "",
) -> None:
    """Raise an educational error when an argument was passed in a context it has no role in."""
    if value is not None:
        msg = f"`{name}={value!r}` was passed but is only valid when {when}."
        if hint:
            msg = f"{msg} {hint}"
        raise ValueError(msg)


def validate_required(
    *,
    name: str,
    value: Any,
    when: str,
) -> None:
    """Raise when an argument is required by the current context but missing."""
    if value is None:
        raise ValueError(f"`{name}` is required when {when}.")

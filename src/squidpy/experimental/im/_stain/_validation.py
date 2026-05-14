"""Validation and canonicalisation helpers for fitted stain matrices."""

from __future__ import annotations

import numpy as np

from squidpy.experimental.im._stain._constants import NEAR_ZERO_NORM, RUIFROK_HE


class StainFittingError(RuntimeError):
    """Raised when a stain matrix fit is unusable.

    Carries the image key (when known) so cohort-fit code can attribute the
    failure to a specific sample rather than failing the whole batch.
    """

    def __init__(self, reason: str, *, image_key: str | None = None) -> None:
        self.reason = reason
        self.image_key = image_key
        prefix = f"[{image_key}] " if image_key else ""
        super().__init__(f"{prefix}{reason}")


def _ensure_2d(W: np.ndarray, *, expected_cols: tuple[int, ...]) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] != 3 or W.shape[1] not in expected_cols:
        cols = " or ".join(f"(3, {n})" for n in expected_cols)
        raise ValueError(f"Stain matrix must have shape {cols}; got {W.shape}.")
    return W


def _normalise_columns(W: np.ndarray, *, image_key: str | None = None) -> np.ndarray:
    norms = np.linalg.norm(W, axis=0)
    if np.any(norms < NEAR_ZERO_NORM):
        raise StainFittingError("Stain matrix has a near-zero column.", image_key=image_key)
    return W / norms


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def reorder_to_canonical(
    W: np.ndarray,
    reference: dict[str, np.ndarray] = RUIFROK_HE,
    *,
    stain_order: tuple[str, str] = ("hematoxylin", "eosin"),
) -> np.ndarray:
    """Reorder the first two columns of ``W`` into canonical ``(H, E, ...)``.

    Macenko and Vahadane both produce stain vectors in arbitrary order; this
    aligns each column with its nearest canonical reference and flips the
    sign so the column points the same way as the canonical vector. Returns
    a new array of the same shape as ``W``; the third column (if present) is
    left untouched.
    """
    W = _ensure_2d(W, expected_cols=(2, 3))
    ref_vectors = [reference[name] for name in stain_order]

    candidates = W[:, :2]
    pairs: list[tuple[int, int]] = [(0, 1), (1, 0)]
    best_pair = max(
        pairs,
        key=lambda p: sum(abs(float(np.dot(candidates[:, p[k]], ref_vectors[k]))) for k in range(2)),
    )

    out = W.copy()
    for k, src in enumerate(best_pair):
        col = candidates[:, src].copy()
        if np.dot(col, ref_vectors[k]) < 0:
            col = -col
        out[:, k] = col
    return out


def complement_third_column(W: np.ndarray) -> np.ndarray:
    """Append a unit-norm column orthogonal to the first two.

    Given ``W`` of shape ``(3, 2)``, return a ``(3, 3)`` matrix whose third
    column is the normalised cross product of columns 0 and 1.
    """
    W = _ensure_2d(W, expected_cols=(2,))
    third = np.cross(W[:, 0], W[:, 1])
    n = np.linalg.norm(third)
    if n < NEAR_ZERO_NORM:
        raise StainFittingError("First two stain vectors are collinear; cannot complement.")
    third = third / n
    return np.column_stack([W, third])


def validate_stain_matrix(
    W: np.ndarray,
    *,
    image_key: str | None = None,
    reference: dict[str, np.ndarray] = RUIFROK_HE,
    stain_order: tuple[str, str] = ("hematoxylin", "eosin"),
    max_angle_deg: float = 45.0,
    min_singular_value: float = 1e-3,
) -> None:
    """Validate a fitted ``(3, 3)`` stain matrix.

    Raises :class:`StainFittingError` if any column is degenerate, the
    matrix is rank-deficient, or the H or E column deviates from its
    canonical reference by more than ``max_angle_deg`` degrees.
    """
    W = _ensure_2d(W, expected_cols=(3,))
    W = _normalise_columns(W, image_key=image_key)

    singular_values = np.linalg.svd(W, compute_uv=False)
    if float(singular_values.min()) < min_singular_value:
        raise StainFittingError(
            f"Stain matrix is rank-deficient (min singular value "
            f"{singular_values.min():.2e} < {min_singular_value:.2e}).",
            image_key=image_key,
        )

    for k, name in enumerate(stain_order):
        ref = reference[name]
        angle = _angle_deg(W[:, k], ref)
        if angle > max_angle_deg and (180.0 - angle) > max_angle_deg:
            raise StainFittingError(
                f"Stain column {k} ({name}) deviates from canonical by {angle:.1f} degrees (>{max_angle_deg:.1f}).",
                image_key=image_key,
            )

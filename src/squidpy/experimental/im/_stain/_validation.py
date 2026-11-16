"""Stain-matrix validation and canonicalisation primitives.

Pure numpy, no ``sdata``, no public export. Shared by the Macenko and
Vahadane fits so both produce a canonical ``(H, E, complement)`` matrix that
downstream apply/decompose code can treat method-agnostically.
"""

from __future__ import annotations

import numpy as np

from squidpy.experimental.im._stain._constants import RUIFROK_HE


class StainFittingError(RuntimeError):
    """A stain-matrix fit produced an invalid or degenerate result.

    Carries ``image_key`` so cohort fitting (a later PR) can attribute a
    failure to a specific slide and skip or flag it by name.
    """

    def __init__(self, reason: str, *, image_key: str | None = None) -> None:
        self.reason = reason
        self.image_key = image_key
        prefix = f"[{image_key}] " if image_key is not None else ""
        super().__init__(f"{prefix}{reason}")


def _canonical_he(reference: dict[str, np.ndarray]) -> np.ndarray:
    """Stack the reference H and E unit vectors as columns of a ``(3, 2)``."""
    return np.stack([reference["hematoxylin"], reference["eosin"]], axis=1)


def angle_between_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Unsigned angle in degrees between two vectors (sign-agnostic)."""
    cos = abs(float(u @ v)) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _unit_columns(matrix: np.ndarray) -> np.ndarray:
    """Scale each column of ``matrix`` to unit L2 norm."""
    return matrix / np.linalg.norm(matrix, axis=0, keepdims=True)


def reorder_to_canonical(matrix: np.ndarray, reference: dict[str, np.ndarray] = RUIFROK_HE) -> np.ndarray:
    """Order a ``(3, 2)`` stain matrix to ``(H, E)`` and fix column signs.

    Macenko's SVD and Vahadane's NMF recover the two stain directions in an
    arbitrary order and sign. We assign each recovered column to whichever of
    the canonical Ruifrok H/E vectors it is most colinear with, then flip its
    sign so it points the same way as that reference (absorbance is positive).
    """
    w = np.asarray(matrix, dtype=np.float64)
    if w.shape != (3, 2):
        raise ValueError(f"stain matrix to reorder must have shape (3, 2); got {w.shape}.")
    canonical = _canonical_he(reference)  # (3, 2): [H, E]
    cols = _unit_columns(w)

    # cosine of each recovered column against each canonical vector
    sim = cols.T @ canonical  # (2 recovered, 2 canonical)
    # assign recovered column 0/1 to H if it favours H more than column 1 does
    h_idx = int(np.argmax(np.abs(sim[:, 0])))
    e_idx = 1 - h_idx
    ordered = np.stack([w[:, h_idx], w[:, e_idx]], axis=1)

    # flip signs so each column points along its canonical reference
    for j in range(2):
        if ordered[:, j] @ canonical[:, j] < 0:
            ordered[:, j] = -ordered[:, j]
    return ordered


def complement_third_column(matrix: np.ndarray) -> np.ndarray:
    """Extend a ``(3, 2)`` H/E matrix to ``(3, 3)`` with a complement column.

    The third column is the unit cross product of the H and E columns: the
    residual direction orthogonal to both, used to capture absorbance not
    explained by either stain.
    """
    w = np.asarray(matrix, dtype=np.float64)
    if w.shape != (3, 2):
        raise ValueError(f"stain matrix to complement must have shape (3, 2); got {w.shape}.")
    third = np.cross(w[:, 0], w[:, 1])
    norm = np.linalg.norm(third)
    if norm < 1e-8:
        raise StainFittingError("H and E stain vectors are colinear; cannot form a complement.")
    third = third / norm
    return np.column_stack([w, third])


def validate_stain_matrix(
    matrix: np.ndarray,
    *,
    reference: dict[str, np.ndarray] = RUIFROK_HE,
    max_angle_deg: float = 45.0,
    image_key: str | None = None,
) -> None:
    """Raise :class:`StainFittingError` if a ``(3, 3)`` matrix is implausible.

    Guards against the failure modes of an unsupervised stain fit: a column
    collapsed to zero, a rank-deficient (single-stain) matrix, or an H/E
    direction rotated far from its Ruifrok canonical (a sign the fit latched
    onto noise or a non-H&E chromogen).
    """
    w = np.asarray(matrix, dtype=np.float64)
    if w.shape != (3, 3):
        raise StainFittingError(f"stain matrix must have shape (3, 3); got {w.shape}.", image_key=image_key)
    if not np.all(np.isfinite(w)):
        raise StainFittingError("stain matrix contains non-finite values.", image_key=image_key)

    norms = np.linalg.norm(w, axis=0)
    if np.any(norms < 1e-8):
        raise StainFittingError("stain matrix has a zero-norm column.", image_key=image_key)
    if np.linalg.matrix_rank(w, tol=1e-6) < 3:
        raise StainFittingError("stain matrix is rank-deficient (stains are not separable).", image_key=image_key)

    canonical = _canonical_he(reference)
    for name, j in (("hematoxylin", 0), ("eosin", 1)):
        angle = angle_between_deg(w[:, j], canonical[:, j])
        if angle > max_angle_deg:
            raise StainFittingError(
                f"{name} stain vector deviates {angle:.1f} deg from its canonical (max {max_angle_deg}).",
                image_key=image_key,
            )

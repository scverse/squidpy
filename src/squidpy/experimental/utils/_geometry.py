"""Shared internal geometry helpers for mask/contour analysis.

Not part of the public API - symbols here are private and may change
without notice.
"""

from __future__ import annotations

import numpy as np
from skimage.measure import find_contours


def equivalent_diameter(area: float) -> float:
    """Diameter of the circle with the given area: ``sqrt(4 * area / pi)``."""
    return float(np.sqrt(4 * area / np.pi))


def largest_contour(padded_mask: np.ndarray, level: float = 0.5) -> np.ndarray | None:
    """Return the longest :func:`skimage.measure.find_contours` contour, or ``None``.

    The mask must be **already 1px zero-padded** by the caller so that cells
    touching the crop edge (e.g. filling their bbox) are traced closed.  Padding
    is left to the caller because its placement relative to other steps (e.g.
    downsampling) is order-sensitive and differs between call sites.  Returned
    coordinates are in the padded mask's frame.
    """
    contours = find_contours(padded_mask, level)
    if not contours:
        return None
    return max(contours, key=len)

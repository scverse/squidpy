"""Fixture definitions shared by squidpy-ports and squidpy.

This module is vendored **byte-identically** into squidpy as
``tests/experimental/methods/_stalign_fixtures.py``. Both sides building their inputs
from the same source is what stops the committed reference bundle and the tests that
consume it from drifting apart; a checksum of this file is stamped into every generated
``.npz`` and asserted on the squidpy side.

Consequently it must depend on **numpy only** -- no torch, no jax, no squidpy_ports.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

__all__ = [
    "LDDMM_PARAMS",
    "N_LANDMARKS",
    "N_POINTS",
    "RASTER_PARAMS",
    "SEED",
    "SHIFT",
    "THETA",
    "Clouds",
    "checksum",
    "make_clouds",
    "rotation",
]

SEED = 20240501
N_POINTS = 800
N_LANDMARKS = 6

# Deliberately not a round angle and not an integer shift.
#
# Upstream `interp` normalises coordinates with (c - x0) / (x[-1] - x0) and then lets
# grid_sample(align_corners=True) denormalise by *(n - 1); squidpy's `_interp` uses
# (c - x0) / (x[1] - x0). Those are algebraically equal but differ by ~1 ulp, so a
# sample landing exactly on a grid line can floor() to different neighbour pairs on the
# two sides -- an O(1) disagreement that says nothing about the port. An "easy" rotation
# angle or an integer shift makes that happen everywhere at once. `test_fixture_samples_
# are_off_grid` asserts we stayed clear of it.
THETA = 0.1503216
SHIFT = (25.371, -15.119)

#: Fraction of the cloud's RMS radius added as isotropic noise to the query.
JITTER_FRAC = 0.03
#: Fraction of query points dropped, so the correspondence is not a bijection.
DROPOUT_FRAC = 0.05

#: Rasterisation controls. Small enough that a full LDDMM solve runs in seconds.
RASTER_PARAMS = {"dx": 30.0, "blur": [2.0, 1.0, 0.5], "expand": 1.1}

#: LDDMM controls. `a=200` keeps the velocity grid ~14x23; everything else is upstream's
#: default so we are comparing the published configuration.
LDDMM_PARAMS = {
    "a": 200.0,
    "p": 2.0,
    "expand": 2.0,
    "nt": 3,
    "diffeo_start": 0,
    "epL": 2e-8,
    "epT": 2e-1,
    "epV": 2e3,
    "sigmaM": 1.0,
    "sigmaB": 2.0,
    "sigmaA": 5.0,
    "sigmaR": 5e5,
    "sigmaP": 2e1,
}


class Clouds:
    """A reference/query point-cloud pair with corresponding landmarks, all in ``(x, y)``."""

    def __init__(
        self,
        ref: np.ndarray,
        query: np.ndarray,
        landmarks_ref: np.ndarray,
        landmarks_query: np.ndarray,
    ) -> None:
        self.ref = ref
        self.query = query
        self.landmarks_ref = landmarks_ref
        self.landmarks_query = landmarks_query

    # `torch.tensor` rejects arrays with negative strides, which a bare `[:, ::-1]`
    # produces, so every row-col view is materialised contiguous.

    @property
    def ref_rc(self) -> np.ndarray:
        """Reference cloud in row-col order, which is what upstream works in."""
        return np.ascontiguousarray(self.ref[:, ::-1])

    @property
    def query_rc(self) -> np.ndarray:
        """Query cloud in row-col order."""
        return np.ascontiguousarray(self.query[:, ::-1])

    @property
    def landmarks_ref_rc(self) -> np.ndarray:
        """Reference landmarks in row-col order."""
        return np.ascontiguousarray(self.landmarks_ref[:, ::-1])

    @property
    def landmarks_query_rc(self) -> np.ndarray:
        """Query landmarks in row-col order."""
        return np.ascontiguousarray(self.landmarks_query[:, ::-1])


def rotation(theta: float) -> np.ndarray:
    """A 2x2 rotation matrix acting on ``(x, y)`` column vectors."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def make_clouds() -> Clouds:
    """Build the deterministic reference/query pair.

    The reference is three structures -- an anisotropic Gaussian lobe, a smaller offset
    lobe, and an annulus -- so the rasterised density has real structure rather than one
    featureless blob, and the registration has something to lock onto. The query is that
    cloud rotated, shifted, jittered and randomly thinned.
    """
    rng = np.random.default_rng(SEED)

    n_lobe, n_small = 400, 200
    n_ring = N_POINTS - n_lobe - n_small

    lobe = rng.normal(size=(n_lobe, 2)) @ np.diag([150.0, 100.0]) @ rotation(0.4).T
    small = rng.normal(size=(n_small, 2)) @ np.diag([80.0, 80.0]) + np.array([400.0, 250.0])

    angle = rng.uniform(0.0, 2.0 * np.pi, size=n_ring)
    radius = 300.0 + rng.normal(scale=25.0, size=n_ring)
    ring = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1) + np.array([100.0, -200.0])

    ref = np.concatenate([lobe, small, ring], axis=0)

    rms_radius = float(np.sqrt(np.mean(np.sum((ref - ref.mean(0)) ** 2, axis=1))))
    moved = ref @ rotation(THETA).T + np.asarray(SHIFT)
    jittered = moved + rng.normal(scale=JITTER_FRAC * rms_radius, size=moved.shape)

    keep = rng.permutation(N_POINTS)[: int(round(N_POINTS * (1.0 - DROPOUT_FRAC)))]
    keep.sort()
    query = jittered[keep]

    # Landmarks are spread across the whole cloud rather than clustered, so the affine
    # they induce is well conditioned.
    lm_idx = keep[np.linspace(0, keep.size - 1, N_LANDMARKS).round().astype(int)]
    landmarks_ref = ref[lm_idx]
    landmarks_query = jittered[lm_idx]

    return Clouds(
        ref=np.ascontiguousarray(ref, dtype=float),
        query=np.ascontiguousarray(query, dtype=float),
        landmarks_ref=np.ascontiguousarray(landmarks_ref, dtype=float),
        landmarks_query=np.ascontiguousarray(landmarks_query, dtype=float),
    )


def checksum() -> str:
    """SHA-256 of this file, so both copies can prove they are the same file."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

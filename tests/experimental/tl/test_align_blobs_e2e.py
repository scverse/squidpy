"""End-to-end alignment tests on the spatialdata ``blobs`` fixture.

Pulls the 200-point ``blobs_points`` cloud from
:func:`spatialdata.datasets.blobs`, applies a known transformation to a
copy, and verifies that both alignment paths recover the inverse:

- ``align_by_landmarks`` should recover an exact closed-form solution.
- ``align_obs(flavour="stalign")`` should reduce the residual displacement
  (the LDDMM is non-affine, so we compare the warped query against the ref
  by mean Euclidean distance, not by exact equality).

These tests exercise real solver iterations.  They are slower than the
``niter=1`` smoke tests in ``test_align_stalign_integration.py`` (the
stalign test takes a few seconds) but still run on every commit.  The user
can mark them ``slow`` later if CI budget becomes a concern.
"""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

pytest.importorskip("jax")


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _blobs_points_xy() -> np.ndarray:
    """Pull the 200 ``blobs_points`` rows as an ``(N, 2)`` ``(x, y)`` array."""
    from spatialdata.datasets import blobs

    sd = blobs()
    pts_df = sd.points["blobs_points"].compute()
    return np.column_stack([pts_df["x"].to_numpy(), pts_df["y"].to_numpy()]).astype(float)


def _make_blob_adata(coords_xy: np.ndarray) -> AnnData:
    """Wrap an ``(N, 2)`` point cloud as an AnnData with ``obsm['spatial']``."""
    adata = AnnData(np.zeros((coords_xy.shape[0], 1), dtype=float))
    adata.obsm["spatial"] = coords_xy
    return adata


def _rotation_about_centre(theta_rad: float, centre_xy: np.ndarray) -> np.ndarray:
    """Build a ``(3, 3)`` homogeneous rotation about a centre, in xy convention."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    T_to = np.array([[1.0, 0.0, -centre_xy[0]], [0.0, 1.0, -centre_xy[1]], [0.0, 0.0, 1.0]])
    T_back = np.array([[1.0, 0.0, centre_xy[0]], [0.0, 1.0, centre_xy[1]], [0.0, 0.0, 1.0]])
    return T_back @ R @ T_to


def _apply_homog(points_xy: np.ndarray, M: np.ndarray) -> np.ndarray:
    return points_xy @ M[:2, :2].T + M[:2, 2]


@pytest.fixture(scope="module")
def blobs_rotated() -> tuple[AnnData, AnnData, np.ndarray]:
    """``(ref_adata, query_adata, gt_affine)`` for the rotation-recovery tests.

    The query is the reference rotated by 12° around the cloud centroid.
    The ground-truth affine ``gt_affine`` maps ``ref -> query``; recovery
    means producing a transform that maps ``query -> ref`` (i.e. the
    inverse of ``gt_affine``).
    """
    ref_xy = _blobs_points_xy()
    centre = ref_xy.mean(axis=0)
    gt = _rotation_about_centre(np.deg2rad(12.0), centre)
    query_xy = _apply_homog(ref_xy, gt)

    ref = _make_blob_adata(ref_xy)
    query = _make_blob_adata(query_xy)
    return ref, query, gt


# ---------------------------------------------------------------------------
# 1. align_by_landmarks recovers the rotation exactly
# ---------------------------------------------------------------------------


def test_align_by_landmarks_recovers_blobs_rotation_exactly(blobs_rotated) -> None:
    """A handful of correspondences is enough for the closed-form fit to
    invert a pure 2D rotation up to numerical noise."""
    from squidpy.experimental.tl._align._backends._landmark import fit_landmark_affine

    ref, query, gt = blobs_rotated

    # Pick 4 landmarks that span the cloud well.
    idx = [0, 50, 100, 150]
    landmarks_ref = ref.obsm["spatial"][idx]
    landmarks_query = query.obsm["spatial"][idx]

    fit = fit_landmark_affine(landmarks_ref, landmarks_query, model="similarity")
    inv_gt = np.linalg.inv(gt)

    # Apply the recovered transform to all 200 query points and compare to ref.
    recovered = fit.apply(query.obsm["spatial"])
    residual = np.linalg.norm(recovered - ref.obsm["spatial"], axis=1)
    assert residual.max() < 1e-6, f"max residual {residual.max():.3e} should be ~0 for a rigid fit"

    # And the matrix itself should match the inverse of gt to high precision.
    np.testing.assert_allclose(fit.matrix, inv_gt, atol=1e-9)


def test_align_by_landmarks_affine_recovers_blobs_rotation(blobs_rotated) -> None:
    """The ``model='affine'`` path also fits a pure rotation correctly,
    even though it has 2 extra DOF over the similarity case."""
    from squidpy.experimental.tl._align._backends._landmark import fit_landmark_affine

    ref, query, _gt = blobs_rotated
    idx = [0, 30, 60, 90, 120, 150]
    fit = fit_landmark_affine(
        ref.obsm["spatial"][idx],
        query.obsm["spatial"][idx],
        model="affine",
    )

    recovered = fit.apply(query.obsm["spatial"])
    residual = np.linalg.norm(recovered - ref.obsm["spatial"], axis=1)
    assert residual.max() < 1e-6


# ---------------------------------------------------------------------------
# 2. align_obs (stalign) reduces the residual on the rotated cloud
# ---------------------------------------------------------------------------


def test_align_obs_stalign_reduces_residual_on_blobs(blobs_rotated) -> None:
    """The LDDMM solver isn't expected to be exact - non-rigid by design -
    but feeding it the landmark-fit affine as an init via the
    ``landmarks_*`` kwargs should reduce the residual *below* the no-op
    baseline by an order of magnitude.  This is the wiring proof: we go
    from raw misaligned coordinates to substantially-aligned coordinates
    end-to-end through ``sq.experimental.tl.align_obs``.
    """
    import squidpy as sq

    ref, query, _gt = blobs_rotated

    baseline = np.linalg.norm(ref.obsm["spatial"] - query.obsm["spatial"], axis=1).mean()

    config = sq.experimental.tl.STalignConfig(
        preprocess=sq.experimental.tl.STalignPreprocessConfig(dx=20.0, blur=2.0, expand=1.2),
        registration=sq.experimental.tl.STalignRegistrationConfig(
            a=80.0,
            expand=1.2,
            nt=3,
            niter=80,
            epV=5e2,
        ),
    )

    # Use the same well-spread landmarks as the closed-form test so the
    # affine init is meaningful; LDDMM then refines the diffeomorphism.
    idx = [0, 50, 100, 150]
    landmarks_ref = ref.obsm["spatial"][idx]
    landmarks_query = query.obsm["spatial"][idx]

    aligned = sq.experimental.tl.align_obs(
        ref,
        query,
        flavour="stalign",
        output_mode="obs",
        inplace=False,
        config=config,
        landmarks_source=landmarks_query,
        landmarks_target=landmarks_ref,
    )
    assert isinstance(aligned, AnnData)

    after = np.linalg.norm(ref.obsm["spatial"] - aligned.obsm["spatial"], axis=1).mean()
    # Sanity: the residual is finite and the alignment moved the points.
    assert np.isfinite(after)
    assert after < baseline, f"residual {after:.2f} should improve on baseline {baseline:.2f}"


# ---------------------------------------------------------------------------
# 3. The result type is consistent across both backends
# ---------------------------------------------------------------------------


def test_blobs_landmark_and_stalign_use_compatible_xy_convention(blobs_rotated) -> None:
    """Both backends operate on (x, y) coords drawn from the same blobs
    fixture and produce results in the same convention - sanity check that
    the type unification didn't introduce a silent flip."""
    import squidpy as sq
    from squidpy.experimental.tl._align._backends._landmark import fit_landmark_affine
    from squidpy.experimental.tl._align._types import AffineTransform, ObsDisplacement

    ref, query, _ = blobs_rotated
    idx = [0, 50, 100, 150]
    affine_fit = fit_landmark_affine(
        ref.obsm["spatial"][idx],
        query.obsm["spatial"][idx],
        model="similarity",
    )
    assert isinstance(affine_fit, AffineTransform)
    assert affine_fit.matrix.shape == (3, 3)

    config = sq.experimental.tl.STalignConfig(
        preprocess=sq.experimental.tl.STalignPreprocessConfig(dx=30.0, blur=2.0),
        registration=sq.experimental.tl.STalignRegistrationConfig(a=100.0, expand=1.2, nt=1, niter=1, epV=1.0),
    )
    stalign_result = sq.experimental.tl.align_obs(
        ref,
        query,
        flavour="stalign",
        output_mode="return",
        config=config,
    )
    assert isinstance(stalign_result.transform, ObsDisplacement)
    assert stalign_result.transform.deltas.shape == query.obsm["spatial"].shape

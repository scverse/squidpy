"""Happy-path integration tests for the stalign backend.

These exercise the lift from scverse/squidpy#1150 through the
``align_obs`` API and the ``output_mode`` writeback paths.  Tiny synthetic
fixtures with ``niter=1`` keep them fast enough to run on every commit; they
verify wiring and shapes only, **not** solver quality.  Numeric / visual
end-to-end verification on a rotation-recovery fixture is a separate
follow-up (see the plan file).
"""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

pytest.importorskip("jax")


def _make_xy_adata() -> AnnData:
    """Five-point synthetic AnnData with an ``obsm['spatial']`` cloud."""
    points_xy = np.array(
        [
            [10.0, 1.0],
            [12.0, 1.0],
            [11.0, 2.0],
            [10.0, 3.0],
            [12.0, 3.0],
        ]
    )
    adata = AnnData(np.zeros((points_xy.shape[0], 1)))
    adata.obsm["spatial"] = points_xy
    return adata


def _tiny_config():
    """Single-iteration LDDMM hyperparameters - smallest possible solve."""
    import squidpy as sq

    return sq.experimental.tl.STalignConfig(
        preprocess=sq.experimental.tl.STalignPreprocessConfig(dx=0.5, blur=1.0),
        registration=sq.experimental.tl.STalignRegistrationConfig(
            a=1.0,
            expand=1.0,
            nt=1,
            niter=1,
            epV=1.0,
        ),
    )


def test_align_obs_stalign_return_mode_yields_obs_displacement() -> None:
    """Wiring smoke test: dispatch -> stalign LDDMM -> ObsDisplacement."""
    import squidpy as sq
    from squidpy.experimental.tl._align._types import AlignResult, ObsDisplacement

    ref = _make_xy_adata()
    query = _make_xy_adata()
    result = sq.experimental.tl.align_obs(
        ref,
        query,
        flavour="stalign",
        output_mode="return",
        config=_tiny_config(),
    )

    assert isinstance(result, AlignResult)
    assert isinstance(result.transform, ObsDisplacement)
    assert result.transform.deltas.shape == query.obsm["spatial"].shape
    assert np.all(np.isfinite(result.transform.deltas))
    assert result.metadata["flavour"] == "stalign"
    # The escape hatch: the full STalignResult is preserved for power users.
    assert "stalign_result" in result.metadata


def test_align_obs_stalign_obs_mode_writes_new_anndata() -> None:
    """``output_mode='obs'`` materialises a new AnnData in the ref cs."""
    import squidpy as sq

    ref = _make_xy_adata()
    query = _make_xy_adata()
    aligned = sq.experimental.tl.align_obs(
        ref,
        query,
        flavour="stalign",
        output_mode="obs",
        inplace=False,
        config=_tiny_config(),
    )

    assert isinstance(aligned, AnnData)
    assert aligned.obsm["spatial"].shape == query.obsm["spatial"].shape
    # The writer stamps `align` metadata on uns so callers can introspect.
    assert "align" in aligned.uns
    assert aligned.uns["align"]["flavour"] == "stalign"


def test_align_obs_stalign_affine_mode_errors_for_non_affine_fit() -> None:
    """LDDMM is non-affine; ``output_mode='affine'`` must error helpfully."""
    import squidpy as sq

    ref = _make_xy_adata()
    query = _make_xy_adata()
    with pytest.raises(TypeError, match="requires the backend to return an AffineTransform"):
        sq.experimental.tl.align_obs(
            ref,
            query,
            flavour="stalign",
            output_mode="affine",
            config=_tiny_config(),
        )


def test_align_obs_stalign_with_landmarks() -> None:
    """Landmark-guided affine init reaches the solver via flavour_kwargs."""
    import squidpy as sq

    ref = _make_xy_adata()
    query = _make_xy_adata()
    landmarks_xy = ref.obsm["spatial"][:3]

    result = sq.experimental.tl.align_obs(
        ref,
        query,
        flavour="stalign",
        output_mode="return",
        config=_tiny_config(),
        landmarks_source=landmarks_xy,
        landmarks_target=landmarks_xy,
    )
    assert "stalign_result" in result.metadata
    assert result.transform.deltas.shape == query.obsm["spatial"].shape

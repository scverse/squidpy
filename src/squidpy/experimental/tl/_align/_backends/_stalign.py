"""STalign backend.

Wraps the JAX LDDMM solver lifted from scverse/squidpy#1150 (Selman Özleyen)
into the :class:`AlignBackend` Protocol. Only ``align_obs`` is implemented
today; ``align_images`` raises until upstream support exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from squidpy.experimental.tl._align._types import AlignPair, AlignResult


class StAlignBackend:
    name = "stalign"
    requires_jax = True

    def align_obs(
        self,
        pair: AlignPair,
        *,
        device: Literal["cpu", "gpu"] | None = None,
        config: Any | None = None,
        landmarks_source: np.ndarray | None = None,
        landmarks_target: np.ndarray | None = None,
        **kwargs: Any,
    ) -> AlignResult:
        from anndata import AnnData

        from squidpy.experimental.tl._align._jax import require_jax

        # Resolve JAX *before* importing the lifted _tools module, because
        # _tools does `import jax.numpy as jnp` at module level. If we let
        # that import fire first, callers without JAX get a confusing
        # `ModuleNotFoundError: import of jax halted; None in sys.modules`
        # instead of the clean `ImportError("JAX is required ...")` from
        # _jax.require_jax.
        require_jax(device)

        from squidpy.experimental.tl._align._backends._stalign_tools import stalign_points
        from squidpy.experimental.tl._align._types import AlignResult, ObsDisplacement

        if not isinstance(pair.ref, AnnData) or not isinstance(pair.query, AnnData):
            raise TypeError(
                "stalign backend `align_obs` only supports AnnData / table inputs; "
                f"got ref={type(pair.ref).__name__}, query={type(pair.query).__name__}."
            )
        if "spatial" not in pair.query.obsm or "spatial" not in pair.ref.obsm:
            raise KeyError("Both ref and query must carry an `obsm['spatial']` point cloud.")

        src_xy = np.asarray(pair.query.obsm["spatial"], dtype=float)
        tgt_xy = np.asarray(pair.ref.obsm["spatial"], dtype=float)

        # stalign_points runs internally in row_col (yx); obsm["spatial"] is xy
        # by squidpy convention -- swap axes at the boundary.
        src_rc = src_xy[:, [1, 0]]
        tgt_rc = tgt_xy[:, [1, 0]]
        landmarks_src_rc = None if landmarks_source is None else np.asarray(landmarks_source)[:, [1, 0]]
        landmarks_tgt_rc = None if landmarks_target is None else np.asarray(landmarks_target)[:, [1, 0]]

        stalign_result = stalign_points(
            source_points=src_rc,
            target_points=tgt_rc,
            config=config,
            landmarks_source=landmarks_src_rc,
            landmarks_target=landmarks_tgt_rc,
        )

        aligned_rc = np.asarray(stalign_result.aligned_points)
        aligned_xy = aligned_rc[:, [1, 0]]
        deltas_xy = aligned_xy - src_xy

        return AlignResult(
            transform=ObsDisplacement(
                deltas=deltas_xy,
                source_cs=pair.query_cs,
                target_cs=pair.ref_cs,
            ),
            metadata={
                "flavour": "stalign",
                # Escape hatch: the full STalignResult (velocity field,
                # velocity grid, affine init) for power users who need
                # the diffeomorphic map.  This keeps the JAX arrays alive
                # in memory -- callers who only need the displacement
                # should drop this key or use ``output_mode='obs'``.
                "stalign_result": stalign_result,
            },
        )

    def align_images(
        self,
        pair: AlignPair,
        *,
        device: Literal["cpu", "gpu"] | None = None,
        **kwargs: Any,
    ) -> AlignResult:
        raise NotImplementedError(
            "stalign image alignment is not yet implemented; PR #1150 only ships "
            "point-cloud alignment. Use `flavour='stalign'` with `align_obs`."
        )

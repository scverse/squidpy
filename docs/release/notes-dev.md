# Squidpy dev (the-future)

## Features

- {func}`squidpy.experimental.im.calculate_image_features` now featurizes tiles on a shared dask engine: `n_jobs > 1` runs worker processes via a `dask.distributed.LocalCluster` (or an active `Client`), and per-tile BLAS/OpenMP threads are pinned to avoid oversubscription. This also speeds up the serial path. {func}`squidpy.experimental.tl.calculate_tiling_qc` shares the same engine. Adds `distributed` and `threadpoolctl` as dependencies.
- Add a numerical reference suite for the experimental STalign port, comparing it against the
  original PyTorch implementation. Deselected by default; runs on the scheduled job. Reference
  values are generated out of band by
  [theislab/squidpy-ports](https://github.com/theislab/squidpy-ports), so `torch` is not a squidpy
  dependency.
  [#1243](https://github.com/scverse/squidpy/issues/1243)
- Add experimental sample alignment, one public function per method (the
  `calculate_niche_*` shape): {func}`squidpy.experimental.tl.align_stalign_obs` aligns
  point clouds and {func}`squidpy.experimental.tl.align_stalign_image` aligns images
  with the STalign diffeomorphic solver; {func}`squidpy.experimental.tl.align_landmarks`
  fits a closed-form `"similarity"` or `"affine"` transform from paired landmarks. Data
  is addressed with conventional key arguments (`spatial_key`, `table_key`, `image_key`,
  `landmark_key`), each accepting a `(ref, query)` pair; `key_added=None` (the default)
  returns the fitted alignment and writes nothing, and `inplace=False` writes into a
  returned copy. LDDMM solver tuning is passed as flat, typed keyword arguments
  (`squidpy.experimental.methods.StalignSolverKwargs`). A landmark fit can either
  transform coordinates (`spatial_key` + `key_added`) or be registered on a whole
  SpatialData coordinate system (`target_coordinate_system`); registration refuses when
  the reference shares the query's coordinate system, since it would be dragged along.
  A fitted diffeomorphism has no SpatialData transformation type, so
  `align_stalign_image` materialises the warped image instead. The array-in/array-out
  estimators live in `squidpy.experimental.methods`.
- The experimental STalign solver now runs its whole gradient descent as a single compiled
  `lax.while_loop` instead of a Python loop around a jitted step, about **4.6x faster** per
  iteration (2.20 to 0.46 ms on the reference fixture, so `niter=5000` drops from ~11s to ~2.4s).
  Numerically unchanged: the reference suite still matches the original implementation at 1, 5, 50
  and 500 iterations.
- The reference suite now covers `fit_stalign_image`, the last part of the experimental
  STalign port without one. It reproduces the original to ~4e-12 on the affine and ~2e-12 on the
  velocity field over a full trajectory.
- The experimental STalign solver returns the per-iteration `energies` trace and `n_iter`, and
  accepts optional `tol` / `patience` early stopping. Off by default. Note the objective changes
  definition at iteration 50, when the mixture-weight E step engages, so the convergence window
  deliberately never spans that point.

## Bugfixes

- The experimental STalign estimator no longer differentiates through the contrast-transform ridge
  solve. That solve is an expectation-maximisation M step and must be held constant; treating it as
  part of the objective changed the search direction. Gradients now agree with the original
  implementation to ~1e-15, previously ~1e-3.
- **Breaking (experimental):** the rasterisation and velocity grids in the experimental STalign
  estimator were one sample longer per axis than intended, and their length varied with
  floating-point rounding. Output shapes change accordingly.
- The experimental STalign rasteriser now deposits each point bilinearly rather than snapping it to
  the nearest cell, and conserves mass at the image border. Relative error against the original
  implementation drops from 6.2 %/2.0 %/6.0 % to 4.1 %/0.8 %/2.9 % across the default blur scales.
- `lddmm(niter=0)` no longer raises `UnboundLocalError`.

- Fix {func}`squidpy.tl.var_by_distance` behaviour when providing {mod}`numpy` arrays of coordinates as anchor point.
- Update :attr:`squidpy.pl.var_by_distance` to show multiple variables on same plot.
  [@LLehner](https://github.com/LLehner)
  [#929](https://github.com/scverse/squidpy/pull/929)

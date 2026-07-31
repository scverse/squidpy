# Squidpy dev (the-future)

## Features

- {func}`squidpy.experimental.im.calculate_image_features` now featurizes tiles on a shared dask engine: `n_jobs > 1` runs worker processes via a `dask.distributed.LocalCluster` (or an active `Client`), and per-tile BLAS/OpenMP threads are pinned to avoid oversubscription. This also speeds up the serial path. {func}`squidpy.experimental.tl.calculate_tiling_qc` shares the same engine. Adds `distributed` and `threadpoolctl` as dependencies.
- Add a numerical reference suite for the experimental STalign port, comparing it against the
  original PyTorch implementation. Deselected by default; runs on the scheduled job. Reference
  values are generated out of band by
  [theislab/squidpy-ports](https://github.com/theislab/squidpy-ports), so `torch` is not a squidpy
  dependency.
  [#1243](https://github.com/scverse/squidpy/issues/1243)
- **Breaking (experimental):** {func}`squidpy.experimental.tl.align` now locates data with `in_`
  and `out` paths instead of a stack of key arguments. `ref_key`, `query_key`, `spatial_key`,
  `key_added` and `output_mode` are replaced by `in_` (e.g. `"obsm/spatial"`,
  `"tables/slice1/obsm/spatial"`, `"images/he"`, `"shapes/landmarks"`), `out`, and `copy`.
  `out=None` (the default) returns the fitted alignment and writes nothing. This follows the shape
  proposed for scanpy in [scanpy#4007](https://github.com/scverse/scanpy/issues/4007).
- **Breaking (experimental):** `squidpy.experimental.tl.align_by_landmarks` is folded into
  {func}`squidpy.experimental.tl.align` as `by="landmarks"`. `on` becomes `by`, gaining a
  `"landmarks"` value alongside `"obs"` and `"images"`; `in_` then names the correspondences and
  `apply_to` names what moves. Writing to `out="cs/<name>"` registers the fitted affine on a whole
  SpatialData coordinate system instead of materialising anything.
- **Breaking (experimental):** the `align_samples`, `align_images` and `align_landmarks` registries
  collapse into a single `ALIGN` registry of `AlignMethod` records, each declaring which modalities
  it implements. Asking for one a method does not support now fails immediately and says what it
  does support. `fit_stalign` is renamed `fit_stalign_obs` for symmetry with `fit_stalign_image`.
- {func}`squidpy.experimental.tl.align` can now align on images. The fitted diffeomorphism cannot
  be expressed as a SpatialData transformation, so writing to an `images/...` path materialises the
  warped image rather than registering it lazily. Adds an `align_images` method family and
  `squidpy.experimental.methods.align_samples.fit_stalign_image`.
- The experimental STalign solver now runs its whole gradient descent as a single compiled
  `lax.while_loop` instead of a Python loop around a jitted step, about **4.6x faster** per
  iteration (2.20 to 0.46 ms on the reference fixture, so `niter=5000` drops from ~11s to ~2.4s).
  Numerically unchanged: the reference suite still matches the original implementation at 1, 5, 50
  and 500 iterations.
- The experimental STalign solver returns the per-iteration `energies` trace and `n_iter`, and
  accepts optional `tol` / `patience` early stopping. Off by default. Note the objective changes
  definition at iteration 50, when the mixture-weight E step engages, so the convergence window
  deliberately never spans that point.
- **Breaking (experimental):** {func}`squidpy.experimental.tl.align_by_landmarks` takes `in_` /
  `out` / `copy` in place of `spatial_key` / `key_added` / `output_mode`, and its coordinate-system
  arguments are renamed `cs_ref` / `cs_query`. Because `out` is always named explicitly, the guard
  that refused to overwrite an auto-derived key is gone.

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

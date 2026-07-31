# Squidpy dev (the-future)

## Features

- {func}`squidpy.experimental.im.calculate_image_features` now featurizes tiles on a shared dask engine: `n_jobs > 1` runs worker processes via a `dask.distributed.LocalCluster` (or an active `Client`), and per-tile BLAS/OpenMP threads are pinned to avoid oversubscription. This also speeds up the serial path. {func}`squidpy.experimental.tl.calculate_tiling_qc` shares the same engine. Adds `distributed` and `threadpoolctl` as dependencies.
- Add a numerical reference suite for the experimental STalign port, comparing it against the
  original PyTorch implementation. Deselected by default; runs on the scheduled job. Reference
  values are generated out of band by
  [scverse/squidpy-ports](https://github.com/scverse/squidpy-ports), so `torch` is not a squidpy
  dependency.
  [#1243](https://github.com/scverse/squidpy/issues/1243)

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

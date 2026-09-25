# Squidpy dev (the-future)

## Features

- {func}`squidpy.experimental.im.calculate_image_features` now featurizes tiles on a shared dask engine: `n_jobs > 1` runs worker processes via a `dask.distributed.LocalCluster` (or an active `Client`), and per-tile BLAS/OpenMP threads are pinned to avoid oversubscription. This also speeds up the serial path. {func}`squidpy.experimental.tl.calculate_tiling_qc` shares the same engine. Adds `distributed` and `threadpoolctl` as dependencies.
- Fix {func}`squidpy.tl.var_by_distance` behaviour when providing {mod}`numpy` arrays of coordinates as anchor point.
- Update :attr:`squidpy.pl.var_by_distance` to show multiple variables on same plot.
  [@LLehner](https://github.com/LLehner)
  [#929](https://github.com/scverse/squidpy/pull/929)

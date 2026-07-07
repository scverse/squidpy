# Squidpy dev (the-future)

## Breaking changes

- The dataset registry and downloader now come from
  [scverse-misc](https://github.com/scverse/scverse-misc); downloads cache under
  `<datasetdir>/<type>/`. Visium samples move from `visium/` to `visium_10x/` and
  images from `images/` to `image/`, so existing caches are re-downloaded once into
  the new layout. `sq.datasets.*` signatures and return types are unchanged.
  [#1213](https://github.com/scverse/squidpy/pull/1213)

## Features

- Fix {func}`squidpy.tl.var_by_distance` behaviour when providing {mod}`numpy` arrays of coordinates as anchor point.
- Update :attr:`squidpy.pl.var_by_distance` to show multiple variables on same plot.
  [@LLehner](https://github.com/LLehner)
  [#929](https://github.com/scverse/squidpy/pull/929)

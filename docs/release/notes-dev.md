# Squidpy dev (the-future)

## Features

- Add {func}`squidpy.experimental.tl.assign_stitch_groups` (with {class}`squidpy.experimental.tl.StitchParams`)
  to group tile-cut cell pieces flagged by {func}`squidpy.experimental.tl.calculate_tiling_qc`, scoring candidate
  pairs with a transparent weighted mean of five geometric features.
  [@timtreis](https://github.com/timtreis)
- Fix {func}`squidpy.tl.var_by_distance` behaviour when providing {mod}`numpy` arrays of coordinates as anchor point.
- Update :attr:`squidpy.pl.var_by_distance` to show multiple variables on same plot.
  [@LLehner](https://github.com/LLehner)
  [#929](https://github.com/scverse/squidpy/pull/929)

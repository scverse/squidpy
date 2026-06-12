# Squidpy dev (the-future)

## Features

- Add {func}`squidpy.experimental.im.make_stitched_labels` to materialise a stitched labels element (and an optional collapsed table) from an {func}`squidpy.experimental.tl.assign_stitch_groups` result, completing the tile-cut stitching workflow.
- Fix {func}`squidpy.tl.var_by_distance` behaviour when providing {mod}`numpy` arrays of coordinates as anchor point.
- Update :attr:`squidpy.pl.var_by_distance` to show multiple variables on same plot.
  [@LLehner](https://github.com/LLehner)
  [#929](https://github.com/scverse/squidpy/pull/929)

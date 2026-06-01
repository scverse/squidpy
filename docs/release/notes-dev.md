# Squidpy dev (the-future)

## Features

- Fix {func}`squidpy.tl.var_by_distance` behaviour when providing {mod}`numpy` arrays of coordinates as anchor point.
- Update :attr:`squidpy.pl.var_by_distance` to show multiple variables on same plot.
  [@LLehner](https://github.com/LLehner)
  [#929](https://github.com/scverse/squidpy/pull/929)

## Bugfixes

- Use the correct normality variance for Geary's C in {func}`squidpy.gr.spatial_autocorr`,
  fixing a miscalibrated analytic p-value (`pval_norm`) for `mode="geary"`.
  [#1183](https://github.com/scverse/squidpy/issues/1183)

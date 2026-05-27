# Squidpy 1.3.0 (2023-06-16)

## Features

- Add {func}`squidpy.tl.var_by_distance` to calculate distances to anchor points and store results in a design matrix.
- Add {func}`squidpy.pl.var_by_distance` to visualize variables such as gene expression by distance to an anchor point.
  [@LLehner](https://github.com/LLehner)
  [#591](https://github.com/scverse/squidpy/pull/591)


## Bugfixes

- Fix {mod}`pandas` inf {func}`squidpy.pl.ligrec`.
  [@michalk8](https://github.com/michalk8)
  [#625](https://github.com/scverse/squidpy/pull/625)

- Remove column assignment to improve compatibility with new cell metadata.
  [@cornhundred](https://github.com/cornhundred)
  [#648](https://github.com/scverse/squidpy/pull/648)

- Fix {func}`squidpy.pl.extract` on views.
  [@michalk8](https://github.com/michalk8)
  [#663](https://github.com/scverse/squidpy/pull/663)

- Set coordinates' index type to same as in :attr:`anndata.AnnData.obs` in {func}`squidpy.read.vizgen`
  and {func}`squidpy.read.visium`.
  [@michalk8](https://github.com/michalk8)
  [#665](https://github.com/scverse/squidpy/pull/665)

- Update cell metadata index conversion.
  [@djlee1](https://github.com/djlee1)
  [#679](https://github.com/scverse/squidpy/pull/679)

- Fix previously updated cell metadata index conversion.
  [@dfhannum](https://github.com/dfhannum)
  [#692](https://github.com/scverse/squidpy/pull/692)


## Miscellaneous

- Update pre-commits and unpin numba and numpy.
  [@giovp](https://github.com/giovp)
  [#643](https://github.com/scverse/squidpy/pull/643)

- Add `attr`: option to {func}`squidpy.gr.spatial_autocorr` to select values from :attr:`anndata.AnnData.obs` or :attr:`anndata.AnnData.obsm`.
  [@michalk8](https://github.com/michalk8)
  [#664](https://github.com/scverse/squidpy/pull/664), [#672](https://github.com/scverse/squidpy/pull/672)

- Add `percentile` option to {func}`squidpy.gr.spatial_neighbors` to filter neighbor graph using percentile of distances threshold.
  [@LLehner](https://github.com/LLehner)
  [#690](https://github.com/scverse/squidpy/pull/690)

- Add {class}`spatialdata.SpatialData` as possible input for graph functions.
  [@LLehner](https://github.com/LLehner)
  [#701](https://github.com/scverse/squidpy/pull/701)


## Documentation

- Fix CI badges and tox.
  [@michalk8](https://github.com/michalk8)
  [#627](https://github.com/scverse/squidpy/pull/627)

- Changed tutorial directory structure.
  [@LLehner](https://github.com/LLehner)
  [#113](https://github.com/scverse/squidpy_notebooks/pull/113)

- Updated the quality control tutorials for Vizgen, Xenium and Nanostring.
  [@pakiessling](https://github.com/pakiessling)
  [#110](https://github.com/scverse/squidpy_notebooks/pull/110)

- Improved example for {func}`squidpy.tl.var_by_distance` and {func}`squidpy.pl.var_by_distance`.
  [@LLehner](https://github.com/LLehner)
  [#115](https://github.com/scverse/squidpy_notebooks/pull/115)

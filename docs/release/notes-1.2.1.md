# Squidpy 1.2.1 (2022-05-14)

## Features

- Refactor :meth:`squidpy.im.ImageContainer.subset` to return a view.
  [@michalk8](https://github.com/michalk8)
  [#534](https://github.com/scverse/squidpy/pull/534)


## Bugfixes

- Add discourse, zulip badges and links.
  [@giovp](https://github.com/giovp)
  [#525](https://github.com/scverse/squidpy/pull/525)

- Fix not correctly subsetting {class}`anndata.AnnData` when interactively visualizing it.
  [@michalk8](https://github.com/michalk8)
  [#531](https://github.com/scverse/squidpy/pull/531)

- Close #536. Set consistent image resolution key in {func}`squidpy.read.visium`.
  [@giovp](https://github.com/giovp)
  [#537](https://github.com/scverse/squidpy/pull/537)

- Fix alpha in {func}`squidpy.pl.spatial_scatter` when keys are categorical.
  [@michalk8](https://github.com/michalk8)
  [#542](https://github.com/scverse/squidpy/pull/542)

- {func}`squidpy.read.nanostring` reads only image file extensions.
  [@dineshpalli](https://github.com/dineshpalli)
  [#546](https://github.com/scverse/squidpy/pull/546)

- Return ``cell_id`` for segmentation masks in {func}`squidpy.read.nanostring`.
  [@giovp](https://github.com/giovp)
  [#547](https://github.com/scverse/squidpy/pull/547)

- Add prettier pre-commit check, remove python 3.7 and add mac-os python 3.9 .
  [@giovp](https://github.com/giovp)
  [#548](https://github.com/scverse/squidpy/pull/548)

- Rename default branch from ``master`` to ``main``.
  [@giovp](https://github.com/giovp)
  [#549](https://github.com/scverse/squidpy/pull/549)


## Miscellaneous

- Fix news fragment generation checks.
  [@michalk8](https://github.com/michalk8)
  [#550](https://github.com/scverse/squidpy/pull/550)

Squidpy 1.2.1 (2022-05-14)
==========================

Features
--------

- Refactor :meth:`squidpy.im.ImageContainer.subset` to return a view.
  `@michalk8 <https://github.com/michalk8>`__
  `#534 <https://github.com/scverse/squidpy/pull/534>`__


Bugfixes
--------

- Add discourse, zulip badges and links.
  `@giovp <https://github.com/giovp>`__
  `#525 <https://github.com/scverse/squidpy/pull/525>`__

- Fix not correctly subsetting :class:`anndata.AnnData` when interactively visualizing it.
  `@michalk8 <https://github.com/michalk8>`__
  `#531 <https://github.com/scverse/squidpy/pull/531>`__

- Close #536. Set consistent image resolution key in :func:`squidpy.read.visium`.
  `@giovp <https://github.com/giovp>`__
  `#537 <https://github.com/scverse/squidpy/pull/537>`__

- Fix alpha in :func:`squidpy.pl.spatial_scatter` when keys are categorical.
  `@michalk8 <https://github.com/michalk8>`__
  `#542 <https://github.com/scverse/squidpy/pull/542>`__

- :func:`squidpy.read.nanostring` reads only image file extensions.
  `@dineshpalli <https://github.com/dineshpalli>`__
  `#546 <https://github.com/scverse/squidpy/pull/546>`__

- Return ``cell_id`` for segmentation masks in :func:`squidpy.read.nanostring`.
  `@giovp <https://github.com/giovp>`__
  `#547 <https://github.com/scverse/squidpy/pull/547>`__

- Add prettier pre-commit check, remove python 3.7 and add mac-os python 3.9 .
  `@giovp <https://github.com/giovp>`__
  `#548 <https://github.com/scverse/squidpy/pull/548>`__

- Rename default branch from ``master`` to ``main``.
  `@giovp <https://github.com/giovp>`__
  `#549 <https://github.com/scverse/squidpy/pull/549>`__


Miscellaneous
-------------

- Fix news fragment generation checks.
  `@michalk8 <https://github.com/michalk8>`__
  `#550 <https://github.com/scverse/squidpy/pull/550>`__

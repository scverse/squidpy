Squidpy dev (2022-05-13)
========================

Features
--------

- Refactor :meth:`squidpy.im.ImageContainer.subset` to return a view.
  `@michalk8 <https://github.com/michalk8>`__
  `#534 <https://github.com/theislab/squidpy/pull/534>`__


Bugfixes
--------

- Add discourse, zulip badges and links.
  `@giovp <https://github.com/giovp>`__
  `#525 <https://github.com/theislab/squidpy/pull/525>`__

- Fix not correctly subsetting :class:`anndata.AnnData` when interactively visualizing it.
  `@michalk8 <https://github.com/michalk8>`__
  `#531 <https://github.com/theislab/squidpy/pull/531>`__

- Close #536. Set consistent image resolution key in :func:`squidpy.read.visium`.
  `@giovp <https://github.com/giovp>`__
  `#537 <https://github.com/theislab/squidpy/pull/537>`__

- Fix alpha in :func:`squidpy.pl.spatial_scatter` when keys are categorical.
  `@michalk8 <https://github.com/michalk8>`__
  `#542 <https://github.com/theislab/squidpy/pull/542>`__

- ``sq.read.nanostring`` reads only image file extensions.
  `@dineshpalli <https://github.com/dineshpalli>`__
  `#546 <https://github.com/theislab/squidpy/pull/546>`__

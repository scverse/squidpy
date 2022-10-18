Squidpy dev (2022-10-18)
========================

Bugfixes
--------

- Fix plotting non-unique categorical colors in :func:`squidpy.pl.spatial_scatter`.
  `@michalk8 <https://github.com/michalk8>`__
  `#561 <https://github.com/scverse/squidpy/pull/561>`__

- Fixes :func:`squidpy.read.nanostring` . Closes #566 .
  `@giovp <https://github.com/giovp>`__
  `#567 <https://github.com/scverse/squidpy/pull/567>`__

- Fix :func:`squidpy.read.vizgen`.
  `@giovp <https://github.com/giovp>`__
  `#568 <https://github.com/scverse/squidpy/pull/568>`__

- Fix passing :class:`matplotlib.colors.ListedColorMap` as palette to
  :func:`squidpy.pl.spatial_scatter`.
  `@michalk8 <https://github.com/michalk8>`__
  `#580 <https://github.com/scverse/squidpy/pull/580>`__

- This PR updates squidpy to accommodate the latest changes made in spaceranger 2.0 which will break
  the released version of squidpy. Will provide backwards compatibility to pre 2.0 releases.
  `@stephenwilliams22 <https://github.com/stephenwilliams22>`__
  `#583 <https://github.com/scverse/squidpy/pull/583>`__


Miscellaneous
-------------

- Better error message for handling palette in  :func:`squidpy.pl.spatial_scatter`.
  `@giovp <https://github.com/giovp>`__
  `#562 <https://github.com/scverse/squidpy/pull/562>`__


Documentation
-------------

- Add tutorials on analysis of Vizgen and Nanostring data.
  Remove reference of ``scanpy.pl.spatial`` plotting in examples.
  `@dineshpalli <https://github.com/dineshpalli>`__
  `#569 <https://github.com/scverse/squidpy/pull/569>`__

- New tutorial for 10x Genomics Xenium data.
  `@LLehner <https://github.com/LLehner>`__
  `#615 <https://github.com/scverse/squidpy/pull/615>`__

- New tutorial for Vizgen mouse liver data.
  `@cornhundred <https://github.com/cornhundred>`__
  `#616 <https://github.com/scverse/squidpy/pull/616>`__

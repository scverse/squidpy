Squidpy dev (2022-08-02)
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

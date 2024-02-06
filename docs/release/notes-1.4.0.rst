Squidpy 1.4.0 (2024-02-05)
==========================

Bugfixes
--------

- Fix building graph in ``knn`` and ``delaunay`` mode.
  `@michalk8 <https://github.com/michalk8>`__
  `#792 <https://github.com/scverse/squidpy/pull/792>`__

- Correct shuffling of annotations in ``sq.gr.nhood_enrichment``.
  `@giovp <https://github.com/giovp>`__
  `#775 <https://github.com/scverse/squidpy/pull/775>`__


Miscellaneous
-------------

- Fix napari installation.
  `@giovp <https://github.com/giovp>`__
  `#767 <https://github.com/scverse/squidpy/pull/767>`__

- Made nanostring reader more flexible by adjusting loading of images.
  `@FrancescaDr <https://github.com/FrancescaDr>`__
  `#766 <https://github.com/scverse/squidpy/pull/766>`__

- Fix ``sq.tl.var_by_distance`` method to support ``pandas 2.2.0``.
  `@LLehner <https://github.com/LLehner>`__
  `#794 <https://github.com/scverse/squidpy/pull/794>`__

Development Version
===================

Features
--------

- Add an experimental, opt-in ``spatialdata-plot`` delegation backend for
  :func:`squidpy.pl.spatial_scatter` and :func:`squidpy.pl.spatial_segment`,
  enabled with the ``SQUIDPY_USE_SDATAPLOT=1`` environment variable. It accepts
  native :class:`spatialdata.SpatialData` input (in addition to AnnData) and
  renders through ``spatialdata-plot`` instead of the legacy matplotlib path.
  On SpatialData input, ``shapes_layer`` / ``labels_layer`` / ``points_layer`` /
  ``image_layer`` / ``table`` select the element to render when a coordinate
  system holds more than one candidate.

Deprecations
------------

- Passing an :class:`anndata.AnnData` to the spatial plotting functions now emits
  a :class:`DeprecationWarning` under the delegation backend; pass a
  :class:`spatialdata.SpatialData` instead. AnnData input is slated for removal in
  squidpy v2.0.

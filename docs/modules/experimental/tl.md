# Tools `tl`

#### Alignment

```{eval-rst}
.. module:: squidpy.experimental.tl
.. currentmodule:: squidpy.experimental
.. autosummary::
    :toctree: ../../api

    tl.stalign_align_obs
    tl.stalign_align_image
    tl.stalign_align_volume
    tl.align_landmarks
    tl.apply_affine
```

#### Fits

What an alignment returns: a frozen object carrying the operations that apply it. One class
per entry point, because what a fit can do follows from what it was fitted from -- only the
two that carry a raster frame offer ``deformation_grid``, and only the rank-2 image fit
offers ``warp_image``.

```{eval-rst}
.. currentmodule:: squidpy.experimental
.. autosummary::
    :toctree: ../../api

    tl.StalignFit
    tl.StalignObsFit
    tl.StalignImageFit
    tl.StalignVolumeFit
```

#### Tiling and stitching

```{eval-rst}
.. currentmodule:: squidpy.experimental
.. autosummary::
    :toctree: ../../api

    tl.calculate_tiling_qc
    tl.assign_stitch_groups
    tl.make_stitched_labels
```

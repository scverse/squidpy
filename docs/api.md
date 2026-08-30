# API

```{eval-rst}
.. module:: squidpy
```

Import Squidpy as:

```python
import squidpy as sq
```

## Graph `gr`
```{eval-rst}
.. module:: squidpy.gr
.. currentmodule:: squidpy
.. autosummary::
    :toctree: api

    gr.spatial_neighbors
    gr.spatial_neighbors_from_builder
    gr.spatial_neighbors_knn
    gr.spatial_neighbors_radius
    gr.spatial_neighbors_delaunay
    gr.spatial_neighbors_grid
    gr.mask_graph
    gr.nhood_enrichment
    gr.co_occurrence
    gr.centrality_scores
    gr.interaction_matrix
    gr.ripley
    gr.ligrec
    gr.spatial_autocorr
    gr.sepal
    gr.calculate_niche
    gr.calculate_niche_neighborhood
    gr.calculate_niche_utag
    gr.calculate_niche_cellcharter
    gr.calculate_niche_spatialleiden
```

### **neighbors**

See the {doc}`extensibility guide </extensibility>` for how to implement a custom graph
builder. ``GraphMatrixT`` is the type variable those interfaces are generic over; it is
documented here rather than beside the ``gr`` functions, where a bare type variable read as
public API.

```{eval-rst}
.. module:: squidpy.gr.neighbors
.. currentmodule:: squidpy.gr
.. autosummary::
    :toctree: api

    neighbors.GraphBuilder
    neighbors.GraphBuilderCSR
    neighbors.GraphPostprocessor
    neighbors.DistanceIntervalPostprocessor
    neighbors.PercentilePostprocessor
    neighbors.TransformPostprocessor
    neighbors.KNNBuilder
    neighbors.RadiusBuilder
    neighbors.DelaunayBuilder
    neighbors.GridBuilder
    neighbors.GraphMatrixT
```

## Image `im`
```{eval-rst}
.. module:: squidpy.im
.. currentmodule:: squidpy
.. autosummary::
    :toctree: api

    im.process
    im.segment
    im.calculate_image_features
    im.SegmentationModel
```

## Plotting `pl`
```{eval-rst}
.. module:: squidpy.pl
.. currentmodule:: squidpy
.. autosummary::
    :toctree: api

    pl.spatial_scatter
    pl.spatial_segment
    pl.nhood_enrichment
    pl.centrality_scores
    pl.interaction_matrix
    pl.ligrec
    pl.ripley
    pl.co_occurrence
    pl.extract
    pl.var_by_distance
```

## Reading `read`
```{eval-rst}
.. module:: squidpy.read
.. currentmodule:: squidpy
.. autosummary::
    :toctree: api

    read.visium
    read.vizgen
    read.nanostring
```

## Tools `tl`
```{eval-rst}
.. module:: squidpy.tl
.. currentmodule:: squidpy
.. autosummary::
    :toctree: api

    tl.sliding_window
    tl.var_by_distance
```

## Datasets `datasets`
```{eval-rst}
.. module:: squidpy.datasets
.. currentmodule:: squidpy
.. autosummary::
    :toctree: api

    datasets.four_i
    datasets.imc
    datasets.seqfish
    datasets.merfish
    datasets.mibitof
    datasets.slideseqv2
    datasets.sc_mouse_cortex
    datasets.visium
    datasets.visium_hne_adata
    datasets.visium_hne_adata_crop
    datasets.visium_fluo_adata
    datasets.visium_fluo_adata_crop
    datasets.visium_hne_image
    datasets.visium_hne_image_crop
    datasets.visium_fluo_image_crop
```

## Experimental `experimental`

```{eval-rst}
.. module:: squidpy.experimental
```

Under active development: names and signatures here may change without a deprecation cycle.
Laid out by submodule, the way the stable API is.

### Images `im`
#### Features, tiling and rasterization
```{eval-rst}
.. module:: squidpy.experimental.im
.. currentmodule:: squidpy.experimental
.. autosummary::
    :toctree: api

    im.calculate_image_features
    im.rasterize_points
    im.sample_volume
    im.make_tiles
    im.make_tiles_from_spots
```

#### Quality control
```{eval-rst}
.. currentmodule:: squidpy.experimental
.. autosummary::
    :toctree: api

    im.qc_image
```

#### Tissue detection
```{eval-rst}
.. currentmodule:: squidpy.experimental
.. autosummary::
    :toctree: api

    im.detect_tissue
```

#### Stain normalization
```{eval-rst}
.. currentmodule:: squidpy.experimental
.. autosummary::
    :toctree: api

    im.fit_stain_reference
    im.normalize_stains
    im.decompose_stains
    im.estimate_white_point
    im.StainReference
```

### Tools `tl`
#### Alignment
```{eval-rst}
.. module:: squidpy.experimental.tl
.. currentmodule:: squidpy.experimental
.. autosummary::
    :toctree: api

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
    :toctree: api

    tl.StalignFit
    tl.StalignObsFit
    tl.StalignImageFit
    tl.StalignVolumeFit
```

#### Tiling and stitching
```{eval-rst}
.. currentmodule:: squidpy.experimental
.. autosummary::
    :toctree: api

    tl.calculate_tiling_qc
    tl.assign_stitch_groups
    tl.make_stitched_labels
```

### Plotting `pl`
```{eval-rst}
.. module:: squidpy.experimental.pl
.. currentmodule:: squidpy.experimental
.. autosummary::
    :toctree: api

    pl.tiling_qc
    pl.qc_image
```

## Types `types`

The parameter bags and result tuples, collected by kind rather than by domain. Anything
carrying behaviour is a fit and lives with the functions that produce it.

### Parameters

```{eval-rst}
.. module:: squidpy.types
.. currentmodule:: squidpy
.. autosummary::
    :toctree: api

    types.StalignObsParams
    types.StalignImageParams
    types.StalignVolumeParams
    types.TilingQCParams
    types.StitchParams
    types.BackgroundDetectionParams
    types.FelzenszwalbParams
    types.WekaParams
    types.ReinhardParams
    types.MacenkoParams
    types.VahadaneParams
```

### Results

```{eval-rst}
.. currentmodule:: squidpy
.. autosummary::
    :toctree: api

    types.SpatialNeighborsResult
    types.NhoodEnrichmentResult
```

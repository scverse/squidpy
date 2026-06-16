# API

```{eval-rst}
.. module:: squidpy
```

Import Squidpy as:

```python
import squidpy as sq
```

## Graph
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
    gr.GraphMatrixT
    gr.SpatialNeighborsResult
    gr.mask_graph
    gr.nhood_enrichment
    gr.NhoodEnrichmentResult
    gr.co_occurrence
    gr.centrality_scores
    gr.interaction_matrix
    gr.ripley
    gr.ligrec
    gr.spatial_autocorr
    gr.sepal
    gr.calculate_niche
```

## Image
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

## Plotting
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

## Reading
```{eval-rst}
.. module:: squidpy.read
.. currentmodule:: squidpy
.. autosummary::
    :toctree: api

    read.visium
    read.vizgen
    read.nanostring
```

## Tools
```{eval-rst}
.. module:: squidpy.tl
.. currentmodule:: squidpy
.. autosummary::
    :toctree: api

    tl.sliding_window
    tl.var_by_distance
```

## Datasets
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

## Extensibility

See the {doc}`extensibility guide </extensibility>` for how to implement a custom graph builder.

```{eval-rst}
.. module:: squidpy.gr.neighbors
.. currentmodule:: squidpy
.. autosummary::
    :toctree: api

    gr.neighbors.GraphBuilder
    gr.neighbors.GraphBuilderCSR
    gr.neighbors.GraphMatrixT
    gr.neighbors.GraphPostprocessor
    gr.neighbors.DistanceIntervalPostprocessor
    gr.neighbors.PercentilePostprocessor
    gr.neighbors.TransformPostprocessor
    gr.neighbors.KNNBuilder
    gr.neighbors.RadiusBuilder
    gr.neighbors.DelaunayBuilder
    gr.neighbors.GridBuilder
```

## Experimental
```{eval-rst}
.. module:: squidpy.experimental
.. currentmodule:: squidpy
.. autosummary::
    :toctree: api

    experimental.im.calculate_image_features
    experimental.tl.calculate_tiling_qc
    experimental.tl.TilingQCParams
    experimental.tl.align
    experimental.tl.align_by_landmarks
    experimental.tl.AlignResult
    experimental.tl.assign_stitch_groups
    experimental.tl.StitchParams
    experimental.pl.tiling_qc
    experimental.im.fit_stain_reference
    experimental.im.normalize_stains
    experimental.im.decompose_stains
    experimental.im.estimate_white_point
    experimental.im.StainReference
    experimental.im.ReinhardParams
    experimental.im.MacenkoParams
    experimental.im.VahadaneParams
```

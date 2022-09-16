API
===

Import Squidpy as::

    import squidpy as sq

Graph
~~~~~

.. module:: squidpy.gr
.. currentmodule:: squidpy

.. autosummary::
    :toctree: api

    gr.spatial_neighbors
    gr.nhood_enrichment
    gr.co_occurrence
    gr.centrality_scores
    gr.interaction_matrix
    gr.ripley
    gr.ligrec
    gr.spatial_autocorr
    gr.sepal

Image
~~~~~

.. module:: squidpy.im
.. currentmodule:: squidpy

.. autosummary::
    :toctree: api

    im.process
    im.segment
    im.calculate_image_features

Plotting
~~~~~~~~

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

Reading
~~~~~~~

.. module:: squidpy.read
.. currentmodule:: squidpy

.. autosummary::
    :toctree: api

    read.visium
    read.vizgen
    read.nanostring

Datasets
~~~~~~~~

.. module:: squidpy.datasets
.. currentmodule:: squidpy

.. autosummary::
    :toctree: api

    four_i
    imc
    seqfish
    merfish
    mibitof
    slideseqv2
    sc_mouse_cortex
    visium
    visium_hne_adata
    visium_hne_adata_crop
    visium_fluo_adata
    visium_fluo_adata_crop
    visium_hne_image
    visium_hne_image_crop
    visium_fluo_image_cro

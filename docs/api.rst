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
    pl.var_by_distance

Reading
~~~~~~~

.. module:: squidpy.read
.. currentmodule:: squidpy

.. autosummary::
    :toctree: api

    read.visium
    read.vizgen
    read.nanostring

Tools
~~~~~~~~

.. module:: squidpy.tl
.. currentmodule:: squidpy

.. autosummary::
    :toctree: api

    tl.var_by_distance

Datasets
~~~~~~~~

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

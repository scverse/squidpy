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
    gr.centrality_scores
    gr.interaction_matrix
    gr.ligrec
    gr.spatial_autocorr
    gr.ripley_k
    gr.co_occurrence

Image
~~~~~

.. py:module:: squidpy.im
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

    pl.nhood_enrichment
    pl.centrality_scores
    pl.interaction_matrix
    pl.ligrec
    pl.ripley_k
    pl.co_occurrence
    pl.extract

Datasets
~~~~~~~~

.. module:: squidpy.datasets
.. currentmodule:: squidpy

.. autosummary::
    :toctree: api

    datasets.four_i
    datasets.imc
    datasets.seqfish
    datasets.visium_hne_adata
    datasets.visium_hne_adata_crop
    datasets.visium_fluo_adata
    datasets.visium_fluo_adata_crop
    datasets.visium_hne_image
    datasets.visium_hne_image_crop
    datasets.visium_fluo_image_crop

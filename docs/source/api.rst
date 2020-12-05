API
===

Import Squidpy as::

    import squidpy as sp

Graph
~~~~~

.. module::squidpy.gr
.. currentmodule:: squidpy

.. autosummary::
    :toctree: api

    gr.spatial_connectivity
    gr.nhood_enrichment
    gr.centrality_scores
    gr.interaction_matrix
    gr.perm_test
    gr.moran
    gr.ripley_k

Image
~~~~~

.. py:module::squidpy.im
.. currentmodule:: squidpy

.. autosummary::
    :toctree: api

    im.get_color_hist
    im.get_hog_features
    im.get_summary_stats
    im.calculate_image_features
    im.get_grey_texture_features
    im.process_img
    im.segment_img
    im.crop_img

Plotting
~~~~~~~~

.. module::squidpy.pl
.. currentmodule:: squidpy

.. autosummary::
    :toctree: api

    pl.plot_ripley_k
    pl.nhood_enrichment
    pl.centrality_scores
    pl.interaction_matrix
    pl.plot_segmentation

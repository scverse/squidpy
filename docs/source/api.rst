API
===

Import Squidpy as::

    import squidpy as sp

Graph
~~~~~

.. module::squidpy.graph
.. currentmodule:: squidpy

.. autosummary::
    :toctree: api

    graph.spatial_connectivity
    graph.nhood_enrichment
    graph.centrality_scores
    graph.interaction_matrix
    graph.perm_test
    graph.moran
    graph.ripley_k

Image
~~~~~

.. module::squidpy.image
.. currentmodule:: squidpy

.. autosummary::
    :toctree: api

    image.get_color_hist
    image.get_hog_features
    image.get_summary_stats
    image.calculate_image_features
    image.get_grey_texture_features
    image.ImageContainer
    image.segment_img
    image.process_img

Plotting
~~~~~~~~

.. module::squidpy.plotting
.. currentmodule:: squidpy

.. autosummary::
    :toctree: api

    plotting.plot_ripley_k
    plotting.nhood_enrichment
    plotting.centrality_scores
    plotting.interaction_matrix
    plotting.plot_segmentation

Reading
~~~~~~~

.. module::squidpy
.. currentmodule:: squidpy

.. autosummary::
    :toctree: api

    read_seqfish
    read_visium_data

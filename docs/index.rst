|PyPI| |Downloads| |CI| |Docs| |Coverage| |Discourse| |Zulip|

Squidpy - Spatial Single Cell Analysis in Python
================================================

**Squidpy** is a tool for the analysis and visualization of spatial molecular data.
It builds on top of `scanpy`_ and `anndata`_, from which it inherits modularity and scalability.
It provides analysis tools that leverages the spatial coordinates of the data, as well as
tissue images if available.

.. image:: https://raw.githubusercontent.com/scverse/squidpy/main/docs/_static/img/figure1.png
    :alt: Squidpy title figure
    :width: 400px
    :align: center
    :target: https://doi.org/10.1038/s41592-021-01358-2

Manuscript
----------
Please see our manuscript :cite:`palla:22` in **Nature Methods** to learn more.

Squidpy's key applications
--------------------------
- Build and analyze the neighborhood graph from spatial coordinates.
- Compute spatial statistics for cell-types and genes.
- Efficiently store, analyze and visualize large tissue images, leveraging `skimage`_.
- Interactively explore `anndata`_ and large tissue images in `napari`_.

Getting started with Squidpy
----------------------------
- Browse :doc:`notebooks/tutorials/index` and :doc:`notebooks/examples/index`.
- Discuss usage on `discourse`_ and development on `github`_.

Contributing to Squidpy
-----------------------
We are happy about any contributions! Before you start, check out our `contributing guide`_.

.. toctree::
    :caption: General
    :maxdepth: 2
    :hidden:

    installation
    api
    classes
    release_notes
    references

.. toctree::
    :caption: Gallery
    :maxdepth: 2
    :hidden:

    notebooks/tutorials/index
    notebooks/examples/index

.. |PyPI| image:: https://img.shields.io/pypi/v/squidpy.svg
    :target: https://pypi.org/project/squidpy/
    :alt: PyPI

.. |CI| image:: https://img.shields.io/github/actions/workflow/status/scverse/squidpy/test.yml?branch=main
    :target: https://github.com/scverse/squidpy/actions
    :alt: CI

.. |Docs| image:: https://img.shields.io/readthedocs/squidpy
    :target: https://squidpy.readthedocs.io/en/stable/
    :alt: Documentation

.. |Coverage| image:: https://codecov.io/gh/scverse/squidpy/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/scverse/squidpy
    :alt: Coverage

.. |Downloads| image:: https://pepy.tech/badge/squidpy
    :target: https://pepy.tech/project/squidpy
    :alt: Downloads

.. |Discourse| image:: https://img.shields.io/discourse/posts?color=yellow&logo=discourse&server=https%3A%2F%2Fdiscourse.scverse.org
    :target: https://discourse.scverse.org/
    :alt: Discourse

.. |Zulip| image:: https://img.shields.io/badge/zulip-join_chat-%2367b08f.svg
    :target: https://scverse.zulipchat.com
    :alt: Zulip

.. _scanpy: https://scanpy.readthedocs.io/en/stable/
.. _anndata: https://anndata.readthedocs.io/en/stable/
.. _napari: https://napari.org/
.. _skimage: https://scikit-image.org/
.. _contributing guide: https://github.com/scverse/squidpy/blob/main/CONTRIBUTING.rst
.. _discourse: https://discourse.scverse.org/
.. _github: https://github.com/scverse/squidpy

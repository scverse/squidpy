|PyPI| |Downloads| |CI| |Notebooks| |Docs| |Coverage|

Squidpy - Spatial Single Cell Analysis in Python
================================================

**Squidpy** is a tool for the analysis and visualization of spatial molecular data.
It builds on top of `scanpy`_ and `anndata`_, from which it inherits modularity and scalability.
It provides analysis tools that leverages the spatial coordinates of the data, as well as
tissue images if available.

.. image:: https://raw.githubusercontent.com/theislab/squidpy/main/docs/source/_static/img/figure1.png
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
- Browse :doc:`tutorials </tutorials>` and :doc:`examples </examples>`.
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

    tutorials
    examples

.. |PyPI| image:: https://img.shields.io/pypi/v/squidpy.svg
    :target: https://pypi.org/project/squidpy/
    :alt: PyPI

.. |CI| image:: https://img.shields.io/github/workflow/status/theislab/squidpy/Test/main
    :target: https://github.com/theislab/squidpy/actions
    :alt: CI

.. |Notebooks| image:: https://img.shields.io/github/workflow/status/theislab/squidpy_notebooks/CI/main?label=notebooks
    :target: https://github.com/theislab/squidpy_notebooks/actions
    :alt: Notebooks CI

.. |Docs| image:: https://img.shields.io/readthedocs/squidpy
    :target: https://squidpy.readthedocs.io/en/stable/
    :alt: Documentation

.. |Coverage| image:: https://codecov.io/gh/theislab/squidpy/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/theislab/squidpy
    :alt: Coverage

.. |Downloads| image:: https://pepy.tech/badge/squidpy
    :target: https://pepy.tech/project/squidpy
    :alt: Downloads

.. _scanpy: https://scanpy.readthedocs.io/en/stable/
.. _anndata: https://anndata.readthedocs.io/en/stable/
.. _napari: https://napari.org/
.. _skimage: https://scikit-image.org/
.. _contributing guide: https://github.com/theislab/squidpy/blob/main/CONTRIBUTING.rst
.. _discourse: https://discourse.scverse.org/
.. _github: https://github.com/theislab/squidpy

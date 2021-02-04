|PyPI| |CI| |Notebooks| |Docs| |Coverage|

squidpy: spatial molecular data analysis in python
===================================================

.. image:: <>
   :width: 400px
   :align: center

**squidpy** is a tool for the analysis and visualization of spatial molecular data.
It builds on top of `scanpy`_ and `anndata`_, from which it inherits modularity and scalability.
It provides analysis tools that leverages the spatial coordinates of the data, as well as
microscopy images if available.
Visit the documentation [TODO] for docs, tutorials and examples.

Manuscript
----------
Please see our `preprint`_ on **bioRxiv** to learn more.

Latest additions
----------------
.. include:: latest_additions.rst

squidpy's key applications
--------------------------
- Build and analyze the neighborhood graph from spatial coordinates.
- Compute spatial statistics for cell-types and genes.
- Efficiently store, analyze and visualize large microscopy images.
- Explore `anndata`_ and the large microscopy image in `napari`_.

Contributing to squidpy
-----------------------
If you wish to contribute to ``squidpy``, please make sure you're familiar with our
`Contributing guide <CONTRIBUTING.rst>`_.

.. toctree::
    :caption: General
    :maxdepth: 2
    :hidden:

    installation
    api
    classes
    references

.. toctree::
    :caption: Gallery
    :maxdepth: 2
    :hidden:

    tutorials
    examples

.. |PyPI| image:: https://img.shields.io/pypi/v/squidpy.svg
    :target: https://img.shields.io/pypi/v/squidpy.svg
    :alt: PyPI

.. |CI| image:: https://img.shields.io/github/workflow/status/theislab/squidpy/CI/master
    :target: https://github.com/theislab/squidpy/actions
    :alt: CI

.. |Notebooks| image:: https://img.shields.io/github/workflow/status/theislab/squidpy_notebooks/CI/master
    :target: https://github.com/theislab/squidpy_notebooks/actions
    :alt: Notebooks CI

.. |Docs| image:: https://img.shields.io/readthedocs/squidpy
    :target: https://img.shields.io/readthedocs/squidpy
    :alt: Documentation

.. |Coverage| image:: https://codecov.io/gh/theislab/squidpy/branch/master/graph/badge.svg?token=JQZA3UZ94Y
    :target: https://codecov.io/gh/theislab/squidpy
    :alt: Coverage

.. _preprint: VERY SOON
.. _scanpy: https://scanpy.readthedocs.io/en/latest/
.. _anndata: https://anndata.readthedocs.io/en/latest/
.. _napari: https://napari.org/

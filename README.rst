|PyPI| |Downloads| |CI| |Notebooks| |Docs| |Coverage|

Squidpy - Spatial Molecular Data Analysis in Python
===================================================

.. image:: https://raw.githubusercontent.com/theislab/squidpy/master/docs/source/_static/img/squidpy_horizontal.png
    :alt: Title figure
    :width: 400px
    :align: center

**Squidpy** is a tool for the analysis and visualization of spatial molecular data.
It builds on top of `scanpy`_ and `anndata`_, from which it inherits modularity and scalability.
It provides analysis tools that leverages the spatial coordinates of the data, as well as
microscopy images if available.

Visit our `documentation`_ for installation, tutorials, examples and more.

Manuscript
----------
Please see our `preprint`_ on **bioRxiv** to learn more.

Squidpy's key applications
--------------------------
- Build and analyze the neighborhood graph from spatial coordinates.
- Compute spatial statistics for cell-types and genes.
- Efficiently store, analyze and visualize large microscopy images, leveraging `skimage`_.
- Explore `anndata`_ and the large microscopy image in `napari`_.

Installation
------------
Install Squidpy via PyPI by running::

    pip install squidpy

Contributing to Squidpy
-----------------------
If you wish to contribute to ``Squidpy``, please make sure you're familiar with our
`Contributing guide <CONTRIBUTING.rst>`_.

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
    :target: https://squidpy.readthedocs.io/en/latest/
    :alt: Documentation

.. |Coverage| image:: https://codecov.io/gh/theislab/squidpy/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/theislab/squidpy
    :alt: Coverage

.. |Downloads| image:: https://pepy.tech/badge/squidpy
    :target: https://pepy.tech/project/squidpy
    :alt: Downloads

.. _preprint: VERY SOON
.. _scanpy: https://scanpy.readthedocs.io/en/latest/
.. _anndata: https://anndata.readthedocs.io/en/latest/
.. _napari: https://napari.org/
.. _skimage: https://scikit-image.org/
.. _documentation: https://squidpy.readthedocs.io/en/latest/

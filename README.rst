|PyPI| |Downloads| |CI| |Notebooks| |Docs| |Coverage|

Squidpy - Spatial Single Cell Analysis in Python
================================================

.. raw:: html

    <p align="center">
        <a href="https://squidpy.readthedocs.io/en/stable/">
            <img src="https://raw.githubusercontent.com/theislab/squidpy/master/docs/source/_static/img/squidpy_horizontal.png"
             width="400px" alt="Squidpy logo">
        </a>
    </p>


**Squidpy** is a tool for the analysis and visualization of spatial molecular data.
It builds on top of `scanpy`_ and `anndata`_, from which it inherits modularity and scalability.
It provides analysis tools that leverages the spatial coordinates of the data, as well as
tissue images if available.

.. raw:: html

    <p align="center">
        <a href="https://www.biorxiv.org/content/10.1101/2021.02.19.431994v2">
            <img src="https://raw.githubusercontent.com/theislab/squidpy/master/docs/source/_static/img/figure1.png"
             width="400px" alt="Squidpy title figure">
        </a>
    </p>

Visit our `documentation`_ for installation, tutorials, examples and more.

Manuscript
----------
Please see our `preprint`_ on **bioRxiv** to learn more.

Squidpy's key applications
--------------------------
- Build and analyze the neighborhood graph from spatial coordinates.
- Compute spatial statistics for cell-types and genes.
- Efficiently store, analyze and visualize large tissue images, leveraging `skimage`_.
- Explore `anndata`_ and the large tissue image in `napari`_.

Installation
------------
Install Squidpy via PyPI by running::

    pip install squidpy
    # or with napari included
    pip install 'squidpy[interactive]'

Contributing to Squidpy
-----------------------
We are happy about any contributions! Before you start, check out our `contributing guide <CONTRIBUTING.rst>`_.

.. |PyPI| image:: https://img.shields.io/pypi/v/squidpy.svg
    :target: https://img.shields.io/pypi/v/squidpy.svg
    :alt: PyPI

.. |CI| image:: https://img.shields.io/github/workflow/status/theislab/squidpy/Test/dev
    :target: https://github.com/theislab/squidpy/actions
    :alt: CI

.. |Notebooks| image:: https://img.shields.io/github/workflow/status/theislab/squidpy_notebooks/CI/dev?label=notebooks
    :target: https://github.com/theislab/squidpy_notebooks/actions
    :alt: Notebooks CI

.. |Docs| image:: https://img.shields.io/readthedocs/squidpy
    :target: https://squidpy.readthedocs.io/en/stable/
    :alt: Documentation

.. |Coverage| image:: https://codecov.io/gh/theislab/squidpy/branch/dev/graph/badge.svg
    :target: https://codecov.io/gh/theislab/squidpy
    :alt: Coverage

.. |Downloads| image:: https://pepy.tech/badge/squidpy
    :target: https://pepy.tech/project/squidpy
    :alt: Downloads

.. _preprint: https://www.biorxiv.org/content/10.1101/2021.02.19.431994v2
.. _scanpy: https://scanpy.readthedocs.io/en/stable/
.. _anndata: https://anndata.readthedocs.io/en/stable/
.. _napari: https://napari.org/
.. _skimage: https://scikit-image.org/
.. _documentation: https://squidpy.readthedocs.io/en/stable/

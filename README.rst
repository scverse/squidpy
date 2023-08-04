|PyPI| |Downloads| |CI| |Docs| |Coverage| |Discourse| |Zulip| |NumFOCUS|

Squidpy - Spatial Single Cell Analysis in Python
================================================

**Squidpy** is a tool for the analysis and visualization of spatial molecular data.
It builds on top of `scanpy`_ and `anndata`_, from which it inherits modularity and scalability.
It provides analysis tools that leverages the spatial coordinates of the data, as well as
tissue images if available.

Visit our `documentation`_ for installation, tutorials, examples and more.

Squidpy is part of the scverse project (`website`_, `governance`_) and is fiscally sponsored by `NumFOCUS`_.
Please consider making a tax-deductible `donation`_ to help the project pay for developer time, professional services, travel, workshops, and a variety of other needs.

|NumFOCUS logo|

Manuscript
----------
Please see our manuscript `Palla, Spitzer et al. (2022)`_ in **Nature Methods** to learn more.

Squidpy's key applications
--------------------------
- Build and analyze the neighborhood graph from spatial coordinates.
- Compute spatial statistics for cell-types and genes.
- Efficiently store, analyze and visualize large tissue images, leveraging `skimage`_.
- Interactively explore `anndata`_ and large tissue images in `napari`_.

Installation
------------
Install Squidpy via PyPI by running::

    pip install squidpy
    # or with napari included
    pip install 'squidpy[interactive]'

or via Conda as::

    conda install -c conda-forge squidpy

Contributing to Squidpy
-----------------------
We are happy about any contributions! Before you start, check out our `contributing guide <CONTRIBUTING.rst>`_.

.. |PyPI| image:: https://img.shields.io/pypi/v/squidpy.svg
    :target: https://pypi.org/project/squidpy/
    :alt: PyPI

.. |CI| image:: https://img.shields.io/github/actions/workflow/status/scverse/squidpy/test.yml?branch=main
    :target: https://github.com/scverse/squidpy/actions
    :alt: CI

.. |Pre-commit| image:: https://results.pre-commit.ci/badge/github/scverse/squidpy/main.svg
   :target: https://results.pre-commit.ci/latest/github/scverse/squidpy/main
   :alt: pre-commit.ci status

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

.. |NumFOCUS| image:: https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A
    :target: http://numfocus.org
    :alt: NumFOCUS

.. |NumFOCUS logo| image:: https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png
    :target: https://numfocus.org/project/scverse
    :width: 200

.. _Palla, Spitzer et al. (2022): https://doi.org/10.1038/s41592-021-01358-2
.. _scanpy: https://scanpy.readthedocs.io/en/stable/
.. _anndata: https://anndata.readthedocs.io/en/stable/
.. _napari: https://napari.org/
.. _skimage: https://scikit-image.org/
.. _documentation: https://squidpy.readthedocs.io/en/stable/
.. _website: https://scverse.org/
.. _governance: https://scverse.org/about/roles/
.. _NumFOCUS: https://numfocus.org/
.. _donation: https://numfocus.org/donate-to-scverse/

[![PyPI](https://img.shields.io/pypi/v/squidpy.svg)](https://pypi.org/project/squidpy/)
[![Downloads](https://pepy.tech/badge/squidpy)](https://pepy.tech/project/squidpy)
[![CI](https://img.shields.io/github/actions/workflow/status/scverse/squidpy/test.yml?branch=main)](https://github.com/scverse/squidpy/actions)
[![Docs](https://img.shields.io/readthedocs/squidpy)](https://squidpy.readthedocs.io/en/stable/)
[![Coverage](https://codecov.io/gh/scverse/squidpy/branch/main/graph/badge.svg)](https://codecov.io/gh/scverse/squidpy)
[![Discourse](https://img.shields.io/discourse/posts?color=yellow&logo=discourse&server=https%3A%2F%2Fdiscourse.scverse.org)](https://discourse.scverse.org/)
[![Zulip](https://img.shields.io/badge/zulip-join_chat-%2367b08f.svg)](https://scverse.zulipchat.com)
[![NumFOCUS](https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](http://numfocus.org)

# Squidpy - Spatial Single Cell Analysis in Python

**Squidpy** is a tool for the analysis and visualization of spatial molecular data.
It builds on top of [scanpy](https://scanpy.readthedocs.io/en/stable/) and [anndata](https://anndata.readthedocs.io/en/stable/), from which it inherits modularity and scalability.
It provides analysis tools that leverages the spatial coordinates of the data, as well as tissue images if available.

Visit our [documentation](https://squidpy.readthedocs.io/en/stable/) for installation, tutorials, examples and more.

## Manuscript

Please see our manuscript [Palla, Spitzer et al. (2022)](https://doi.org/10.1038/s41592-021-01358-2) in **Nature Methods** to learn more.

## Squidpy's key applications

- Build and analyze the neighborhood graph from spatial coordinates.
- Compute spatial statistics for cell-types and genes.
- Efficiently store, analyze and visualize large tissue images, leveraging [skimage](https://scikit-image.org/).
- Interactively explore [anndata](https://anndata.readthedocs.io/en/stable/) and large tissue images in [napari](https://napari.org/).

## Installation

Install Squidpy via PyPI by running:

```bash
pip install squidpy
# or with napari included
pip install 'squidpy[interactive]'
```

or via Conda as:

```bash
conda install -c conda-forge squidpy
```

## Contributing to Squidpy

We are happy about any contributions! Before you start, check out our [contributing guide](CONTRIBUTING.rst).

---

Squidpy is part of the scverse® project ([website](https://scverse.org/), [governance](https://scverse.org/about/roles/)) and is fiscally sponsored by [NumFOCUS](https://numfocus.org/).
Please consider making a tax-deductible [donation](https://numfocus.org/donate-to-scverse/) to help the project pay for developer time, professional services, travel, workshops, and a variety of other needs.

<div align="center">
  <a href="https://numfocus.org/project/scverse">
    <img src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png" width="200" alt="NumFOCUS">
  </a>
</div>

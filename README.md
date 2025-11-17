[![Build](https://github.com/scverse/squidpy/actions/workflows/build.yaml/badge.svg)](https://github.com/scverse/squidpy/actions/workflows/build.yaml)
[![Test](https://github.com/scverse/squidpy/actions/workflows/test.yaml/badge.svg)](https://github.com/scverse/squidpy/actions/workflows/test.yaml)
[![codecov](https://codecov.io/gh/scverse/squidpy/graph/badge.svg)](https://codecov.io/gh/scverse/squidpy)
[![License](https://img.shields.io/github/license/scverse/squidpy)](https://opensource.org/licenses/BSD-3-Clause)
[![PyPI](https://img.shields.io/pypi/v/squidpy.svg)](https://pypi.org/project/squidpy/)
[![Python Version](https://img.shields.io/pypi/pyversions/squidpy)](https://pypi.org/project/squidpy/)
[![Read the Docs](https://img.shields.io/readthedocs/squidpy/latest.svg?label=Read%20the%20Docs)](https://squidpy.readthedocs.io/en/stable)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

# Squidpy - Spatial Single Cell Analysis in Python

Squidpy is the scverse toolkit for scalable analysis and visualization of spatial molecular data.
It builds on [scanpy](https://scanpy.readthedocs.io/en/stable/) and [anndata](https://anndata.readthedocs.io/en/stable/), providing streamlined APIs for feature extraction, spatial statistics, and interactive exploration of tissue sections together with microscopy images.

![Squidpy overview](https://raw.githubusercontent.com/scverse/squidpy/main/docs/_static/img/figure1.png)

## Documentation

Head over to the [documentation](https://squidpy.readthedocs.io/en/stable/) for installation instructions, tutorials, how-to guides, and reference material.

## Installation

We recommend running Squidpy on a recent Linux or macOS system with Python ≥3.11, but it also works on Windows via WSL.

Install from [PyPI](https://pypi.org/project/squidpy) with:

```console
pip install squidpy
```

or from [conda-forge](https://anaconda.org/conda-forge/squidpy):

```console
conda install -c conda-forge squidpy
```

### Interactive visualization

To get optional dependencies required for the napari-based interactive plotting APIs, install the `interactive` extra:

```console
pip install 'squidpy[interactive]'
```

## Key capabilities

- Build and analyze spatial neighbor graphs directly from Visium, Slide-seq, Xenium, and other spatial omics assays.
- Compute spatial statistics for cell types and genes, including neighborhood enrichment, co-occurrence, and Moran's I.
- Efficiently store, featurize, and visualize high-resolution tissue microscopy images via [scikit-image](https://scikit-image.org/).
- Explore annotated datasets interactively with [napari](https://napari.org/) and scverse visualization tooling.

## Contributing

Contributions are welcome! Please read the [contributing guide](CONTRIBUTING.rst) for instructions on setting up your environment, running tests, and submitting pull requests.

## Citation

If you use Squidpy in your research, cite the original publication:

```bibtex
@article{palla:22,
    author = {Palla, Giovanni and Spitzer, Hannah and Klein, Michal and Fischer, David and Schaar, Anna Christina
              and Kuemmerle, Louis Benedikt and Rybakov, Sergei and Ibarra, Ignacio L. and Holmberg, Olle
              and Virshup, Isaac and Lotfollahi, Mohammad and Richter, Sabrina and Theis, Fabian J.},
    title = {Squidpy: a scalable framework for spatial omics analysis},
    journal = {Nature Methods},
    year = {2022},
    month = {Feb},
    volume = {19},
    number = {2},
    pages = {171--178},
    issn = {1548-7105},
    doi = {10.1038/s41592-021-01358-2},
}
```

---

Squidpy is part of the scverse® project ([website](https://scverse.org/), [governance](https://scverse.org/about/roles/)) and is fiscally sponsored by [NumFOCUS](https://numfocus.org/).
Please consider making a tax-deductible [donation](https://numfocus.org/donate-to-scverse/) to help the project pay for developer time, professional services, travel, workshops, and a variety of other needs.

<div align="center">
  <a href="https://numfocus.org/project/scverse">
    <img src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png" width="200" alt="NumFOCUS">
  </a>
</div>

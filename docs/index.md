[![PyPI](https://img.shields.io/pypi/v/squidpy.svg)](https://pypi.org/project/squidpy/)
[![Downloads](https://pepy.tech/badge/squidpy)](https://pepy.tech/project/squidpy)
[![CI](https://img.shields.io/github/actions/workflow/status/scverse/squidpy/test.yml?branch=main)](https://github.com/scverse/squidpy/actions)
[![Docs](https://img.shields.io/readthedocs/squidpy)](https://squidpy.readthedocs.io/en/stable/)
[![Coverage](https://codecov.io/gh/scverse/squidpy/branch/main/graph/badge.svg)](https://codecov.io/gh/scverse/squidpy)
[![Discourse](https://img.shields.io/discourse/posts?color=yellow&logo=discourse&server=https%3A%2F%2Fdiscourse.scverse.org)](https://discourse.scverse.org/)
[![Zulip](https://img.shields.io/badge/zulip-join_chat-%2367b08f.svg)](https://scverse.zulipchat.com)

# Squidpy - Spatial Single Cell Analysis in Python

**Squidpy** is a tool for the analysis and visualization of spatial molecular data.
It builds on top of [scanpy](https://scanpy.readthedocs.io/en/stable/) and [anndata](https://anndata.readthedocs.io/en/stable/), from which it inherits modularity and scalability.
It provides analysis tools that leverage the spatial coordinates of the data, as well as tissue images if available.

[![Squidpy title figure](https://raw.githubusercontent.com/scverse/squidpy/main/docs/_static/img/figure1.png)](https://doi.org/10.1038/s41592-021-01358-2)
```{warning}
🚨🚨🚨 **Warning!** 🚨🚨🚨

The original napari-plugin of Squidpy has been moved to [napari-spatialdata].

All the functionalities previously available are also implemented in the new plugin, which also has many additional new features.

You can find a rich set of [documentation and examples], and we suggest starting with the [napari-spatialdata tutorial].

If you are new to SpatialData, we invite you to take a look at the [spatialdata tutorials].
```

[napari-spatialdata]: https://github.com/scverse/napari-spatialdata
[documentation and examples]: https://spatialdata.scverse.org/projects/napari/en/latest/index.html
[napari-spatialdata tutorial]: https://spatialdata.scverse.org/projects/napari/en/latest/notebooks/spatialdata.html
[spatialdata tutorials]: https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks.html

Squidpy is part of the scverse® project ([website](https://scverse.org/), [governance](https://scverse.org/about/roles/)) and is fiscally sponsored by [NumFOCUS](https://numfocus.org/).
Please consider making a tax-deductible [donation](https://numfocus.org/donate-to-scverse/) to help the project pay for developer time, professional services, travel, workshops, and a variety of other needs.

[![NumFOCUS](https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png)](https://numfocus.org/project/scverse)

## Manuscript

Please see our manuscript [Palla, Spitzer et al. (2022)](https://doi.org/10.1038/s41592-021-01358-2) in **Nature Methods** to learn more.

## Squidpy's key applications

- Build and analyze the neighborhood graph from spatial coordinates
- Compute spatial statistics for cell-types and genes
- Efficiently store, analyze and visualize large tissue images, leveraging [scikit-image](https://scikit-image.org/)
- Interactively explore [anndata](https://anndata.readthedocs.io/en/stable/) and large tissue images in [napari](https://napari.org/)

## Getting started with Squidpy

- Browse tutorials and examples
- Discuss usage on [discourse](https://discourse.scverse.org/) and development on [github](https://github.com/scverse/squidpy)

## Contributing to Squidpy

We are happy about any contributions! Before you start, check out our [contributing guide](https://github.com/scverse/squidpy/blob/main/CONTRIBUTING.rst).
```{eval-rst}
.. toctree::
    :caption: General
    :maxdepth: 2
    :hidden:

    installation
    api
    classes
    release_notes
    references
    contributing

.. toctree::
    :caption: Gallery
    :maxdepth: 2
    :hidden:

    notebooks/tutorials/index
    notebooks/examples/index
    notebooks/deprecated_features/index
```

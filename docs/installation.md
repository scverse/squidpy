# Installation

Squidpy requires Python version >= 3.11 to run.

## PyPI

Install Squidpy by running::

    pip install squidpy

## Conda

Install Squidpy via Conda as::

    conda install -c conda-forge squidpy

## Development version

To install Squidpy from GitHub, run::

    pip install git+https://github.com/scverse/squidpy@main

## STalign alignment

The `stalign_align_*` functions in {mod}`squidpy.experimental.tl` need JAX. Install it with the `jax` extra::

    pip install "squidpy[jax]"

This installs the CPU build. For a GPU, add the matching JAX build to the same command, e.g. for an
NVIDIA GPU on Linux::

    pip install "squidpy[jax]" "jax[cuda13]"

For other GPUs, TPUs and the driver each build needs, see the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Interactive visualization

For interactive visualization with napari, please use [napari-spatialdata](https://github.com/scverse/napari-spatialdata).
The original napari plugin from Squidpy has been deprecated and replaced by napari-spatialdata,
which offers improved functionality and support for the SpatialData ecosystem.

Install napari-spatialdata with::

    pip install napari-spatialdata

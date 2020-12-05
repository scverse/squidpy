# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
from pathlib import Path
from datetime import datetime

from sphinx_gallery.directives import MiniGallery
from sphinx_gallery.gen_gallery import DEFAULT_GALLERY_CONF

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))  # this way, we don't have to install squidpy
sys.path.insert(0, os.path.abspath("_ext"))

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import squidpy  # noqa: E402

needs_sphinx = "3.0"

# -- Project information -----------------------------------------------------

project = "squidpy"
author = squidpy.__author__
copyright = f"{datetime.now():%Y}, {author}"
github_repo = "https://github.com/theislab/squidpy"

# The full version, including alpha/beta/rc tags
release = f"master ({squidpy.__version__})"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx_last_updated_by_git",
    "sphinx_gallery.load_style",
    "edit_on_github",
]
intersphinx_mapping = dict(  # noqa: C408
    python=("https://docs.python.org/3", None),
    numpy=("https://docs.scipy.org/doc/numpy/", None),
    statsmodels=("https://www.statsmodels.org/stable/", None),
    scipy=("https://docs.scipy.org/doc/scipy/reference/", None),
    pandas=("https://pandas.pydata.org/pandas-docs/stable/", None),
    anndata=("https://anndata.readthedocs.io/en/stable/", None),
    scanpy=("https://scanpy.readthedocs.io/en/stable/", None),
    matplotlib=("https://matplotlib.org/", None),
    seaborn=("https://seaborn.pydata.org/", None),
    joblib=("https://joblib.readthedocs.io/en/latest/", None),
    networkx=("https://networkx.github.io/documentation/stable/", None),
    astropy=("https://docs.astropy.org/en/stable/", None),
    esda=("https://pysal.org/esda/", None),
    dask=("https://docs.dask.org/en/latest/", None),
    rasterio=("https://rasterio.readthedocs.io/en/latest/", None),
    skimage=("https://scikit-image.org/docs/stable/", None),
    numba=("https://numba.readthedocs.io/en/stable/", None),
    xarray=("https://xarray.pydata.org/en/stable/", None),
    omnipath=("https://omnipath.readthedocs.io/en/latest", None),
)

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
source_suffix = [".rst", ".ipynb"]
master_doc = "index"
pygments_style = "sphinx"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["**.ipynb", "**.md5", "**.py", "**.ipynb_checkpoints"]  # ignore anything that isn't .rst


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
autosummary_generate = True
autodoc_member_order = "alphabetical"
autodoc_typehints = "signature"
autodoc_docstring_signature = True
autodoc_follow_wrapped = False
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]
todo_include_todos = False
# 403 Client Error: Forbidden for url: https://www.jstor.org/stable/2332142?origin=crossref
linkcheck_ignore = ["https://doi.org/10.2307/2332142"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = dict(navigation_depth=4, logo_only=True)  # noqa: C408
html_show_sphinx = False


def setup(app):  # noqa: D103
    DEFAULT_GALLERY_CONF["backreferences_dir"] = "gen_modules/backreferences"
    DEFAULT_GALLERY_CONF["show_signature"] = False

    app.add_config_value("sphinx_gallery_conf", DEFAULT_GALLERY_CONF, "html")
    app.add_directive("minigallery", MiniGallery)
    app.add_css_file("css/custom.css")

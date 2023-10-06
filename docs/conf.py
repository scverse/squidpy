# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys
from datetime import datetime

# from importlib.metadata import metadata
from pathlib import Path

from sphinx.application import Sphinx

HERE = Path(__file__).parent
# sys.path.insert(0, str(HERE.parent.parent))  # this way, we don't have to install squidpy
# sys.path.insert(0, os.path.abspath("_ext"))

sys.path.insert(0, str(HERE / "_ext"))

# -- Project information -----------------------------------------------------

import squidpy  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

# -- Project information -----------------------------------------------------

project = squidpy.__name__
author = squidpy.__author__
version = squidpy.__version__
copyright = f"{datetime.now():%Y}, scverse"

# info = metadata("squidpy")
# project_name = info["Name"]
# author = info["Author"]
# copyright = f"{datetime.now():%Y}, {author}."
# version = info["Version"]
# release = info["Version"]

# # project = squidpy.__name__
# # author = squidpy.__author__
# # copyright = f"{datetime.now():%Y}, {author}"  # noqa: A001

# github_org = "scverse"
# github_repo = "squidpy"
# github_ref = "main"

# # The full version, including alpha/beta/rc tags
# # release = github_ref
# # version = f"{release} ({squidpy.__version__})"

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
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "myst_nb",
    "nbsphinx",
    "typed_returns",
    "IPython.sphinxext.ipython_console_highlighting",
]
intersphinx_mapping = dict(  # noqa: C408
    python=("https://docs.python.org/3", None),
    numpy=("https://numpy.org/doc/stable/", None),
    statsmodels=("https://www.statsmodels.org/stable/", None),
    scipy=("https://docs.scipy.org/doc/scipy/", None),
    pandas=("https://pandas.pydata.org/pandas-docs/stable/", None),
    anndata=("https://anndata.readthedocs.io/en/stable/", None),
    scanpy=("https://scanpy.readthedocs.io/en/stable/", None),
    matplotlib=("https://matplotlib.org/stable/", None),
    seaborn=("https://seaborn.pydata.org/", None),
    joblib=("https://joblib.readthedocs.io/en/latest/", None),
    networkx=("https://networkx.org/documentation/stable/", None),
    dask=("https://docs.dask.org/en/latest/", None),
    skimage=("https://scikit-image.org/docs/stable/", None),
    sklearn=("https://scikit-learn.org/stable/", None),
    numba=("https://numba.readthedocs.io/en/stable/", None),
    xarray=("https://xarray.pydata.org/en/stable/", None),
    omnipath=("https://omnipath.readthedocs.io/en/latest", None),
    napari=("https://napari.org/", None),
    spatialdata=("https://spatialdata.scverse.org/en/latest", None),
)

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
source_suffix = {".rst": "restructuredtext", ".ipynb": "myst-nb"}
master_doc = "index"
pygments_style = "sphinx"

# myst
nb_execution_mode = "off"
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
]
myst_heading_anchors = 2

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "notebooks/README.rst",
    "notebooks/CONTRIBUTING.rst",
    "release/changelog/*",
    "**.ipynb_checkpoints",
    "build",
]
suppress_warnings = ["download.not_readable", "git.too_shallow"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
autosummary_generate = True
autodoc_member_order = "groupwise"
autodoc_typehints = "signature"
autodoc_docstring_signature = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
todo_include_todos = False

# bibliography
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "alpha"

# spelling
spelling_lang = "en_US"
spelling_warning = True
spelling_word_list_filename = "spelling_wordlist.txt"
spelling_add_pypi_package_names = True
spelling_show_suggestions = True
spelling_exclude_patterns = ["references.rst"]
# see: https://pyenchant.github.io/pyenchant/api/enchant.tokenize.html
spelling_filters = [
    "enchant.tokenize.URLFilter",
    "enchant.tokenize.EmailFilter",
    "docs.source.utils.ModnameFilter",
    "docs.source.utils.SignatureFilter",
    "enchant.tokenize.MentionFilter",
]
# see the solution from: https://github.com/sphinx-doc/sphinx/issues/7369
linkcheck_ignore = [
    # 403 Client Error
    "https://doi.org/10.1126/science.aar7042",
    "https://doi.org/10.1126/science.aau5324",
    "https://doi.org/10.1093/bioinformatics/btab164",
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2716260/",
    "https://raw.githubusercontent.com/scverse/squidpy/main/docs/_static/img/figure1.png",
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/img/squidpy_horizontal.png"
html_theme_options = {"navigation_depth": 4, "logo_only": True}
html_show_sphinx = False


def setup(app: Sphinx) -> None:
    app.add_css_file("css/custom.css")
    app.add_css_file("css/sphinx_gallery.css")
    app.add_css_file("css/nbsphinx.css")
    app.add_css_file("css/dataframe.css")  # had to add this manually

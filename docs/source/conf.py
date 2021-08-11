# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from pathlib import Path
from datetime import datetime
from sphinx.application import Sphinx
from sphinx_gallery.gen_gallery import DEFAULT_GALLERY_CONF

# -- Path setup --------------------------------------------------------------
import os
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))  # this way, we don't have to install squidpy
sys.path.insert(0, os.path.abspath("_ext"))

from docs.source.utils import (  # noqa: E402
    _is_dev,
    _get_thumbnails,
    _fetch_notebooks,
    MaybeMiniGallery,
)
import squidpy  # noqa: E402

needs_sphinx = "3.0"

# -- Project information -----------------------------------------------------

project = "squidpy"
author = squidpy.__author__
copyright = f"{datetime.now():%Y}, {author}"  # noqa: A001

github_org = "theislab"
github_repo = "squidpy"
github_ref = "dev" if _is_dev() else "master"
github_nb_repo = "squidpy_notebooks"
_fetch_notebooks(repo_url=f"https://github.com/{github_org}/{github_nb_repo}")

# The full version, including alpha/beta/rc tags
release = github_ref
version = f"{release} ({squidpy.__version__})"

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
    "nbsphinx",
    "sphinxcontrib.bibtex",
    "edit_on_github",
    "typed_returns",
    "IPython.sphinxext.ipython_console_highlighting",
]
intersphinx_mapping = dict(  # noqa: C408
    python=("https://docs.python.org/3", None),
    numpy=("https://docs.scipy.org/doc/numpy/", None),
    statsmodels=("https://www.statsmodels.org/stable/", None),
    scipy=("https://docs.scipy.org/doc/scipy/reference/", None),
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
    napari=("https://napari.org/docs/dev/", None),
)

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
source_suffix = [".rst", ".ipynb"]
master_doc = "index"
pygments_style = "sphinx"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "auto_*/**.ipynb",
    "auto_*/**.md5",
    "auto_*/**.py",
    "release/changelog/*",
    "**.ipynb_checkpoints",
]
suppress_warnings = ["download.not_readable"]

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
# problematic entry: andersson2021
# see the solution from: https://github.com/sphinx-doc/sphinx/issues/7369
user_agent = "Mozilla/5.0 (X11; Linux x86_64; rv:25.0) Gecko/20100101 Firefox/25.0"
# TODO: has been fixed on notebooks' dev, remove once it's merged in master
linkcheck_ignore = [r"\.\./\.\./external_tutorials/tutorial_napari.html"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/img/squidpy_horizontal.png"
html_theme_options = {"navigation_depth": 4, "logo_only": True}
html_show_sphinx = False

nbsphinx_thumbnails = {**_get_thumbnails("auto_tutorials"), **_get_thumbnails("auto_examples")}
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'png', 'pdf'}",  # correct figure resize
    "--InlineBackend.rc={'figure.dpi': 96}",
]
nbsphinx_prolog = r"""
{% set docname = 'docs/source/' + env.doc2path(env.docname, base=None) %}
.. raw:: html

    <div class="binder-badge docutils container">
        <a class="reference external image-reference"
           href="https://mybinder.org/v2/gh/theislab/squidpy_notebooks/{{ env.config.release|e }}?filepath={{ docname|e }}">
        <img alt="Launch binder" src="https://mybinder.org/badge_logo.svg" width="150px">
        </a>
    </div>
"""  # noqa: E501


def setup(app: Sphinx) -> None:
    DEFAULT_GALLERY_CONF["src_dir"] = str(HERE)
    DEFAULT_GALLERY_CONF["backreferences_dir"] = "gen_modules/backreferences"
    DEFAULT_GALLERY_CONF["download_all_examples"] = False
    DEFAULT_GALLERY_CONF["show_signature"] = False
    DEFAULT_GALLERY_CONF["log_level"] = {"backreference_missing": "info"}
    DEFAULT_GALLERY_CONF["gallery_dirs"] = ["auto_examples", "auto_tutorials"]
    DEFAULT_GALLERY_CONF["default_thumb_file"] = "docs/source/_static/img/squidpy_vertical.png"

    app.add_config_value("sphinx_gallery_conf", DEFAULT_GALLERY_CONF, "html")
    app.add_directive("minigallery", MaybeMiniGallery)
    app.add_css_file("css/custom.css")
    app.add_css_file("css/sphinx_gallery.css")
    app.add_css_file("css/nbsphinx.css")
    app.add_css_file("css/dataframe.css")  # had to add this manually

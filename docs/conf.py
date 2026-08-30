# -- Path setup --------------------------------------------------------------
from __future__ import annotations

from datetime import datetime
from functools import cache
from importlib.metadata import metadata

from sphinx.application import Sphinx

# -- Project information -----------------------------------------------------
info = metadata("squidpy")
project_name = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}"
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
repository_url = urls["Source"]

# The full version, including alpha/beta/rc tags
release = info["Version"]

needs_sphinx = "4.0"

# -- General configuration ---------------------------------------------------
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
    "sphinx_design",
    "sphinx_tabs.tabs",
    "myst_nb",
    "nbsphinx",
    "scverse_misc.sphinx_ext",
    "IPython.sphinxext.ipython_console_highlighting",
]
intersphinx_mapping = dict(  # noqa: C408
    python=("https://docs.python.org/3", None),
    numpy=("https://numpy.org/doc/stable", None),
    statsmodels=("https://www.statsmodels.org/stable", None),
    scipy=("https://docs.scipy.org/doc/scipy", None),
    pandas=("https://pandas.pydata.org/pandas-docs/stable", None),
    anndata=("https://anndata.readthedocs.io/en/stable", None),
    scanpy=("https://scanpy.readthedocs.io/en/stable", None),
    matplotlib=("https://matplotlib.org/stable", None),
    cycler=("https://matplotlib.org/cycler", None),
    seaborn=("https://seaborn.pydata.org", None),
    joblib=("https://joblib.readthedocs.io/en/latest", None),
    networkx=("https://networkx.org/documentation/stable", None),
    dask=("https://docs.dask.org/en/latest", None),
    skimage=("https://scikit-image.org/docs/stable", None),
    sklearn=("https://scikit-learn.org/stable", None),
    numba=("https://numba.readthedocs.io/en/stable", None),
    xarray=("https://docs.xarray.dev/en/stable", None),
    omnipath=("https://omnipath.readthedocs.io/en/latest", None),
    napari=("https://napari.org", None),
    spatialdata=("https://spatialdata.scverse.org/en/latest", None),
    shapely=("https://shapely.readthedocs.io/en/stable", None),
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
# Keep documented objects (every params key, every attribute) out of the left nav:
# it should list pages and sections, not one entry per key.
toc_object_entries = False
autosummary_generate = True
autodoc_member_order = "groupwise"
autodoc_typehints = "signature"
# jax is an optional extra; mock it so the fit classes' annotations render without it
autodoc_mock_imports = ["jax"]
# show each parameter's default next to its type, as scanpy does
typehints_defaults = "braces"
autodoc_docstring_signature = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = False
# fold the return type into the Returns section rather than giving it its own
typehints_use_rtype = False
napoleon_use_param = True
todo_include_todos = False

# bibliography
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "alpha"

# sphinx linktime checkouts (esp bioRxiv is spotty)
linkcheck_timeout = 90
linkcheck_retries = 2

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

# Link checking
nitpicky = True  # this is linkcheck for Sphinx.
nitpick_ignore = [
    ("py:func", "numba.prange"),  # no reference for this function
    ("py:class", "matplotlib_scalebar.ScaleBar"),  # this project has no sphinx docs
    # TODO: fix using scanpydoc.elegant_typehints
    ("py:class", "pathlib._local.Path"),
    ("py:data", "typing.Union"),
    # there seems to be a bug with autodoc for NamedTuple attributes
    ("py:class", "NDArray"),
    # numpy.typing.NDArray canonicalizes to this private path, which has no doc target
    ("py:class", "numpy._typing._array_like.NDArray"),
    ("py:class", "numpy._typing._array_like.ArrayLike"),
    # `npt.ArrayLike` in a signature resolves to its union members, which are private
    ("py:class", "numpy._typing._array_like._SupportsArray"),
    ("py:class", "numpy._typing._nested_sequence._NestedSequence"),
    # optional dep mocked at build time (see autodoc_mock_imports), so no resolvable target
    ("py:class", "jax.Array"),
    ("py:class", "JaxArray"),
    ("py:class", "np.number"),
    ("py:class", "csr_matrix"),
    # no idea why those aren’t exported
    ("py:class", "squidpy._constants._constants.SpatialAutocorr"),
    ("py:class", "squidpy._constants._constants.CoordType"),
    ("py:class", "squidpy._constants._constants.Transform"),
    ("py:class", "pandas.core.frame.DataFrame"),
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
html_theme_options = {"navigation_depth": 5, "logo_only": True}
html_show_sphinx = False


# Each params key carries its default in the `Annotated` metadata of its
# declaration (`squidpy.experimental.utils._params.Default`). Read it from there --
# one source of truth, no defaults restated in docstrings where they could drift.


@cache
def _params_defaults() -> dict[str, dict[str, object]]:
    """Read every params key's ``Default`` -- and then hide it from autodoc.

    Autodoc renders `Annotated` metadata verbatim (`Annotated[float, Default(1.0)]`),
    so once the defaults are in hand each annotation is replaced by its bare type.
    `_stack_attribute_defaults` puts the default back, on the key's signature.
    """
    from typing import get_type_hints

    from squidpy import types
    from squidpy.experimental.utils._params import defaults_of

    defaults = {}
    # only the `*Params` types carry per-key defaults; the result types do not
    for name in (n for n in types.__all__ if n.endswith("Params")):
        cls = getattr(types, name)
        defaults[name] = dict(defaults_of(cls))
        cls.__annotations__ = {
            key: hint.__origin__ if hasattr(hint, "__metadata__") else hint
            for key, hint in get_type_hints(cls, include_extras=True).items()
        }
    return defaults


def _default_of(name: str) -> object:
    """The declared default of a params key, or ``_NO_DEFAULT`` if it has none."""
    cls_path, _, key = name.rpartition(".")
    if not cls_path:
        return _NO_DEFAULT
    defaults = _params_defaults().get(cls_path.rpartition(".")[2], {})
    return defaults.get(key, _NO_DEFAULT)


_NO_DEFAULT = object()


def _stack_attribute_defaults(app, doctree, docname) -> None:  # type: ignore[no-untyped-def]
    """Show a params key's default on its signature, as a parameter's is shown.

    A key and a parameter are the same thing to a caller, so they should read the same:
    ``name : type (default: value)``. Autodoc gives an attribute no default of its own --
    a `TypedDict` key holds no value -- so it is appended to the signature here.
    """
    from docutils import nodes
    from sphinx import addnodes

    for sig in doctree.findall(addnodes.desc_signature):
        parent = sig.parent
        if not isinstance(parent, addnodes.desc) or parent.get("objtype") != "attribute":
            continue
        default = _default_of("".join(sig.get("ids", [""])[:1]))
        if default is _NO_DEFAULT:
            continue
        sig += addnodes.desc_annotation("", "", nodes.Text(f" (default: {default!r})"))


def _skip_dict_api(app, what, name, obj, skip, options) -> bool | None:  # type: ignore[no-untyped-def]
    """Hide the mapping API a params TypedDict inherits from :class:`dict`."""
    return True if getattr(obj, "__qualname__", "").startswith("dict.") else None


def _stack_parameter_types(app, doctree, docname) -> None:  # type: ignore[no-untyped-def]
    """Render a parameter as ``name : type`` with its description on the next line.

    Sphinx's Python domain renders a typed field inline -- ``name (type) - description`` --
    which puts the three things a reader scans for on one run-on line. Rewrite each entry
    into two paragraphs so the name and its type read as a term and the prose sits under it.
    """
    from docutils import nodes

    for field in doctree.findall(nodes.field):
        name = field.next_node(nodes.field_name)
        if name is None or not name.astext().startswith("Parameters"):
            continue
        for item in field.findall(nodes.list_item):
            para = item.next_node(nodes.paragraph)
            if para is None or not para.children:
                continue
            split = None
            for i, child in enumerate(para.children):
                if isinstance(child, nodes.Text) and "\u2013" in child:
                    split = i
                    break
            if split is None:
                continue
            sep = para.children[split].astext()
            head = para.children[:split]
            # the separator text also carries the closing paren of the type
            rest = sep.split("\u2013", 1)[1].lstrip()
            tail = ([nodes.Text(rest)] if rest else []) + para.children[split + 1 :]
            # `name (type)` -> `name : type`, leaving any inner parens (a default) alone
            parens = [i for i, c in enumerate(head) if isinstance(c, nodes.Text) and c.astext().strip() in "()"]
            opening = next((i for i in parens if head[i].astext().strip() == "("), None)
            closing = next((i for i in reversed(parens) if head[i].astext().strip() == ")"), None)
            if opening is not None:
                head[opening] = nodes.Text(" : ")
            if closing is not None:
                del head[closing]
            term = nodes.paragraph("", "", *head, classes=["param-term"])
            body = nodes.paragraph("", "", *tail, classes=["param-desc"])
            item.clear()
            item += [term, body] if tail else [term]


def setup(app: Sphinx) -> None:
    app.connect("builder-inited", lambda _app: _params_defaults())
    app.connect("autodoc-skip-member", _skip_dict_api)
    app.connect("doctree-resolved", _stack_parameter_types)
    app.connect("doctree-resolved", _stack_attribute_defaults)
    app.add_css_file("css/custom.css")
    app.add_css_file("css/sphinx_gallery.css")
    app.add_css_file("css/nbsphinx.css")
    app.add_css_file("css/dataframe.css")  # had to add this manually

"""Based on gist.github.com/MantasVaitkunas/7c16de233812adcb7028."""
from typing import Any, Dict, Optional
from sphinx.application import Sphinx
import os
import warnings

__licence__ = "BSD (3 clause)"


def get_github_repo(app: Sphinx, path: str) -> str:
    if path.endswith(".ipynb"):
        return str(app.config.github_nb_repo)
    if "auto_examples" in path:
        return str(app.config.github_nb_repo)
    if "auto_tutorials" in path:
        return str(app.config.github_nb_repo)
    return str(app.config.github_repo)


def _html_page_context(
    app: Sphinx, _pagename: str, templatename: str, context: Dict[str, Any], doctree: Optional[Any]
) -> None:
    # doctree is None - otherwise viewcode fails
    if templatename != "page.html" or doctree is None:
        return

    if not app.config.github_repo:
        warnings.warn("`github_repo` not specified")
        return

    if not app.config.github_ref:
        app.config.github_org = "theislab"
    if not app.config.github_nb_repo:
        nb_repo = f"{app.config.github_repo}_notebooks"
        warnings.warn(f"`github_nb_repo `not specified. Setting to `{nb_repo}`")
        app.config.github_nb_repo = nb_repo
    if not app.config.github_ref:
        warnings.warn("`github_ref` not specified. Setting to `'master'`")
        app.config.github_ref = "master"

    path = os.path.relpath(doctree.get("source"), app.builder.srcdir)
    repo = get_github_repo(app, path)

    # For sphinx_rtd_theme.
    context["display_github"] = True
    context["github_user"] = app.config.github_org
    context["github_version"] = app.config.github_ref
    context["github_repo"] = repo
    context["conf_py_path"] = "/docs/source/"


def setup(app: Sphinx) -> None:
    app.add_config_value("github_org", "", True)
    app.add_config_value("github_repo", "", True)
    app.add_config_value("github_ref", "", True)
    app.add_config_value("github_nb_repo", "", True)
    app.connect("html-page-context", _html_page_context)

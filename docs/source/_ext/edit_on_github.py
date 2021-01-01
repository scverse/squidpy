"""Based on gist.github.com/MantasVaitkunas/7c16de233812adcb7028."""

import warnings

__licence__ = "BSD (3 clause)"


def _html_page_context(app, _pagename, templatename, context, doctree):
    # doctree is None - otherwise viewcode fails
    if templatename != "page.html" or doctree is None:
        return

    if not app.config.github_repo:
        warnings.warn("`github_repo` not specified")
        return

    # For sphinx_rtd_theme.
    context["display_github"] = True
    context["github_user"] = "theislab"
    context["github_version"] = "master"
    context["github_repo"] = app.config.github_repo
    context["conf_py_path"] = "/docs/source/"


def setup(app):
    app.add_config_value("github_repo", "", True)
    app.connect("html-page-context", _html_page_context)

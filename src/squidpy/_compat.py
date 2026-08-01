from __future__ import annotations

from importlib import import_module
from importlib.metadata import version
from typing import Any

from packaging.version import Version


def _scanpy_plotting_layout() -> tuple[str, Any | None, Any | None]:
    try:
        legacy = import_module("scanpy.plotting.legacy")
    except ModuleNotFoundError as error:
        if error.name != "scanpy.plotting.legacy":
            raise
        return "scanpy.plotting", None, import_module("scanpy").settings
    return "scanpy.plotting.legacy", legacy.mpl_settings, None


_scanpy_plotting_path, _scanpy_mpl_settings, _scanpy_settings = _scanpy_plotting_layout()
_scanpy_plotting_utils = import_module(f"{_scanpy_plotting_path}._utils")
_scanpy_scatterplots = import_module(f"{_scanpy_plotting_path}._tools.scatterplots")
_scanpy_palettes = import_module(f"{_scanpy_plotting_path}.palettes")

add_categorical_legend = _scanpy_scatterplots._add_categorical_legend
panel_grid = _scanpy_scatterplots._panel_grid
default_palette = _scanpy_palettes.default_102

__all__ = [
    # scanpy
    "set_default_colors_for_categorical_obs",
    "add_categorical_legend",
    "panel_grid",
    "add_colors_for_categorical_sample_annotation",
    "default_palette",
    "scanpy_frameon",
    "scanpy_vector_friendly",
    # anndata
    "ArrayView",
    "SparseCSCView",
    "SparseCSRView",
]

add_colors_for_categorical_sample_annotation = _scanpy_plotting_utils.add_colors_for_categorical_sample_annotation
set_default_colors_for_categorical_obs = getattr(
    _scanpy_plotting_utils,
    "_set_default_colors_for_categorical_obs",
    None,
)
if set_default_colors_for_categorical_obs is None:
    set_default_colors_for_categorical_obs = _scanpy_plotting_utils.set_default_colors_for_categorical_obs


def scanpy_frameon() -> bool:
    if _scanpy_mpl_settings is not None:
        return _scanpy_mpl_settings.FRAMEON
    assert _scanpy_settings is not None
    return _scanpy_settings._frameon


def scanpy_vector_friendly() -> bool:
    if _scanpy_mpl_settings is not None:
        return _scanpy_mpl_settings.VECTOR_FRIENDLY
    assert _scanpy_settings is not None
    return _scanpy_settings._vector_friendly


CAN_USE_SPARSE_ARRAY = Version(version("anndata")) >= Version("0.11.0rc1")
if CAN_USE_SPARSE_ARRAY:
    from anndata._core.views import ArrayView
    from anndata._core.views import SparseCSCMatrixView as SparseCSCView
    from anndata._core.views import SparseCSRMatrixView as SparseCSRView
else:
    from anndata._core.views import ArrayView, SparseCSCView, SparseCSRView

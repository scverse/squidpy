from __future__ import annotations

from importlib.metadata import version

from packaging.version import Version

__all__ = [
    # scanpy
    "set_default_colors_for_categorical_obs",
    "add_categorical_legend",
    "panel_grid",
    "add_colors_for_categorical_sample_annotation",
    "default_palette",
    "default_frameon",
    "vector_friendly",
    # anndata
    "ArrayView",
    "SparseCSCView",
    "SparseCSRView",
]

# Scanpy 1.13 moved the pre-v2 plotting internals under ``scanpy.plotting.legacy``.
# ``scanpy.plotting.__getattr__`` forwards attribute access there, but submodule
# imports such as ``scanpy.plotting.palettes`` are not covered by it.
try:
    from scanpy.plotting.legacy._tools.scatterplots import _add_categorical_legend as add_categorical_legend
    from scanpy.plotting.legacy._tools.scatterplots import _panel_grid as panel_grid
    from scanpy.plotting.legacy._utils import (
        add_colors_for_categorical_sample_annotation,
        set_default_colors_for_categorical_obs,
    )
    from scanpy.plotting.legacy.palettes import default_102 as default_palette
except ImportError:
    from scanpy.plotting._tools.scatterplots import _add_categorical_legend as add_categorical_legend
    from scanpy.plotting._tools.scatterplots import _panel_grid as panel_grid
    from scanpy.plotting._utils import add_colors_for_categorical_sample_annotation
    from scanpy.plotting.palettes import default_102 as default_palette

    # See https://github.com/scverse/squidpy/issues/1061 for more details.
    # Scanpy 0.11.x-0.12.x renamed set_default_colors_for_categorical_obs to _set_default_colors_for_categorical_obs
    # and then changed it back. Try underscore version first, fall back to non-underscore.
    try:
        from scanpy.plotting._utils import (
            _set_default_colors_for_categorical_obs as set_default_colors_for_categorical_obs,
        )
    except ImportError:
        from scanpy.plotting._utils import set_default_colors_for_categorical_obs


# Scanpy 1.13 also moved these two plotting defaults off ``Settings``, where they were the
# private ``_frameon`` / ``_vector_friendly`` class attributes, and onto module-level
# globals in ``scanpy.plotting.legacy.mpl_settings``. ``scanpy.set_figure_params`` rebinds
# them in either layout, so they must be read at call time rather than imported once.
try:
    from scanpy.plotting.legacy import mpl_settings as _mpl_settings

    def default_frameon() -> bool:
        return _mpl_settings.FRAMEON

    def vector_friendly() -> bool:
        return _mpl_settings.VECTOR_FRIENDLY

except ImportError:
    from scanpy import settings as _sc_settings

    def default_frameon() -> bool:
        return _sc_settings._frameon

    def vector_friendly() -> bool:
        return _sc_settings._vector_friendly


CAN_USE_SPARSE_ARRAY = Version(version("anndata")) >= Version("0.11.0rc1")
if CAN_USE_SPARSE_ARRAY:
    from anndata._core.views import ArrayView
    from anndata._core.views import SparseCSCMatrixView as SparseCSCView
    from anndata._core.views import SparseCSRMatrixView as SparseCSRView
else:
    from anndata._core.views import ArrayView, SparseCSCView, SparseCSRView

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING

from packaging.version import Version
from scanpy.get import obs_df

if TYPE_CHECKING:
    import numpy as np
    from anndata import AnnData

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
    "get_vector",
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


def get_vector(adata: AnnData, key: str, *, layer: str | None = None, use_raw: bool | None = None) -> np.ndarray:
    """Return a 1-D vector of values for ``key`` from ``.obs`` or ``.var_names``.

    Drop-in replacement for ``AnnData.obs_vector`` that avoids the ``FutureWarning``
    emitted by anndata >= 0.13 (see https://github.com/scverse/squidpy/issues/1261).
    Delegates to :func:`scanpy.get.obs_df`, whose contract already covers ``.obs``
    columns and ``.var_names`` genes (honoring ``layer`` / ``use_raw`` and preserving
    ``Categorical`` dtype), mirroring the ``use_raw``-then-``layer`` precedence of the
    original ``obs_vector`` calls.
    """
    if use_raw and key not in adata.obs:
        return obs_df(adata, keys=[key], use_raw=True)[key].values
    return obs_df(adata, keys=[key], layer=layer)[key].values


CAN_USE_SPARSE_ARRAY = Version(version("anndata")) >= Version("0.11.0rc1")
if CAN_USE_SPARSE_ARRAY:
    from anndata._core.views import ArrayView
    from anndata._core.views import SparseCSCMatrixView as SparseCSCView
    from anndata._core.views import SparseCSRMatrixView as SparseCSRView
else:
    from anndata._core.views import ArrayView, SparseCSCView, SparseCSRView

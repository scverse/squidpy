from __future__ import annotations

import os
from functools import partial
from typing import TYPE_CHECKING

import legacy_api_wrap
from anndata._core.views import ArrayView
from anndata._core.views import SparseCSCMatrixView as SparseCSCView
from anndata._core.views import SparseCSRMatrixView as SparseCSRView
from anndata.acc import A
from legacy_api_wrap import legacy_api
from scanpy.get import obs_df
from scanpy.plotting.legacy import mpl_settings as _mpl_settings
from scanpy.plotting.legacy._tools.scatterplots import _add_categorical_legend as add_categorical_legend
from scanpy.plotting.legacy._tools.scatterplots import _panel_grid as panel_grid
from scanpy.plotting.legacy._utils import (
    add_colors_for_categorical_sample_annotation,
    set_default_colors_for_categorical_obs,
)
from scanpy.plotting.legacy.palettes import default_102 as default_palette

if TYPE_CHECKING:
    import numpy as np
    from anndata import AnnData

__all__ = [
    "old_positionals",
    "SKIP_OWN_FRAMES",
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


class _PositionalArgumentWarning(FutureWarning):
    """An argument that became keyword-only was passed positionally; that stops working in v1.9.0."""

    def __init__(self, message: str) -> None:
        super().__init__(f"{message}. Passing them positionally stops working in squidpy v1.9.0.")


#: ``warnings.warn(..., skip_file_prefixes=SKIP_OWN_FRAMES)`` points at the first frame outside
#: squidpy and the ``legacy_api`` shim, however many wrappers sit in between.
SKIP_OWN_FRAMES = (
    os.path.dirname(__file__) + os.sep,
    os.path.dirname(legacy_api_wrap.__file__) + os.sep,
)

#: For arguments that became keyword-only: an old positional call still works, with a
#: ``FutureWarning`` naming them (#1288). List the names in their v1.8.3 order, and put the
#: shim outside ``deprecated_params`` / ``deprecated_randomness_param`` so they see old names.
old_positionals = partial(legacy_api, category=_PositionalArgumentWarning, skip_file_prefixes=SKIP_OWN_FRAMES)


# ``scanpy.set_figure_params`` rebinds these module globals, so read them at call time.
def default_frameon() -> bool:
    return _mpl_settings.FRAMEON


def vector_friendly() -> bool:
    return _mpl_settings.VECTOR_FRIENDLY


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
    return obs_df(adata, keys=[key], use=None if layer is None else A.layers[layer])[key].values

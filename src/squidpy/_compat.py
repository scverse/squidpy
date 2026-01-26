from __future__ import annotations

from importlib.metadata import version

from packaging.version import Version
from scanpy.plotting._tools.scatterplots import _add_categorical_legend as add_categorical_legend
from scanpy.plotting._tools.scatterplots import _panel_grid as panel_grid
from scanpy.plotting._utils import add_colors_for_categorical_sample_annotation

__all__ = [
    # scanpy
    "set_default_colors_for_categorical_obs",
    "add_categorical_legend",
    "panel_grid",
    "add_colors_for_categorical_sample_annotation",
    # anndata
    "ArrayView",
    "SparseCSCView",
    "SparseCSRView",
]

# See https://github.com/scverse/squidpy/issues/1061 for more details
# scanpy around version 0.11.x- 0.12.x changed the function name from set_default_colors_for_categorical_obs to _set_default_colors_for_categorical_obs
# and then changed it back
# so to not track with versioning we use the underscore version first (current), fall back to non-underscore for older/future versions
try:
    from scanpy.plotting._utils import _set_default_colors_for_categorical_obs as set_default_colors_for_categorical_obs
except ImportError:
    from scanpy.plotting._utils import set_default_colors_for_categorical_obs


CAN_USE_SPARSE_ARRAY = Version(version("anndata")) >= Version("0.11.0rc1")
if CAN_USE_SPARSE_ARRAY:
    from anndata._core.views import ArrayView
    from anndata._core.views import SparseCSCMatrixView as SparseCSCView
    from anndata._core.views import SparseCSRMatrixView as SparseCSRView
else:
    from anndata._core.views import ArrayView, SparseCSCView, SparseCSRView

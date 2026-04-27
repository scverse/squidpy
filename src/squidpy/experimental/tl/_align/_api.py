"""Public alignment orchestrators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData

AlignmentMethod = Literal["stalign", "grid-ot"]

__all__ = ["align_by_landmarks", "align_obs"]


def align_obs(
    adata_ref: AnnData,
    adata_query: AnnData,
    *,
    ref_key: str = "spatial",
    query_key: str = "spatial",
    method: AlignmentMethod = "stalign",
    output_mode: Literal["obs", "return"] = "obs",
    key_added: str | None = None,
    inplace: bool = False,
    **method_kwargs: Any,
) -> AnnData | Any:
    """Align query observation coordinates to reference observation coordinates."""
    raise NotImplementedError("`align_obs` will be wired to alignment methods in a follow-up commit.")


def align_by_landmarks(
    sdata_ref: SpatialData,
    sdata_query: SpatialData | None = None,
    *,
    cs_name_ref: str,
    cs_name_query: str,
    landmarks_ref: Sequence[tuple[float, float]] | np.ndarray,
    landmarks_query: Sequence[tuple[float, float]] | np.ndarray,
    model: Literal["similarity", "affine"] = "similarity",
    output_mode: Literal["affine", "return"] = "affine",
    inplace: bool = True,
) -> SpatialData | Any:
    """Align SpatialData coordinate systems by user-provided landmarks."""
    raise NotImplementedError("`align_by_landmarks` will be implemented in a follow-up commit.")


def align_images(*args: Any, **kwargs: Any) -> None:
    """Reserved image alignment entry point."""
    raise NotImplementedError("Image alignment is reserved for a future implementation.")

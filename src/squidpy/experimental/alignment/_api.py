"""Public experimental alignment entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from anndata import AnnData

AlignmentMethod = Literal["stalign", "grid-ot"]

__all__ = ["align_obs"]


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


def align_images(*args: Any, **kwargs: Any) -> None:
    """Reserved image alignment entry point."""
    raise NotImplementedError("Image alignment is reserved for a future implementation.")

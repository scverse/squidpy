from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from anndata import AnnData

from squidpy._docs import d
from squidpy._validators import assert_one_of

from ._spage_impute import SpaGEParams, spage_impute

__all__ = ["impute"]

_ALLOWED_METHODS = ("spage",)


@d.dedent
def impute(
    st_adata: AnnData,
    sc_adata: AnnData,
    *,
    genes: Sequence[str] | None = None,
    method: str = "spage",
    method_params: SpaGEParams | Mapping[str, Any] | None = None,
    n_pv: int = 30,
    n_neighbors: int = 50,
    cosine_threshold: float = 0.3,
    use_raw: bool = False,
    layer: str | None = None,
    key_added: str = "spage",
    n_jobs: int | None = None,
    copy: bool = False,
) -> AnnData:
    """
    Impute spatially unmeasured genes in spatial data using a selected method.

    Parameters
    ----------
    st_adata
        Spatial AnnData object.
    sc_adata
        scRNA-seq AnnData object.
    genes
        Genes to impute. If `None`, uses genes present in `sc_adata` but missing from `st_adata`.
    method
        Imputation method to use. Valid options are:

            - ``"spage"`` - SpaGE imputation.
    method_params
        Optional method-specific parameters. For ``method="spage"``, provide :class:`SpaGEParams`
        or a mapping with matching field names.
    key_added
        Key added to `.obsm` for the imputed genes.
    copy
        Whether to return a copy of `st_adata`.

    Returns
    -------
    AnnData with imputed genes stored in `.obsm[key_added]`.
    """
    assert_one_of(method, _ALLOWED_METHODS, name="method")

    if method == "spage":
        if method_params is None:
            method_params = SpaGEParams(
                n_pv=n_pv,
                n_neighbors=n_neighbors,
                cosine_threshold=cosine_threshold,
                use_raw=use_raw,
                layer=layer,
                n_jobs=n_jobs,
            )
        elif isinstance(method_params, Mapping):
            method_params = SpaGEParams.from_mapping(method_params)

        return spage_impute(
            st_adata,
            sc_adata,
            genes=genes,
            params=method_params,
            key_added=key_added,
            copy=copy,
        )

    raise NotImplementedError(f"Method `{method}` is not yet implemented.")

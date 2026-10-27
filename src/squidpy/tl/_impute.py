from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from anndata import AnnData

from squidpy._docs import d
from squidpy._validators import assert_one_of

from ._spage_impute import spage_impute

__all__ = ["impute"]

_ALLOWED_METHODS = ("spage",)


@d.dedent
def impute(
    st_adata: AnnData,
    sc_adata: AnnData,
    *,
    genes: Sequence[str] | None = None,
    method: str = "spage",
    n_pv: int = 30,
    n_neighbors: int = 50,
    cosine_threshold: float = 0.3,
    use_raw: bool = False,
    layer: str | None = None,
    key_added: str = "spage",
    n_jobs: int | None = None,
    copy: bool = False,
) -> pd.DataFrame | None:
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
    n_pv
        Number of principal vectors used for alignment.
    n_neighbors
        Number of nearest neighbors used for imputation.
    cosine_threshold
        Threshold on cosine similarity to select effective principal vectors.
    use_raw
        Whether to use `.raw` for expression values.
    layer
        Layer to use for expression values.
    key_added
        Key added to `.obsm` for the imputed genes.
    n_jobs
        Number of parallel jobs for nearest neighbors search.
    copy
        If `True`, return the imputed dataframe. Otherwise, save it to `.obsm[key_added]`.
    Returns
    -------
    If ``copy = True``, returns a :class:`pandas.DataFrame` with imputed values.
    Otherwise, stores the result in :attr:`anndata.AnnData.obsm` ``[key_added]`` and returns `None`.
    """
    assert_one_of(method, _ALLOWED_METHODS, name="method")

    if method == "spage":
        return spage_impute(
            st_adata,
            sc_adata,
            genes=genes,
            n_pv=n_pv,
            n_neighbors=n_neighbors,
            cosine_threshold=cosine_threshold,
            use_raw=use_raw,
            layer=layer,
            key_added=key_added,
            n_jobs=n_jobs,
            copy=copy,
        )

    raise NotImplementedError(f"Method `{method}` is not yet implemented.")

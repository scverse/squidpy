from __future__ import annotations

from collections.abc import Sequence

import numba
import numpy as np
import pandas as pd
from anndata import AnnData
from scanpy import logging as logg
from scipy import linalg
from scipy.sparse import issparse, spmatrix
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from squidpy._docs import d
from squidpy._utils import NDArrayA
from squidpy.gr._utils import _extract_expression, _save_data

__all__ = ["spage_impute"]


@d.dedent
def spage_impute(
    st_adata: AnnData,
    sc_adata: AnnData,
    *,
    genes: Sequence[str] | None = None,
    n_pv: int = 30,
    n_neighbors: int = 50,
    cosine_threshold: float = 0.3,
    use_raw: bool = False,
    layer: str | None = None,
    key_added: str = "spage",
    n_jobs: int | None = None,
) -> AnnData:
    """
    Impute spatially unmeasured genes in spatial data using SpaGE.

    Parameters
    ----------
    st_adata
        Spatial AnnData object.
    sc_adata
        scRNA-seq AnnData object.
    genes
        Genes to impute. If `None`, uses genes present in `sc_adata` but missing from `st_adata`.
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

    Returns
    -------
    AnnData with imputed genes stored in `.obsm[key_added]`.
    """
    start = logg.info("Running SpaGE imputation")


    if n_pv <= 0:
        raise ValueError("`n_pv` must be positive.")
    if n_neighbors <= 0:
        raise ValueError("`n_neighbors` must be positive.")
    if cosine_threshold < 0:
        raise ValueError("`cosine_threshold` must be non-negative.")

    genes_to_predict = _resolve_genes_to_predict(st_adata, sc_adata, genes)
    shared_genes = _shared_genes(st_adata, sc_adata)

    if n_pv > len(shared_genes):
        raise ValueError(f"`n_pv` must be <= number of shared genes ({len(shared_genes)}), found `{n_pv}`.")

    sc_shared, _ = _extract_expression(sc_adata, genes=shared_genes, use_raw=use_raw, layer=layer)
    st_shared, _ = _extract_expression(st_adata, genes=shared_genes, use_raw=use_raw, layer=layer)
    sc_target, _ = _extract_expression(sc_adata, genes=genes_to_predict, use_raw=use_raw, layer=layer)

    sc_shared = _standardize(sc_shared)
    st_shared = _standardize(st_shared)

    source_components = _fit_components(sc_shared, n_pv)
    target_components = _fit_components(st_shared, n_pv)

    source_components = _orthonormalize(source_components)
    target_components = _orthonormalize(target_components)

    n_pv_eff = min(n_pv, source_components.shape[0], target_components.shape[0])
    if n_pv_eff <= 0:
        raise ValueError("No principal vectors could be computed.")

    source_pv, target_pv, cosine = _compute_principal_vectors(source_components, target_components, n_pv_eff)

    effective_n_pv = int(np.sum(np.diag(cosine) > cosine_threshold))
    if effective_n_pv <= 0:
        raise ValueError("No effective principal vectors found. Consider lowering `cosine_threshold` or `n_pv`.")

    S = source_pv[:effective_n_pv].T

    sc_proj = _dot(sc_shared, S)
    st_proj = _dot(st_shared, S)

    n_neighbors = min(n_neighbors, sc_proj.shape[0])
    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="auto",
        n_jobs=n_jobs,
    )
    nn.fit(sc_proj)
    distances, indices = nn.kneighbors(st_proj, return_distance=True)

    weights, mask = _compute_weights(distances)
    imputed = _impute_from_neighbors(weights, mask, indices, sc_target)

    result = pd.DataFrame(imputed, index=st_adata.obs_names, columns=genes_to_predict)
    _save_data(st_adata, attr="obsm", key=key_added, data=result, time=start)
    return st_adata


def _resolve_genes_to_predict(
    st_adata: AnnData,
    sc_adata: AnnData,
    genes: Sequence[str] | None,
) -> list[str]:
    if genes is None:
        genes_to_predict = [g for g in sc_adata.var_names if g not in st_adata.var_names]
    else:
        genes_to_predict = [g for g in genes if g in sc_adata.var_names]
        missing = [g for g in genes if g not in sc_adata.var_names]
        if missing:
            raise ValueError(f"Genes not found in `sc_adata`: {missing}")
        genes_to_predict = [g for g in genes_to_predict if g not in st_adata.var_names]
    if not genes_to_predict:
        raise ValueError("No genes to impute. Ensure `genes` are in `sc_adata` and absent from `st_adata`.")
    return genes_to_predict


def _shared_genes(st_adata: AnnData, sc_adata: AnnData) -> list[str]:
    shared = [g for g in st_adata.var_names if g in sc_adata.var_names]
    if not shared:
        raise ValueError("No shared genes between `st_adata` and `sc_adata`.")
    return shared


def _standardize(X: NDArrayA | spmatrix) -> NDArrayA | spmatrix:
    if issparse(X):
        X = X.toarray()
    scaler = StandardScaler(with_mean=True, copy=True)
    return scaler.fit_transform(X)


def _fit_components(X: NDArrayA | spmatrix, n_components: int) -> NDArrayA:
    reducer = PCA(n_components=n_components, svd_solver="arpack", random_state=0)
    reducer.fit(X)
    return reducer.components_


def _orthonormalize(components: NDArrayA) -> NDArrayA:
    return linalg.orth(components.T).T


def _compute_principal_vectors(
    source_factors: NDArrayA,
    target_factors: NDArrayA,
    n_pv: int,
) -> tuple[NDArrayA, NDArrayA, NDArrayA]:
    u, _, v = np.linalg.svd(source_factors @ target_factors.T, full_matrices=False)
    source_pv = (u.T @ source_factors)[:n_pv]
    target_pv = (v @ target_factors)[:n_pv]
    source_pv = _normalize_rows(source_pv)
    target_pv = _normalize_rows(target_pv)
    cosine = source_pv @ target_pv.T
    return source_pv, target_pv, cosine


def _normalize_rows(X: NDArrayA) -> NDArrayA:
    denom = np.linalg.norm(X, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return X / denom


def _dot(X: NDArrayA | spmatrix, S: NDArrayA) -> NDArrayA:
    return X @ S


@numba.njit(cache=True)
def _compute_weights(distances: NDArrayA, threshold: float = 1.0) -> tuple[NDArrayA, NDArrayA]:
    n_obs, n_neighbors = distances.shape
    weights = np.zeros((n_obs, n_neighbors), dtype=np.float64)
    mask = distances < threshold

    for i in range(n_obs):
        denom = 0.0
        count = 0
        for j in range(n_neighbors):
            if mask[i, j]:
                denom += distances[i, j]
                count += 1
        if count <= 1 or denom == 0.0:
            continue
        for j in range(n_neighbors):
            if mask[i, j]:
                weights[i, j] = (1.0 - distances[i, j] / denom) / (count - 1)

    return weights, mask


def _impute_from_neighbors(
    weights: NDArrayA,
    mask: NDArrayA,
    indices: NDArrayA,
    y_train: NDArrayA | spmatrix,
) -> NDArrayA:
    n_obs = weights.shape[0]
    n_genes = y_train.shape[1]
    result = np.zeros((n_obs, n_genes), dtype=np.float64)

    for i in range(n_obs):
        valid = mask[i]
        if not np.any(valid):
            continue
        w = weights[i, valid]
        idx = indices[i, valid]
        y_sub = y_train[idx]
        imputed = w @ y_sub
        result[i] = np.asarray(imputed).ravel()

    return result

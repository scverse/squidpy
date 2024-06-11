from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.spatial import cKDTree
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from spatialdata import SpatialData
from utag import utag
import itertools
from scipy.stats import ranksums

from squidpy._utils import NDArrayA

__all__ = ["calculate_niche"]


def calculate_niche(
    adata: AnnData | SpatialData,
    groups: str,
    flavor: str = "neighborhood",
    library_key: str | None = None,
    radius: float | None = None,
    n_neighbors: int | None = None,
    limit_to: str | list[Any] | None = None,
    table_key: str | None = None,
    spatial_key: str = "spatial",
    copy: bool = False,
) -> AnnData | pd.DataFrame:
    # check whether anndata or spatialdata is provided and if spatialdata, check whether a table with the provided groups is present
    is_sdata = False
    if isinstance(adata, SpatialData):
        is_sdata = True
        if table_key is not None:
            table = adata.tables[table_key]
        else:
            if len(adata.tables) > 1:
                count = 0
                for key in adata.tables.keys():
                    if groups in table.obs:
                        count += 1
                        table_key = key
                if count > 1:
                    raise ValueError(
                        f"Multiple tables in `spatialdata` with group `{groups}` detected. Please specify which table to use in `table_key`."
                    )
                elif count == 0:
                    raise ValueError(
                        f"Group `{groups}` not found in any table in `spatialdata`. Please specify a valid group in `groups`."
                    )
                else:
                    table = adata.tables[table_key]
            else:
                ((key, table),) = adata.tables.items()
                if groups not in table.obs:
                    raise ValueError(
                        f"Group {groups} not found in table in `spatialdata`. Please specify a valid group in `groups`."
                    )
    else:
        table = adata

    # check whether to use radius or knn for neighborhood profile calculation
    if radius is None and n_neighbors is None:
        raise ValueError("Either `radius` or `n_neighbors` must be provided, but both are `None`.")
    if radius is not None and n_neighbors is not None:
        raise ValueError("Either `radius` and `n_neighbors` must be provided, but both were provided.")

    # subset adata if only observations within specified groups are to be considered
    if limit_to is not None:
        if isinstance(limit_to, str):
            limit_to = [limit_to]
        table_subset = table[table.obs[groups].isin([limit_to])]
    else:
        table_subset = table

    if flavor == "neighborhood":
        rel_nhood_profile, abs_nhood_profile = _calculate_neighborhood_profile(
            table, groups, radius, n_neighbors, table_subset, spatial_key
        )
        df = pd.DataFrame(rel_nhood_profile, index=table_subset.obs.index)
        nhood_table = _df_to_adata(df)
        sc.pp.neighbors(nhood_table, n_neighbors=n_neighbors, use_rep="X")
        sc.tl.leiden(nhood_table)
        table.obs["niche"] = nhood_table.obs["leiden"]
        if is_sdata:
            if copy:
                return nhood_table
            adata.tables[f"{flavor}_niche"] = nhood_table
        else:
            if copy:
                return df
            df = df.reindex(table.obs.index)
            print(df.head())
            table.obsm[f"{flavor}_niche"] = df

    elif flavor == "utag":
        result = utag(
            table_subset,
            slide_key=library_key,
            max_dist=10,
            normalization_mode="l1_norm",
            apply_clustering=True,
            clustering_method="leiden",
            resolutions=1.0,
        )
        if is_sdata:
            if copy:
                return result
            adata.tables[f"{flavor}_niche"] = result
        else:
            if copy:
                return result
            df = df.reindex(table.obs.index)
            table.obsm[f"{flavor}_niche"] = df


def _calculate_neighborhood_profile(
    adata: AnnData | SpatialData,
    groups: str,
    radius: float | None,
    n_neighbors: int | None,
    subset: AnnData,
    spatial_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # reset index
    adata.obs = adata.obs.reset_index()

    if n_neighbors is not None:
        # get k-nearest neighbors for each observation
        tree = KDTree(adata.obsm[spatial_key])
        _, indices = tree.query(subset.obsm[spatial_key], k=n_neighbors)
    else:
        # get neighbors within a given radius for each observation
        tree = cKDTree(adata.obsm[spatial_key])
        indices = tree.query_ball_point(subset.obsm[spatial_key], r=radius)

    # get unique categories
    category_arr = adata.obs[groups].values
    unique_categories = np.unique(category_arr)

    # get obs x k matrix where each column is the category of the k-th neighbor
    cat_by_id = np.take(category_arr, indices)

    # in obs x k matrix convert categorical values to numerical values
    cat_indices = {category: index for index, category in enumerate(unique_categories)}
    cat_values = np.vectorize(cat_indices.get)(cat_by_id)

    # For each obs calculate absolute frequency for all (not just k) categories, given the subset of categories present in obs x k matrix
    m, k = cat_by_id.shape
    abs_freq = np.zeros((m, len(unique_categories)), dtype=int)
    np.add.at(abs_freq, (np.arange(m)[:, None], cat_values), 1)

    # normalize by n_neighbors to get relative frequency of each category
    rel_freq = abs_freq / k

    return rel_freq, abs_freq


def _df_to_adata(df: pd.DataFrame) -> AnnData:
    df.index = df.index.map(str)
    adata = AnnData(X=df)
    adata.obs.index = df.index
    return adata

def pairwise_niche_comparison(
        adata: AnnData,
        niche_key: str,
) -> pd.DataFrame:
    niches = adata.obs[niche_key].unique().tolist()
    niche_dict = {}
    for niche in adata.obs[niche_key].unique():
        niche_adata = adata[adata.obs[niche_key] == niche]
        n_cols = niche_adata.X.shape[1]
        arr = np.ones(n_cols)
        for i in range(n_cols):
            col_data = niche_adata.X.getcol(i).data
            percentile_99 = np.percentile(col_data, 99)
            arr[i] = percentile_99
        niche_dict[niche] = arr
    var_by_niche = pd.DataFrame(niche_dict)
    result = pd.DataFrame(index=niches, columns=niches, data=None, dtype=float)
    combinations = list(itertools.combinations_with_replacement(niches, 2))
    for pair in combinations:
        p_val = ranksums(var_by_niche[pair[0]], var_by_niche[pair[1]], alternative="two-sided")[1]
        result.at[pair[0], pair[1]] = p_val
        result.at[pair[1], pair[0]] = p_val
    return result

def mean_fide_score(
    adatas: AnnData | list[AnnData],
    obs_key: str,
    slide_key: str | None = None,
    n_classes: int | None = None,
) -> float:
    """Mean FIDE score over all slides. A low score indicates a great domain continuity."""
    return float(np.mean([fide_score(adata, obs_key, n_classes=n_classes) for adata in _iter_uid(adatas, slide_key)]))


def fide_score(adata: AnnData, obs_key: str, n_classes: int | None = None) -> float:
    """
    F1-score of intra-domain edges (FIDE). A high score indicates a great domain continuity.

    The F1-score is computed for every class, then all F1-scores are averaged. If some classes
    are not predicted, the `n_classes` argument allows to pad with zeros before averaging the F1-scores.
    """
    i_left, i_right = adata.obsp["spatial_connectivities"].nonzero()
    classes_left, classes_right = (
        adata.obs.iloc[i_left][obs_key],
        adata.obs.iloc[i_right][obs_key],
    )

    f1_scores = metrics.f1_score(classes_left, classes_right, average=None)

    if n_classes is None:
        return float(f1_scores.mean())

    assert n_classes >= len(f1_scores), f"Expected {n_classes:=}, but found {len(f1_scores)}, which is greater"

    return float(np.pad(f1_scores, (0, n_classes - len(f1_scores))).mean())


def jensen_shannon_divergence(adatas: AnnData | list[AnnData], obs_key: str, slide_key: str | None = None) -> float:
    """Jensen-Shannon divergence (JSD) over all slides

    Args:
        adata: One or a list of AnnData object(s)
        obs_key: The key containing the clusters
        slide_key: The slide ID obs key

    Returns:
        A float corresponding to the JSD
    """
    distributions = [
        adata.obs[obs_key].value_counts(sort=False).values for adata in _iter_uid(adatas, slide_key, obs_key)
    ]

    return _jensen_shannon_divergence(np.array(distributions))


def _jensen_shannon_divergence(distributions: NDArrayA) -> float:
    """Compute the Jensen-Shannon divergence (JSD) for a multiple probability distributions.

    The lower the score, the better distribution of clusters among the different batches.

    Args:
        distributions: An array of shape (B x C), where B is the number of batches, and C is the number of clusters. For each batch, it contains the percentage of each cluster among cells.

    Returns:
        A float corresponding to the JSD
    """
    distributions = distributions / distributions.sum(1)[:, None]
    mean_distribution = np.mean(distributions, 0)

    return _entropy(mean_distribution) - float(np.mean([_entropy(dist) for dist in distributions]))


def _entropy(distribution: NDArrayA) -> float:
    """Shannon entropy

    Args:
        distribution: An array of probabilities (should sum to one)

    Returns:
        The Shannon entropy
    """
    return float(-(distribution * np.log(distribution + 1e-8)).sum())


def _iter_uid(adatas: AnnData | list[AnnData], slide_key: str | None, obs_key: str | None = None) -> Iterator[AnnData]:
    if isinstance(adatas, AnnData):
        adatas = [adatas]

    if obs_key is not None:
        categories = set.union(*[set(adata.obs[obs_key].unique().dropna()) for adata in adatas])
        for adata in adatas:
            adata.obs[obs_key] = adata.obs[obs_key].astype("category").cat.set_categories(categories)

    for adata in adatas:
        if slide_key is not None:
            for slide_id in adata.obs[slide_key].unique():
                yield adata[adata.obs[slide_key] == slide_id]
        else:
            yield adata

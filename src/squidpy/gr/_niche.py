from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix, vstack
from scipy.stats import ranksums
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
from spatialdata import SpatialData

from squidpy._utils import NDArrayA

__all__ = ["calculate_niche"]


def calculate_niche(
    adata: AnnData | SpatialData,
    groups: str,
    flavor: str = "neighborhood",
    library_key: str | None = None,
    table_key: str | None = None,
    spatial_key: str = "spatial",
    adj_subsets: list[int] | None = None,
    aggregation: str = "mean",
    spatial_connectivities_key: str = "spatial_connectivities",
    spatial_distances_key: str = "spatial_distances",
    copy: bool = False,
) -> AnnData | pd.DataFrame:
    """Calculate niches (spatial clusters) based on a user-defined method in 'flavor'.
    The resulting niche labels with be stored in 'adata.obs'. If flavor = 'all' then all available methods
    will be applied and additionally compared using cluster validation scores.

    Parameters
    ----------
    %(adata)s
    groups
        groups based on which to calculate neighborhood profile.
    flavor
        Method to use for niche calculation. Available options are:

            - `{c.NEIGHBORHOOD.s!r}` - cluster the neighborhood profile.
            - `{c.SPOT.s!r}` - calculate niches using optimal transport.
            - `{c.BANKSY.s!r}`- use Banksy algorithm.
            - `{c.CELLCHARTER.s!r}` - use cellcharter.
            - `{c.UTAG.s!r}` - use utag algorithm (matrix multiplication).
            - `{c.ALL.s!r}` - apply all available methods and compare them using cluster validation scores.
    %(library_key)s
    subset
        Restrict niche calculation to a subset of the data.
    table_key
        Key in `spatialdata.tables` to specify an 'anndata' table. Only necessary if 'sdata' is passed.
    spatial_key
        Location of spatial coordinates in `adata.obsm`.
    %(copy)s
    """

    # check whether anndata or spatialdata is provided and if spatialdata, check whether a table with the provided groups is present
    is_sdata = False
    if isinstance(adata, SpatialData):
        is_sdata = True
        if table_key is not None:
            sdata = adata
            adata = adata.tables[table_key].copy()
        else:
            if len(adata.tables) > 1:
                count = 0
                for table in adata.tables.keys():
                    if groups in table.obs:
                        count += 1
                        table_key = table
                if count > 1:
                    raise ValueError(
                        f"Multiple tables in `spatialdata` with group `{groups}` detected. Please specify which table to use in `table_key`."
                    )
                elif count == 0:
                    raise ValueError(
                        f"Group `{groups}` not found in any table in `spatialdata`. Please specify a valid group in `groups`."
                    )
                else:
                    adata = adata.tables[table_key].copy()
            else:
                ((key, adata),) = adata.tables.items()
                if groups not in adata.obs:
                    raise ValueError(
                        f"Group {groups} not found in table in `spatialdata`. Please specify a valid group in `groups`."
                    )

    if flavor == "neighborhood":
        rel_nhood_profile, abs_nhood_profile = _calculate_neighborhood_profile(
            adata, groups, spatial_connectivities_key
        )
        df = pd.DataFrame(rel_nhood_profile, index=adata.obs.index)
        nhood_table = _df_to_adata(df)
        if copy:
            return df
        else:
            if is_sdata:
                sdata.tables[f"{flavor}_niche"] = nhood_table
            else:
                adata.obsm["neighborhood_profile"] = df

    elif flavor == "utag":
        new_feature_matrix = _utag(adata, normalize_adj=True, spatial_connectivity_key=spatial_connectivities_key)
        if copy:
            return new_feature_matrix
        else:
            if is_sdata:
                sdata.tables[f"{flavor}_niche"] = new_feature_matrix
            else:
                adata.layers["utag"] = new_feature_matrix

    elif flavor == "cellcharter":
        adj_matrix_subsets = []
        if isinstance(adj_subsets, list):
            for k in adj_subsets:
                if k == 0:
                    adj_matrix_subsets.append(adata.obsp[spatial_connectivities_key])
                else:
                    adj_matrix_subsets.append(
                        _get_adj_matrix_subsets(
                            adata.obsp[spatial_connectivities_key], adata.obsp[spatial_distances_key], k
                        )
                    )
            if aggregation == "mean":
                inner_products = [adj_subset.dot(adata.X) for adj_subset in adj_matrix_subsets]
            elif aggregation == "variance":
                inner_products = [
                    _aggregate_var(matrix, adata.obsp[spatial_connectivities_key], adata) for matrix in inner_products
                ]
            else:
                raise ValueError(
                    f"Invalid aggregation method '{aggregation}'. Please choose either 'mean' or 'variance'."
                )
            concatenated_matrix = vstack(inner_products)
            if copy:
                return concatenated_matrix
            else:
                if is_sdata:
                    sdata.tables[f"{flavor}_niche"] = ad.AnnData(concatenated_matrix)
                else:
                    adata.obsm[f"{flavor}_niche"] = concatenated_matrix
        else:
            raise ValueError(
                "Flavor 'cellcharter' requires list of neighbors to build adjacency matrices. Please provide a list of k_neighbors for 'adj_subsets'."
            )


def _calculate_neighborhood_profile(
    adata: AnnData | SpatialData,
    groups: str,
    spatial_connectivities_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # get obs x neighbor matrix from sparse matrix
    matrix = adata.obsp[spatial_connectivities_key].tocoo()
    nonzero_indices = np.split(matrix.col, matrix.row.searchsorted(np.arange(1, matrix.shape[0])))
    neighbor_matrix = pd.DataFrame(nonzero_indices)

    # get unique categories
    category_arr = adata.obs[groups].values
    unique_categories = np.unique(category_arr)

    # get obs x k matrix where each column is the category of the k-th neighbor
    cat_by_id = np.take(category_arr, neighbor_matrix)

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


def _utag(adata: AnnData, normalize_adj: bool, spatial_connectivity_key: str) -> AnnData:
    """Performas inner product of adjacency matrix and feature matrix,
    such that each observation inherits features from its immediate neighbors as described in UTAG paper.

    Parameters
    ----------
    adata
        Annotated data matrix.
    normalize
        If 'True', aggregate by the mean, else aggregate by the sum."""

    adjacency_matrix = adata.obsp[spatial_connectivity_key]

    if normalize_adj:
        return normalize(adjacency_matrix, norm="l1", axis=1) @ adata.X
    else:
        return adjacency_matrix @ adata.X


def _get_adj_matrix_subsets(connectivities: csr_matrix, distances: csr_matrix, k_neighbors: int) -> csr_matrix:
    # Convert the distance matrix to a dense format for easier manipulation
    dist_dense = distances.todense()

    # Find the indices of the k closest neighbors for each row
    closest_neighbors_indices = np.argsort(dist_dense, axis=1)[:, :k_neighbors]

    # Initialize lists to collect data for the new sparse matrix
    rows = []
    cols = []
    data = []

    # Iterate over each row to construct the new adjacency matrix
    for row in range(dist_dense.shape[0]):
        for col in closest_neighbors_indices[row].flat:
            rows.append(row)
            cols.append(col)
            data.append(connectivities[row, col])

    # Create the new sparse matrix with the reduced neighbors
    new_adj_matrix = csr_matrix((data, (rows, cols)), shape=connectivities.shape)
    print(new_adj_matrix.shape)
    return new_adj_matrix


def _df_to_adata(df: pd.DataFrame) -> AnnData:
    df.index = df.index.map(str)
    adata = AnnData(X=df)
    adata.obs.index = df.index
    return adata


def _aggregate_var(product: csr_matrix, connectivities: csr_matrix, adata: AnnData) -> csr_matrix:
    mean_squared = connectivities.dot(adata.X.multiply(adata.X))
    return mean_squared - (product.multiply(product))


def pairwise_niche_comparison(
    adata: AnnData,
    library_key: str,
) -> pd.DataFrame:
    """Do a simple pairwise DE test on the 99th percentile of each gene for each niche.
    Can be used to plot heatmap showing similar (large p-value) or different (small p-value) niches.
    For validating niche results, the niche pairs that are similar in expression are the ones of interest because
    it could hint at niches not being well defined in those cases."""
    niches = adata.obs[library_key].unique().tolist()
    niche_dict = {}
    # for each niche, calculate the 99th percentile of each gene
    for niche in adata.obs[library_key].unique():
        niche_adata = adata[adata.obs[library_key] == niche]
        n_cols = niche_adata.X.shape[1]
        arr = np.ones(n_cols)
        for i in range(n_cols):
            col_data = niche_adata.X.getcol(i).data
            percentile_99 = np.percentile(col_data, 99)
            arr[i] = percentile_99
        niche_dict[niche] = arr
    # create 99th percentile count x niche matrix
    var_by_niche = pd.DataFrame(niche_dict)
    result = pd.DataFrame(index=niches, columns=niches, data=None, dtype=float)
    # construct all pairs (unordered and with pairs of the same niche)
    combinations = list(itertools.combinations_with_replacement(niches, 2))
    # create a p-value matrix for all niche pairs
    for pair in combinations:
        p_val = ranksums(var_by_niche[pair[0]], var_by_niche[pair[1]], alternative="two-sided")[1]
        result.at[pair[0], pair[1]] = p_val
        result.at[pair[1], pair[0]] = p_val
    return result


def mean_fide_score(
    adatas: AnnData | list[AnnData],
    library_key: str,
    slide_key: str | None = None,
    n_classes: int | None = None,
) -> float:
    """Mean FIDE score over all slides. A low score indicates a great domain continuity."""
    return float(
        np.mean([fide_score(adata, library_key, n_classes=n_classes) for adata in _iter_uid(adatas, slide_key)])
    )


def fide_score(adata: AnnData, library_key: str, n_classes: int | None = None) -> float:
    """
    F1-score of intra-domain edges (FIDE). A high score indicates a great domain continuity.

    The F1-score is computed for every class, then all F1-scores are averaged. If some classes
    are not predicted, the `n_classes` argument allows to pad with zeros before averaging the F1-scores.
    """
    i_left, i_right = adata.obsp["spatial_connectivities"].nonzero()
    classes_left, classes_right = (
        adata.obs.iloc[i_left][library_key],
        adata.obs.iloc[i_right][library_key],
    )

    f1_scores = metrics.f1_score(classes_left, classes_right, average=None)

    if n_classes is None:
        return float(f1_scores.mean())

    assert n_classes >= len(f1_scores), f"Expected {n_classes:=}, but found {len(f1_scores)}, which is greater"

    return float(np.pad(f1_scores, (0, n_classes - len(f1_scores))).mean())


def jensen_shannon_divergence(adatas: AnnData | list[AnnData], library_key: str, slide_key: str | None = None) -> float:
    """Jensen-Shannon divergence (JSD) over all slides"""
    distributions = [
        adata.obs[library_key].value_counts(sort=False).values for adata in _iter_uid(adatas, slide_key, library_key)
    ]

    return _jensen_shannon_divergence(np.array(distributions))


def _jensen_shannon_divergence(distributions: NDArrayA) -> float:
    """Compute the Jensen-Shannon divergence (JSD) for a multiple probability distributions.
    The lower the score, the better distribution of clusters among the different batches.

    Parameters
    ----------
    distributions
        An array of shape (B x C), where B is the number of batches, and C is the number of clusters. For each batch, it contains the percentage of each cluster among cells.

    Returns
        JSD (float)
    """
    distributions = distributions / distributions.sum(1)[:, None]
    mean_distribution = np.mean(distributions, 0)

    return _entropy(mean_distribution) - float(np.mean([_entropy(dist) for dist in distributions]))


def _entropy(distribution: NDArrayA) -> float:
    """Shannon entropy

    Parameters
    ----------
        distribution: An array of probabilities (should sum to one)

    Returns
        The Shannon entropy
    """
    return float(-(distribution * np.log(distribution + 1e-8)).sum())


def _iter_uid(
    adatas: AnnData | list[AnnData], slide_key: str | None, library_key: str | None = None
) -> Iterator[AnnData]:
    if isinstance(adatas, AnnData):
        adatas = [adatas]

    if library_key is not None:
        categories = set.union(*[set(adata.obs[library_key].unique().dropna()) for adata in adatas])
        for adata in adatas:
            adata.obs[library_key] = adata.obs[library_key].astype("category").cat.set_categories(categories)

    for adata in adatas:
        if slide_key is not None:
            for slide in adata.obs[slide_key].unique():
                yield adata[adata.obs[slide_key] == slide]
        else:
            yield adata


def _compare_niche_definitions(adata: AnnData, niche_definitions: list[str]) -> pd.DataFrame:
    """Given different clustering results, compare them using different scores."""

    result = pd.DataFrame(index=niche_definitions, columns=niche_definitions, data=None, dtype=float)
    combinations = list(itertools.combinations_with_replacement(niche_definitions, 2))
    scores = {"ARI:": adjusted_rand_score, "NMI": normalized_mutual_info_score, "FMI": fowlkes_mallows_score}

    # for each score, apply it on all pairs of niche definitions
    for score_name, score_func in scores.items():
        for pair in combinations:
            score = score_func(adata.obs[pair[0]], adata.obs[pair[1]])
            result.at[pair[0], pair[1]] = score
            result.at[pair[1], pair[0]] = score
        adata.uns[f"niche_definition_comparison_{score_name}"] = result

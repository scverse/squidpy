from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps
from anndata import AnnData
from scipy.sparse import csr_matrix, hstack, issparse, spdiags
from scipy.stats import ranksums
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, normalize
from spatialdata import SpatialData

from squidpy._utils import NDArrayA

__all__ = ["calculate_niche", "build_consensus_niche"]


def calculate_niche(
    adata: AnnData | SpatialData,
    groups: str,
    flavor: str = "neighborhood",
    library_key: str | None = None,
    table_key: str | None = None,
    mask: pd.core.series.Series = None,
    n_neighbors: int = 15,
    resolutions: int | list[float] | None = None,
    subset_groups: list[str] | None = None,
    min_niche_size: int | None = None,
    scale: bool = True,
    abs_nhood: bool = False,
    adj_subsets: int | list[int] | None = None,
    aggregation: str = "mean",
    n_components: int = 3,
    random_state: int = 42,
    spatial_key: str = "spatial",
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
            - `{c.CELLCHARTER.s!r}` - cluster adjacency matrix with Gaussian Mixture Model (GMM) using CellCharter's approach.
            - `{c.UTAG.s!r}` - use utag algorithm (matrix multiplication).
    %(library_key)s
    subset
        Restrict niche calculation to a subset of the data.
    table_key
        Key in `spatialdata.tables` to specify an 'anndata' table. Only necessary if 'sdata' is passed.
    mask
        Boolean array to filter cells which won't get assigned to a niche.
        Note that if you want to exclude these cells during neighborhood calculation already, you should subset your AnnData table before running 'sq.gr.spatial_neigbors'.
    n_neighbors
        Number of neighbors to use for 'scanpy.pp.neighbors' before clustering using leiden algorithm.
        Required if flavor == 'neighborhood' or flavor == 'UTAG'.
    resolutions
        List of resolutions to use for leiden clustering.
        Required if flavor == 'neighborhood' or flavor == 'UTAG'.
    subset_groups
        Groups (e.g. cell type categories) to ignore when calculating the neighborhood profile.
        Optional if flavor == 'neighborhood'.
        Optional if flavor == 'neighborhood'.
    min_niche_size
        Minimum required size of a niche. Niches with fewer cells will be labeled as 'not_a_niche'.
        Optional if flavor == 'neighborhood'.
    scale
        If 'True', compute z-scores of neighborhood profiles.
        Optional if flavor == 'neighborhood'.
    abs_nhood
        If 'True', calculate niches based on absolute neighborhood profile.
        Optional if flavor == 'neighborhood'.
    adj_subsets
        List of adjacency matrices to use e.g. [1,2,3] for 1,2,3 neighbors respectively.
        Required if flavor == 'cellcharter'.
    aggregation
        How to aggregate count matrices. Either 'mean' or 'variance'.
        Required if flavor == 'cellcharter'.
    n_components
        Number of components to use for GMM.
        Required if flavor == 'cellcharter'.
    random_state
        Random state to use for GMM.
        Required if flavor == 'cellcharter'.
    spatial_key
        Location of spatial coordinates in `adata.obsm`.
    spatial_connectivities_key
        Key in `adata.obsp` where spatial connectivities are stored.
    spatial_distances_key
        Key in `adata.obsp` where spatial distances are stored.
    %(copy)s
    """

    # check whether anndata or spatialdata is provided and if spatialdata, check whether a table with the provided groups is present if no table is specified
    if isinstance(adata, SpatialData):
        if table_key is not None:
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
            adata, groups, subset_groups, spatial_connectivities_key
        )
        if not abs_nhood:
            adata_neighborhood = ad.AnnData(X=rel_nhood_profile)
        else:
            adata_neighborhood = ad.AnnData(X=abs_nhood_profile)

        if scale:
            sc.pp.scale(adata_neighborhood, zero_center=True)

        if mask is not None:
            if subset_groups is not None:
                mask = mask[mask.index.isin(adata_neighborhood.obs.index)]
            adata_neighborhood = adata_neighborhood[mask]

        sc.pp.neighbors(adata_neighborhood, n_neighbors=n_neighbors, use_rep="X")

        if resolutions is not None:
            if not isinstance(resolutions, list):
                resolutions = [resolutions]
        else:
            raise ValueError("Please provide resolutions for leiden clustering.")

        for res in resolutions:
            sc.tl.leiden(adata_neighborhood, resolution=res, key_added=f"neighborhood_niche_res={res}")
            adata.obs[f"neighborhood_niche_res={res}"] = adata.obs.index.map(
                adata_neighborhood.obs[f"neighborhood_niche_res={res}"]
            ).fillna("not_a_niche")
            if min_niche_size is not None:
                counts_by_niche = adata.obs[f"neighborhood_niche_res={res}"].value_counts()
                to_filter = counts_by_niche[counts_by_niche < min_niche_size].index
                adata.obs[f"neighborhood_niche_res={res}"] = adata.obs[f"neighborhood_niche_res={res}"].apply(
                    lambda x, to_filter=to_filter: "not_a_niche" if x in to_filter else x
                )

    elif flavor == "utag":
        new_feature_matrix = _utag(adata, normalize_adj=True, spatial_connectivity_key=spatial_connectivities_key)
        adata_utag = ad.AnnData(X=new_feature_matrix)
        sc.tl.pca(adata_utag)
        sc.pp.neighbors(adata_utag, n_neighbors=n_neighbors, use_rep="X_pca")

        if resolutions is not None:
            if not isinstance(resolutions, list):
                resolutions = [resolutions]
        else:
            raise ValueError("Please provide resolutions for leiden clustering.")

        for res in resolutions:
            sc.tl.leiden(adata_utag, resolution=res, key_added=f"utag_res={res}")
            adata.obs[f"utag_res={res}"] = adata_utag.obs[f"utag_res={res}"].values

    elif flavor == "cellcharter":
        adjacency_matrix = adata.obsp[spatial_connectivities_key]
        if not isinstance(adj_subsets, list):
            if adj_subsets is not None:
                adj_subsets = list(range(adj_subsets + 1))
            else:
                raise ValueError(
                    "flavor 'cellcharter' requires adj_subsets to not be None. Specify list of values or maximum value of neighbors to use."
                )

        aggregated_matrices = []
        adj_hop = _setdiag(adjacency_matrix, 0)  # Remove self-loops, set diagonal to 0
        adj_visited = _setdiag(adjacency_matrix.copy(), 1)  # Track visited neighbors
        for k in adj_subsets:
            if k == 0:
                # If k == 0, we're using the original cell features (no neighbors)
                aggregated_matrices.append(adata.X)
            else:
                if k > 1:
                    adj_hop, adj_visited = _hop(adj_hop, adjacency_matrix, adj_visited)

                adj_hop_norm = _normalize(adj_hop)  # Normalize adjacency matrix for current hop

                # Apply aggregation, default to "mean" unless specified otherwise
                aggregated_matrix = _aggregate(adata, adj_hop_norm, aggregation)

                # Collect the aggregated matrices
                aggregated_matrices.append(aggregated_matrix)

        concatenated_matrix = hstack(aggregated_matrices)  # Stack all matrices horizontally
        arr = concatenated_matrix.toarray()  # Densify the sparse matrix

        niches = _get_GMM_clusters(arr, n_components, random_state)

        adata.obs[f"{flavor}_niche"] = pd.Categorical(niches)


def _calculate_neighborhood_profile(
    adata: AnnData,
    groups: str,
    subset_groups: list[str] | None,
    spatial_connectivities_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if subset_groups:
        adjacency_matrix = adata.obsp[spatial_connectivities_key].tocsc()
        obs_mask = ~adata.obs[groups].isin(subset_groups)
        adata = adata[obs_mask]
        adata = adata[obs_mask]

        # Update adjacency matrix such that it only contains connections to filtered observations
        adjacency_matrix = adjacency_matrix[obs_mask, :][:, obs_mask]
        adata.obsp[spatial_connectivities_key] = adjacency_matrix.tocsr()

    # get obs x neighbor matrix from sparse matrix
    matrix = adata.obsp[spatial_connectivities_key].tocoo()
    nonzero_indices = np.split(matrix.col, matrix.row.searchsorted(np.arange(1, matrix.shape[0])))
    neighbor_matrix = pd.DataFrame(nonzero_indices)

    # get unique categories
    unique_categories = np.unique(adata.obs[groups].values)

    # get obs x k matrix where each column is the category of the k-th neighbor
    indices_with_nan = neighbor_matrix.to_numpy()
    valid_indices = neighbor_matrix.fillna(-1).astype(int).to_numpy()
    cat_by_id = adata.obs[groups].values[valid_indices]
    cat_by_id[indices_with_nan == -1] = np.nan
    # cat_by_id = np.take(category_arr, neighbor_matrix)

    # in obs x k matrix convert categorical values to numerical values
    cat_indices = {category: index for index, category in enumerate(unique_categories)}
    cat_values = np.vectorize(cat_indices.get)(cat_by_id)

    # get obx x category matrix where each column is the absolute amount of a category in the neighborhood
    m, k = cat_by_id.shape
    abs_freq = np.zeros((m, len(unique_categories)), dtype=int)
    np.add.at(abs_freq, (np.arange(m)[:, None], cat_values), 1)

    # normalize by n_neighbors to get relative frequency of each category
    rel_freq = abs_freq / k

    return pd.DataFrame(rel_freq, index=adata.obs.index), pd.DataFrame(abs_freq, index=adata.obs.index)


def _utag(adata: AnnData, normalize_adj: bool, spatial_connectivity_key: str) -> AnnData:
    """Performs inner product of adjacency matrix and feature matrix,
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


def _setdiag(adjacency_matrix: sps.spmatrix, value: int) -> sps.spmatrix:
    if issparse(adjacency_matrix):
        adjacency_matrix = adjacency_matrix.tolil()
    adjacency_matrix.setdiag(value)
    adjacency_matrix = adjacency_matrix.tocsr()
    if value == 0:
        adjacency_matrix.eliminate_zeros()
    return adjacency_matrix


def _hop(
    adj_hop: sps.spmatrix, adj: sps.spmatrix, adj_visited: sps.spmatrix = None
) -> tuple[sps.spmatrix, sps.spmatrix]:
    adj_hop = adj_hop @ adj

    if adj_visited is not None:
        adj_hop = adj_hop > adj_visited
        adj_visited = adj_visited + adj_hop

    return adj_hop, adj_visited


def _normalize(adj: sps.spmatrix) -> sps.spmatrix:
    deg = np.array(np.sum(adj, axis=1)).squeeze()
    with np.errstate(divide="ignore"):
        deg_inv = 1 / deg
    deg_inv[deg_inv == float("inf")] = 0

    return spdiags(deg_inv, 0, len(deg_inv), len(deg_inv)) * adj


def _aggregate(adata: AnnData, normalized_adjacency_matrix: sps.spmatrix, aggregation: str = "mean") -> Any:
    if aggregation == "mean":
        aggregated_matrix = normalized_adjacency_matrix @ adata.X
    elif aggregation == "variance":
        mean_matrix = normalized_adjacency_matrix @ adata.X
        mean_squared_matrix = normalized_adjacency_matrix @ (adata.X * adata.X)
        aggregated_matrix = mean_squared_matrix - mean_matrix * mean_matrix
    else:
        raise ValueError(f"Invalid aggregation method '{aggregation}'. Please choose either 'mean' or 'variance'.")

    return aggregated_matrix


def _get_GMM_clusters(A: np.ndarray[float, Any], n_components: int, random_state: int) -> Any:
    """Returns niche labels generated by GMM clustering.
    Compared to cellcharter this approach is simplified by using sklearn's GaussianMixture model without stability analysis."""

    gmm = GaussianMixture(n_components=n_components, random_state=random_state, init_params="random_from_data")
    gmm.fit(A)
    labels = gmm.predict(A)

    return labels


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


def _get_subset_indices(df: pd.DataFrame, column: str) -> pd.Series:
    return df.groupby(column).apply(lambda x: set(x.index))


def _find_best_match(subset: set[str], other_subsets: dict[str, set[str]], exclude: set[str]) -> tuple[str, float]:
    """Find best matching niche pair between two sets of niche definitions.
    Niches which have already been matched, are excluded from further comparisons."""

    best_match = ""
    max_overlap = 0.0
    for other_subset, indices in other_subsets.items():
        if other_subset in exclude:
            continue  # Skip excluded matches
        overlap = len(subset & indices) / len(subset | indices)  # jaccard index
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = other_subset
    return (best_match, max_overlap)


def _get_initial_niches(niche_definitions: list[dict[str, set[str]]]) -> dict[str, set[str]]:
    """Select the niche definition with the fewest amount of unique niches."""

    min_niches = {}
    min_niche_count = float("inf")

    for niches in niche_definitions:
        niche_count = len(niches)

        if niche_count < min_niche_count:
            min_niches = niches
            min_niche_count = niche_count

    return min_niches


def _filter_overlap(initial_consensus: dict[str, set[str]]) -> dict[str, str]:
    """ "Remove labels which are present in multiple niches. Labels are always kept in the niche with higher average jaccard index."""

    filtered_consensus = {}
    processed_elements: set[str] = set()

    for key, values in initial_consensus.items():
        unique_values = values - processed_elements  # Remove already processed elements
        for value in unique_values:
            filtered_consensus[value] = key  # Swap key and value to make further processing easier
        processed_elements.update(unique_values)  # Mark value as processed

    return filtered_consensus


def build_consensus_niche(adata: AnnData, niche_definitions: list[str], merge: str = "union") -> AnnData:
    """Given multiple niche definitions, construct a consensus niche using set matching.
    Each niche definition is treated as a set of subsets. For each subset in set A we look for the best matching subset in set B.
    Once a match has been found, these sets are merged either by union or intersection. This merged set is then used as the new set A for the next iteration.
    The final consensus niches are filtered for overlapping labels and stored as a new column in `adata.obs`.
    Parameters
    ----------
    %(adata)s
    niche_definitions
        Name of columns in `adata.obs` where previously calculated niches are stored.
    merge
        - `{c.union.s!r}`- merge niche matches via union join.
        - `{c.intersection.s!r}` - merge niche matches by their intersection.
    """

    list_of_sets = []
    for definition in niche_definitions:
        list_of_sets.append(_get_subset_indices(adata.obs, definition))

    union_of_matches = _get_initial_niches(list_of_sets)

    avg_jaccard = np.zeros(len(union_of_matches))  # the jaccard index is tracked to order the consensus niches later on

    for set_of_sets in range(len(list_of_sets) - 1):
        current_matches = {}
        used_matches: set[str] = set()
        matches_A_B = {
            subset: _find_best_match(indices, list_of_sets[set_of_sets + 1], exclude=used_matches)
            for subset, indices in union_of_matches.items()
        }
        ranked_matches = sorted(matches_A_B.items(), key=lambda x: x[1][1], reverse=True)
        for subset_A, (match, jaccard_index) in ranked_matches:
            if match not in used_matches:
                current_matches[subset_A] = (match, jaccard_index)
                used_matches.add(match)
            else:
                new_match, new_jaccard = _find_best_match(
                    union_of_matches[subset_A], list_of_sets[set_of_sets + 1], exclude=used_matches
                )
                if new_match:
                    current_matches[subset_A] = (new_match, new_jaccard)
                    used_matches.add(new_match)

        jaccard = np.asarray([jaccard_index for _, (_, jaccard_index) in current_matches.items()])
        avg_jaccard = (avg_jaccard + jaccard) / (set_of_sets + 1)

        if merge == "union":
            consensus = {
                subset_A: union_of_matches[subset_A] | list_of_sets[set_of_sets + 1][match]
                for subset_A, (match, _) in current_matches.items()
            }
        if merge == "intersection":
            consensus = {
                subset_A: union_of_matches[subset_A] & list_of_sets[set_of_sets + 1][match]
                for subset_A, (match, _) in current_matches.items()
            }

    niche_categories = list(consensus.keys())
    consensus_by_jaccard = dict(zip(niche_categories, avg_jaccard))

    sorted_by_jaccard = dict(
        sorted(consensus_by_jaccard.items(), key=lambda item: item[1], reverse=True),
    )
    sorted_consensus = {key: consensus[key] for key in sorted_by_jaccard}
    filtered_consensus = _filter_overlap(sorted_consensus)

    adata.obs["consensus_niche"] = adata.obs.index.map(filtered_consensus).fillna("None")

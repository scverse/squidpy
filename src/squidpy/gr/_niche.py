from __future__ import annotations

import warnings
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps
from anndata import AnnData
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, hstack, issparse, spdiags
from scipy.spatial import distance
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from spatialdata import SpatialData

from squidpy._constants._constants import NicheDefinitions
from squidpy._docs import d, inject_docs

__all__ = ["calculate_niche"]


@d.dedent
@inject_docs(fla=NicheDefinitions)
def calculate_niche(
    adata: AnnData | SpatialData,
    flavor: Literal["neighborhood", "utag", "cellcharter"] = "neighborhood",
    library_key: str | None = None,  # TODO: calculate niches on a per-slide basis
    table_key: str | None = None,
    mask: pd.core.series.Series = None,
    groups: str | None = None,
    n_neighbors: int | None = None,
    resolutions: float | list[float] | None = None,
    subset_groups: list[str] | None = None,
    min_niche_size: int | None = None,
    scale: bool = True,
    abs_nhood: bool = False,
    distance: int = 1,
    n_hop_weights: list[float] | None = None,
    aggregation: str = "mean",
    n_components: int | None = None,
    random_state: int = 42,
    spatial_connectivities_key: str = "spatial_connectivities",
) -> AnnData | pd.DataFrame:
    """
    Calculate niches (spatial clusters) based on a user-defined method in 'flavor'.
    The resulting niche labels with be stored in 'adata.obs'. If flavor = 'all' then all available methods
    will be applied and additionally compared using cluster validation scores.
    Parameters
    ----------
    %(adata)s
    flavor
        Method to use for niche calculation. Available options are:
            - `{fla.NEIGHBORHOOD.s!r}` - cluster the neighborhood profile.
            - `{fla.UTAG.s!r}` - use utag algorithm (matrix multiplication).
            - `{fla.CELLCHARTER.s!r}` - cluster adjacency matrix with Gaussian Mixture Model (GMM) using CellCharter's approach.
            - `{fla.SPOT.s!r}` - calculate niches using optimal transport. (coming soon)
            - `{fla.BANKSY.s!r}`- use Banksy algorithm. (coming soon)
    %(library_key)s
    table_key
        Key in `spatialdata.tables` to specify an 'anndata' table. Only necessary if 'sdata' is passed.
    mask
        Boolean array to filter cells which won't get assigned to a niche.
        Note that if you want to exclude these cells during neighborhood calculation already, you should subset your AnnData table before running 'sq.gr.spatial_neigbors'.
    groups
        Groups based on which to calculate neighborhood profile (E.g. columns of cell type annotations in adata.obs).
        Required if flavor == `{fla.NEIGHBORHOOD.s!r}`.
    n_neighbors
        Number of neighbors to use for 'scanpy.pp.neighbors' before clustering using leiden algorithm.
        Required if flavor == `{fla.NEIGHBORHOOD.s!r}` or flavor == `{fla.UTAG.s!r}`.
    resolutions
        List of resolutions to use for leiden clustering.
        Required if flavor == `{fla.NEIGHBORHOOD.s!r}` or flavor == `{fla.UTAG.s!r}`.
    subset_groups
        Groups (e.g. cell type categories) to ignore when calculating the neighborhood profile.
        Optional if flavor == `{fla.NEIGHBORHOOD.s!r}`.
    min_niche_size
        Minimum required size of a niche. Niches with fewer cells will be labeled as 'not_a_niche'.
        Optional if flavor == `{fla.NEIGHBORHOOD.s!r}`.
    scale
        If 'True', compute z-scores of neighborhood profiles.
        Optional if flavor == `{fla.NEIGHBORHOOD.s!r}`.
    abs_nhood
        If 'True', calculate niches based on absolute neighborhood profile.
        Optional if flavor == `{fla.NEIGHBORHOOD.s!r}`.
    distance
        n-hop neighbor adjacency matrices to use e.g. [1,2,3] for 1-hop,2-hop,3-hop neighbors respectively or "5" for 1-hop,...,5-hop neighbors. 0 (self) is always included.
        Required if flavor == `{fla.CELLCHARTER.s!r}`.
        Optional if flavor == `{fla.NEIGHBORHOOD.s!r}`.
    n_hop_weights
        How to weight subsequent n-hop adjacency matrices. E.g. [1, 0.5, 0.25] for weights of 1-hop, 2-hop, 3-hop adjacency matrices respectively.
        Optional if flavor == `{fla.NEIGHBORHOOD.s!r}` and `distance` > 1.
    aggregation
        How to aggregate count matrices. Either 'mean' or 'variance'.
        Required if flavor == `{fla.CELLCHARTER.s!r}`.
    n_components
        Number of components to use for GMM.
        Required if flavor == `{fla.CELLCHARTER.s!r}`.
    random_state
        Random state to use for GMM.
        Optional if flavor == `{fla.CELLCHARTER.s!r}`.
    spatial_connectivities_key
        Key in `adata.obsp` where spatial connectivities are stored.
    """

    # check whether anndata or spatialdata is provided and if spatialdata, check whether table_key is provided
    if isinstance(adata, SpatialData):
        if table_key is not None:
            adata = adata.tables[table_key].copy()
        else:
            raise ValueError("Please specify which table to use with `table_key`.")
    else:
        adata = adata

    # check whether neighborhood graph exists
    if spatial_connectivities_key not in adata.obsp.keys():
        raise KeyError(
            f"Key '{spatial_connectivities_key}' not found in `adata.obsp`. If you haven't computed a spatial neighborhood graph yet, use `sq.gr.spatial_neighbors`."
        )

    _validate_niche_args(
        adata,
        mask,
        flavor,
        groups,
        n_neighbors,
        resolutions,
        subset_groups,
        min_niche_size,
        scale,
        abs_nhood,
        distance,
        n_hop_weights,
        aggregation,
        n_components,
        random_state,
        spatial_connectivities_key,
    )


def _get_nhood_profile_niches(
    adata: AnnData,
    mask: pd.core.series.Series | None,
    groups: str,
    n_neighbors: int,
    resolutions: float | list[float],
    subset_groups: list[str] | None,
    min_niche_size: int | None,
    scale: bool,
    abs_nhood: bool,
    distance: int,
    n_hop_weights: list[float] | None,
    spatial_connectivities_key: str,
) -> None:
    """
    adapted from https://github.com/immunitastx/monkeybread/blob/main/src/monkeybread/calc/_neighborhood_profile.py
    """
    # If subsetting, filter connections from adjacency matrix
    if subset_groups:
        adjacency_matrix = adata.obsp[spatial_connectivities_key].tocsc()
        obs_mask = ~adata.obs[groups].isin(subset_groups)
        adata = adata[obs_mask]

        adjacency_matrix = adjacency_matrix[obs_mask, :][:, obs_mask]
        adata.obsp[spatial_connectivities_key] = adjacency_matrix.tocsr()

    # get obs x neighbor matrix from sparse matrix
    matrix = adata.obsp[spatial_connectivities_key].tocoo()

    # get obs x category matrix where each column is the absolute/relative frequency of a category in the neighborhood
    nhood_profile = _calculate_neighborhood_profile(adata, groups, matrix, abs_nhood)

    # Additionally use n-hop neighbors if distance > 1. This sums up the (weighted) neighborhood profiles of all n-hop neighbors.
    if distance > 1:
        n_hop_adjacency_matrix = adata.obsp[spatial_connectivities_key].copy()
        # if no weights are provided, use 1 for all n_hop neighbors
        if n_hop_weights is None:
            n_hop_weights = [1] * distance
        # if weights are provided, start with applying weight to the original neighborhood profile
        else:
            nhood_profile = n_hop_weights[0] * nhood_profile
        # get n_hop neighbor adjacency matrices by multiplying the original adjacency matrix with itself n times and get corresponding neighborhood profiles.
        for n_hop in range(distance - 1):
            n_hop_adjacency_matrix = n_hop_adjacency_matrix @ adata.obsp[spatial_connectivities_key]
            matrix = n_hop_adjacency_matrix.tocoo()
            nhood_profile += n_hop_weights[n_hop + 1] * _calculate_neighborhood_profile(
                adata, groups, matrix, abs_nhood
            )
        if not abs_nhood:
            nhood_profile = nhood_profile / sum(n_hop_weights)

    # create AnnData object from neighborhood profile to perform scanpy functions
    adata_neighborhood = ad.AnnData(X=nhood_profile)

    # reason for scaling see https://monkeybread.readthedocs.io/en/latest/notebooks/tutorial.html#niche-analysis
    if scale:
        sc.pp.scale(adata_neighborhood, zero_center=True)

    # mask obs to exclude cells for which no niche shall be assigned
    if mask is not None:
        mask = mask[mask.index.isin(adata_neighborhood.obs.index)]
        adata_neighborhood = adata_neighborhood[mask]

    # required for leiden clustering (note: no dim reduction performed in original implementation)
    sc.pp.neighbors(adata_neighborhood, n_neighbors=n_neighbors, use_rep="X")

    resolutions = [resolutions] if not isinstance(resolutions, list) else resolutions

    # For each resolution, apply leiden on neighborhood profile. Each cluster label equals to a niche label
    for res in resolutions:
        sc.tl.leiden(adata_neighborhood, resolution=res, key_added=f"neighborhood_niche_res={res}")
        adata.obs[f"neighborhood_niche_res={res}"] = adata.obs.index.map(
            adata_neighborhood.obs[f"neighborhood_niche_res={res}"]
        ).fillna("not_a_niche")

        # filter niches with n_cells < min_niche_size
        if min_niche_size is not None:
            counts_by_niche = adata.obs[f"neighborhood_niche_res={res}"].value_counts()
            to_filter = counts_by_niche[counts_by_niche < min_niche_size].index
            adata.obs[f"neighborhood_niche_res={res}"] = adata.obs[f"neighborhood_niche_res={res}"].apply(
                lambda x, to_filter=to_filter: "not_a_niche" if x in to_filter else x
            )

    return


def _get_utag_niches(
    adata: AnnData,
    subset_groups: list[str] | None,
    n_neighbors: int,
    resolutions: float | list[float] | None,
    spatial_connectivities_key: str,
) -> None:
    """
    Adapted from https://github.com/ElementoLab/utag/blob/main/utag/segmentation.py
    """

    new_feature_matrix = _utag(adata, normalize_adj=True, spatial_connectivity_key=spatial_connectivities_key)
    adata_utag = ad.AnnData(X=new_feature_matrix)
    sc.tl.pca(adata_utag)  # note: unlike with flavor 'neighborhood' dim reduction is performed here
    sc.pp.neighbors(adata_utag, n_neighbors=n_neighbors, use_rep="X_pca")

    if resolutions is not None:
        if not isinstance(resolutions, list):
            resolutions = [resolutions]
    else:
        raise ValueError("Please provide resolutions for leiden clustering.")

    # For each resolution, apply leiden on neighborhood profile. Each cluster label equals to a niche label
    for res in resolutions:
        sc.tl.leiden(adata_utag, resolution=res, key_added=f"utag_res={res}")
        adata.obs[f"utag_res={res}"] = adata_utag.obs[f"utag_res={res}"].values
    return


def _get_cellcharter_niches(
    adata: AnnData,
    subset_groups: list[str] | None,
    distance: int,
    aggregation: str,
    n_components: int,
    random_state: int,
    spatial_connectivities_key: str,
) -> None:
    """adapted from https://github.com/CSOgroup/cellcharter/blob/main/src/cellcharter/gr/_aggr.py
    and https://github.com/CSOgroup/cellcharter/blob/main/src/cellcharter/tl/_gmm.py"""

    adjacency_matrix = adata.obsp[spatial_connectivities_key]
    layers = list(range(distance + 1))

    aggregated_matrices = []
    adj_hop = _setdiag(adjacency_matrix, 0)  # Remove self-loops, set diagonal to 0
    adj_visited = _setdiag(adjacency_matrix.copy(), 1)  # Track visited neighbors
    for k in layers:
        if k == 0:
            # get original count matrix (not aggregated)
            aggregated_matrices.append(adata.X)
        else:
            # get count and adjacency matrix for k-hop (neighbor of neighbor of neighbor ...) and aggregate them
            if k > 1:
                adj_hop, adj_visited = _hop(adj_hop, adjacency_matrix, adj_visited)
            adj_hop_norm = _normalize(adj_hop)
            aggregated_matrix = _aggregate(adata, adj_hop_norm, aggregation)
            aggregated_matrices.append(aggregated_matrix)

    concatenated_matrix = hstack(aggregated_matrices)  # Stack all matrices horizontally
    arr = concatenated_matrix.toarray()  # Densify
    arr_ad = ad.AnnData(X=arr)
    sc.tl.pca(arr_ad)

    # cluster concatenated matrix with GMM, each cluster label equals to a niche label
    niches = _get_GMM_clusters(arr_ad.obsm["X_pca"], n_components, random_state)

    adata.obs["cellcharter_niche"] = pd.Categorical(niches)
    return


def _calculate_neighborhood_profile(
    adata: AnnData,
    groups: str,
    matrix: coo_matrix,
    abs_nhood: bool,
) -> pd.DataFrame:
    """
    Returns an obs x category matrix where each column is the absolute/relative frequency of a category in the neighborhood
    """

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

    if abs_nhood:
        return pd.DataFrame(abs_freq, index=adata.obs.index)
    else:
        return pd.DataFrame(rel_freq, index=adata.obs.index)


def _utag(adata: AnnData, normalize_adj: bool, spatial_connectivity_key: str) -> AnnData:
    """
    Performs inner product of adjacency matrix and feature matrix,
    such that each observation inherits features from its immediate neighbors as described in UTAG paper.
    """

    adjacency_matrix = adata.obsp[spatial_connectivity_key]

    if normalize_adj:
        return normalize(adjacency_matrix, norm="l1", axis=1) @ adata.X
    else:
        return adjacency_matrix @ adata.X


def _setdiag(adjacency_matrix: sps.spmatrix, value: int) -> sps.spmatrix:
    """remove self-loops"""

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
    """get nearest neighbor of neighbors"""

    adj_hop = adj_hop @ adj

    if adj_visited is not None:
        adj_hop = adj_hop > adj_visited
        adj_visited = adj_visited + adj_hop

    return adj_hop, adj_visited


def _normalize(adj: sps.spmatrix) -> sps.spmatrix:
    """normalize adjacency matrix such that nodes with high degree don't disproportionately affect aggregation"""

    deg = np.array(np.sum(adj, axis=1)).squeeze()
    with np.errstate(divide="ignore"):
        deg_inv = 1 / deg
    deg_inv[deg_inv == float("inf")] = 0

    return spdiags(deg_inv, 0, len(deg_inv), len(deg_inv)) * adj


def _aggregate(adata: AnnData, normalized_adjacency_matrix: sps.spmatrix, aggregation: str = "mean") -> Any:
    """aggregate count and adjacency matrix either by mean or variance"""
    # TODO: add support for other aggregation methods
    if aggregation == "mean":
        aggregated_matrix = normalized_adjacency_matrix @ adata.X
    elif aggregation == "variance":
        mean_matrix = (normalized_adjacency_matrix @ adata.X).toarray()
        X_to_arr = adata.X.toarray()
        mean_squared_matrix = normalized_adjacency_matrix @ (X_to_arr * X_to_arr)
        aggregated_matrix = mean_squared_matrix - mean_matrix * mean_matrix
    else:
        raise ValueError(f"Invalid aggregation method '{aggregation}'. Please choose either 'mean' or 'variance'.")

    return aggregated_matrix


def _get_GMM_clusters(A: NDArray[np.float64], n_components: int, random_state: int) -> Any:
    """Returns niche labels generated by GMM clustering.
    Compared to cellcharter this approach is simplified by using sklearn's GaussianMixture model without stability analysis."""

    gmm = GaussianMixture(n_components=n_components, random_state=random_state, init_params="random_from_data")
    gmm.fit(A)
    labels = gmm.predict(A)

    return labels


def _fide_score(adata: AnnData, niche_key: str, average: bool) -> Any:
    """
    F1-score of intra-domain edges (FIDE). A high score indicates a great domain continuity.

    The F1-score is computed for every class, then all F1-scores are averaged. If some classes
    are not predicted, the `n_classes` argument allows to pad with zeros before averaging the F1-scores.
    """
    i, j = adata.obsp["spatial_connectivities"].nonzero()  # get row and column indices of non-zero elements
    niche_labels, neighbor_niche_labels = (
        adata.obs.iloc[i][niche_key],
        adata.obs.iloc[j][niche_key],
    )

    if not average:
        fide = f1_score(niche_labels, neighbor_niche_labels, average=None)
    else:
        fide = f1_score(niche_labels, neighbor_niche_labels, average="macro")

    return fide


def _jensen_shannon_divergence(adata: AnnData, niche_key: str, library_key: str) -> Any:
    """
    Calculate Jensen-Shannon divergence (JSD) over all slides.
    This metric measures how well niche label distributions match across different slides.
    """
    niche_labels = sorted(adata.obs[niche_key].unique())
    label_distributions = []

    for _, slide in adata.obs.groupby(library_key):
        counts = slide[niche_key].value_counts(normalize=True)
        relative_freq = [counts.get(label, 0) for label in niche_labels]
        label_distributions.append(relative_freq)

    return distance.jensenshannon(np.array(label_distributions))


def _validate_niche_args(
    adata: AnnData,
    mask: pd.core.series.Series | None,
    flavor: Literal["neighborhood", "utag", "cellcharter"],
    groups: str | None,
    n_neighbors: int | None,
    resolutions: float | list[float] | None,
    subset_groups: list[str] | None,
    min_niche_size: int | None,
    scale: bool,
    abs_nhood: bool,
    distance: int,
    n_hop_weights: list[float] | None,
    aggregation: str,
    n_components: int | None,
    random_state: int,
    spatial_connectivities_key: str,
) -> str | None:
    """
    Validate whether necessary arguments are provided for a given niche flavor.
    If required arguments are provided, run respective niche calculation function.
    Also warns whether unnecessary optional arguments are supplied.
    """
    if flavor == "neighborhood":
        if any(arg is not None for arg in ([random_state])):
            warnings.warn("param 'random_state' is not used for neighborhood flavor.", stacklevel=2)
        if groups is not None and n_neighbors is not None and resolutions is not None:
            _get_nhood_profile_niches(
                adata,
                mask,
                groups,
                n_neighbors,
                resolutions,
                subset_groups,
                min_niche_size,
                scale,
                abs_nhood,
                distance,
                n_hop_weights,
                spatial_connectivities_key,
            )
        else:
            raise ValueError(
                "One of required args 'groups', 'n_neighbors' and 'resolutions' for flavor 'neighborhood' is 'None'."
            )
    elif flavor == "utag":
        if any(arg is not None for arg in (subset_groups, min_niche_size, scale, abs_nhood, random_state)):
            warnings.warn(
                "param 'subset_groups', 'min_niche_size', 'scale', 'abs_nhood', 'random_state' are not used for utag flavor.",
                stacklevel=2,
            )
        if n_neighbors is not None and resolutions is not None:
            _get_utag_niches(adata, subset_groups, n_neighbors, resolutions, spatial_connectivities_key)
        else:
            raise ValueError("One of required args 'n_neighbors' and 'resolutions' for flavor 'utag' is 'None'.")
    elif flavor == "cellcharter":
        if any(arg is not None for arg in (groups, subset_groups, min_niche_size, scale, abs_nhood)):
            warnings.warn(
                "param 'groups', 'subset_groups', 'min_niche_size', 'scale', 'abs_nhood' are not used for cellcharter flavor.",
                stacklevel=2,
            )
        if distance is not None and aggregation is not None and n_components is not None:
            _get_cellcharter_niches(
                adata, subset_groups, distance, aggregation, n_components, random_state, spatial_connectivities_key
            )
        else:
            raise ValueError(
                "One of required args 'distance', 'aggregation' and 'n_components' for flavor 'cellcharter' is 'None'."
            )
    else:
        raise ValueError(f"Invalid flavor '{flavor}'. Please choose one of 'neighborhood', 'utag', 'cellcharter'.")

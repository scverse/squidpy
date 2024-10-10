from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps
from anndata import AnnData
from scipy.sparse import hstack, issparse, spdiags
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from spatialdata import SpatialData

from squidpy._utils import NDArrayA

__all__ = ["calculate_niche"]


def calculate_niche(
    adata: AnnData | SpatialData,
    flavor: str = "neighborhood",
    library_key: str | None = None,  # TODO: calculate niches on a per-slide basis
    table_key: str | None = None,
    mask: pd.core.series.Series = None,
    groups: str | None = None,
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
    spatial_connectivities_key: str = "spatial_connectivities",
) -> AnnData | pd.DataFrame:
    """Calculate niches (spatial clusters) based on a user-defined method in 'flavor'.
    The resulting niche labels with be stored in 'adata.obs'. If flavor = 'all' then all available methods
    will be applied and additionally compared using cluster validation scores.
    Parameters
    ----------
    %(adata)s
    flavor
        Method to use for niche calculation. Available options are:
            - `{c.NEIGHBORHOOD.s!r}` - cluster the neighborhood profile.
            - `{c.UTAG.s!r}` - use utag algorithm (matrix multiplication).
            - `{c.CELLCHARTER.s!r}` - cluster adjacency matrix with Gaussian Mixture Model (GMM) using CellCharter's approach.
            - `{c.SPOT.s!r}` - calculate niches using optimal transport. (coming soon)
            - `{c.BANKSY.s!r}`- use Banksy algorithm. (coming soon)
    %(library_key)s
    subset
        Restrict niche calculation to a subset of the data.
    table_key
        Key in `spatialdata.tables` to specify an 'anndata' table. Only necessary if 'sdata' is passed.
    mask
        Boolean array to filter cells which won't get assigned to a niche.
        Note that if you want to exclude these cells during neighborhood calculation already, you should subset your AnnData table before running 'sq.gr.spatial_neigbors'.
    groups
        Groups based on which to calculate neighborhood profile (E.g. columns of cell type annotations in adata.obs).
        Required if flavor == 'neighborhood'.
    n_neighbors
        Number of neighbors to use for 'scanpy.pp.neighbors' before clustering using leiden algorithm.
        Required if flavor == 'neighborhood' or flavor == 'UTAG'.
    resolutions
        List of resolutions to use for leiden clustering.
        Required if flavor == 'neighborhood' or flavor == 'UTAG'.
    subset_groups
        Groups (e.g. cell type categories) to ignore when calculating the neighborhood profile.
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
        List of adjacency matrices to use e.g. [1,2,3] for 1-hop,2-hop,3-hop neighbors respectively or "5" for 1-hop,...,5-hop neighbors. 0 (self) is always included.
        Required if flavor == 'cellcharter'.
    aggregation
        How to aggregate count matrices. Either 'mean' or 'variance'.
        Required if flavor == 'cellcharter'.
    n_components
        Number of components to use for GMM.
        Required if flavor == 'cellcharter'.
    random_state
        Random state to use for GMM.
        Optional if flavor == 'cellcharter'.
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

    if flavor == "neighborhood":
        """adapted from https://github.com/immunitastx/monkeybread/blob/main/src/monkeybread/calc/_neighborhood_profile.py"""

        # calculate the neighborhood profile for each cell (relative and absolute proportion of e.g. each cell type in the neighborhood)
        rel_nhood_profile, abs_nhood_profile = _calculate_neighborhood_profile(
            adata, groups, subset_groups, spatial_connectivities_key
        )
        # create AnnData object from neighborhood profile to perform scanpy functions
        if not abs_nhood:
            adata_neighborhood = ad.AnnData(X=rel_nhood_profile)
        else:
            adata_neighborhood = ad.AnnData(X=abs_nhood_profile)

        # reason for scaling see https://monkeybread.readthedocs.io/en/latest/notebooks/tutorial.html#niche-analysis
        if scale:
            sc.pp.scale(adata_neighborhood, zero_center=True)

        # mask obs to exclude cells for which no niche shall be assigned
        if mask is not None:
            mask = mask[mask.index.isin(adata_neighborhood.obs.index)]
            adata_neighborhood = adata_neighborhood[mask]

        # required for leiden clustering (note: no dim reduction performed in original implementation)
        sc.pp.neighbors(adata_neighborhood, n_neighbors=n_neighbors, use_rep="X")

        if resolutions is not None:
            if not isinstance(resolutions, list):
                resolutions = [resolutions]
        else:
            raise ValueError("Please provide resolutions for leiden clustering.")

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

    elif flavor == "utag":
        """adapted from https://github.com/ElementoLab/utag/blob/main/utag/segmentation.py"""

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

    elif flavor == "cellcharter":
        """adapted from https://github.com/CSOgroup/cellcharter/blob/main/src/cellcharter/gr/_aggr.py
        and https://github.com/CSOgroup/cellcharter/blob/main/src/cellcharter/tl/_gmm.py"""

        adjacency_matrix = adata.obsp[spatial_connectivities_key]
        if not isinstance(adj_subsets, list):
            if adj_subsets is not None:
                adj_subsets = list(range(adj_subsets + 1))
            else:
                raise ValueError(
                    "flavor 'cellcharter' requires adj_subsets to not be None. Specify list of values or maximum value of neighbors to use."
                )
        else:
            if 0 not in adj_subsets:
                adj_subsets.insert(0, 0)
        if any(x < 0 for x in adj_subsets):
            raise ValueError("adj_subsets must contain non-negative integers.")

        aggregated_matrices = []
        adj_hop = _setdiag(adjacency_matrix, 0)  # Remove self-loops, set diagonal to 0
        adj_visited = _setdiag(adjacency_matrix.copy(), 1)  # Track visited neighbors
        for k in adj_subsets:
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

        # cluster concatenated matrix with GMM, each cluster label equals to a niche label
        niches = _get_GMM_clusters(arr, n_components, random_state)

        adata.obs[f"{flavor}_niche"] = pd.Categorical(niches)


def _calculate_neighborhood_profile(
    adata: AnnData,
    groups: str | None,
    subset_groups: list[str] | None,
    spatial_connectivities_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """returns an obs x category matrix where each column is the absolute/relative frequency of a category in the neighborhood"""

    if groups is None:
        raise ValueError("Please specify 'groups' based on which to calculate neighborhood profile.")
    if subset_groups:
        adjacency_matrix = adata.obsp[spatial_connectivities_key].tocsc()
        obs_mask = ~adata.obs[groups].isin(subset_groups)
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
    such that each observation inherits features from its immediate neighbors as described in UTAG paper."""

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


def _get_GMM_clusters(A: np.ndarray[np.float64, Any], n_components: int, random_state: int) -> Any:
    """Returns niche labels generated by GMM clustering.
    Compared to cellcharter this approach is simplified by using sklearn's GaussianMixture model without stability analysis."""

    gmm = GaussianMixture(n_components=n_components, random_state=random_state, init_params="random_from_data")
    gmm.fit(A)
    labels = gmm.predict(A)

    return labels


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

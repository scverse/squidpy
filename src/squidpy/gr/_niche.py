from __future__ import annotations

import contextlib
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
from spatialdata._logging import logger as logg

from squidpy._constants._constants import NicheDefinitions
from squidpy._docs import d, inject_docs

__all__ = ["calculate_niche"]


@d.dedent
@inject_docs(fla=NicheDefinitions)
def calculate_niche(
    data: AnnData | SpatialData,
    flavor: Literal["neighborhood", "utag", "cellcharter"],
    library_key: str | None = None,
    table_key: str | None = None,
    mask: pd.core.series.Series = None,
    groups: str | None = None,
    n_neighbors: int | None = None,
    resolutions: float | list[float] | None = None,
    min_niche_size: int | None = None,
    scale: bool = True,
    abs_nhood: bool = False,
    distance: int | None = None,
    n_hop_weights: list[float] | None = None,
    aggregation: str | None = None,
    n_components: int | None = None,
    random_state: int = 42,
    spatial_connectivities_key: str = "spatial_connectivities",
    inplace: bool = True,
) -> AnnData:
    """
    Calculate niches (spatial clusters) based on a user-defined method in 'flavor'.
    The resulting niche labels with be stored in 'adata.obs'.

    Parameters
    ----------
    %(adata)s
    flavor
        Method to use for niche calculation. Available options are:
            - `{fla.NEIGHBORHOOD.s!r}` - cluster the neighborhood profile.
            - `{fla.UTAG.s!r}` - use utag algorithm (matrix multiplication).
            - `{fla.CELLCHARTER.s!r}` - cluster adjacency matrix with Gaussian Mixture Model (GMM) using CellCharter's approach.
    %(library_key)s
        If provided, niches will be calculated separately for each unique value in this column.
        Each niche will be prefixed with the library identifier.
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
    inplace
        If 'True', perform the operation in place.
        If 'False', return a new AnnData object with the niche labels.
    """

    if flavor == "cellcharter" and aggregation is None:
        aggregation = "mean"

    _validate_niche_args(
        data,
        flavor,
        library_key,
        table_key,
        groups,
        n_neighbors,
        resolutions,
        min_niche_size,
        scale,
        abs_nhood,
        distance,
        n_hop_weights,
        aggregation,
        n_components,
        random_state,
        inplace,
    )

    if resolutions is None:
        resolutions = [0.5]

    if distance is None:
        distance = 1

    if isinstance(data, SpatialData):
        orig_adata = data.tables[table_key]
        adata = orig_adata.copy()
    else:
        orig_adata = data
        adata = data.copy()

    if spatial_connectivities_key not in adata.obsp.keys():
        raise KeyError(
            f"Key '{spatial_connectivities_key}' not found in `adata.obsp`. "
            "If you haven't computed a spatial neighborhood graph yet, use `sq.gr.spatial_neighbors`."
        )

    result_columns = _get_result_columns(
        flavor=flavor,
        resolutions=resolutions,
        library_key=None,
        libraries=None,
    )

    if library_key is not None:
        if library_key not in adata.obs.columns:
            raise KeyError(f"'{library_key}' not found in `adata.obs`.")

        logg.info(f"Stratifying by library_key '{library_key}'")

        for col in result_columns:
            adata.obs[col] = "not_a_niche"

        for lib_id in adata.obs[library_key].unique():
            logg.info(f"Processing library '{lib_id}'")

            lib_indices = adata.obs[adata.obs[library_key] == lib_id].index

            if len(lib_indices) == 0:
                logg.warning(f"Library '{lib_id}' contains no cells, skipping")
                continue

            lib_adata = adata[lib_indices].copy()

            lib_mask = None
            if mask is not None:
                lib_mask = mask[mask.index.isin(lib_indices)]

            lib_result = calculate_niche(
                lib_adata,
                flavor=flavor,
                library_key=None,
                mask=lib_mask,
                groups=groups,
                n_neighbors=n_neighbors,
                resolutions=None if flavor == "cellcharter" else resolutions,
                min_niche_size=min_niche_size,
                scale=scale,
                abs_nhood=abs_nhood,
                distance=None if flavor == "utag" else distance,
                n_hop_weights=n_hop_weights,
                aggregation=aggregation,
                n_components=n_components,
                random_state=random_state,
                spatial_connectivities_key=spatial_connectivities_key,
                inplace=False,
            )

            for col in result_columns:
                if col in lib_result.obs.columns:
                    prefixed_values = lib_result.obs[col].apply(
                        lambda x, lib=lib_id: (f"lib={lib}_{x}" if x != "not_a_niche" else x)
                    )

                    adata.obs.loc[lib_indices, col] = prefixed_values.values

    else:
        _calculate_niches(
            adata,
            mask,
            flavor,
            groups,
            n_neighbors,
            resolutions,
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

    if not inplace:
        return adata
    # For SpatialData, update the table directly
    if isinstance(data, SpatialData):
        data.tables[table_key] = adata
    else:
        # For AnnData, copy results back to original object
        for col in result_columns:
            if col in orig_adata.obs.columns:
                logg.info(f"Overwriting existing column '{col}'")
                with contextlib.suppress(KeyError):
                    del orig_adata.obs[col]
            if f"{col}_colors" in orig_adata.uns.keys():
                with contextlib.suppress(KeyError):
                    del orig_adata.uns[f"{col}_colors"]

            orig_adata.obs[col] = adata.obs[col]

    return None


def _get_result_columns(
    flavor: str,
    resolutions: float | list[float],
    library_key: str | None,
    libraries: list[str] | None,
) -> list[str]:
    """Get the column names that will be populated based on flavor and resolutions."""

    library_str = f"_{library_key}" if library_key is not None else ""

    if flavor == "cellcharter":
        base_column = "cellcharter_niche"
        if library_key is None:
            return [base_column]
        elif libraries is not None and len(libraries) > 0:
            return [f"{base_column}_{lib}" for lib in libraries]

    # For neighborhood and utag, we need to handle resolutions
    if not isinstance(resolutions, list):
        resolutions = [resolutions]

    prefix = f"nhood_niche{library_str}" if flavor == "neighborhood" else f"utag_niche{library_str}"
    if library_key is None:
        return [f"{prefix}_res={res}" for res in resolutions]
    else:
        assert isinstance(libraries, list)  # for mypy
        return [f"{prefix}_{lib}_res={res}" for lib in libraries for res in resolutions]


def _calculate_niches(
    adata: AnnData,
    mask: pd.core.series.Series | None,
    flavor: str,
    groups: str | None,
    n_neighbors: int | None,
    resolutions: float | list[float],
    min_niche_size: int | None,
    scale: bool,
    abs_nhood: bool,
    distance: int,
    n_hop_weights: list[float] | None,
    aggregation: str | None,
    n_components: int | None,
    random_state: int,
    spatial_connectivities_key: str,
) -> None:
    """Calculate niches using the specified flavor and parameters."""
    if flavor == "neighborhood":
        _get_nhood_profile_niches(
            adata,
            mask,
            groups,
            n_neighbors,
            resolutions,
            min_niche_size,
            scale,
            abs_nhood,
            distance,
            n_hop_weights,
            spatial_connectivities_key,
        )
    elif flavor == "utag":
        _get_utag_niches(adata, n_neighbors, resolutions, spatial_connectivities_key)
    elif flavor == "cellcharter":
        assert isinstance(aggregation, str)  # for mypy
        assert isinstance(n_components, int)  # for mypy
        _get_cellcharter_niches(
            adata,
            distance,
            aggregation,
            n_components,
            random_state,
            spatial_connectivities_key,
        )


def _get_nhood_profile_niches(
    adata: AnnData,
    mask: pd.core.series.Series | None,
    groups: str | None,
    n_neighbors: int | None,
    resolutions: float | list[float],
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

    adata_masked = adata

    # get obs x neighbor matrix from sparse matrix
    matrix = adata_masked.obsp[spatial_connectivities_key].tocoo()

    # get obs x category matrix where each column is the absolute/relative frequency of a category in the neighborhood
    nhood_profile = _calculate_neighborhood_profile(adata_masked, groups, matrix, abs_nhood)

    # Additionally use n-hop neighbors if distance > 1. This sums up the (weighted) neighborhood profiles of all n-hop neighbors.
    if distance > 1:
        n_hop_adjacency_matrix = adata_masked.obsp[spatial_connectivities_key].copy()
        # if no weights are provided, use 1 for all n_hop neighbors
        if n_hop_weights is None:
            n_hop_weights = [1] * distance
        # if weights are provided, start with applying weight to the original neighborhood profile
        elif len(n_hop_weights) < distance:
            # Extend weights if too few provided
            n_hop_weights = n_hop_weights + [n_hop_weights[-1]] * (distance - len(n_hop_weights))
            logg.debug(f"Extended weights to match distance: {n_hop_weights}")

        # Apply first weight to base profile
        weighted_profile = n_hop_weights[0] * nhood_profile

        # Calculate higher-order hop profiles
        n_hop_adjacency_matrix = adata_masked.obsp[spatial_connectivities_key].copy()

        # get n_hop neighbor adjacency matrices by multiplying the original adjacency matrix with itself n times and get corresponding neighborhood profiles.
        for n_hop in range(1, distance):
            logg.debug(f"Calculating {n_hop + 1}-hop neighbors")
            # Multiply adjacency matrix by itself to get n+1 hop adjacency
            n_hop_adjacency_matrix = n_hop_adjacency_matrix @ adata_masked.obsp[spatial_connectivities_key]
            matrix = n_hop_adjacency_matrix.tocoo()

            # Calculate and add weighted profile
            hop_profile = _calculate_neighborhood_profile(adata_masked, groups, matrix, abs_nhood)
            weighted_profile += n_hop_weights[n_hop] * hop_profile

        if not abs_nhood:
            weighted_profile = weighted_profile / sum(n_hop_weights)

        nhood_profile = weighted_profile

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

    resolutions = resolutions if isinstance(resolutions, list) else [resolutions]

    # For each resolution, apply leiden on neighborhood profile. Each cluster label equals to a niche label
    for res in resolutions:
        niche_key = f"nhood_niche_res={res}"

        if niche_key in adata_masked.obs.columns:
            del adata_masked.obs[niche_key]

        if f"{niche_key}_colors" in adata_masked.uns.keys():
            del adata_masked.uns[f"{niche_key}_colors"]
        # print(adata_masked.obs[niche_key])

        sc.tl.leiden(
            adata_neighborhood,
            resolution=res,
            key_added=niche_key,
        )

        adata_masked.obs[niche_key] = "not_a_niche"

        neighborhood_clusters = dict(zip(adata_neighborhood.obs.index, adata_neighborhood.obs[niche_key], strict=False))

        mask_indices = adata_masked.obs.index
        adata_masked.obs.loc[mask_indices, niche_key] = [
            neighborhood_clusters.get(idx, "not_a_niche") for idx in mask_indices
        ]

        # filter niches with n_cells < min_niche_size
        if min_niche_size is not None:
            counts_by_niche = adata_masked.obs[niche_key].value_counts()
            to_filter = counts_by_niche[counts_by_niche < min_niche_size].index
            adata_masked.obs[niche_key] = adata_masked.obs[niche_key].apply(
                lambda x, to_filter=to_filter: "not_a_niche" if x in to_filter else x
            )
            adata_masked.obs[niche_key] = adata_masked.obs.index.map(adata_masked.obs[niche_key]).fillna("not_a_niche")

    return


def _get_utag_niches(
    adata: AnnData,
    n_neighbors: int | None,
    resolutions: float | list[float],
    spatial_connectivities_key: str,
) -> None:
    """
    Adapted from https://github.com/ElementoLab/utag/blob/main/utag/segmentation.py
    """

    new_feature_matrix = _utag(adata, normalize_adj=True, spatial_connectivity_key=spatial_connectivities_key)
    adata_utag = ad.AnnData(X=new_feature_matrix)
    sc.tl.pca(adata_utag)  # note: unlike with flavor 'neighborhood' dim reduction is performed here
    sc.pp.neighbors(adata_utag, n_neighbors=n_neighbors, use_rep="X_pca")

    if not isinstance(resolutions, list):
        resolutions = [resolutions]
    # For each resolution, apply leiden on neighborhood profile. Each cluster label equals to a niche label
    for res in resolutions:
        sc.tl.leiden(adata_utag, resolution=res, key_added=f"utag_niche_res={res}")
        adata.obs[f"utag_niche_res={res}"] = adata_utag.obs[f"utag_niche_res={res}"].values

    return


def _get_cellcharter_niches(
    adata: AnnData,
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
    groups: str | None,
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
    Compared to cellcharter this approach is simplified by using sklearn's GaussianMixture model without stability analysis.
    """

    gmm = GaussianMixture(
        n_components=n_components,
        random_state=random_state,
        init_params="random_from_data",
    )
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
    data: AnnData | SpatialData,
    flavor: Literal["neighborhood", "utag", "cellcharter"],
    library_key: str | None,
    table_key: str | None,
    groups: str | None,
    n_neighbors: int | None,
    resolutions: float | list[float] | None,
    min_niche_size: int | None,
    scale: bool,
    abs_nhood: bool,
    distance: int | None,
    n_hop_weights: list[float] | None,
    aggregation: str | None,
    n_components: int | None,
    random_state: int,
    inplace: bool,
) -> None:
    """
    Validate whether necessary arguments are provided for a given niche flavor.
    Also warns whether unnecessary optional arguments are supplied.

    Raises
    ------
    ValueError
        If required arguments for the specified flavor are missing or have incorrect values.
    TypeError
        If arguments are of incorrect type.
    """
    if not isinstance(data, AnnData | SpatialData):
        raise TypeError(f"'data' must be an AnnData or SpatialData object, got {type(data).__name__}")

    if flavor not in ["neighborhood", "utag", "cellcharter"]:
        raise ValueError(f"Invalid flavor '{flavor}'. Please choose one of 'neighborhood', 'utag', 'cellcharter'.")

    if library_key is not None:
        if not isinstance(library_key, str):
            raise TypeError(f"'library_key' must be a string, got {type(library_key).__name__}")
        if isinstance(data, AnnData):
            if library_key not in data.obs.columns:
                raise ValueError(f"'library_key' must be a column in 'adata.obs', got {library_key}")
        elif isinstance(data, SpatialData):
            if table_key is None:
                raise ValueError("'table_key' is required when 'data' is a SpatialData object")
            if table_key not in data.tables:
                raise ValueError(f"'table_key' must be a valid table key in 'data', got {table_key}")
            if library_key not in data.tables[table_key].obs.columns:
                raise ValueError(f"'library_key' must be a column in 'adata.obs', got {library_key}")

    if n_neighbors is not None and not isinstance(n_neighbors, int):
        raise TypeError(f"'n_neighbors' must be an integer, got {type(n_neighbors).__name__}")

    if resolutions is not None:
        if not isinstance(resolutions, float | list):
            raise TypeError(f"'resolutions' must be a float or list of floats, got {type(resolutions).__name__}")
        if isinstance(resolutions, list) and not all(isinstance(res, float) for res in resolutions):
            raise TypeError("All elements in 'resolutions' list must be floats")

    if n_hop_weights is not None and not isinstance(n_hop_weights, list):
        raise TypeError(f"'n_hop_weights' must be a list of floats, got {type(n_hop_weights).__name__}")

    if not isinstance(scale, bool):
        raise TypeError(f"'scale' must be a boolean, got {type(scale).__name__}")

    if not isinstance(abs_nhood, bool):
        raise TypeError(f"'abs_nhood' must be a boolean, got {type(abs_nhood).__name__}")

    # Define parameters used by each flavor
    flavor_param_specs = {
        "neighborhood": {
            "required": ["groups", "n_neighbors", "resolutions"],
            "optional": [
                "min_niche_size",
                "scale",
                "abs_nhood",
                "distance",
                "n_hop_weights",
            ],
            "unused": ["aggregation", "n_components", "random_state"],
        },
        "utag": {
            "required": ["n_neighbors", "resolutions"],
            "optional": [],
            "unused": [
                "groups",
                "min_niche_size",
                "scale",
                "abs_nhood",
                "distance",
                "n_hop_weights",
                "aggregation",
                "n_components",
                "random_state",
            ],
        },
        "cellcharter": {
            "required": ["distance", "aggregation", "n_components", "random_state"],
            "optional": [],
            "unused": [
                "groups",
                "min_niche_size",
                "scale",
                "abs_nhood",
                "n_neighbors",
                "resolutions",
                "n_hop_weights",
            ],
        },
    }

    for param_name in flavor_param_specs[flavor]["required"]:
        param_value = locals()[param_name]
        if param_value is None:
            raise ValueError(f"'{param_name}' is required for flavor '{flavor}'")

    _check_unnecessary_args(
        flavor,
        {
            "groups": groups,
            "n_neighbors": n_neighbors,
            "resolutions": resolutions,
            "min_niche_size": min_niche_size,
            "scale": scale,
            "abs_nhood": abs_nhood,
            "distance": distance,
            "n_hop_weights": n_hop_weights,
            "aggregation": aggregation,
            "n_components": n_components,
            "random_state": random_state,
        },
        flavor_param_specs[flavor],
    )

    # Flavor-specific validations
    if flavor == "neighborhood":
        if not isinstance(groups, str):
            raise TypeError(f"'groups' must be a string, got {type(groups).__name__}")

        if min_niche_size is not None and not isinstance(min_niche_size, int):
            raise TypeError(f"'min_niche_size' must be an integer, got {type(min_niche_size).__name__}")

        if distance is not None and isinstance(distance, int) and distance < 1:
            raise ValueError(f"'distance' must be at least 1, got {distance}")

    elif flavor == "cellcharter":
        if distance is not None and not isinstance(distance, int):
            raise TypeError(f"'distance' must be an integer, got {type(distance).__name__}")
        if distance is not None and distance < 1:
            raise ValueError(f"'distance' must be at least 1, got {distance}")

        if aggregation is not None and not isinstance(aggregation, str):
            raise TypeError(f"'aggregation' must be a string, got {type(aggregation).__name__}")
        if aggregation not in ["mean", "variance"]:
            raise ValueError(f"'aggregation' must be one of 'mean' or 'variance', got {aggregation}")

        if not isinstance(n_components, int):
            raise TypeError(f"'n_components' must be an integer, got {type(n_components).__name__}")
        if n_components < 1:
            raise ValueError(f"'n_components' must be at least 1, got {n_components}")

        if not isinstance(random_state, int):
            raise TypeError(f"'random_state' must be an integer, got {type(random_state).__name__}")

        # for mypy
        if resolutions is None:
            resolutions = [0.0]

    if not isinstance(inplace, bool):
        raise TypeError(f"'inplace' must be a boolean, got {type(inplace).__name__}")


def _check_unnecessary_args(flavor: str, param_dict: dict[str, Any], param_specs: dict[str, Any]) -> None:
    """
    Check for unnecessary arguments that were provided but not used by the given flavor.

    Parameters
    ----------
    flavor
        The flavor being used ('neighborhood', 'utag', or 'cellcharter')
    param_dict
        Dictionary of parameter names to their values
    param_specs
        Dictionary with 'required', 'optional', and 'unused' parameter lists for the flavor
    """
    unnecessary_args = []

    for param_name in param_specs["unused"]:
        param_value = param_dict.get(param_name)

        # Special handling for boolean parameters with default values
        if param_name == "scale" and param_value is True:
            continue
        if param_name == "abs_nhood" and param_value is False:
            continue
        if param_name == "random_state" and param_value == 42:
            continue

        if param_value is not None:
            unnecessary_args.append(param_name)

    if unnecessary_args:
        logg.warn(
            f"Parameters {', '.join([f'{arg}' for arg in unnecessary_args])} are not used for flavor '{flavor}'.",
        )

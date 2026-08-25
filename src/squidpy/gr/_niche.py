from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps
from anndata import AnnData
from scipy.sparse import coo_matrix, hstack, issparse, lil_matrix, spdiags
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from spatialdata import SpatialData, sanitize_table
from spatialdata._logging import logger as logg

from squidpy._constants._constants import NicheDefinitions
from squidpy._docs import d, inject_docs
from squidpy._utils import NDArrayA, RNGLike, SeedLike, deprecated_randomness_param, legacy_random
from squidpy._validators import assert_isinstance, assert_key_in_adata, assert_one_of
from squidpy.gr._utils import extract_adata_if_sdata

__all__ = [
    "calculate_niche",
    "calculate_niche_neighborhood",
    "calculate_niche_utag",
    "calculate_niche_cellcharter",
    "calculate_niche_spatialleiden",
]


@d.dedent
@inject_docs(fla=NicheDefinitions)
@deprecated_randomness_param
def calculate_niche(
    data: AnnData | SpatialData,
    flavor: Literal["neighborhood", "utag", "cellcharter", "spatialleiden"],
    library_key: str | None = None,
    mask: pd.core.series.Series = None,
    groups: str | None = None,
    n_neighbors: int | None = None,
    resolutions: float | tuple[float, float] | list[float | tuple[float, float]] | None = None,
    min_niche_size: int | None = None,
    scale: bool = True,
    abs_nhood: bool = False,
    distance: int | None = None,
    n_hop_weights: list[float] | None = None,
    aggregation: str | None = None,
    n_components: int | None = None,
    spatial_connectivities_key: str = "spatial_connectivities",
    latent_connectivities_key: str = "connectivities",
    layer_ratio: float = 1.0,
    n_iterations: int = -1,
    use_weights: bool | tuple[bool, bool] = True,
    use_rep: str | None = None,
    inplace: bool = True,
    *,
    table_key: str | None = None,
    rng: SeedLike | RNGLike | None = None,
) -> AnnData | None:
    """
    Calculate niches (spatial clusters) based on a user-defined method in 'flavor'.
    The resulting niche labels with be stored in 'adata.obs'.

    .. deprecated:: 1.8.4
        ``calculate_niche`` is deprecated and will be removed in squidpy
        v1.9.0. Use one of the flavor-specific functions instead:

        - :func:`calculate_niche_neighborhood`
        - :func:`calculate_niche_utag`
        - :func:`calculate_niche_cellcharter`
        - :func:`calculate_niche_spatialleiden`

    See Also
    --------
    calculate_niche_neighborhood : Neighborhood-profile flavor with an explicit signature.
    calculate_niche_utag : UTAG flavor with an explicit signature.
    calculate_niche_cellcharter : CellCharter flavor with an explicit signature.
    calculate_niche_spatialleiden : SpatialLeiden flavor with an explicit signature.

    Parameters
    ----------
    %(adata)s
    flavor
        Method to use for niche calculation. Available options are:
            - `{fla.NEIGHBORHOOD.s!r}` - cluster the neighborhood profile.
            - `{fla.UTAG.s!r}` - use utag algorithm (matrix multiplication).
            - `{fla.SPATIALLEIDEN.s!r}` - cluster spatially resolved omics data using Multiplex Leiden.
            - `{fla.CELLCHARTER.s!r}` - a simplified version of CellCharter's approach, using PCA for dimensionality reduction. An arbitrary embedding can be used instead of PCA by setting the `use_rep` parameter which will try to find the embedding in `adata.obsm`.
    %(library_key)s
        If provided, niches will be calculated separately for each unique value in this column.
        Each niche will be prefixed with the library identifier.
    %(table_key)s
    mask
        Boolean array to filter cells which won't get assigned to a niche.
        Note that if you want to exclude these cells during neighborhood calculation already, you should subset your AnnData table before running 'sq.gr.spatial_neigbors'.
        Mask can look like the following. Here, the index values would correspond to adata.obs.index.
        The entries that are False are the ones ignored.
        mask = Series([False, False, True], index = ["a", "b", "c"])
    groups
        Groups based on which to calculate neighborhood profile (E.g. columns of cell type annotations in adata.obs).
        Required if flavor == `{fla.NEIGHBORHOOD.s!r}`.
    n_neighbors
        Number of neighbors to use for 'scanpy.pp.neighbors' before clustering using leiden algorithm.
        Required if flavor == `{fla.NEIGHBORHOOD.s!r}` or flavor == `{fla.UTAG.s!r}`.
    resolutions
        List of resolutions to use for leiden clustering.
        In the case of spatialleiden you can pass a tuple. Resolution for the latent space and spatial layer, respectively. A single float applies to both layers.
        Required if flavor == `{fla.NEIGHBORHOOD.s!r}` or flavor == `{fla.UTAG.s!r}`.
        Optional if flavor == `{fla.SPATIALLEIDEN.s!r}`.
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
    %(rng)s
        Optional if flavor == `{fla.CELLCHARTER.s!r}` or flavor == `{fla.SPATIALLEIDEN.s!r}`.
    spatial_connectivities_key
        Key in `adata.obsp` where spatial connectivities are stored.
        Required if flavor == `{fla.SPATIALLEIDEN.s!r}`.
    latent_connectivities_key
        Key in `adata.obsp` where gene expression connectivities are stored.
        Required if flavor == `{fla.SPATIALLEIDEN.s!r}`.
    layer_ratio
        The ratio of the weighting of the layers; latent space vs spatial. A higher ratio will increase relevance of the spatial neighbors and lead to more spatially homogeneous clusters.
        Optional if flavor == `{fla.SPATIALLEIDEN.s!r}`.
    n_iterations
        Number of iterations to run the Leiden algorithm. If the number is negative it runs until convergence.
        Optional if flavor == `{fla.SPATIALLEIDEN.s!r}`.
    use_weights
        Whether to use weights for the edges for latent space and spatial neighbors, respectively. A single bool applies to both layers.
        Optional if flavor == `{fla.SPATIALLEIDEN.s!r}`.
    use_rep
        Key in `adata.obsm` where the embedding is stored. If provided, this embedding will be used instead of PCA for dimensionality reduction.
        Optional if flavor == `{fla.CELLCHARTER.s!r}`.
    inplace
        If 'True', perform the operation in place.
        If 'False', return a new AnnData object with the niche labels.
    """

    warnings.warn(
        "Calling `calculate_niche` is deprecated and will be removed in squidpy "
        "v1.9.0. Use `calculate_niche_neighborhood`, `calculate_niche_utag`, "
        "`calculate_niche_cellcharter`, or `calculate_niche_spatialleiden` instead.",
        FutureWarning,
        stacklevel=2,
    )

    # cellcharter-only defaults stay guarded: filling them for other flavors would trip
    # the "not used for flavor" warning in _check_unnecessary_args
    if flavor == "cellcharter":
        if aggregation is None:
            aggregation = "mean"
        if n_components is None:
            n_components = 10
    if distance is None:
        distance = 3 if flavor == "cellcharter" else 1

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
        rng,
        spatial_connectivities_key,
        latent_connectivities_key,
        layer_ratio,
        n_iterations,
        use_weights,
        use_rep,
        inplace,
    )

    if resolutions is None:
        resolutions = [0.5]

    if flavor == "neighborhood":
        return calculate_niche_neighborhood(
            data,
            groups,
            resolutions,
            n_neighbors,
            spatial_connectivities_key,
            scale,
            distance,
            abs_nhood,
            n_hop_weights,
            min_niche_size,
            mask,
            library_key,
            inplace,
            table_key,
        )

    elif flavor == "utag":
        return calculate_niche_utag(
            data,
            resolutions,
            n_neighbors,
            spatial_connectivities_key,
            min_niche_size,
            mask,
            library_key,
            inplace,
            table_key,
        )

    elif flavor == "cellcharter":
        return calculate_niche_cellcharter(
            data,
            distance,
            aggregation,
            rng,
            spatial_connectivities_key,
            n_components,
            use_rep,
            min_niche_size,
            mask,
            library_key,
            inplace,
            table_key,
        )

    elif flavor == "spatialleiden":
        return calculate_niche_spatialleiden(
            data,
            resolutions,
            latent_connectivities_key,
            spatial_connectivities_key,
            layer_ratio,
            n_iterations,
            use_weights,
            rng,
            min_niche_size,
            mask,
            prefix=None,
            library_key=library_key,
            inplace=inplace,
            table_key=table_key,
        )

    return


@d.dedent
def calculate_niche_neighborhood(
    data: AnnData | SpatialData,
    groups: str,
    resolutions: float | list[float],
    n_neighbors: int = 15,
    spatial_connectivities_key: str = "spatial_connectivities",
    scale: bool = True,
    distance: int = 1,
    abs_nhood: bool = False,
    n_hop_weights: list[float] | None = None,
    min_niche_size: int | None = None,
    mask: pd.Series | None = None,
    library_key: str | None = None,
    inplace: bool = True,
    table_key: str | None = None,
) -> AnnData | None:
    """Compute niche neighborhoods using a neighborhood profile embedding and Leiden clustering.

    Each observation is represented by the frequency of ``groups`` labels in its
    spatial neighborhood, which is then clustered with the Leiden algorithm.

    Parameters
    ----------
    %(adata)s
    groups
        Column in ``adata.obs`` defining categorical groups (e.g. cell types)
        used to compute neighborhood composition profiles.
    n_neighbors
        Number of neighbors used when constructing the graph for Leiden clustering.
    resolutions
        Resolution parameter(s) for Leiden clustering. Can be a single float or a list.
    %(niche_spatial_conn_key)s
    scale
        Whether to z-score the neighborhood profile prior to clustering.
    distance
        Number of hops to consider when constructing neighborhood profiles.
        Values greater than ``1`` incorporate higher-order neighbors.
    abs_nhood
        If ``True``, use absolute counts; otherwise normalize to proportions.
    n_hop_weights
        Weights for combining neighborhood profiles across hops.
    %(niche_common_params)s
    %(table_key)s

    Returns
    -------
    If ``inplace = True``, modifies ``adata`` in place and returns ``None``.
    Otherwise, returns a copy of ``adata`` with niche annotations added to ``.obs``.

    """

    # Create instance of _NhoodProfileEmbedder using provided inputs
    embedder = _NhoodProfileEmbedder(
        groups,
        spatial_connectivities_key,
        scale,
        distance,
        abs_nhood,
        n_hop_weights,
    )

    # Create instance of _LeidenClusterer using provided inputs
    clusterer = _LeidenClusterer(n_neighbors, resolutions, "nhood_niche")

    return _calculate_niche_custom(
        data,
        embedder,
        clusterer,
        min_niche_size=min_niche_size,
        mask=mask,
        library_key=library_key,
        inplace=inplace,
        table_key=table_key,
    )


@d.dedent
def calculate_niche_utag(
    data: AnnData | SpatialData,
    resolutions: float | list[float],
    n_neighbors: int = 15,
    spatial_connectivities_key: str = "spatial_connectivities",
    min_niche_size: int | None = None,
    mask: pd.Series | None = None,
    library_key: str | None = None,
    inplace: bool = True,
    table_key: str | None = None,
) -> AnnData | None:
    """Compute niche assignments using a UTAG-style neighborhood embedding.

    Features are propagated over the spatial graph so each observation inherits
    information from its immediate neighbors, then clustered with the Leiden algorithm.

    Parameters
    ----------
    %(adata)s
    n_neighbors
        Number of neighbors used when constructing the graph for Leiden clustering.
    resolutions
        Resolution parameter(s) for Leiden clustering. Can be a single float or a list.
    %(niche_spatial_conn_key)s
    %(niche_common_params)s
    %(table_key)s

    Returns
    -------
    If ``inplace = True``, modifies ``adata`` in place and returns ``None``.
    Otherwise, returns a copy of ``adata`` with niche annotations added to ``.obs``.

    """

    embedder = _UtagEmbedder(spatial_connectivities_key)

    clusterer = _LeidenClusterer(n_neighbors, resolutions, "utag_niche")

    return _calculate_niche_custom(
        data,
        embedder,
        clusterer,
        min_niche_size=min_niche_size,
        mask=mask,
        library_key=library_key,
        inplace=inplace,
        table_key=table_key,
    )


@d.dedent
def calculate_niche_cellcharter(
    data: AnnData | SpatialData,
    distance: int = 3,
    aggregation: str = "mean",
    rng: SeedLike | RNGLike | None = None,
    spatial_connectivities_key: str = "spatial_connectivities",
    n_components: int = 10,
    use_rep: str | None = None,
    min_niche_size: int | None = None,
    mask: pd.Series | None = None,
    library_key: str | None = None,
    inplace: bool = True,
    table_key: str | None = None,
) -> AnnData | None:
    """Compute niche assignments using a CellCharter-style aggregation embedding.

    Features are aggregated across multi-hop spatial neighborhoods, then clustered
    with a Gaussian mixture model.

    Parameters
    ----------
    %(adata)s
    distance
        Number of neighborhood hops to aggregate when building the embedding.
    aggregation
        Aggregation mode used for neighborhood features, typically ``"mean"`` or
        ``"variance"``.
    %(rng)s
        Seeds the Gaussian mixture clustering step. When stratifying by ``library_key``,
        every library is fitted with an independent rng derived from it.
    %(niche_spatial_conn_key)s
    n_components
        Number of embedding components to retain when ``use_rep`` is provided,
        or number of mixture components used by the clusterer.
    use_rep
        Key in ``adata.obsm`` pointing to a precomputed representation to use
        instead of deriving a spatially aggregated embedding.
    %(niche_common_params)s
    %(table_key)s

    Returns
    -------
    If ``inplace = True``, modifies ``adata`` in place and returns ``None``.
    Otherwise, returns a copy of ``adata`` with niche annotations added to ``.obs``.

    """

    embedder = _CellcharterEmbedder(distance, aggregation, spatial_connectivities_key, n_components, use_rep)

    clusterer = _GMMClusterer(n_components, np.random.default_rng(rng), base_colname="cellcharter_niche")

    return _calculate_niche_custom(
        data,
        embedder,
        clusterer,
        min_niche_size=min_niche_size,
        mask=mask,
        library_key=library_key,
        inplace=inplace,
        table_key=table_key,
    )


@d.dedent
def calculate_niche_spatialleiden(
    data: AnnData | SpatialData,
    resolutions: float | tuple[float, float] | list[float | tuple[float, float]],
    latent_connectivities_key: str = "connectivities",
    spatial_connectivities_key: str = "spatial_connectivities",
    layer_ratio: float = 1.0,
    n_iterations: int = -1,
    use_weights: bool | tuple[bool, bool] = True,
    rng: SeedLike | RNGLike | None = None,
    min_niche_size: int | None = None,
    mask: pd.Series | None = None,
    prefix: str | None = None,
    library_key: str | None = None,
    inplace: bool = True,
    table_key: str | None = None,
) -> AnnData | None:
    """Compute niche assignments using the SpatialLeiden algorithm.

    This is a wrapper around the `SpatialLeiden <https://github.com/HiDiHlabs/SpatialLeiden>`_
    algorithm that uses :class:`~anndata.AnnData` as input and works with two layers; one latent
    space and one spatial layer.
    Adapted from https://github.com/HiDiHlabs/SpatialLeiden/.

    Parameters
    ----------
    %(adata)s
    latent_connectivities_key
        Key in ``adata.obsp`` containing the latent-space connectivity matrix.
    %(niche_spatial_conn_key)s
    resolutions
        Resolution parameter(s) for the Leiden optimization. Can be a single
        float or a list of floats.
    layer_ratio
        Relative weight assigned to the latent and spatial layers.
    n_iterations
        Number of optimization iterations used by SpatialLeiden.
    use_weights
        Whether to use edge weights during clustering.
    %(rng)s
        Each resolution — and each library when stratifying by ``library_key`` — is
        clustered with an independent rng derived from it.
    %(niche_min_niche_size)s
    %(niche_mask)s
    prefix
        Prefix added to niche labels produced by SpatialLeiden.
        When stratifying by ``library_key``, a library-specific prefix is added
        automatically (something like "lib=").
    %(library_key)s
    %(niche_inplace)s
    %(table_key)s

    Returns
    -------
    If ``inplace = True``, modifies ``adata`` in place and returns ``None``.
    Otherwise, returns a copy of ``adata`` with niche annotations added to ``.obs``.

    Notes
    -----
    If ``library_key`` is provided, clustering is performed independently for
    each library and the results are merged back into the parent object.
    """

    try:
        import spatialleiden as sl
    except ImportError as e:
        msg = "Please install the spatialleiden algorithm: `pip install squidpy[leiden]` or `conda install bioconda::spatialleiden` or `pip install spatialleiden`."
        raise ImportError(msg) from e

    # obtain adata if data was of sdata type
    orig_adata = extract_adata_if_sdata(data, table_key=table_key)

    if inplace:
        adata = orig_adata
    else:
        adata = orig_adata.copy()

    # normalise once here; everything below this point works with rngs only
    rng = np.random.default_rng(rng)

    if library_key is not None:
        # first assert that library_key was there in adata.obs, and then, stratify the object according to that library_key and
        # then re-call calculate_niche_spatialleiden for each subpart, with library_key = None and prefix with appropriate information like "lib="
        assert_key_in_adata(adata, library_key, attr="obs")
        logg.info(f"Stratifying by library_key '{library_key}'")

        # each library is an independent clustering problem, so it gets its own rng
        # (indexed by `itr` so that skipped empty libraries don't shift the others)
        library_ids = adata.obs[library_key].unique()
        library_rngs = rng.spawn(len(library_ids))

        # go through each library_id and process the corresponding adata subset
        for itr, lib_id in enumerate(library_ids):
            logg.info(f"Processing library '{lib_id}'")

            lib_indices = adata.obs[adata.obs[library_key] == lib_id].index

            if len(lib_indices) == 0:
                logg.warning(f"Library '{lib_id}' contains no cells, skipping")
                continue

            lib_adata = adata[lib_indices].copy()

            # give prefix appropriate value so that the niche values indicate lib id.
            calculate_niche_spatialleiden(
                lib_adata,
                resolutions,
                latent_connectivities_key,
                spatial_connectivities_key,
                layer_ratio,
                n_iterations,
                use_weights,
                library_rngs[itr],
                min_niche_size,
                mask,
                prefix=f"lib={lib_id}_",
                library_key=None,
                inplace=True,  # to save memory
                table_key=table_key,
            )

            # from itr==1 onwards, adata will hold the columns that are being added hence,
            # added_columns will be empty. Hence only obtain added_columns when itr==0
            if itr == 0:
                added_columns = list(set(lib_adata.obs.columns) - set(adata.obs.columns))

            for col in added_columns:
                # ensure that adata has the columns in which we are adding the information
                if col not in adata.obs:
                    adata.obs[col] = "not_a_niche"
                adata.obs.loc[lib_indices, col] = list(lib_adata.obs[col].astype("str"))

    else:
        # Simply call sl.spatialleiden with the provided arguments
        if not isinstance(resolutions, list):
            resolutions = [resolutions]

        # every resolution is a separate clustering run, so seed each one independently
        resolution_rngs = rng.spawn(len(resolutions))

        for res, res_rng in zip(resolutions, resolution_rngs, strict=True):
            sl.spatialleiden(
                adata,
                resolution=res,
                use_weights=use_weights,
                n_iterations=n_iterations,
                layer_ratio=layer_ratio,
                latent_neighbors_key=latent_connectivities_key,
                spatial_neighbors_key=spatial_connectivities_key,
                random_state=legacy_random(res_rng),
                directed=False,
                key_added=f"spatialleiden_res={res}",
            )

        # obtain the result_columns, which are basically the difference in columns in orig_adata and adata
        result_columns = [f"spatialleiden_res={res}" for res in resolutions]

        _postprocess_niche_results(adata, result_columns, mask, min_niche_size, prefix)

    # For SpatialData, the column names shouldn't have = sign. Hence, run sanitize_table.
    # TODO: In future, change the naming standard of any niche columns added to not have '=' to be compatible with spatialdata naming
    if isinstance(data, SpatialData):
        sanitize_table(adata)

    if inplace:
        return None
    else:
        return adata


@d.dedent
def _calculate_niche_custom(
    data: AnnData | SpatialData,
    embedder: _NicheEmbedder,
    clusterer: _NicheClusterer,
    min_niche_size: int | None = None,
    mask: pd.Series | None = None,
    library_key: str | None = None,
    inplace: bool = True,
    table_key: str | None = None,
) -> AnnData | None:
    """Compute niche assignments using user-defined embedding, clustering, and postprocessing.

    This function provides a flexible pipeline where embedding, clustering,
    and postprocessing are decoupled and customizable.

    Parameters
    ----------
    %(adata)s
    embedder
        Instance of ``_NicheEmbedder`` used to compute an embedding from ``adata``.
    clusterer
        Instance of ``_NicheClusterer`` used to assign niches based on the embedding.
    %(niche_common_params)s
    %(table_key)s

    Returns
    -------
    If ``inplace = True``, modifies ``adata`` in place and returns ``None``.
    Otherwise, returns a copy of ``adata`` with niche annotations added to ``.obs``.

    Notes
    -----
    If ``library_key`` is provided, the computation is performed independently
    for each library and results are merged back into ``adata``.

    See Also
    --------
    calculate_niche_neighborhood : Convenience wrapper for neighborhood flavor niche analysis.
    calculate_niche_utag : Convenience wrapper for utag flavor niche analysis.
    calculate_niche_cellcharter : Convenience wrapper for cellcharter flavor niche analysis.
    calculate_niche_spatialleiden : Convenience wrapper for spatialleiden flavor niche analysis.
    _NicheEmbedder : Base class for embedding strategies.
    _NicheClusterer : Base class for clustering strategies.
    """

    # obtain adata if data was of sdata type
    orig_adata = extract_adata_if_sdata(data, table_key=table_key)

    if inplace:
        adata = orig_adata
    else:
        adata = orig_adata.copy()

    if library_key is not None:
        assert_key_in_adata(adata, library_key, attr="obs")
        logg.info(f"Stratifying by library_key '{library_key}'")

        # go through each library_id and process the corresponding adata subset
        for itr, lib_id in enumerate(adata.obs[library_key].unique()):
            logg.info(f"Processing library '{lib_id}'")

            lib_indices = adata.obs[adata.obs[library_key] == lib_id].index

            if len(lib_indices) == 0:
                logg.warning(f"Library '{lib_id}' contains no cells, skipping")
                continue

            lib_adata = adata[lib_indices].copy()

            _run_niche_pipeline(
                lib_adata, embedder, clusterer, mask=mask, min_niche_size=min_niche_size, prefix=f"lib={lib_id}_"
            )

            # from itr==1 onwards, adata will hold the columns that are being added hence,
            # added_columns will be empty. Hence only obtain added_columns when itr==0
            if itr == 0:
                added_columns = list(set(lib_adata.obs.columns) - set(adata.obs.columns))

            for col in added_columns:
                # ensure that adata has the columns in which we are adding the information
                if col not in adata.obs:
                    adata.obs[col] = "not_a_niche"
                adata.obs.loc[lib_indices, col] = list(lib_adata.obs[col].astype("str"))

    else:
        _run_niche_pipeline(adata, embedder, clusterer, mask=mask, min_niche_size=min_niche_size)

    # For SpatialData, the column names shouldn't have = sign. Hence, run sanitize_table.
    # TODO: In future, change the naming standard of any niche columns added to not have '=' to be compatible with spatialdata naming
    if isinstance(data, SpatialData):
        sanitize_table(adata)

    if inplace:
        return None
    else:
        return adata


def _run_niche_pipeline(
    adata: AnnData,
    embedder: _NicheEmbedder,
    clusterer: _NicheClusterer,
    mask: pd.Series | None,
    min_niche_size: int | None,
    prefix: str | None = None,
) -> None:
    """Embed, cluster, and postprocess ``adata`` in place."""
    embedding = embedder.get_embedding(adata)
    result_columns = clusterer.cluster(adata, embedding)
    _postprocess_niche_results(adata, result_columns, mask, min_niche_size, prefix)


def _validate_niche_args(
    data: AnnData | SpatialData,
    flavor: Literal["neighborhood", "utag", "cellcharter", "spatialleiden"],
    library_key: str | None,
    table_key: str | None,
    groups: str | None,
    n_neighbors: int | None,
    resolutions: float | tuple[float, float] | list[float | tuple[float, float]] | None,
    min_niche_size: int | None,
    scale: bool,
    abs_nhood: bool,
    distance: int | None,
    n_hop_weights: list[float] | None,
    aggregation: str | None,
    n_components: int | None,
    # the one internal that sees a raw `rng`: it reports on what the caller passed, and
    # `None` must stay `None` here so the "unused for this flavor" check can spot it
    rng: SeedLike | RNGLike | None,
    spatial_connectivities_key: str,
    latent_connectivities_key: str,
    layer_ratio: float,
    n_iterations: int,
    use_weights: bool | tuple[bool, bool],
    use_rep: str | None,
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
    assert_isinstance(data, (AnnData, SpatialData), name="data")

    assert_one_of(flavor, ["neighborhood", "utag", "cellcharter", "spatialleiden"], name="flavor")

    if isinstance(data, SpatialData) and table_key is None:
        raise TypeError("missing required keyword-only argument: 'table_key'")

    if library_key is not None:
        assert_isinstance(library_key, str, name="library_key")
        adata = extract_adata_if_sdata(data, table_key=table_key)
        if library_key not in adata.obs.columns:
            raise ValueError(f"'library_key' must be a column in 'adata.obs', got {library_key}")

    if n_neighbors is not None:
        assert_isinstance(n_neighbors, int, name="n_neighbors")

    if resolutions is not None:
        if not isinstance(resolutions, float | tuple | list):
            raise TypeError(
                f"'resolutions' must be a float, a tuple of floats, a list of floats, or a list containing floats and/or tuples of floats, got {type(resolutions).__name__}"
            )

        if isinstance(resolutions, tuple):
            if not all(isinstance(x, float) for x in resolutions):
                raise TypeError("All elements in the tuple 'resolutions' must be floats.")
        elif isinstance(resolutions, list):
            for item in resolutions:
                if not (
                    isinstance(item, float) or (isinstance(item, tuple) and all(isinstance(i, float) for i in item))
                ):
                    raise TypeError("Each item in the list 'resolutions' must be a float or a tuple of floats.")

    if n_hop_weights is not None:
        assert_isinstance(n_hop_weights, list, name="n_hop_weights")

    assert_isinstance(scale, bool, name="scale")

    assert_isinstance(abs_nhood, bool, name="abs_nhood")

    # Define parameters used by each flavor
    flavor_param_specs = {
        "neighborhood": {
            "required": ["groups", "n_neighbors", "resolutions", "spatial_connectivities_key"],
            "optional": [
                "min_niche_size",
                "scale",
                "abs_nhood",
                "distance",
                "n_hop_weights",
            ],
            "unused": [
                "aggregation",
                "n_components",
                "rng",
                "latent_connectivities_key",
                "layer_ratio",
                "n_iterations",
                "use_weights",
                "use_rep",
            ],
        },
        "utag": {
            "required": ["n_neighbors", "resolutions", "spatial_connectivities_key"],
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
                "rng",
                "latent_connectivities_key",
                "layer_ratio",
                "n_iterations",
                "use_weights",
                "use_rep",
            ],
        },
        "cellcharter": {
            "required": ["distance", "aggregation", "spatial_connectivities_key"],
            # `rng` is optional: `None` is a valid value meaning "draw from OS entropy"
            "optional": ["n_components", "use_rep", "rng"],
            "unused": [
                "groups",
                "min_niche_size",
                "scale",
                "abs_nhood",
                "n_neighbors",
                "resolutions",
                "n_hop_weights",
                "latent_connectivities_key",
                "layer_ratio",
                "n_iterations",
                "use_weights",
            ],
        },
        "spatialleiden": {
            "required": ["latent_connectivities_key", "spatial_connectivities_key"],
            "optional": [
                "resolutions",
                "layer_ratio",
                "n_iterations",
                "use_weights",
                "rng",
            ],
            "unused": ["groups", "min_niche_size", "scale", "abs_nhood", "n_neighbors", "n_hop_weights", "use_rep"],
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
            "rng": rng,
            "use_rep": use_rep,
        },
        flavor_param_specs[flavor],
    )

    # Flavor-specific validations
    if flavor == "neighborhood":
        assert_isinstance(groups, str, name="groups")

        if min_niche_size is not None:
            assert_isinstance(min_niche_size, int, name="min_niche_size")

        if distance is not None and isinstance(distance, int) and distance < 1:
            raise ValueError(f"'distance' must be at least 1, got {distance}")

    elif flavor == "cellcharter":
        if distance is not None:
            assert_isinstance(distance, int, name="distance")
        if distance is not None and distance < 1:
            raise ValueError(f"'distance' must be at least 1, got {distance}")

        if aggregation is not None:
            assert_isinstance(aggregation, str, name="aggregation")
            assert_one_of(aggregation, ["mean", "variance"], name="aggregation")

        assert_isinstance(n_components, int, name="n_components")
        if n_components < 1:
            raise ValueError(f"'n_components' must be at least 1, got {n_components}")

        if use_rep is not None:
            assert_isinstance(use_rep, str, name="use_rep")

        # for mypy
        if resolutions is None:
            resolutions = [0.0]

    elif flavor == "spatialleiden":
        assert_isinstance(latent_connectivities_key, str, name="latent_connectivities_key")
        assert_isinstance(spatial_connectivities_key, str, name="spatial_connectivities_key")

        assert_isinstance(layer_ratio, (float, int), name="layer_ratio")
        assert_isinstance(n_iterations, int, name="n_iterations")
        if not (
            isinstance(use_weights, bool)
            or (
                isinstance(use_weights, tuple)
                and len(use_weights) == 2
                and all(isinstance(x, bool) for x in use_weights)
            )
        ):
            raise TypeError(f"'use_weights' must be a bool or a tuple of two bools, got {use_weights!r}")

        if resolutions is None:
            resolutions = [1.0]

    assert_isinstance(inplace, bool, name="inplace")


def _check_unnecessary_args(flavor: str, param_dict: dict[str, Any], param_specs: dict[str, Any]) -> None:
    """
    Check for unnecessary arguments that were provided but not used by the given flavor.

    Parameters
    ----------
    flavor
        The flavor being used ('neighborhood', 'utag', 'cellcharter', or 'spatialleiden')
    param_dict
        Dictionary of parameter names to their values
    param_specs
        Dictionary with 'required', 'optional', and 'unused' parameter lists for the flavor
    """
    unnecessary_args = []

    for param_name in param_specs["unused"]:
        param_value = param_dict.get(param_name)

        # Special handling for parameters whose default is not None
        if param_name == "scale" and param_value is True:
            continue
        if param_name == "abs_nhood" and param_value is False:
            continue

        if param_value is not None:
            unnecessary_args.append(param_name)

    if unnecessary_args:
        logg.warning(
            f"Parameters {', '.join([f'{arg}' for arg in unnecessary_args])} are not used for flavor '{flavor}'.",
        )


############
### embedder classes
############


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
    adj_hop: sps.spmatrix,
    adj: sps.spmatrix,
    adj_visited: sps.spmatrix = None,
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


class _NicheEmbedder(ABC):
    """Base class for computing embeddings used in niche analysis.

    Subclasses must implement :meth:`get_embedding`, which transforms an
    :class:`~anndata.AnnData` object into a feature matrix suitable for clustering.
    The 0-index dimension of returned embedding (embedding.shape[0])
    should correspond to the rows in adata.obs (and adata.X), meaning in
    the same order and having same length.
    """

    @abstractmethod
    def get_embedding(self, adata: AnnData) -> NDArrayA:
        """return an embedding matrix, with cells as rows"""


@d.dedent
class _NhoodProfileEmbedder(_NicheEmbedder):
    """Compute neighborhood composition profiles as embeddings.

    Each observation is represented by the frequency of categorical labels
    (e.g. cell types) in its spatial neighborhood. Optionally, higher-order
    neighborhoods (multi-hop) can be incorporated.

    Parameters
    ----------
    groups
        Column in ``adata.obs`` defining categorical labels.
    %(niche_spatial_conn_key)s
    scale
        Whether to z-score the resulting embedding.
    distance
        Number of hops to consider for neighborhood aggregation.
    abs_nhood
        If ``True``, use absolute counts of categories in neighborhood; otherwise
        normalize to proportions.
    n_hop_weights
        Weights for combining profiles across neighborhood hops.

    Notes
    -----
    For ``distance > 1``, neighborhood profiles are iteratively aggregated using
    powers of the adjacency matrix, optionally weighted per hop.
    """

    def __init__(
        self,
        groups: str,
        spatial_connectivities_key: str,
        scale: bool,
        distance: int,
        abs_nhood: bool,
        n_hop_weights: list[float] | None,
    ):
        self.groups = groups
        self.spatial_connectivities_key = spatial_connectivities_key
        self.scale = scale
        self.distance = distance
        self.abs_nhood = abs_nhood
        self.n_hop_weights = n_hop_weights

    def _calculate_neighborhood_profile(
        self,
        adata: AnnData,
        matrix: coo_matrix,
    ) -> pd.DataFrame:
        """
        Returns an obs x category matrix where each column is the absolute/relative frequency of a category in the neighborhood
        """

        # ensure that adata.obs[group] is of categorical type, as that makes it explicit, which cols of the returned profile_df
        # correspond to which categories in group
        if adata.obs[self.groups].dtype.name != "category":
            warnings.warn(
                "Since adata.obs[groups] does not already have categorical dtype, converting it into categorical type.",
                stacklevel=2,
            )
            adata.obs[self.groups] = adata.obs[self.groups].astype("category")

        # ensure matrix is in csc format for efficient column slicing
        if matrix.format != "csc":
            matrix = matrix.tocsc()

        # get cell categories in order
        categories_order = adata.obs[self.groups].cat.categories
        n_categories = len(categories_order)

        # map category to column index
        category_to_idx = {ct: i for i, ct in enumerate(categories_order)}

        # pre allocate sparse LIL matrix for efficient assignment (n_cells x n_categories)
        profile_sparse = lil_matrix((matrix.shape[0], n_categories), dtype=np.float64)

        # for each category, sum over cells of that category
        for ct in categories_order:
            ct_mask = adata.obs[self.groups] == ct  # boolean mask for cells of this category
            col_indices = np.where(ct_mask)[0]  # indices of those cells
            if len(col_indices) > 0:
                col_slice = matrix[:, col_indices]  # sparse submatrix
                profile_sparse[:, category_to_idx[ct]] = col_slice.sum(axis=1).A1

        # convert to dataframe (csr for final storage, dense for pandas)
        profile_df = pd.DataFrame(
            profile_sparse.tocsr().todense(), index=adata.obs[self.groups].index, columns=categories_order
        )

        # now according to parameter abs_nhood, make raw counts into proportions or not
        if not self.abs_nhood:
            total_neighs = profile_df.sum(axis=1)
            profile_df = profile_df.div(total_neighs, axis=0)
            # this may lead to some values being nan, as some cells might have had no neighbors. Make those values as 0
            profile_df = profile_df.fillna(0.0)

        return profile_df

    def get_embedding(self, adata: AnnData) -> NDArrayA:
        """
        adapted from https://github.com/immunitastx/monkeybread/blob/main/src/monkeybread/calc/_neighborhood_profile.py
        """

        # get obs x neighbor matrix from sparse matrix
        matrix = adata.obsp[self.spatial_connectivities_key].tocoo()

        # get obs x category matrix where each column is the absolute/relative frequency of a category in the neighborhood
        nhood_profile = self._calculate_neighborhood_profile(adata, matrix)

        # Additionally use n-hop neighbors if distance > 1. This sums up the (weighted) neighborhood profiles of all n-hop neighbors.
        if self.distance > 1:
            # keep weights local: the same embedder instance is reused across libraries
            if self.n_hop_weights is None:
                weights = [1.0] * self.distance
            elif len(self.n_hop_weights) < self.distance:
                # Extend weights if too few provided
                weights = self.n_hop_weights + [self.n_hop_weights[-1]] * (self.distance - len(self.n_hop_weights))
                logg.debug(f"Extended weights to match distance: {weights}")
            else:
                weights = self.n_hop_weights

            # Apply first weight to base profile
            weighted_profile = weights[0] * nhood_profile

            # Calculate higher-order hop profiles
            n_hop_adjacency_matrix = adata.obsp[self.spatial_connectivities_key].copy()

            # get n_hop neighbor adjacency matrices by multiplying the original adjacency matrix with itself n times and get corresponding neighborhood profiles.
            for n_hop in range(1, self.distance):
                logg.debug(f"Calculating {n_hop + 1}-hop neighbors")
                # Multiply adjacency matrix by itself to get n+1 hop adjacency
                n_hop_adjacency_matrix = n_hop_adjacency_matrix @ adata.obsp[self.spatial_connectivities_key]
                matrix = n_hop_adjacency_matrix.tocoo()

                # Calculate and add weighted profile
                hop_profile = self._calculate_neighborhood_profile(adata, matrix)
                weighted_profile += weights[n_hop] * hop_profile

            if not self.abs_nhood:
                weighted_profile = weighted_profile / sum(weights)

            nhood_profile = weighted_profile

        # create AnnData object from neighborhood profile to perform scanpy functions
        # Use .to_numpy(copy=True) to ensure the array is writeable (required for pandas CoW compatibility)
        # Preserve the DataFrame index for later matching with adata_masked
        adata_neighborhood = ad.AnnData(
            X=nhood_profile.to_numpy(copy=True), obs=pd.DataFrame(index=nhood_profile.index)
        )

        # reason for scaling see https://monkeybread.readthedocs.io/en/latest/notebooks/tutorial.html#niche-analysis
        if self.scale:
            sc.pp.scale(adata_neighborhood, zero_center=True)
        return adata_neighborhood.X


@d.dedent
class _UtagEmbedder(_NicheEmbedder):
    """Compute a UTAG-style embedding by propagating features over spatial neighbors.

    The embedding is constructed by normalizing the spatial connectivity matrix,
    multiplying it by ``adata.X``, and then applying PCA to the propagated
    feature matrix.

    Parameters
    ----------
    %(niche_spatial_conn_key)s

    Notes
    -----
    This follows the general UTAG idea that each observation inherits information
    from its immediate spatial neighborhood before dimensionality reduction.
    """

    def __init__(
        self,
        spatial_connectivities_key: str,
    ):
        self.spatial_connectivities_key = spatial_connectivities_key

    def get_embedding(self, adata: AnnData) -> NDArrayA:
        """
        Performs inner product of adjacency matrix and feature matrix,
        such that each observation inherits features from its immediate neighbors as described in UTAG paper.
        """

        adjacency_matrix = adata.obsp[self.spatial_connectivities_key]
        new_feature_matrix = normalize(adjacency_matrix, norm="l1", axis=1) @ adata.X
        adata_utag = ad.AnnData(X=new_feature_matrix)
        sc.tl.pca(adata_utag)  # note: unlike with flavor 'neighborhood' dim reduction is performed here
        return adata_utag.obsm["X_pca"]


# TODO: This function requires some work later on. Right now keeping the implementation just like how
# it was before the refactor, and in that case, when use_rep was provided, then it simply returned
# that as the embedding, so no cellcharter algorithm used in that case
@d.dedent
class _CellcharterEmbedder(_NicheEmbedder):
    """Compute a CellCharter-style embedding from spatially aggregated features.

    The embedding can either be derived from a precomputed representation in
    ``adata.obsm`` or constructed by aggregating features across multi-hop
    spatial neighborhoods.

    Parameters
    ----------
    distance
        Number of neighborhood hops to aggregate.
    aggregation
        Aggregation strategy to apply to neighborhood features, such as
        ``"mean"`` or ``"variance"``.
    %(niche_spatial_conn_key)s
    n_components
        Number of components to keep from the input representation when ``use_rep``
        is provided.
    use_rep
        Key in ``adata.obsm`` pointing to the representation to use. If ``None``,
        a spatially aggregated embedding is constructed from ``adata.X``.

    Notes
    -----
    When ``use_rep`` is ``None``, PCA is applied to the concatenated aggregated
    feature matrix to produce the final embedding.
    """

    def __init__(
        self,
        distance: int,
        aggregation: str,
        spatial_connectivities_key: str,
        n_components: int,
        use_rep: str | None,
    ):
        self.distance = distance
        self.aggregation = aggregation
        self.spatial_connectivities_key = spatial_connectivities_key
        self.n_components = n_components
        self.use_rep = use_rep

    # this will hold an if block checking if use_rep is not None. If not None, then it will simply
    # return that representation from adata
    # Also a note for user, n_components is only used when use_rep is not None. It is the number of
    # components from that representation to use as the embedding
    # aggregation is only used when use_rep is None
    def get_embedding(self, adata: AnnData) -> NDArrayA:
        """adapted from https://github.com/CSOgroup/cellcharter/blob/main/src/cellcharter/gr/_aggr.py
        and https://github.com/CSOgroup/cellcharter/blob/main/src/cellcharter/tl/_gmm.py"""

        if self.use_rep is not None:
            # Use provided embedding from adata.obsm
            assert_key_in_adata(adata, self.use_rep, attr="obsm")
            embedding = adata.obsm[self.use_rep]
            # Ensure embedding has the right number of components
            if embedding.shape[1] < self.n_components:
                raise ValueError(
                    f"Embedding has {embedding.shape[1]} components, but n_components={self.n_components}. Please provide an embedding with at least {self.n_components} components."
                )
            # Use only the first n_components
            embedding = embedding[:, : self.n_components]
        else:
            logg.warning(
                "CellCharter recommends to use a dimensionality reduced embedding of the data, e.g. a scVI embedding. Since 'use_rep' is not provided, PCA will be used as proxy - performance may be suboptimal."
            )
            adjacency_matrix = adata.obsp[self.spatial_connectivities_key]
            layers = list(range(self.distance + 1))

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
                    aggregated_matrix = _aggregate(adata, adj_hop_norm, self.aggregation)
                    aggregated_matrices.append(aggregated_matrix)

            concatenated_matrix = hstack(aggregated_matrices)  # Stack all matrices horizontally
            arr = concatenated_matrix.toarray()  # Densify

            arr_ad = ad.AnnData(X=arr)
            sc.tl.pca(arr_ad)
            embedding = arr_ad.obsm["X_pca"]

        return embedding


############
### clusterer classes
############


class _NicheClusterer(ABC):
    """Base class for clustering embeddings into niche assignments.

    Subclasses must implement :meth:`cluster`, which assigns cluster labels
    and stores them in ``adata.obs``.
    """

    @abstractmethod
    def cluster(self, adata: AnnData, embedding: NDArrayA) -> list[str]:
        """Adds column/s in adata.obs with the clustering done. Returns the names of the columns just added."""


@d.dedent
class _LeidenClusterer(_NicheClusterer):
    """Cluster embeddings using the Leiden algorithm.

    Parameters
    ----------
    n_neighbors
        Number of neighbors used to construct the kNN graph.
    resolutions
        Resolution parameter(s) for Leiden clustering. Can be a single
        float value or list of floats.
    base_colname
        Base name for columns added to ``adata.obs``. Resolution is
        appended to this to unique identify columns for each resolution.

    Notes
    -----
    A separate clustering is computed for each resolution, producing multiple
    niche annotation columns.
    """

    def __init__(
        self,
        n_neighbors: int,
        resolutions: float | list[float],
        base_colname: str = "niche_leiden",
    ):
        self.n_neighbors = n_neighbors
        self.resolutions = resolutions if isinstance(resolutions, list) else [resolutions]
        self.base_colname = base_colname

    def cluster(self, adata: AnnData, embedding: NDArrayA) -> list:
        # first create an adata object using the embedding provided
        adata_embedding = ad.AnnData(X=embedding, obs=pd.DataFrame(index=adata.obs.index))

        # required for leiden clustering (note: no dim reduction performed in original implementation)
        sc.pp.neighbors(adata_embedding, n_neighbors=self.n_neighbors, use_rep="X")

        # For each resolution, apply leiden on neighborhood profile. Each cluster label equals to a niche label
        niche_keys = []
        for res in self.resolutions:
            niche_key = f"{self.base_colname}_res={res}"
            niche_keys.append(niche_key)

            if niche_key in adata.obs.columns:
                logg.info(f"Overwriting existing column '{niche_key}'")

            sc.tl.leiden(
                adata_embedding,
                resolution=res,
                key_added=niche_key,
            )

            adata.obs[niche_key] = list(
                adata_embedding.obs[niche_key]
            )  # since constrain all embedders to return embedding with numrows==numcells and in same order, this should be fine

        return niche_keys


@d.dedent
class _GMMClusterer(_NicheClusterer):
    """Cluster embeddings with a Gaussian mixture model.

    Parameters
    ----------
    n_components
        Number of mixture components.
    rng
        rng supplying the seed of every mixture fit.
    base_colname
        Name of the output column added to ``adata.obs``.

    Notes
    -----
    Cluster assignments are stored as categorical niche labels in ``adata.obs``.

    One instance may be reused for several fits (e.g. once per library when stratifying
    by ``library_key``). Each :meth:`cluster` call draws a fresh seed from ``rng``, so the
    fits are seeded independently while remaining reproducible as a sequence.
    """

    def __init__(
        self,
        n_components: int,
        rng: np.random.Generator,
        base_colname: str = "niche_gmm",
    ):
        self.n_components = n_components
        self.rng = rng
        self.base_colname = base_colname

    def cluster(self, adata: AnnData, embedding: NDArrayA) -> list:
        """Returns niche labels generated by GMM clustering.
        Compared to cellcharter this approach is simplified by using sklearn's GaussianMixture model without stability analysis.
        """
        # cluster concatenated matrix with GMM, each cluster label equals to a niche label
        gmm = GaussianMixture(
            n_components=self.n_components,
            random_state=legacy_random(self.rng),
            init_params="random_from_data",
        )
        gmm.fit(embedding)
        niches = gmm.predict(embedding)

        if self.base_colname in adata.obs.columns:
            logg.info(f"Overwriting existing column '{self.base_colname}'")

        adata.obs[self.base_colname] = pd.Categorical(niches)
        return [self.base_colname]


############
### postprocessing
############


def _postprocess_niche_results(
    adata: AnnData,
    result_columns: list[str],
    mask: pd.Series | None = None,
    min_niche_size: int | None = None,
    prefix: str | None = None,
) -> None:
    """Refine niche assignments in place, rewriting each column in ``result_columns``.

    Parameters
    ----------
    adata
        Annotated data matrix.
    result_columns
        Columns in ``adata.obs`` holding the niche assignments to refine.
    mask
        Boolean :class:`~pandas.Series` indexed like ``adata.obs``. Observations that
        are ``False`` get the label ``"not_a_niche"``, e.g.
        ``Series([False, False, True], index=["a", "b", "c"])``.
    min_niche_size
        Niches with fewer than this many observations are relabeled ``"not_a_niche"``.
    prefix
        Prepended to every niche label, used to keep labels unique across libraries.

    Notes
    -----
    Columns are modified in place, so the niche column name does not depend on
    which of these options were supplied.
    """
    if mask is None and min_niche_size is None and prefix is None:
        return

    for col in result_columns:
        # str, so that "not_a_niche" and prefixed labels can be assigned regardless of the clusterer's dtype
        labels = adata.obs[col].astype(str)

        if mask is not None:
            aligned = mask[mask.index.isin(adata.obs.index)]
            labels[~aligned] = "not_a_niche"

        if min_niche_size is not None:
            counts = labels.value_counts()
            too_small = counts[counts < min_niche_size].index
            labels[labels.isin(too_small)] = "not_a_niche"

        if prefix is not None:
            labels = prefix + labels

        adata.obs[col] = labels

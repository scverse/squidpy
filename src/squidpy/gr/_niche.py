from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps
from anndata import AnnData
from fast_array_utils.types import HasArrayNamespace as Array
from sklearn.base import clone
from spatialdata import SpatialData, sanitize_table
from spatialdata._logging import logger as logg

from squidpy._constants._constants import NicheDefinitions
from squidpy._docs import d, inject_docs
from squidpy._utils import RNGLike, SeedLike, deprecated_randomness_param, legacy_random
from squidpy._validators import assert_isinstance, assert_key_in_adata, assert_one_of
from squidpy.gr._autok import _gmm, check_model_params
from squidpy.gr._clusterers import _AutoKClusterer, _LeidenClusterer
from squidpy.gr._nhood import _nhood_aggregate
from squidpy.gr._utils import extract_adata_if_sdata
from squidpy.types import Clusterer, SweepableClusterer

#: Turns an ``AnnData`` into the matrix -- or graph -- its clusterer consumes, one row per
#: observation, in ``adata.obs`` order.
NicheEmbedder = Callable[[AnnData], Array]

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
            library_key=library_key,
            copy=not inplace,
            table_key=table_key,
            n_iterations=n_iterations,
            rng=rng,
        )

    elif flavor == "utag":
        return calculate_niche_utag(
            data,
            resolutions,
            n_neighbors,
            spatial_connectivities_key,
            min_niche_size,
            mask,
            library_key=library_key,
            copy=not inplace,
            table_key=table_key,
            n_iterations=n_iterations,
            rng=rng,
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
            library_key=library_key,
            copy=not inplace,
            table_key=table_key,
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
            copy=not inplace,
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
    copy: bool = False,
    table_key: str | None = None,
    *,
    flavor: Literal["igraph", "leidenalg"] = "igraph",
    n_iterations: int = -1,
    rng: SeedLike | RNGLike | None = None,
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
    %(niche_leiden_params)s

    Returns
    -------
    If ``copy = True``, returns a copy of ``adata`` with niche annotations added to ``.obs``.
    Otherwise, modifies ``adata`` in place and returns ``None``.

    """

    # Create instance of _NhoodProfileEmbedder using provided inputs
    # the graph and the partitions are the two stochastic stages, so they get independent
    # streams: changing one does not shift the other's draws
    graph_rng, clusterer_rng = np.random.default_rng(rng).spawn(2)

    embedder = partial(
        _latent_graph_embedding,
        embedder=partial(
            _nhood_profile_embedding,
            groups=groups,
            spatial_connectivities_key=spatial_connectivities_key,
            scale=scale,
            distance=distance,
            abs_nhood=abs_nhood,
            n_hop_weights=n_hop_weights,
        ),
        n_neighbors=n_neighbors,
        rng=graph_rng,
    )

    clusterers = _leiden_clusterers("nhood_niche", resolutions, flavor=flavor, n_iterations=n_iterations)

    return _calculate_niche_custom(
        data,
        embedder,
        clusterers,
        clusterer_rng,
        min_niche_size=min_niche_size,
        mask=mask,
        library_key=library_key,
        copy=copy,
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
    copy: bool = False,
    table_key: str | None = None,
    *,
    flavor: Literal["igraph", "leidenalg"] = "igraph",
    n_iterations: int = -1,
    rng: SeedLike | RNGLike | None = None,
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
    %(niche_leiden_params)s

    Returns
    -------
    If ``copy = True``, returns a copy of ``adata`` with niche annotations added to ``.obs``.
    Otherwise, modifies ``adata`` in place and returns ``None``.

    """

    # both stages are stochastic here, so they get independent streams: changing one does not
    # shift the other's draws, and results stay comparable across versions
    embedder_rng, graph_rng, clusterer_rng = np.random.default_rng(rng).spawn(3)

    embedder = partial(
        _latent_graph_embedding,
        embedder=partial(_utag_embedding, spatial_connectivities_key=spatial_connectivities_key, rng=embedder_rng),
        n_neighbors=n_neighbors,
        rng=graph_rng,
    )

    clusterers = _leiden_clusterers("utag_niche", resolutions, flavor=flavor, n_iterations=n_iterations)

    return _calculate_niche_custom(
        data,
        embedder,
        clusterers,
        clusterer_rng,
        min_niche_size=min_niche_size,
        mask=mask,
        library_key=library_key,
        copy=copy,
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
    copy: bool = False,
    table_key: str | None = None,
    *,
    n_clusters: int | tuple[int, int] | Sequence[int] | None = None,
    max_runs: int = 10,
    convergence_tol: float = 1e-2,
    store_labels: bool = False,
    model_params: Mapping[str, Any] | None = None,
) -> AnnData | None:
    """Compute niche assignments using a CellCharter-style aggregation embedding.

    Features are aggregated across multi-hop spatial neighborhoods, then clustered
    with a Gaussian mixture model. The number of mixture components can either be fixed
    or selected by the stability sweep of :func:`~squidpy.gr.cluster_auto_k`.

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
    n_clusters
        Number of mixture components. ``None`` falls back to ``n_components``, an ``int``
        fits that number directly, and a ``(min, max)`` tuple or a sequence of candidates
        selects the most stable K by fitting each candidate ``max_runs`` times, see
        ``expand_n_clusters``.
    max_runs
        Maximum number of repetitions per candidate K. Only used when ``n_clusters``
        requests a sweep.
    convergence_tol
        Stop the sweep early once the mean absolute percentage error between the mean
        stability curves of consecutive runs falls below this value.
    store_labels
        Also keep the labeling of every fitted K as ``cellcharter_niche_k{K}`` columns in
        ``adata.obs``, for comparing resolutions.
    model_params
        Extra keyword arguments for :class:`~sklearn.mixture.GaussianMixture`, e.g.
        ``{'reg_covar': 1e-4}`` when a component collapses. The mapping is never modified.
        ``n_components`` and ``random_state`` are controlled by ``n_clusters`` and ``rng``.

    Returns
    -------
    If ``copy = True``, returns a copy of ``adata`` with niche annotations added to ``.obs``.
    Otherwise, modifies ``adata`` in place and returns ``None``.

    When ``n_clusters`` requests a sweep, per-K diagnostics are stored in
    ``adata.uns["cellcharter_niche_autok"]`` (keyed by library id when ``library_key`` is
    given, since each library selects its own K). The niche column is always
    ``cellcharter_niche``, independent of the selected K.
    """

    # both stages are stochastic here, so they get independent streams: changing one does not
    # shift the other's draws, and results stay comparable across versions
    embedder_rng, rng = np.random.default_rng(rng).spawn(2)

    # `use_rep` short-circuits the whole recipe, so it picks the embedder rather than
    # branching inside one; see scverse/squidpy#1277
    embedder: NicheEmbedder
    if use_rep is not None:
        embedder = partial(_rep_embedding, use_rep=use_rep, n_components=n_components)
    else:
        logg.warning(
            "CellCharter recommends to use a dimensionality reduced embedding of the data, e.g. a scVI "
            "embedding. Since 'use_rep' is not provided, PCA will be used as proxy - performance may be "
            "suboptimal."
        )
        embedder = partial(
            _cellcharter_embedding,
            distance=distance,
            aggregation=aggregation,
            spatial_connectivities_key=spatial_connectivities_key,
            rng=embedder_rng,
        )

    # up front, so a bad one is rejected before the embedding is computed
    check_model_params(model_params or {})

    clusterer: Clusterer
    if n_clusters is None or isinstance(n_clusters, int):
        # `n_components` doubles as rep width above; `n_clusters` takes over the mixture count
        clusterer = _gmm(model_params, n_components=n_components if n_clusters is None else n_clusters)
    else:
        clusterer = _AutoKClusterer(
            n_clusters=n_clusters,
            max_runs=max_runs,
            convergence_tol=convergence_tol,
            store_labels=store_labels,
            uns_key="cellcharter_niche_autok",
            model_params=model_params,
        )

    return _calculate_niche_custom(
        data,
        embedder,
        {"cellcharter_niche": clusterer},
        rng,
        min_niche_size=min_niche_size,
        mask=mask,
        library_key=library_key,
        copy=copy,
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
    copy: bool = False,
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
    %(copy)s
    %(table_key)s

    Returns
    -------
    If ``copy = True``, returns a copy of ``adata`` with niche annotations added to ``.obs``.
    Otherwise, modifies ``adata`` in place and returns ``None``.

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

    adata = orig_adata.copy() if copy else orig_adata

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
                copy=False,  # to save memory
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

    return adata if copy else None


@d.dedent
def _calculate_niche_custom(
    data: AnnData | SpatialData,
    embedder: NicheEmbedder,
    clusterers: Mapping[str, Clusterer],
    rng: np.random.Generator,
    min_niche_size: int | None = None,
    mask: pd.Series | None = None,
    library_key: str | None = None,
    copy: bool = False,
    table_key: str | None = None,
) -> AnnData | None:
    """Compute niche assignments using user-defined embedding, clustering, and postprocessing.

    This function provides a flexible pipeline where embedding, clustering,
    and postprocessing are decoupled and customizable.

    Parameters
    ----------
    %(adata)s
    embedder
        Any ``(AnnData) -> Array`` callable, returning one row per observation in
        ``adata.obs`` order. See :data:`NicheEmbedder`.
    clusterers
        The ``adata.obs`` column each :class:`~squidpy.types.Clusterer` labels, keyed by
        column name. Any scikit-learn clusterer will do; several entries means several
        niche columns, as a scan over Leiden resolutions does.
    rng
        Seeds every fit: one draw per clusterer, per library.
    %(niche_common_params)s
    %(table_key)s

    Returns
    -------
    If ``copy = True``, returns a copy of ``adata`` with niche annotations added to ``.obs``.
    Otherwise, modifies ``adata`` in place and returns ``None``.

    Notes
    -----
    If ``library_key`` is provided, the computation is performed independently
    for each library and results are merged back into ``adata``.

    Diagnostics a clusterer leaves in a ``niche_uns_`` attribute are written to
    ``adata.uns``; see :mod:`squidpy.gr._clusterers`.

    See Also
    --------
    calculate_niche_neighborhood : Convenience wrapper for neighborhood flavor niche analysis.
    calculate_niche_utag : Convenience wrapper for utag flavor niche analysis.
    calculate_niche_cellcharter : Convenience wrapper for cellcharter flavor niche analysis.
    calculate_niche_spatialleiden : Convenience wrapper for spatialleiden flavor niche analysis.
    squidpy.types.Clusterer : What the clusterers have to implement.
    """

    # obtain adata if data was of sdata type
    orig_adata = extract_adata_if_sdata(data, table_key=table_key)

    adata = orig_adata.copy() if copy else orig_adata

    if library_key is not None:
        assert_key_in_adata(adata, library_key, attr="obs")
        logg.info(f"Stratifying by library_key '{library_key}'")

        diagnostics_per_library: dict[str, dict[str, Any]] = {}

        # go through each library_id and process the corresponding adata subset
        for itr, lib_id in enumerate(adata.obs[library_key].unique()):
            logg.info(f"Processing library '{lib_id}'")

            lib_indices = adata.obs[adata.obs[library_key] == lib_id].index

            if len(lib_indices) == 0:
                logg.warning(f"Library '{lib_id}' contains no cells, skipping")
                continue

            lib_adata = adata[lib_indices].copy()

            diagnostics = _run_niche_pipeline(
                lib_adata,
                embedder,
                clusterers,
                rng,
                mask=mask,
                min_niche_size=min_niche_size,
                prefix=f"lib={lib_id}_",
            )
            if diagnostics is not None:
                diagnostics_per_library[str(lib_id)] = diagnostics

            # from itr==1 onwards, adata will hold the columns that are being added hence,
            # added_columns will be empty. Hence only obtain added_columns when itr==0
            if itr == 0:
                added_columns = list(set(lib_adata.obs.columns) - set(adata.obs.columns))

            for col in added_columns:
                # ensure that adata has the columns in which we are adding the information
                if col not in adata.obs:
                    adata.obs[col] = "not_a_niche"
                adata.obs.loc[lib_indices, col] = list(lib_adata.obs[col].astype("str"))

        # per-library `uns` is discarded by the merge, so write once here
        for uns_key in {key for per_library in diagnostics_per_library.values() for key in per_library}:
            adata.uns[uns_key] = {
                lib_id: per_library[uns_key]
                for lib_id, per_library in diagnostics_per_library.items()
                if uns_key in per_library
            }

    else:
        diagnostics = _run_niche_pipeline(adata, embedder, clusterers, rng, mask=mask, min_niche_size=min_niche_size)
        if diagnostics is not None:
            adata.uns.update(diagnostics)

    # For SpatialData, the column names shouldn't have = sign. Hence, run sanitize_table.
    # TODO: In future, change the naming standard of any niche columns added to not have '=' to be compatible with spatialdata naming
    if isinstance(data, SpatialData):
        sanitize_table(adata)

    return adata if copy else None


def _leiden_clusterers(
    base_colname: str,
    resolutions: float | list[float],
    *,
    flavor: Literal["igraph", "leidenalg"],
    n_iterations: int,
) -> dict[str, Clusterer]:
    """One Leiden clusterer per resolution, keyed by the niche column each labels.

    They all partition the one graph the embedder produced, which is what makes a scan
    over resolutions comparable -- and cheap.
    """
    values = resolutions if isinstance(resolutions, list) else [resolutions]
    return {
        f"{base_colname}_res={res}": _LeidenClusterer(resolution=res, flavor=flavor, n_iterations=n_iterations)
        for res in values
    }


def _seeded_clone(clusterer: Clusterer, rng: np.random.Generator) -> Clusterer:
    """A clone of *clusterer* for one fit, reseeded if it takes a seed.

    A clone per fit, so the caller's estimator is never mutated and no two fits -- one per
    column, per library -- share a seed. Estimators that take no ``random_state`` are
    cloned but not reseeded: :class:`~sklearn.cluster.AgglomerativeClustering`, say, has
    none because nothing about it varies between runs. One that cannot be cloned at all is
    fitted as given.
    """
    if not isinstance(clusterer, SweepableClusterer):
        return clusterer
    clusterer = clone(clusterer)
    if "random_state" in clusterer.get_params():
        clusterer.set_params(random_state=legacy_random(rng))
    return clusterer


def _run_niche_pipeline(
    adata: AnnData,
    embedder: NicheEmbedder,
    clusterers: Mapping[str, Clusterer],
    rng: np.random.Generator,
    mask: pd.Series | None,
    min_niche_size: int | None,
    prefix: str | None = None,
) -> dict[str, Any] | None:
    """Embed, cluster, postprocess in place; returns clusterer diagnostics as {uns_key: payload}."""
    embedding = embedder(adata)

    columns: dict[str, Array] = {}
    diagnostics: dict[str, Any] = {}
    for colname, clusterer in clusterers.items():
        fitted = _seeded_clone(clusterer, rng)
        columns[colname] = fitted.fit_predict(embedding)
        # the two fitted attributes a niche flavor needs beyond one label column; see
        # `squidpy.gr._clusterers`. A plain scikit-learn clusterer has neither.
        columns |= {f"{colname}_{suffix}": extra for suffix, extra in getattr(fitted, "niche_columns_", {}).items()}
        diagnostics |= getattr(fitted, "niche_uns_", {})

    for colname, labels in columns.items():
        if colname in adata.obs.columns:
            logg.info(f"Overwriting existing column '{colname}'")
        adata.obs[colname] = pd.Categorical(labels)

    _postprocess_niche_results(adata, list(columns), mask, min_niche_size, prefix)
    return diagnostics or None


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
                "rng",
                "n_iterations",
            ],
            "unused": [
                "aggregation",
                "n_components",
                "latent_connectivities_key",
                "layer_ratio",
                "use_weights",
                "use_rep",
            ],
        },
        "utag": {
            "required": ["n_neighbors", "resolutions", "spatial_connectivities_key"],
            "optional": ["rng", "n_iterations"],
            "unused": [
                "groups",
                "min_niche_size",
                "scale",
                "abs_nhood",
                "distance",
                "n_hop_weights",
                "aggregation",
                "n_components",
                "latent_connectivities_key",
                "layer_ratio",
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


# An embedder is any ``(AnnData) -> Array`` callable: adata in, one row per observation
# out, in the same order. A function rather than a class, since all any of these needs is
# its arguments bound -- and a function is something backend dispatch can attach to, which
# a third-party subclass is not. See scverse/squidpy#1151.


def _nhood_profile_embedding(
    adata: AnnData,
    *,
    groups: str,
    spatial_connectivities_key: str,
    scale: bool,
    distance: int,
    abs_nhood: bool,
    n_hop_weights: list[float] | None,
) -> Array:
    """Neighborhood composition: how frequent each category is around each observation.

    Adapted from
    https://github.com/immunitastx/monkeybread/blob/main/src/monkeybread/calc/_neighborhood_profile.py
    """
    profile = _nhood_aggregate(
        adata,
        groups=groups,
        connectivity_key=spatial_connectivities_key,
        # every hop counts walks of every length up to it, and the hops are summed into one
        # profile rather than concatenated
        hops=range(1, distance + 1),
        hop_mode="power",
        combine="sum",
        hop_weights=n_hop_weights,
        aggregation="sum" if abs_nhood else "mean",
    )
    # reason for scaling see https://monkeybread.readthedocs.io/en/latest/notebooks/tutorial.html#niche-analysis
    return sc.pp.scale(profile, zero_center=True) if scale else profile


def _utag_embedding(adata: AnnData, *, spatial_connectivities_key: str, rng: np.random.Generator) -> Array:
    """Each observation inherits the mean features of its immediate neighbors, as in the UTAG paper."""
    aggregated = _nhood_aggregate(adata, connectivity_key=spatial_connectivities_key, hops=(1,))
    # note: unlike with flavor 'neighborhood' dim reduction is performed here. A fresh draw per
    # call, so each library gets its own PCA seed, as the clusterers do.
    return sc.tl.pca(aggregated, random_state=legacy_random(rng))


def _cellcharter_embedding(
    adata: AnnData,
    *,
    distance: int,
    aggregation: str,
    spatial_connectivities_key: str,
    rng: np.random.Generator,
) -> Array:
    """Disjoint hop rings of aggregated features, concatenated and reduced.

    Adapted from https://github.com/CSOgroup/cellcharter/blob/main/src/cellcharter/gr/_aggr.py
    and https://github.com/CSOgroup/cellcharter/blob/main/src/cellcharter/tl/_gmm.py
    """
    aggregated = _nhood_aggregate(
        adata,
        connectivity_key=spatial_connectivities_key,
        # hop 0 is the observation's own counts; the rings are disjoint, and are concatenated
        # so that each keeps its own columns
        hops=range(distance + 1),
        hop_mode="shell",
        combine="concat",
        aggregation=aggregation,
    )
    return sc.tl.pca(aggregated, random_state=legacy_random(rng))


def _rep_embedding(adata: AnnData, *, use_rep: str, n_components: int) -> Array:
    """The first *n_components* of an existing representation, aggregated by whoever made it."""
    assert_key_in_adata(adata, use_rep, attr="obsm")
    embedding = adata.obsm[use_rep]
    if embedding.shape[1] < n_components:
        raise ValueError(
            f"Embedding has {embedding.shape[1]} components, but n_components={n_components}. "
            f"Please provide an embedding with at least {n_components} components."
        )
    return embedding[:, :n_components]


def _latent_graph_embedding(
    adata: AnnData, *, embedder: NicheEmbedder, n_neighbors: int, rng: np.random.Generator
) -> sps.csr_matrix:
    """The connectivities of the kNN graph over another embedder's matrix.

    The step between an embedding and a graph clusterer such as
    :class:`~squidpy.gr._clusterers._LeidenClusterer`. It sits here rather than inside the
    clusterer so that one graph serves any number of resolutions -- building it is the
    expensive part, and every resolution has to partition the *same* graph for the scan to
    mean anything.
    """
    # no dimension reduction, as in the original implementations
    embedding = ad.AnnData(X=np.asarray(embedder(adata)))
    sc.pp.neighbors(embedding, n_neighbors=n_neighbors, use_rep="X", random_state=legacy_random(rng))
    return embedding.obsp["connectivities"]


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

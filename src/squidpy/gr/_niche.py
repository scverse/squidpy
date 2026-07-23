from __future__ import annotations

import contextlib
import warnings
from typing import Any, Literal
from abc import abstractmethod

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps
from anndata import AnnData
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, hstack, issparse, lil_matrix, spdiags
from scipy.spatial import distance
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from spatialdata import SpatialData, sanitize_table
from spatialdata._logging import logger as logg

from squidpy._constants._constants import NicheDefinitions
from squidpy._docs import d, inject_docs
from squidpy._validators import assert_isinstance, assert_key_in_adata, assert_one_of
from squidpy.gr._utils import extract_adata_if_sdata
from squidpy._utils import NDArrayA

__all__ = ["calculate_niche"]


@d.dedent
@inject_docs(fla=NicheDefinitions)
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
    random_state: int = 42,
    spatial_connectivities_key: str = "spatial_connectivities",
    latent_connectivities_key: str = "connectivities",
    layer_ratio: float = 1.0,
    n_iterations: int = -1,
    use_weights: bool | tuple[bool, bool] = True,
    use_rep: str | None = None,
    inplace: bool = True,
    *,
    table_key: str | None = None,
) -> AnnData | None:
    """
    Calculate Niche
    """

    if flavor == "cellcharter" and aggregation is None:
        aggregation = "mean"

    if distance is None:
        distance = 3 if flavor == "cellcharter" else 1

    if flavor == "cellcharter" and n_components is None:
        n_components = 10

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

    if flavor == 'neighborhood':
        return calculate_niche_neighborhood(
            data,
            groups,
            n_neighbors,
            resolutions,
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

    elif flavor == 'utag':
        return calculate_niche_utag(
            data,
            n_neighbors,
            resolutions,
            spatial_connectivities_key,
            min_niche_size,
            mask,
            library_key,
            inplace,
            table_key,
        )

    elif flavor == 'cellcharter':
        return calculate_niche_cellcharter(
            data,
            distance,
            aggregation,
            random_state,
            spatial_connectivities_key,
            n_components,
            use_rep,
            min_niche_size,
            mask,
            library_key,
            inplace,
            table_key,
        )

    elif flavor == 'spatialleiden':
        return calculate_niche_spatialleiden(
            data,
            latent_connectivities_key,
            spatial_connectivities_key,
            resolutions,
            layer_ratio,
            n_iterations,
            use_weights,
            random_state,
            min_niche_size,
            mask,
            prefix = None,
            library_key = library_key,
            inplace = inplace,
            table_key = table_key,
        )

    return

def calculate_niche_neighborhood(
    data,
    groups,
    n_neighbors,
    resolutions,
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
) -> AnnData | None:

    # Create instance of NhoodProfileEmbedder using provided inputs
    embedder = NhoodProfileEmbedder(
        groups,
        spatial_connectivities_key,
        scale,
        distance,
        abs_nhood,
        n_hop_weights,
    )

    # Create instance of LeidenClusterer using provided inputs
    clusterer = LeidenClusterer(n_neighbors, resolutions, 'nhood_niche')

    # generate the list of postprocessor objects using the supplied args
    postprocessors_list = []
    if mask is not None:
        mask_postprocessor = MaskPostprocessor(mask)
        postprocessors_list.append(mask_postprocessor)
    if min_niche_size is not None:
        min_niche_size_postprocessor = MinNicheSizePostprocessor(min_niche_size)
        postprocessors_list.append(min_niche_size_postprocessor)

    return calculate_niche_custom(
        data,
        embedder,
        clusterer,
        postprocessors_list,
        library_key,
        inplace,
        table_key
    )

def calculate_niche_utag(
    data,
    n_neighbors,
    resolutions,
    spatial_connectivities_key,
    min_niche_size,
    mask,
    library_key,
    inplace,
    table_key,
) -> AnnData | None:

    embedder = UtagEmbedder(
        spatial_connectivities_key
    )

    clusterer = LeidenClusterer(n_neighbors, resolutions, 'utag_niche')

    # generate the list of postprocessor objects using the supplied args
    postprocessors_list = []
    if mask is not None:
        mask_postprocessor = MaskPostprocessor(mask)
        postprocessors_list.append(mask_postprocessor)
    if min_niche_size is not None:
        min_niche_size_postprocessor = MinNicheSizePostprocessor(min_niche_size)
        postprocessors_list.append(min_niche_size_postprocessor)

    return calculate_niche_custom(
        data,
        embedder,
        clusterer,
        postprocessors_list,
        library_key,
        inplace,
        table_key
    )

def calculate_niche_cellcharter(
    data,
    distance,
    aggregation,
    random_state,
    spatial_connectivities_key,
    n_components,
    use_rep,
    min_niche_size,
    mask,
    library_key,
    inplace,
    table_key,
) -> AnnData | None:

    embedder = CellcharterEmbedder(
        distance,
        aggregation,
        spatial_connectivities_key,
        n_components,
        use_rep
    )

    clusterer = GMMClusterer(n_components, random_state, base_colname='cellcharter_niche')

    # generate the list of postprocessor objects using the supplied args
    postprocessors_list = []
    if mask is not None:
        mask_postprocessor = MaskPostprocessor(mask)
        postprocessors_list.append(mask_postprocessor)
    if min_niche_size is not None:
        min_niche_size_postprocessor = MinNicheSizePostprocessor(min_niche_size)
        postprocessors_list.append(min_niche_size_postprocessor)

    return calculate_niche_custom(
        data,
        embedder,
        clusterer,
        postprocessors_list,
        library_key,
        inplace,
        table_key
    )

def calculate_niche_spatialleiden(
    data,
    latent_connectivities_key,
    spatial_connectivities_key,
    resolutions,
    layer_ratio,
    n_iterations,
    use_weights,
    random_state,
    min_niche_size,
    mask,
    prefix, # default value will be None
    library_key,
    inplace,
    table_key,
) -> AnnData | None:
    """
    Perform SpatialLeiden clustering.
    This is a wrapper around :py:func:`spatialleiden.multiplex_leiden` that uses :py:class:`anndata.AnnData` as input and works with two layers; one latent space and one spatial layer.
    Adapted from https://github.com/HiDiHlabs/SpatialLeiden/.

    Parameters
    ----------
    prefix
        What to add as a prefix in the names of niches identified. Used implicitly when library_key is not None (adds "lib=").
    """
    try:
        import spatialleiden as sl
    except ImportError as e:
        msg = "Please install the spatialleiden algorithm: `pip install squidpy[leiden]` or `conda install bioconda::spatialleiden` or `pip install spatialleiden`."
        raise ImportError(msg) from e
    
    # obtain adata if data was of sdata type
    orig_adata = extract_adata_if_sdata(data, table_key=table_key)
    # make a copy of the adata object, with which we will work
    adata = orig_adata.copy()

    if library_key is not None:
        # first assert that library_key was there in adata.obs, and then, stratify the object according to that library_key and
        # then re-call calculate_niche_spatialleiden for each subpart, with library_key = None and prefix with appropriate information like "lib="
        assert_key_in_adata(adata, library_key, attr="obs")
        logg.info(f"Stratifying by library_key '{library_key}'")

        # go through each library_id and process the corresponding adata subset
        for lib_id in adata.obs[library_key].unique():
            logg.info(f"Processing library '{lib_id}'")

            lib_indices = adata.obs[adata.obs[library_key] == lib_id].index

            if len(lib_indices) == 0:
                logg.warning(f"Library '{lib_id}' contains no cells, skipping")
                continue

            lib_adata = adata[lib_indices].copy()

            # give prefix appropriate value so that the niche values indicate lib id.
            lib_result = calculate_niche_spatialleiden(
                lib_adata,
                latent_connectivities_key,
                spatial_connectivities_key,
                resolutions,
                layer_ratio,
                n_iterations,
                use_weights,
                random_state,
                min_niche_size,
                mask,
                prefix = f'lib={lib_id}',
                library_key = None,
                inplace = False,
                table_key = table_key,
            )

            added_columns = list(set(lib_result.obs.columns) - set(adata.obs.columns))

            for col in added_columns:
                # ensure that adata has the columns in which we are adding the information
                if col not in adata.obs:
                    adata.obs[col] = 'not_a_niche'
                adata.obs.loc[lib_indices, col] = list(lib_result.obs[col])
        
    else:
        # Simply call sl.spatialleiden with the provided arguments
        if not isinstance(resolutions, list):
            resolutions = [resolutions]
        
        for res in resolutions:
            sl.spatialleiden(
                adata,
                resolution=res,
                use_weights=use_weights,
                n_iterations=n_iterations,
                layer_ratio=layer_ratio,
                latent_neighbors_key=latent_connectivities_key,
                spatial_neighbors_key=spatial_connectivities_key,
                random_state=random_state,
                directed=False,
                key_added=f"spatialleiden_res={res}",
            )

        # obtain the result_columns, which are basically the difference in columns in orig_adata and adata
        result_columns = list(set(adata.obs.columns) - set(orig_adata.obs.columns))

        # generate the list of postprocessor objects using the supplied args
        postprocessors_list = []
        if mask is not None:
            mask_postprocessor = MaskPostprocessor(mask)
            postprocessors_list.append(mask_postprocessor)
        if min_niche_size is not None:
            min_niche_size_postprocessor = MinNicheSizePostprocessor(min_niche_size)
            postprocessors_list.append(min_niche_size_postprocessor)
        if prefix is not None:
            renaming_postprocessor = RenamePostprocessor(prefix)
            postprocessors_list.append(renaming_postprocessor)

        postprocess_niche_results(adata, result_columns, postprocessors_list)

    return return_niche_output(data, orig_adata, adata, inplace, table_key)

def calculate_niche_custom(
    data,
    embedder,
    clusterer,
    postprocessors_list,
    library_key,
    inplace,
    table_key,
) -> AnnData | None:

    # obtain adata if data was of sdata type
    orig_adata = extract_adata_if_sdata(data, table_key=table_key)
    # make a copy of the adata object, with which we will work
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

            # append a renaming postprocessor to postprocessors_list_lib
            renaming_postprocessor = RenamePostprocessor(prefix_for_niches = f'lib={lib_id}_')
            postprocessors_list_lib = postprocessors_list + [renaming_postprocessor]

            lib_result = calculate_niche_custom(
                lib_adata,
                embedder,
                clusterer,
                postprocessors_list_lib,
                library_key = None,
                inplace = False,
                table_key = None,
            )

            # from itr==1 onwards, adata will hold the columns that are being added hence, 
            # added_columns will be empty. Hence only obtain added_columns when itr==0
            if itr == 0:
                added_columns = list(set(lib_result.obs.columns) - set(adata.obs.columns))

            for col in added_columns:
                # ensure that adata has the columns in which we are adding the information
                if col not in adata.obs:
                    adata.obs[col] = 'not_a_niche'
                adata.obs.loc[lib_indices, col] = list(lib_result.obs[col])

    else:
        # supply the adata object to the embedder object, and obtain appropriate embedding matrix
        embedding = embedder.get_embedding(adata)

        # Supply to the clusterer object, the embedding matrix just obtained, and get the appropriate clustering.
        result_columns = clusterer.cluster(adata, embedding)

        # do postprocessing
        postprocess_niche_results(adata, result_columns, postprocessors_list)

    return return_niche_output(data, orig_adata, adata, inplace, table_key)

def return_niche_output(data, orig_adata, adata, inplace, table_key):
    if not inplace:
        return adata

    # result_columns are the columns that are added to adata compared to orig_adata
    result_columns = list(set(adata.obs.columns) - set(orig_adata.obs.columns))

    # For SpatialData, update the table directly
    if isinstance(data, SpatialData):
        sanitize_table(adata)
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

def postprocess_niche_results(adata, result_columns, postprocessors_list):

    # go through each postprocessor object, and apply it to the adata, and store
    # results in the form of new columns in adata
    for postprocessor in postprocessors_list:
        # obtain the new columns in this process
        result_columns = postprocessor.postprocess(adata, result_columns)

    return

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
    random_state: int,
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
                "random_state",
                "latent_connectivities_key",
                "layer_ratio",
                "n_iterations",
                "use_weights",
                "use_rep"
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
                "random_state",
                "latent_connectivities_key",
                "layer_ratio",
                "n_iterations",
                "use_weights",
                "use_rep"
            ],
        },
        "cellcharter": {
            "required": ["distance", "aggregation", "random_state", "spatial_connectivities_key"],
            "optional": ["n_components", "use_rep"],
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
                "random_state",
            ],
            "unused": [
                "groups",
                "min_niche_size",
                "scale",
                "abs_nhood",
                "n_neighbors",
                "n_hop_weights",
                "use_rep"
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

        assert_isinstance(random_state, int, name="random_state")

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
        assert_isinstance(random_state, int, name="random_state")

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
        logg.warning(
            f"Parameters {', '.join([f'{arg}' for arg in unnecessary_args])} are not used for flavor '{flavor}'.",
        )

############
### embedder classes
############

class NicheEmbedder():

    @abstractmethod
    def get_embedding(self, adata: AnnData) -> NDArrayA:
        """return an embedding matrix, with cells as rows"""

class NhoodProfileEmbedder(NicheEmbedder):

    def __init__(
        self,
        groups,
        spatial_connectivities_key,
        scale,
        distance,
        abs_nhood,
        n_hop_weights,
    ):
        super().__init__()
        self.groups = groups
        self.spatial_connectivities_key = spatial_connectivities_key
        self.scale = scale
        self.distance = distance
        self.abs_nhood = abs_nhood
        self.n_hop_weights = n_hop_weights

    def _calculate_neighborhood_profile(
        self,
        adata: AnnData,
        groups: str | None,
        matrix: coo_matrix,
        abs_nhood: bool,
    ) -> pd.DataFrame:
        """
        Returns an obs x category matrix where each column is the absolute/relative frequency of a category in the neighborhood
        """

        # ensure that adata.obs[group] is of categorical type, as that makes it explicit, which cols of the returned profile_df
        # correspond to which categories in group
        if adata.obs[groups].dtype.name != "category":
            warnings.warn(
                "Since adata.obs[groups] does not already have categorical dtype, converting it into categorical type.",
                stacklevel=2,
            )
            adata.obs[groups] = adata.obs[groups].astype("category")

        # ensure matrix is in csc format for efficient column slicing
        if matrix.format != "csc":
            matrix = matrix.tocsc()

        # get cell categories in order
        categories_order = adata.obs[groups].cat.categories
        n_categories = len(categories_order)

        # map category to column index
        category_to_idx = {ct: i for i, ct in enumerate(categories_order)}

        # pre allocate sparse LIL matrix for efficient assignment (n_cells x n_categories)
        profile_sparse = lil_matrix((matrix.shape[0], n_categories), dtype=np.float64)

        # for each category, sum over cells of that category
        for ct in categories_order:
            ct_mask = adata.obs[groups] == ct  # boolean mask for cells of this category
            col_indices = np.where(ct_mask)[0]  # indices of those cells
            if len(col_indices) > 0:
                col_slice = matrix[:, col_indices]  # sparse submatrix
                profile_sparse[:, category_to_idx[ct]] = col_slice.sum(axis=1).A1

        # convert to dataframe (csr for final storage, dense for pandas)
        profile_df = pd.DataFrame(profile_sparse.tocsr().todense(), index=adata.obs[groups].index, columns=categories_order)

        # now according to parameter abs_nhood, make raw counts into proportions or not
        if not abs_nhood:
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
        nhood_profile = self._calculate_neighborhood_profile(adata, self.groups, matrix, self.abs_nhood)

        # Additionally use n-hop neighbors if distance > 1. This sums up the (weighted) neighborhood profiles of all n-hop neighbors.
        if self.distance > 1:
            n_hop_adjacency_matrix = adata.obsp[self.spatial_connectivities_key].copy()
            # if no weights are provided, use 1 for all n_hop neighbors
            if self.n_hop_weights is None:
                self.n_hop_weights = [1] * self.distance
            # if weights are provided, start with applying weight to the original neighborhood profile
            elif len(self.n_hop_weights) < self.distance:
                # Extend weights if too few provided
                self.n_hop_weights = self.n_hop_weights + [self.n_hop_weights[-1]] * (self.distance - len(self.n_hop_weights))
                logg.debug(f"Extended weights to match distance: {self.n_hop_weights}")

            # Apply first weight to base profile
            weighted_profile = self.n_hop_weights[0] * nhood_profile

            # Calculate higher-order hop profiles
            n_hop_adjacency_matrix = adata.obsp[self.spatial_connectivities_key].copy()

            # get n_hop neighbor adjacency matrices by multiplying the original adjacency matrix with itself n times and get corresponding neighborhood profiles.
            for n_hop in range(1, self.distance):
                logg.debug(f"Calculating {n_hop + 1}-hop neighbors")
                # Multiply adjacency matrix by itself to get n+1 hop adjacency
                n_hop_adjacency_matrix = n_hop_adjacency_matrix @ adata.obsp[self.spatial_connectivities_key]
                matrix = n_hop_adjacency_matrix.tocoo()

                # Calculate and add weighted profile
                hop_profile = self._calculate_neighborhood_profile(adata, self.groups, matrix, self.abs_nhood)
                weighted_profile += self.n_hop_weights[n_hop] * hop_profile

            if not self.abs_nhood:
                weighted_profile = weighted_profile / sum(self.n_hop_weights)

            nhood_profile = weighted_profile

        # create AnnData object from neighborhood profile to perform scanpy functions
        # Use .to_numpy(copy=True) to ensure the array is writeable (required for pandas CoW compatibility)
        # Preserve the DataFrame index for later matching with adata_masked
        adata_neighborhood = ad.AnnData(X=nhood_profile.to_numpy(copy=True), obs=pd.DataFrame(index=nhood_profile.index))

        # reason for scaling see https://monkeybread.readthedocs.io/en/latest/notebooks/tutorial.html#niche-analysis
        if self.scale:
            sc.pp.scale(adata_neighborhood, zero_center=True)
        return adata_neighborhood.X

class UtagEmbedder(NicheEmbedder):
    def __init__(
        self, 
        spatial_connectivities_key,
    ):
        super().__init__()
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
        return adata_utag.obsm['X_pca']
    
# TODO: This function requires some work later on. Right now keeping the implementation just like how 
# it was before the refactor, and in that case, when use_rep was provided, then it simply returned 
# that as the embedding, so no cellcharter algorithm used in that case
class CellcharterEmbedder(NicheEmbedder):
    def __init__(
        self,
        distance,
        aggregation,
        spatial_connectivities_key,
        n_components,
        use_rep,
    ):
        super().__init__()
        self.distance = distance
        self.aggregation = aggregation
        self.spatial_connectivities_key = spatial_connectivities_key
        self.n_components = n_components
        self.use_rep = use_rep

    def _setdiag(self, adjacency_matrix: sps.spmatrix, value: int) -> sps.spmatrix:
        """remove self-loops"""

        if issparse(adjacency_matrix):
            adjacency_matrix = adjacency_matrix.tolil()
        adjacency_matrix.setdiag(value)
        adjacency_matrix = adjacency_matrix.tocsr()
        if value == 0:
            adjacency_matrix.eliminate_zeros()
        return adjacency_matrix

    def _hop(
        self,
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
    
    def _normalize(self, adj: sps.spmatrix) -> sps.spmatrix:
        """normalize adjacency matrix such that nodes with high degree don't disproportionately affect aggregation"""

        deg = np.array(np.sum(adj, axis=1)).squeeze()
        with np.errstate(divide="ignore"):
            deg_inv = 1 / deg
        deg_inv[deg_inv == float("inf")] = 0

        return spdiags(deg_inv, 0, len(deg_inv), len(deg_inv)) * adj
    
    def _aggregate(self, adata: AnnData, normalized_adjacency_matrix: sps.spmatrix, aggregation: str = "mean") -> Any:
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
            embedding = embedding[:, :self.n_components]
        else:
            logg.warning(
                "CellCharter recommends to use a dimensionality reduced embedding of the data, e.g. a scVI embedding. Since 'use_rep' is not provided, PCA will be used as proxy - performance may be suboptimal."
            )
            adjacency_matrix = adata.obsp[self.spatial_connectivities_key]
            layers = list(range(self.distance + 1))

            aggregated_matrices = []
            adj_hop = self._setdiag(adjacency_matrix, 0)  # Remove self-loops, set diagonal to 0
            adj_visited = self._setdiag(adjacency_matrix.copy(), 1)  # Track visited neighbors
            for k in layers:
                if k == 0:
                    # get original count matrix (not aggregated)
                    aggregated_matrices.append(adata.X)
                else:
                    # get count and adjacency matrix for k-hop (neighbor of neighbor of neighbor ...) and aggregate them
                    if k > 1:
                        adj_hop, adj_visited = self._hop(adj_hop, adjacency_matrix, adj_visited)
                    adj_hop_norm = self._normalize(adj_hop)
                    aggregated_matrix = self._aggregate(adata, adj_hop_norm, self.aggregation)
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

class NicheClusterer():
    @abstractmethod
    def cluster(self, adata: AnnData, embedding: NDArrayA) -> list:
        """Adds column/s in adata.obs with the clustering done. Returns the names of the columns just added."""

class LeidenClusterer(NicheClusterer):
    def __init__(self, n_neighbors, resolutions: float | list[float], base_colname: str = 'niche_leiden'):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.resolutions = resolutions if isinstance(resolutions, list) else [resolutions]
        self.base_colname = base_colname

    def cluster(self, adata: AnnData, embedding: NDArrayA) -> list:
        # first create an adata object using the embedding provided
        adata_embedding = ad.AnnData(X=embedding, obs=pd.DataFrame(index=adata.obs.index)) # TODO: is supplying obs necessary here?

        # required for leiden clustering (note: no dim reduction performed in original implementation)
        sc.pp.neighbors(adata_embedding, n_neighbors=self.n_neighbors, use_rep="X")

        # For each resolution, apply leiden on neighborhood profile. Each cluster label equals to a niche label
        niche_keys = []
        for res in self.resolutions:
            niche_key = f"{self.base_colname}_res={res}"
            niche_keys.append(niche_key)

            if niche_key in adata.obs.columns:
                logg.info(f"Overwriting existing column '{niche_key}'")
                del adata.obs[niche_key]

            sc.tl.leiden(
                adata_embedding,
                resolution=res,
                key_added=niche_key,
            )

            adata.obs[niche_key] = list(adata_embedding.obs[niche_key]) # since constrain all embedders to return embedding with numrows==numcells and in same order, this should be fine

        return niche_keys

class GMMClusterer(NicheClusterer):

    def __init__(self, n_components, random_state, base_colname = 'niche_gmm'):
        super().__init__()
        self.n_components = n_components
        self.random_state = random_state
        self.base_colname = base_colname

    def cluster(self, adata: AnnData, embedding: NDArrayA) -> list:
        """Returns niche labels generated by GMM clustering.
        Compared to cellcharter this approach is simplified by using sklearn's GaussianMixture model without stability analysis.
        """
        # cluster concatenated matrix with GMM, each cluster label equals to a niche label
        gmm = GaussianMixture(
            n_components=self.n_components,
            random_state=self.random_state,
            init_params="random_from_data",
        )
        gmm.fit(embedding)
        niches = gmm.predict(embedding)

        adata.obs[self.base_colname] = pd.Categorical(niches)
        return [self.base_colname]

############
### postprocessor classes
############

class NichePostprocessor():
    def __init__(self, suffix):
        self.suffix = suffix

    @abstractmethod
    def postprocess(self, adata: AnnData, result_columns: list[str]) -> list[str]:
        """Logic to postprocess adata and return the names of columns added."""
        # should append add self.suffix to the columns added

class MinNicheSizePostprocessor(NichePostprocessor):

    def __init__(self, min_niche_size, suffix = '_size_filter'):
        super().__init__(suffix = suffix)
        self.min_niche_size = min_niche_size

    def postprocess(self, adata: AnnData, result_columns: list[str]) -> list[str]:
        new_result_columns = []
        # filter niches with n_cells < min_niche_size
        for result_column in result_columns:
            # copy into new column
            new_result_column = result_column + self.suffix
            new_result_columns.append(new_result_column)
            adata.obs[new_result_column] = list(adata.obs[result_column])

            counts_by_niche = adata.obs[new_result_column].value_counts()
            to_filter = counts_by_niche[counts_by_niche < self.min_niche_size].index
            adata.obs[new_result_column] = adata.obs[new_result_column].apply(
                lambda x, to_filter=to_filter: "not_a_niche" if x in to_filter else x
            )
            adata.obs[new_result_column] = adata.obs.index.map(adata.obs[new_result_column]).fillna("not_a_niche")

        return new_result_columns

class MaskPostprocessor(NichePostprocessor):

    def __init__(self, mask, suffix = '_mask'):
        super().__init__(suffix = suffix)
        self.mask = mask

    def postprocess(self, adata: AnnData, result_columns: list[str]) -> list[str]:
        new_result_columns = []
        # mask obs to exclude cells for which no niche shall be assigned
        for result_column in result_columns:
            # copy into new column
            new_result_column = result_column + self.suffix
            new_result_columns.append(new_result_column)
            adata.obs[new_result_column] = list(adata.obs[result_column])

            to_filter = self.mask[self.mask.index.isin(adata.obs.index)]
            adata.obs[new_result_column] = adata.obs[new_result_column].apply(
                lambda x, to_filter=to_filter: "not_a_niche" if x in to_filter else x
            )

        return new_result_columns

class RenamePostprocessor(NichePostprocessor):

    def __init__(self, prefix_for_niches, suffix = '_renamed'):
        super().__init__(suffix = suffix)
        self.prefix_for_niches = prefix_for_niches

    def postprocess(self, adata: AnnData, result_columns: list[str]) -> list[str]:
        new_result_columns = []
        for result_column in result_columns:
            # copy into new column
            new_result_column = result_column + self.suffix
            new_result_columns.append(new_result_column)

            adata.obs[new_result_column] = self.prefix_for_niches + adata.obs[result_column].astype(str)

        return new_result_columns


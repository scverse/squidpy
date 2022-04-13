from __future__ import annotations

from typing import Any
from pathlib import Path
import os
import re
import json

from scanpy import logging as logg
from anndata import AnnData

from scipy.sparse import csr_matrix
import pandas as pd

from squidpy.read._utils import _load_image, _read_counts
from squidpy.datasets._utils import PathLike
from squidpy._constants._pkg_constants import Key

__all__ = ["visium", "vizgen", "nanostring"]


def visium(
    path: PathLike,
    *,
    counts_file: str = "filtered_feature_bc_matrix.h5",
    library_id: str | None = None,
    load_images: bool = True,
    source_image_path: PathLike | None = None,
    **kwargs: Any,
) -> AnnData:
    """
    Read 10x-Genomics-formatted Visium dataset.

    In addition to reading regular 10x output,
    this looks for the `spatial` folder and loads images,
    coordinates and scale factors.
    Based on the `Space Ranger output docs`_.
    See :func:`scanpy.pl.spatial` for a compatible plotting function.
    .. _Space Ranger output docs:
    https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/overview

    Parameters
    ----------
    path
        Path to directory for Visium data files.
    counts_file
        Which file in the passed directory to use as the count file. Typically would be one of:
        'filtered_feature_bc_matrix.h5' or 'raw_feature_bc_matrix.h5'.
    library_id
        Identifier for the Visium library. Can be modified when concatenating multiple adata objects.
    kwargs
        Keyword arguments for TODO.

    Returns
    -------
    Annotated data matrix ``adata``, where observations/cells are named by their
    barcode and variables/genes by gene name. Stores the following information:

        - :attr:`anndata.AnnData.obsm` ``['spatial']`` - spatial spot coordinates.
        - :attr:`anndata.AnnData.uns` ``['spatial']['{library_id}']['images']`` - *hires* and *lowres* images.
        - :attr:`anndata.AnnData.uns` ``['spatial']['{library_id}']['scalefactors']`` - scale factors for the spots.
        - :attr:`anndata.AnnData.uns` ``['spatial']['{library_id}']['metadata']`` - metadata.
    """
    path = Path(path)
    adata, library_id = _read_counts(path, count_file=counts_file, library_id=library_id, **kwargs)

    if not load_images:
        return adata

    adata.uns[Key.uns.spatial][library_id][Key.uns.image_key] = {
        f"{res}_image": _load_image(path / f"{Key.uns.spatial}/tissue_{res}_image.png") for res in ["hires", "lowres"]
    }
    adata.uns[Key.uns.spatial][library_id]["scalefactors"] = json.loads(
        (path / f"{Key.uns.spatial}/scalefactors_json.json").read_bytes()
    )

    # fmt: off
    coords = pd.read_csv(path / f"{Key.uns.spatial}/tissue_positions_list.csv", index_col=0, header=None)
    coords.columns = ["in_tissue", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"]
    # fmt: on

    adata.obs = pd.merge(adata.obs, coords, how="left", left_index=True, right_index=True)
    adata.obsm[Key.obsm.spatial] = adata.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values
    adata.obs.drop(columns=["pxl_row_in_fullres", "pxl_col_in_fullres"], inplace=True)

    if source_image_path is not None:
        source_image_path = Path(source_image_path).absolute()
        if not source_image_path.exists():
            logg.warning(f"Path to the high-resolution tissue image `{source_image_path}` does not exist")
        adata.uns["spatial"][library_id]["metadata"]["source_image_path"] = str(source_image_path)

    return adata


def vizgen(
    path: str | Path,
    *,
    counts_file: str,
    meta_file: str,
    transformation_file: str | None = None,
    library_id: str = "library",
    **kwargs: Any,
) -> AnnData:
    r"""
    Read Vizgen formatted dataset.

    In addition to reading the regular Vizgen output,
    this function loads metadata file to load spatial coordinates and loads transformation matrix.
    .. _Vizgen sample output docs:
    https://f.hubspotusercontent40.net/hubfs/9150442/Vizgen%20MERFISH%20Mouse%20Receptor%20Map%20File%20Descriptions%20.pdf?__hstc=&__hssc=

    Parameters
    ----------
    path
        Path to directory for Vizgen data files.
    counts_file
        Which file in the passed directory to use as the count file. Typically would be one of:
        '_cell_by_gene.csv'.
    meta_file
        This metadata file has the spatial coordinates of each of the detected cells.
    transformation_file
        Transformation matrix file for converting micron coordinates into pixels in images.
    library_id
        Identifier for the Vizgen library. Can be modified when concatenating multiple adata objects.

    Returns
    -------
    Annotated data matrix ``adata``, where observations/cells are named by their
    barcode and variables/genes by gene name. Stores the following information:
    # TODO
    :attr:`~anndata.AnnData.uns`\\ `['spatial']`
        Dict of Vizgen output files with 'library_id' as key
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['scalefactors']['transformation_matrix']`
        Transformation matrix for converting micron coordinates to pixels
    :attr:`~anndata.AnnData.obsm`\\ `['spatial']`
        Spatial coordinates of center of cells in micron coordinates, usable as `basis` by :func:`~scanpy.pl.embedding`.
    """
    path = Path(path)
    adata, library_id = _read_counts(
        path=path, count_file=counts_file, library_id=library_id, delimiter=",", first_column_names=True, **kwargs
    )
    adata.X = csr_matrix(adata.X)

    # fmt: off
    coords = pd.read_csv(path / meta_file, header=0, index_col=0)
    coords.columns = ["fov", "volume", "center_x", "center_y", "min_x", "max_x", "min_y", "max_y"]
    # fmt: on

    adata.obs = pd.merge(adata.obs, coords, how="left", left_index=True, right_index=True)
    adata.obsm[Key.obsm.spatial] = adata.obs[["center_x", "center_y"]].values
    adata.obs.drop(columns=["center_x", "center_y"], inplace=True)

    if transformation_file is not None:
        matrix = pd.read_csv(path / f"images/{transformation_file}", sep=" ", header=None)
        adata.uns[Key.uns.spatial][library_id]["scalefactors"] = {"transformation_matrix": matrix}

    return adata


def nanostring(
    path: str | Path,
    *,
    counts_file: str,
    meta_file: str,
    fov_file: str | None = None,
) -> AnnData:
    """
    Read Nanostring formatted dataset.

    In addition to reading the regular Nanostring output,
    loading metadata file to load spatial coordinates,
    this function reads fov_file to load coordinates of fields of view
    and looks for `CellCompsite` folder and loads images.

    Parameters
    ----------
    path
        Path to directory for Nanostring data files.
    counts_file
        Which file in the passed directory to use as the count file. Typically would be one of:
        '_exprMat_file.csv'.
    meta_file
        Which metadata file in the passed directory to use as the obs file. Typically would be one of:
        '_metadata_file.csv'.
    fov_file
        This file includes the coordinates of all the fields of view.

    Returns
    -------
    Annotated data matrix ``adata``, where observations/cells are named by their
    barcode and variables/genes by gene name. Stores the following information:
    :attr:`~anndata.AnnData.X`
        The data matrix is stored
    :attr:`~anndata.AnnData.obs`
        Table with rows that correspond to the cells, including area, local and global cell coordinates
    :attr:`~anndata.AnnData.obs_names`
        Cell names
    :attr:`~anndata.AnnData.var`
        Gene names
    :attr:`~anndata.AnnData.obsm` `['spatial']`
        Local coordinates of the centers of cells
    :attr:`~anndata.AnnData.obsm` `['spatial_fov']``
        Global coordinates of the centers of cells in the field of view (FOV)
    :attr:`~anndata.AnnData.uns` `['spatial']`
        Dict of Nanostring output files with 'fov_positions', cell composite images with "FOV_<number>" as keys
    :attr:`~anndata.AnnData.uns` `['spatial'][FOV_id]['images']`
        Dict of images
    """
    path, fov = Path(path), "fov"

    counts = pd.read_csv(path / counts_file, header=0, index_col="cell_ID")
    counts.index = counts.index.astype(str).str.cat(counts.pop(fov).astype(str).values, sep="_")

    obs = pd.read_csv(path / meta_file, header=0, index_col="cell_ID")
    obs[fov] = pd.Categorical(obs[fov].astype(str))
    obs.index = obs.index.astype(str).str.cat(obs[fov].values, sep="_")

    adata = AnnData(csr_matrix(counts.values), dtype=counts.values.dtype, uns={Key.uns.spatial: {}})
    adata.obs_names = counts.index
    adata.var_names = counts.columns
    adata.obs = pd.merge(adata.obs, obs, left_index=True, right_index=True, how="left")

    adata.obsm[Key.obsm.spatial] = adata.obs[["CenterX_local_px", "CenterY_local_px"]].values
    adata.obsm["spatial_fov"] = adata.obs[["CenterX_global_px", "CenterY_global_px"]].values
    adata.obs.drop(columns=["CenterX_local_px", "CenterY_local_px"], inplace=True)

    for fov in adata.obs[fov].cat.categories:
        adata.uns[Key.uns.spatial][fov] = {
            "images": {},
            "scalefactors": {"tissue_hires_scalef": 1, "spot_diameter_fullres": 1},
        }

    pat = re.compile(r".*_F(\d+)")
    for subdir in ["CellComposite", "CellLabels"]:
        kind = "hires" if subdir == "CellComposite" else "segmentation"
        for fname in os.listdir(path / subdir):
            fov = str(int(pat.findall(fname)[0]))
            adata.uns[Key.uns.spatial][fov]["images"][kind] = _load_image(path / subdir / fname)

    if fov_file is not None:
        fov_positions = pd.read_csv(path / fov_file, header=0, index_col=fov)
        for fov, row in fov_positions.iterrows():
            adata.uns[Key.uns.spatial][str(fov)] = row.to_dict()

    return adata

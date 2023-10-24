from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import (
    Any,
    Union,  # noqa: F401
)

import numpy as np
import pandas as pd
from anndata import AnnData
from scanpy import logging as logg
from scipy.sparse import csr_matrix

from squidpy._constants._pkg_constants import Key
from squidpy.datasets._utils import PathLike
from squidpy.read._utils import _load_image, _read_counts

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
    Read *10x Genomics* Visium formatted dataset.

    In addition to reading the regular *Visium* output, it looks for the *spatial* directory and loads the images,
    spatial coordinates and scale factors.

    .. seealso::

        - `Space Ranger output <https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/overview>`_.
        - :func:`squidpy.pl.spatial_scatter` on how to plot spatial data.

    Parameters
    ----------
    path
        Path to the root directory containing *Visium* files.
    counts_file
        Which file in the passed directory to use as the count file. Typically either *filtered_feature_bc_matrix.h5* or
        *raw_feature_bc_matrix.h5*.
    library_id
        Identifier for the *Visium* library. Useful when concatenating multiple :class:`anndata.AnnData` objects.
    kwargs
        Keyword arguments for :func:`scanpy.read_10x_h5`, :func:`anndata.read_mtx` or :func:`read_text`.

    Returns
    -------
    Annotated data object with the following keys:

        - :attr:`anndata.AnnData.obsm` ``['spatial']`` - spatial spot coordinates.
        - :attr:`anndata.AnnData.uns` ``['spatial']['{library_id}']['images']`` - *hires* and *lowres* images.
        - :attr:`anndata.AnnData.uns` ``['spatial']['{library_id}']['scalefactors']`` - scale factors for the spots.
        - :attr:`anndata.AnnData.uns` ``['spatial']['{library_id}']['metadata']`` - various metadata.
    """  # noqa: E501
    path = Path(path)
    adata, library_id = _read_counts(path, count_file=counts_file, library_id=library_id, **kwargs)

    if not load_images:
        return adata

    adata.uns[Key.uns.spatial][library_id][Key.uns.image_key] = {
        res: _load_image(path / f"{Key.uns.spatial}/tissue_{res}_image.png") for res in ["hires", "lowres"]
    }
    adata.uns[Key.uns.spatial][library_id]["scalefactors"] = json.loads(
        (path / f"{Key.uns.spatial}/scalefactors_json.json").read_bytes()
    )

    tissue_positions_file = (
        path / "spatial/tissue_positions.csv"
        if (path / "spatial/tissue_positions.csv").exists()
        else path / "spatial/tissue_positions_list.csv"
    )

    coords = pd.read_csv(
        tissue_positions_file,
        header=1 if tissue_positions_file.name == "tissue_positions.csv" else None,
        index_col=0,
    )
    coords.columns = ["in_tissue", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"]
    # https://github.com/scverse/squidpy/issues/657
    coords.set_index(coords.index.astype(adata.obs.index.dtype), inplace=True)

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
    """
    Read *Vizgen* formatted dataset.

    In addition to reading the regular *Vizgen* output, it loads the metadata file and optionally loads
    the transformation matrix.

    .. seealso::

        - `Vizgen data release program <https://vizgen.com/data-release-program/>`_.
        - :func:`squidpy.pl.spatial_scatter` on how to plot spatial data.

    Parameters
    ----------
    path
        Path to the root directory containing *Vizgen* files.
    counts_file
        File containing the counts. Typically ends with *_cell_by_gene.csv*.
    meta_file
        File containing the spatial coordinates and additional cell-level metadata.
    transformation_file
        Transformation matrix file for converting micron coordinates into pixels in images.
    library_id
        Identifier for the *Vizgen* library. Useful when concatenating multiple :class:`anndata.AnnData` objects.

    Returns
    -------
    Annotated data object with the following keys:

        - :attr:`anndata.AnnData.obsm` ``['spatial']`` - spatial spot coordinates in microns.
        - :attr:`anndata.AnnData.obsm` ``['blank_genes']`` - blank genes from Vizgen platform.
        - :attr:`anndata.AnnData.uns` ``['spatial']['{library_id}']['scalefactors']['transformation_matrix']`` -
          transformation matrix for converting micron coordinates to pixels.
          Only present if ``transformation_file != None``.
    """
    path = Path(path)
    adata, library_id = _read_counts(
        path=path, count_file=counts_file, library_id=library_id, delimiter=",", first_column_names=True, **kwargs
    )
    blank_genes = np.array(["Blank" in v for v in adata.var_names])
    adata.obsm["blank_genes"] = pd.DataFrame(
        adata[:, blank_genes].X.copy(), columns=adata.var_names[blank_genes], index=adata.obs_names
    )
    adata = adata[:, ~blank_genes].copy()

    adata.X = csr_matrix(adata.X)

    coords = pd.read_csv(path / meta_file, header=0, index_col=0)
    # https://github.com/scverse/squidpy/issues/657
    coords.set_index(coords.index.astype("str"), inplace=True)

    adata.obs = pd.merge(adata.obs, coords, how="left", left_index=True, right_index=True)
    adata.obsm[Key.obsm.spatial] = adata.obs[["center_x", "center_y"]].values
    adata.obs.drop(columns=["center_x", "center_y"], inplace=True)

    if transformation_file is not None:
        matrix = pd.read_csv(path / f"images/{transformation_file}", sep=" ", header=None)
        # https://github.com/scverse/squidpy/issues/727
        matrix.columns = matrix.columns.astype(str)
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
    Read *Nanostring* formatted dataset.

    In addition to reading the regular *Nanostring* output, it loads the metadata file, if present *CellComposite* and *CellLabels*
    directories containing the images and optionally the field of view file.

    .. seealso::

        - `Nanostring Spatial Molecular Imager <https://nanostring.com/products/cosmx-spatial-molecular-imager/>`_.
        - :func:`squidpy.pl.spatial_scatter` on how to plot spatial data.

    Parameters
    ----------
    path
        Path to the root directory containing *Nanostring* files.
    counts_file
        File containing the counts. Typically ends with *_exprMat_file.csv*.
    meta_file
        File containing the spatial coordinates and additional cell-level metadata.
        Typically ends with *_metadata_file.csv*.
    fov_file
        File containing the coordinates of all the fields of view.

    Returns
    -------
    Annotated data object with the following keys:

        - :attr:`anndata.AnnData.obsm` ``['spatial']`` -  local coordinates of the centers of cells.
        - :attr:`anndata.AnnData.obsm` ``['spatial_fov']`` - global coordinates of the centers of cells in the
          field of view.
        - :attr:`anndata.AnnData.uns` ``['spatial']['{fov}']['images']`` - *hires* and *segmentation* images.
        - :attr:`anndata.AnnData.uns` ``['spatial']['{fov}']['metadata']]['{x,y}_global_px']`` - coordinates of the field of view.
          Only present if ``fov_file != None``.
    """  # noqa: E501
    path, fov_key = Path(path), "fov"
    cell_id_key = "cell_ID"
    counts = pd.read_csv(path / counts_file, header=0, index_col=cell_id_key)
    counts.index = counts.index.astype(str).str.cat(counts.pop(fov_key).astype(str).values, sep="_")

    obs = pd.read_csv(path / meta_file, header=0, index_col=cell_id_key)
    obs[fov_key] = pd.Categorical(obs[fov_key].astype(str))
    obs[cell_id_key] = obs.index.astype(np.int64)
    obs.rename_axis(None, inplace=True)
    obs.index = obs.index.astype(str).str.cat(obs[fov_key].values, sep="_")

    common_index = obs.index.intersection(counts.index)

    adata = AnnData(
        csr_matrix(counts.loc[common_index, :].values),
        dtype=counts.values.dtype,
        obs=obs.loc[common_index, :],
        uns={Key.uns.spatial: {}},
    )
    adata.var_names = counts.columns

    adata.obsm[Key.obsm.spatial] = adata.obs[["CenterX_local_px", "CenterY_local_px"]].values
    adata.obsm["spatial_fov"] = adata.obs[["CenterX_global_px", "CenterY_global_px"]].values
    adata.obs.drop(columns=["CenterX_local_px", "CenterY_local_px"], inplace=True)

    for fov in adata.obs[fov_key].cat.categories:
        adata.uns[Key.uns.spatial][fov] = {
            "images": {},
            "scalefactors": {"tissue_hires_scalef": 1, "spot_diameter_fullres": 1},
        }

    file_extensions = (".jpg", ".png", ".jpeg", ".tif", ".tiff")

    pat = re.compile(r".*_F(\d+)")
    for subdir in ["CellComposite", "CellLabels"]:
        if os.path.exists(path / subdir) and os.path.isdir(path / subdir):
            kind = "hires" if subdir == "CellComposite" else "segmentation"
            for fname in os.listdir(path / subdir):
                if fname.endswith(file_extensions):
                    fov = str(int(pat.findall(fname)[0]))
                    try:
                        adata.uns[Key.uns.spatial][fov]["images"][kind] = _load_image(path / subdir / fname)
                    except KeyError:
                        logg.warning(f"FOV `{str(fov)}` does not exist in {subdir} folder, skipping it.")
                        continue

    if fov_file is not None:
        fov_positions = pd.read_csv(path / fov_file, header=0, index_col=fov_key)
        for fov, row in fov_positions.iterrows():
            try:
                adata.uns[Key.uns.spatial][str(fov)]["metadata"] = row.to_dict()
            except KeyError:
                logg.warning(f"FOV `{str(fov)}` does not exist, skipping it.")
                continue

    return adata

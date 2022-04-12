from __future__ import annotations

from typing import Dict, Sequence
from imageio import imread
from pathlib import Path
import os
import json

from anndata import AnnData

from scipy.sparse import csc_matrix
import pandas as pd

from squidpy._docs import d
from squidpy.read._utils import _read_count, _read_coords, _read_images
from squidpy._constants._pkg_constants import Key


@d.dedent
def read_visium(
    path: str | Path,
    genome: str | None = None,
    *,
    count_file: str = "filtered_feature_bc_matrix.h5",
    library_id: str | None = None,
    load_images: bool | None = True,
    source_image_path: str | Path | None = None,
) -> AnnData:
    r"""
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
    genome
        Filter expression to genes within this genome.
    count_file
        Which file in the passed directory to use as the count file. Typically would be one of:
        'filtered_feature_bc_matrix.h5' or 'raw_feature_bc_matrix.h5'.
    library_id
        Identifier for the Visium library. Can be modified when concatenating multiple adata objects.
    source_image_path
        Path to the high-resolution tissue image. Path will be included in
        `.uns["spatial"][library_id]["metadata"]["source_image_path"]`.

    Returns
    -------
    Annotated data matrix ``adata``, where observations/cells are named by their
    barcode and variables/genes by gene name. Stores the following information:
    :attr:`~anndata.AnnData.X`
        The data matrix is stored
    :attr:`~anndata.AnnData.obs`
        Table with rows that correspond to the spots, including row, column coordinates and pixel coordinates
    :attr:`~anndata.AnnData.obs_names`
        Cell names
    :attr:`~anndata.AnnData.var_names`
        Gene names
    :attr:`~anndata.AnnData.var`\\ `['gene_ids']`
        Gene IDs
    :attr:`~anndata.AnnData.var`\\ `['feature_types']`
        Feature types
    :attr:`~anndata.AnnData.uns`\\ `['spatial']`
        Dict of spaceranger output files with 'library_id' as key
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['images']`
        Dict of images (`'hires'` and `'lowres'`)
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['scalefactors']`
        Scale factors for the spots
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['metadata']`
        Files metadata: 'chemistry_description', 'software_version', 'source_image_path'
    :attr:`~anndata.AnnData.obsm`\\ `['spatial']`
        Spatial spot coordinates, usable as `basis` by :func:`~scanpy.pl.embedding`.
    """
    if isinstance(path, str):
        path = Path(path)

    adata = _read_count(path, count_file, genome)
    library_id = Key.uns.library_id(adata, Key.uns.spatial, library_id)
    if library_id is None:
        raise ValueError(f"Invalid value for `library_id: {library_id}`. Cannot be None.")

    image_files = {
        "hires_image": path / f"{Key.uns.spatial}/tissue_hires_image.png",
        "lowres_image": path / f"{Key.uns.spatial}/tissue_lowres_image.png",
    }
    image_dic = _read_images(image_files)

    adata.uns[Key.uns.spatial][library_id][Key.uns.image_key] = image_dic

    coords_scalefactors_files = {
        "tissue_positions_file": path / f"{Key.uns.spatial}/tissue_positions_list.csv",
        "scalefactors_json_file": path / f"{Key.uns.spatial}/scalefactors_json.json",
    }

    adata.uns[Key.uns.spatial][library_id]["scalefactors"] = json.loads(
        coords_scalefactors_files["scalefactors_json_file"].read_bytes()
    )

    columns = [
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_col_in_fullres",
        "pxl_row_in_fullres",
    ]

    coords = _read_coords(
        coords_scalefactors_files["tissue_positions_file"],
        adata.shape[0],
        columns,
        **{"header": None},
    )

    coords.index = coords["barcode"].values  # type: ignore

    adata.obs = adata.obs.join(coords, how="left")

    adata.obsm[Key.uns.spatial] = adata.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy()
    adata.obs.drop(
        columns=["barcode", "pxl_row_in_fullres", "pxl_col_in_fullres"],
        inplace=True,
    )

    # put image path in uns
    if source_image_path is not None:
        # get an absolute path
        source_image_path = str(Path(source_image_path).resolve())
        adata.uns[Key.uns.spatial][library_id]["metadata"]["source_image_path"] = str(source_image_path)
    return adata


def read_vizgen(
    path: str | Path,
    *,
    count_file: str,
    obs_file: str,
    transformation_file: str,
    library_id: str | None = None,
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
    count_file
        Which file in the passed directory to use as the count file. Typically would be one of:
        '_cell_by_gene.csv'.
    obs_file
        This metadata file has the spatial coordinates of each of the detected cells.
    transformation_file
        Transformation matrix file for converting micron coordinates into pixels in images.
    library_id
        Identifier for the Vizgen library. Can be modified when concatenating multiple adata objects.

    Returns
    -------
    Annotated data matrix ``adata``, where observations/cells are named by their
    barcode and variables/genes by gene name. Stores the following information:
    :attr:`~anndata.AnnData.X`
        The data matrix is stored
    :attr:`~anndata.AnnData.obs`
        Spatial metadata containing the volume and coordinates of center of cells
    :attr:`~anndata.AnnData.obs_names`
        Cell ids
    :attr:`~anndata.AnnData.var_names`
        Gene names
    :attr:`~anndata.AnnData.var`
        Gene IDs
    :attr:`~anndata.AnnData.uns`\\ `['spatial']`
        Dict of Vizgen output files with 'library_id' as key
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['scalefactors']['transformation_matrix']`
        Transformation matrix for converting micron coordinates to pixels
    :attr:`~anndata.AnnData.obsm`\\ `['spatial']`
        Spatial coordinates of center of cells in micron coordinates, usable as `basis` by :func:`~scanpy.pl.embedding`.
    """
    if isinstance(path, str):
        path = Path(path)

    text_kwargs = {
        "first_column_names": True,
        "delimiter": ",",
    }

    adata = _read_count(path=path, count_file=count_file, text_kwargs=text_kwargs)

    transformation_matrix = {
        "transformation_matrix": pd.read_csv(path / f"images/{transformation_file}", sep=" ", header=None)
    }

    if library_id is not None:
        adata.uns[Key.uns.spatial] = {library_id: {"scalefactors": transformation_matrix}}
    else:
        adata.uns[Key.uns.spatial] = {"scalefactors": transformation_matrix}

    coords_file = {
        "metadata_file": path / obs_file,
    }

    columns = [
        "fov",
        "volume",
        "center_x",
        "center_y",
        "min_x",
        "max_x",
        "min_y",
        "max_y",
    ]

    coords = _read_coords(
        coords_file["metadata_file"],
        n_obs=adata.shape[0],
        cols=columns,
        header=0,
        index_col=0,
    )

    adata.obs = adata.obs.join(coords, how="left")

    adata.obsm[Key.uns.spatial] = adata.obs[["center_x", "center_y"]].to_numpy()

    return adata


def read_nanostring(
    path: str | Path,
    *,
    count_file: str,
    obs_file: str,
    fov_file: str,
) -> AnnData:
    r"""
    Read Nanostring formatted dataset.

    In addition to reading the regular Nanostring output,
    loading metadata file to load spatial coordinates,
    this function reads fov_file to load coordinates of fields of view
    and looks for `CellCompsite` folder and loads images.

    Parameters
    ----------
    path
        Path to directory for Nanostring data files.
    count_file
        Which file in the passed directory to use as the count file. Typically would be one of:
        '_exprMat_file.csv'.
    obs_file
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
    :attr:`~anndata.AnnData.obsm`\\ `['spatial']`
        Local coordinates of the centers of cells
    :attr:`~anndata.AnnData.obsm`\\ `['spatial_fov']``
        Global coordinates of the centers of cells in the FOV
    :attr:`~anndata.AnnData.uns`\\ `['spatial']`
        Dict of Nanostring output files with 'fov_positions', cell composite images with "FOV_<number>" as keys
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][FOV_id]['images']`
        Dict of images
    """
    if isinstance(path, str):
        path = Path(path)

    counts = pd.read_csv(path / count_file)
    counts.index = counts.fov.astype(str) + "_" + counts.cell_ID.astype(str)

    obs = pd.read_csv(path / obs_file)
    obs.index = obs.fov.astype(str) + "_" + obs.cell_ID.astype(str)

    fov_positions = {"fov_positions": pd.read_csv(path / fov_file)}

    merged_df = counts.merge(obs, how="left")
    counts.drop(columns=["fov", "cell_ID"], inplace=True)
    obs_columns = obs.columns
    merged_df["fov"] = pd.Categorical(merged_df["fov"].astype(str))

    adata = AnnData(csc_matrix(counts.to_numpy()), obs=merged_df[obs_columns].copy())
    adata.var_names = counts.columns
    adata.obsm[Key.obsm.spatial] = adata.obs[["CenterX_local_px", "CenterY_local_px"]].to_numpy()
    adata.obsm["spatial_fov"] = adata.obs[["CenterX_global_px", "CenterY_global_px"]].to_numpy()

    adata.uns[Key.uns.spatial] = {
        k: {"images": {"hires": None}, "scalefactors": {}} for k in adata.obs.fov.cat.categories
    }
    adata.uns[Key.uns.spatial]["fov"] = fov_positions

    image_files: Sequence[str] = os.listdir(path / "CellComposite")
    images: Dict[str, str] = {str(int(i.strip(".jpg").replace("CellComposite_F", ""))): i for i in image_files}

    for fov, file_name in images.items():
        adata.uns[Key.uns.spatial][fov]["images"]["hires"] = imread(path / "CellComposite" / file_name)

    return adata

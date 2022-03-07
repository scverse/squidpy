from __future__ import annotations

from h5py import File
from types import MappingProxyType
from typing import Any, Mapping, Optional
from pathlib import Path
import json

from scanpy import logging as logg, read_10x_h5
from anndata import AnnData, read_mtx, read_text

import pandas as pd

from matplotlib.image import imread

__all__ = ["read_visium", "read_mtx", "read_text"]

sp = "spatial"


def read_visium(
    path: str | Path,
    genome: Optional[str] = None,
    *,
    count_file: str = "filtered_feature_bc_matrix.h5",
    library_id: Optional[str] = None,
    load_images: Optional[bool] = True,
    source_image_path: Optional[str | Path] = None,
    text_kwargs: Mapping[str, Any] = MappingProxyType({}),
    mtx_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> AnnData:
    r"""
    Read 10x-Genomics-formatted visium dataset.

    In addition to reading regular 10x output,
    this looks for the `spatial` folder and loads images,
    coordinates and scale factors.
    Based on the `Space Ranger output docs`_.
    See :func:`~scanpy.pl.spatial` for a compatible plotting function.
    .. _Space Ranger output docs:
    https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/overview

    Parameters
    ----------
    path
        Path to directory for visium datafiles.
    genome
        Filter expression to genes within this genome.
    count_file
        Which file in the passed directory to use as the count file. Typically would be one of:
        'filtered_feature_bc_matrix.h5' or 'raw_feature_bc_matrix.h5'.
    library_id
        Identifier for the visium library. Can be modified when concatenating multiple adata objects.
    source_image_path
        Path to the high-resolution tissue image. Path will be included in
        `.uns["spatial"][library_id]["metadata"]["source_image_path"]`.

    Returns
    -------
    Annotated data matrix ``adata``, where observations/cells are named by their
    barcode and variables/genes by gene name. Stores the following information:
    :attr:`~anndata.AnnData.X`
        The data matrix is stored
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
    path = Path(path)

    if count_file.endswith(".h5"):
        adata = read_10x_h5(path / count_file, genome=genome)

        adata.uns[sp] = {}

        with File(path / count_file, mode="r") as f:
            attrs = dict(f.attrs)
        if library_id is None:
            library_id = str(attrs.pop("library_ids")[0], "utf-8")

        adata.uns[sp][library_id] = {}

        if load_images:
            files = {
                "tissue_positions_file": path / f"{sp}/tissue_positions_list.csv",
                "scalefactors_json_file": path / f"{sp}/scalefactors_json.json",
                "hires_image": path / f"{sp}/tissue_hires_image.png",
                "lowres_image": path / f"{sp}/tissue_lowres_image.png",
            }

            # check if files exists, continue if images are missing
            for f in files.values():
                if not f.exists():
                    if any(x in str(f) for x in ["hires_image", "lowres_image"]):
                        logg.warning(f"You seem to be missing an image file.\n" f"Could not find '{f}'.")
                    else:
                        raise OSError(f"Could not find '{f}'")

            adata.uns[sp][library_id]["images"] = {}
            for res in ["hires", "lowres"]:
                try:
                    adata.uns[sp][library_id]["images"][res] = imread(str(files[f"{res}_image"]))
                except KeyError:
                    raise KeyError(f"Could not find '{res}_image'")

            # read json scalefactors
            adata.uns[sp][library_id]["scalefactors"] = json.loads(files["scalefactors_json_file"].read_bytes())

            adata.uns[sp][library_id]["metadata"] = {
                k: (str(attrs[k], "utf-8") if isinstance(attrs[k], bytes) else attrs[k])
                for k in ("chemistry_description", "software_version")
                if k in attrs
            }

            # read coordinates
            positions = pd.read_csv(files["tissue_positions_file"], header=None)
            positions.columns = [
                "barcode",
                "in_tissue",
                "array_row",
                "array_col",
                "pxl_col_in_fullres",
                "pxl_row_in_fullres",
            ]
            positions.index = positions["barcode"]

            adata.obs = adata.obs.join(positions, how="left")

            adata.obsm[sp] = adata.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy()
            adata.obs.drop(
                columns=["barcode", "pxl_row_in_fullres", "pxl_col_in_fullres"],
                inplace=True,
            )

            # put image path in uns
            if source_image_path is not None:
                # get an absolute path
                source_image_path = str(Path(source_image_path).resolve())
                adata.uns[sp][library_id]["metadata"]["source_image_path"] = str(source_image_path)
        return adata
    # if the file passed is not a .h5 file, but a text file as .csv or .txt
    elif count_file.endswith((".csv", ".txt")):
        return read_text(count_file, **text_kwargs)
    # if the file passed is not a .h5 file, but a .mtx file
    elif count_file.endswith(".mtx"):
        return read_mtx(count_file, **mtx_kwargs)

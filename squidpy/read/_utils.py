from __future__ import annotations

from h5py import File
from typing import Any, Mapping, Optional, Sequence
from imageio import imread
from pathlib import Path

from scanpy import read_10x_h5
from anndata import AnnData, read_mtx, read_text

import pandas as pd

from squidpy._utils import NDArrayA
from squidpy._constants._pkg_constants import Key


def _read_counts(
    path: str | Path,
    count_file: str,
    library_id: Optional[str] = None,
    **kwargs: Any,
) -> AnnData:
    path = Path(path)
    if count_file.endswith(".h5"):
        adata = read_10x_h5(path / count_file, **kwargs)
        with File(path / count_file, mode="r") as f:
            attrs = dict(f.attrs)
            if library_id is None:
                try:
                    library_id = str(attrs.pop("library_ids")[0], "utf-8")
                except ValueError:
                    raise KeyError(
                        "Unable to extract library id from attributes. Please specify one explicitly."
                    ) from None

            adata.uns[Key.uns.spatial] = {library_id: {"metadata": {}}}  # can overwrite
            for key in ["chemistry_description", "software_version"]:
                if key not in attrs:
                    continue
                metadata = attrs[key].decode("utf-8") if isinstance(attrs[key], bytes) else attrs[key]
                adata.uns[Key.uns.spatial][library_id]["metadata"][key] = metadata

        return adata

    if library_id is None:
        raise ValueError("Please explicitly specify library id.")

    if count_file.endswith((".csv", ".txt")):
        adata = read_text(path / count_file, **kwargs)
    elif count_file.endswith(".mtx"):
        adata = read_mtx(path / count_file, **kwargs)
    else:
        raise NotImplementedError("TODO")

    adata.uns[Key.uns.spatial] = {library_id: {}}  # can overwrite
    return adata


def _read_images(
    image_dic: Mapping[str, Path],
) -> Mapping[str, NDArrayA]:
    """Load images."""
    dic = {}
    for k, f in image_dic.items():
        if not f.exists():
            raise ValueError(f"Could not find: `{f}`")
        dic[k] = imread(f)

    return dic


def _read_coords(
    path: str | Path,
    n_obs: int,
    cols: Sequence[str] | None = None,
    **kwargs: Any,
) -> NDArrayA | pd.DataFrame:
    """Load coordinates."""
    coords = pd.read_csv(path, **kwargs)

    if coords.shape[0] != n_obs:
        raise ValueError(f"Invalid shape of `coordinates` file: `{coords.shape}`.")

    if cols is None:
        return coords.to_numpy()
    else:
        if len(cols) != coords.columns.shape[0]:
            raise ValueError(f"Invalid length for columns: `{cols}`.")
        coords.columns = cols
        return coords

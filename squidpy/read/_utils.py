from __future__ import annotations

from h5py import File
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence
from imageio import imread
from pathlib import Path

from scanpy import read_10x_h5
from anndata import AnnData, read_mtx, read_text

import pandas as pd

from squidpy._utils import NDArrayA
from squidpy._constants._pkg_constants import Key


def _read_count(
    path: str | Path,
    count_file: str,
    genome: Optional[str] = None,
    library_id: Optional[str] = None,
    h5_kwargs: Mapping[str, Any] = MappingProxyType({}),
    text_kwargs: Mapping[str, Any] = MappingProxyType({}),
    mtx_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> AnnData:
    """Load count files."""
    path = Path(path)
    if count_file.endswith(".h5"):
        adata = read_10x_h5(path / count_file, genome=genome, **h5_kwargs)
        with File(path / count_file, mode="r") as f:
            attrs = dict(f.attrs)
            if library_id is None:
                try:
                    library_id = str(attrs.pop("library_ids")[0], "utf-8")
                except ValueError:
                    raise ValueError(f"Invalid value for `library_id: {library_id}`. Cannot be None.")
            adata.uns[Key.uns.spatial] = {library_id: {}}
            metadata_dic = {
                k: (str(attrs[k], "utf-8") if isinstance(attrs[k], bytes) else attrs[k])
                for k in ("chemistry_description", "software_version")
                if k in attrs
            }
            adata.uns[Key.uns.spatial][library_id]["metadata"] = metadata_dic
        return adata
    elif count_file.endswith((".csv", ".txt")):
        return read_text(path / count_file, **text_kwargs)
    elif count_file.endswith(".mtx"):
        return read_mtx(path / count_file, **mtx_kwargs)


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
    coords = pd.read_csv(path, **kwargs)  # nice how you worked with kwargs here

    if coords.shape[0] != n_obs:
        raise ValueError(f"Invalid shape of `coordinates` file: `{coords.shape}`.")

    if cols is None:
        return coords.to_numpy()
    else:
        if len(cols) != coords.columns.shape[0]:
            raise ValueError(f"Invalid length for columns: `{cols}`.")
        coords.columns = cols
        return coords

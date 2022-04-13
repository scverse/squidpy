from __future__ import annotations

from h5py import File
from typing import Any, Mapping, Optional
from imageio import imread
from pathlib import Path

from scanpy import read_10x_h5
from anndata import AnnData, read_mtx, read_text

from squidpy._utils import NDArrayA
from squidpy._constants._pkg_constants import Key


def _read_counts(
    path: str | Path,
    count_file: str,
    library_id: Optional[str] = None,
    **kwargs: Any,
) -> tuple[AnnData, str]:
    path = Path(path)
    if count_file.endswith(".h5"):
        adata: AnnData = read_10x_h5(path / count_file, **kwargs)
        with File(path / count_file, mode="r") as f:
            attrs = dict(f.attrs)
            if library_id is None:
                try:
                    lid = attrs.pop("library_ids")[0]
                    library_id = lid.decode("utf-8") if isinstance(lid, bytes) else str(lid)
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

        return adata, library_id

    if library_id is None:
        raise ValueError("Please explicitly specify library id.")

    if count_file.endswith((".csv", ".txt")):
        adata = read_text(path / count_file, **kwargs)
    elif count_file.endswith(".mtx"):
        adata = read_mtx(path / count_file, **kwargs)
    else:
        raise NotImplementedError("TODO")

    adata.uns[Key.uns.spatial] = {library_id: {"metadata": {}}}  # can overwrite
    return adata, library_id


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

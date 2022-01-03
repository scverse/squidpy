from __future__ import annotations

from typing import Any, Tuple, Mapping, Optional, Sequence

from anndata import AnnData

import numpy as np

from squidpy._utils import NDArrayA
from squidpy._constants._pkg_constants import Key

# def spatial(
#     adata: AnnData,
#     spatial_key: str = Key.obsm.spatial,
#     color: Union[Sequence, str] = None,
#     batch: str = None,
#     cmap: str = "viridis",
#     legend_kwargs: Mapping = None,
#     label_kwargs: Mapping = None,
# ):
#     """Spatial plotting for squidpy."""

#     _sanitize_anndata(adata)

#     return


def _get_spatial_attrs(
    adata: AnnData,
    spatial_key: str = Key.obsm.spatial,
    library_id: Optional[Sequence[str] | None] = None,
    img: Optional[Sequence[NDArrayA] | None] = None,
    img_key: str | None = None,
    scale_factor: Optional[Sequence[float] | None] = None,
    bw: bool = False,
) -> Tuple[Sequence[str], Sequence[float], Sequence[NDArrayA]]:
    """Return lists of image attributes saved in adata for plotting."""
    library_id = Key.uns.library_id(adata, spatial_key, library_id, return_all=True)
    if library_id is None:
        raise ValueError(f"Could not fetch `library_id`, check that `spatial_key={spatial_key}` is correct.")

    image_mapping = Key.uns.library_mapping(adata, spatial_key, Key.uns.image_key, library_id)
    scalefactor_mapping = Key.uns.library_mapping(adata, spatial_key, Key.uns.scalefactor_key, library_id)

    if not (image_mapping.keys() == scalefactor_mapping.keys()):  # check that keys match
        raise KeyError(
            f"Image keys: `{image_mapping.keys()}` and scalefactor keys: `{scalefactor_mapping.keys()}` are not equal."
        )

    if img_key is None:
        img_key = _get_unique_map(image_mapping)  # get intersection of image_mapping.values()
        img_key = img_key[0]  # get first of set
    else:
        if img_key not in image_mapping.values():
            raise ValueError(f"Image key: `{img_key}` does not exist. Available image keys: `{image_mapping.values()}`")

    if scale_factor is None:  # get intersection of scale_factor and match to img_key
        scale_factor_key = _get_unique_map(scalefactor_mapping)
        scale_factor_key = [i for i in scale_factor_key if img_key in i][0]
        if len(scale_factor_key) == 0:
            raise ValueError(f"No `scale_factor` found that could match `img_key`: {img_key}.")
        scale_factor = [adata.uns[Key.uns.spatial][i][Key.uns.scalefactor_key][scale_factor_key] for i in library_id]
    else:
        if len(scale_factor) != len(library_id):
            raise ValueError(
                f"Len of scale_factor list: {len(scale_factor)} is not equal to len of library_id: {len(library_id)}."
            )

    if img is None:
        img = [adata.uns[Key.uns.spatial][i][Key.uns.image_key][img_key] for i in library_id]
    else:
        if len(img) != len(library_id):
            raise ValueError(f"Len of img list: {len(img)} is not equal to len of library_id: {len(library_id)}.")

    if bw:
        img = [np.dot(im[..., :3], [0.2989, 0.5870, 0.1140]) for im in img]

    return library_id, scale_factor, img


def _get_unique_map(dic: Mapping[str, Any]) -> Any:
    """Get intersection of dict values."""
    return sorted(set.intersection(*map(set, dic.values())))

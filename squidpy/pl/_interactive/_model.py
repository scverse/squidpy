from typing import Optional
from dataclasses import field, dataclass

from anndata import AnnData

import numpy as np

from squidpy.im import ImageContainer  # type: ignore[attr-defined]
from squidpy.im._utils import CropCoords
from squidpy.pl._utils import ALayer
from squidpy._constants._pkg_constants import Key


@dataclass
class ImageModel:
    """Model which holds the data for _interactive visualization."""

    adata: AnnData
    container: ImageContainer
    spatial_key: str = field(default=Key.obsm.spatial, repr=False)
    library_id: Optional[str] = None
    spot_diameter: float = field(default=0, init=False)
    coordinates: np.ndarray = field(init=False, repr=False)
    alayer: ALayer = field(init=False, repr=True)

    palette: Optional[str] = field(default=None, repr=False)
    cmap: str = field(default="viridis", repr=False)
    blending: str = field(default="opaque", repr=False)
    key_added: str = "shapes"

    def __post_init__(self) -> None:
        self.coordinates = self.adata.obsm[self.spatial_key][:, ::-1]

        if self.container.data.attrs.get("crop", None) is not None:
            c: CropCoords = self.container.data.attrs["crop"]

            mask = (
                (self.coordinates[:, 0] >= c.x0)
                & (self.coordinates[:, 0] <= c.x1)
                & (self.coordinates[:, 1] >= c.y0)
                & (self.coordinates[:, 1] <= c.y1)
            )

            self.adata = self.adata[mask, :].copy()
            self.coordinates = self.adata.obsm[self.spatial_key][:, ::-1]
            # shift appropriately
            self.coordinates[:, 0] -= c.x0
            self.coordinates[:, 1] -= c.y0

        self.alayer = ALayer(self.adata, is_raw=False, palette=self.palette)

        if self.library_id is None:
            haystack = list(self.adata.uns[self.spatial_key].keys())
            if not len(haystack):
                raise ValueError()
            if len(haystack) > 1:
                raise ValueError()

            self.library_id = haystack[0]

        try:
            self.spot_diameter = float(
                self.adata.uns[self.spatial_key][self.library_id]["scalefactors"]["spot_diameter_fullres"]
            )
        except KeyError:
            raise KeyError(
                f"Unable to get the spot diameter from "
                f"`adata.uns[{self.spatial_key!r}][{self.library_id!r}]['scalefactors'['spot_diameter_fullres']]`"
            ) from None

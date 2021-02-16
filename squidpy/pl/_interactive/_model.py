from typing import Optional
from dataclasses import field, dataclass

from anndata import AnnData

import numpy as np

from squidpy.im import ImageContainer  # type: ignore[attr-defined]
from squidpy.im._utils import CropCoords, _NULL_COORDS
from squidpy.pl._utils import ALayer
from squidpy._constants._constants import Symbol
from squidpy._constants._pkg_constants import Key

__all__ = ["ImageModel"]


@dataclass
class ImageModel:
    """Model which holds the data for interactive visualization."""

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
    symbol: Symbol = Symbol.DISC

    def __post_init__(self) -> None:
        self.symbol = Symbol(self.symbol)
        self.coordinates = self.adata.obsm[self.spatial_key][:, ::-1]

        if self.container.data.attrs.get("coords", _NULL_COORDS) != _NULL_COORDS:
            c: CropCoords = self.container.data.attrs["coords"]

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
        self.library_id = Key.uns.library_id(self.adata, self.spatial_key, self.library_id)
        self.spot_diameter = Key.uns.spot_diameter(self.adata, self.spatial_key, self.library_id)

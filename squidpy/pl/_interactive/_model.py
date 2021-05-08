from typing import Optional
from dataclasses import field, dataclass

from anndata import AnnData

import numpy as np

from squidpy.im import ImageContainer  # type: ignore[attr-defined]
from squidpy.pl._utils import ALayer
from squidpy.im._coords import CropCoords, CropPadding, _NULL_COORDS, _NULL_PADDING
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
        self.library_id = Key.uns.library_id(self.adata, self.spatial_key, self.library_id)
        self.spot_diameter = Key.uns.spot_diameter(self.adata, self.spatial_key, self.library_id)

        s = self.container.data.attrs.get(Key.img.scale, 1)
        if s != 1:
            # update coordinates with image scale
            self.coordinates = self.coordinates * s
            self.spot_diameter *= s

        c: CropCoords = self.container.data.attrs.get(Key.img.coords, _NULL_COORDS)
        p: CropPadding = self.container.data.attrs.get(Key.img.padding, _NULL_PADDING)
        if c != _NULL_COORDS:
            mask = (
                (self.coordinates[:, 0] >= c.y0)
                & (self.coordinates[:, 0] <= c.y1)
                & (self.coordinates[:, 1] >= c.x0)
                & (self.coordinates[:, 1] <= c.x1)
            )

            self.adata = self.adata[mask, :].copy()
            self.coordinates = self.coordinates[mask]
            # shift appropriately
            self.coordinates[:, 0] -= c.y0 - p.y_pre
            self.coordinates[:, 1] -= c.x0 - p.x_pre

        if not self.adata.n_obs:
            raise ValueError("No spots were selected. Please ensure that the image contains at least 1 spot.")

        self.alayer = ALayer(self.adata, is_raw=False, palette=self.palette)

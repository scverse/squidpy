from typing import Optional
from dataclasses import field, dataclass

from anndata import AnnData

import numpy as np

from squidpy.im import ImageContainer  # type: ignore[attr-defined]
from squidpy.gr._utils import _assert_spatial_basis
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
        _assert_spatial_basis(self.adata, self.spatial_key)

        self.symbol = Symbol(self.symbol)
        self.library_id = Key.uns.library_id(self.adata, self.spatial_key, self.library_id)
        self.spot_diameter = Key.uns.spot_diameter(
            self.adata, self.spatial_key, self.library_id
        ) * self.container.data.attrs.get(Key.img.scale, 1)

        self.adata = self.container._subset(self.adata, spatial_key=self.spatial_key, adjust_interactive=True)
        if not self.adata.n_obs:
            raise ValueError("No spots were selected. Please ensure that the image contains at least 1 spot.")

        self.coordinates = self.adata.obsm[self.spatial_key][:, ::-1][:, :2]
        self.alayer = ALayer(self.adata, is_raw=False, palette=self.palette)

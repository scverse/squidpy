from typing import Optional
from dataclasses import field, dataclass

from anndata import AnnData

import numpy as np

from squidpy.im import ImageContainer  # type: ignore[attr-defined]
from squidpy.pl._utils import ALayer
from squidpy.constants._pkg_constants import Key


@dataclass
class ImageModel:
    """Model which holds the data for interactive visualization."""

    adata: AnnData
    container: ImageContainer
    spatial_key: str = field(default=Key.obsm.spatial, repr=False)
    spot_diameter: float = field(default=0, init=False)
    coordinates: np.ndarray = field(init=False, repr=False)
    alayer: ALayer = field(init=False, repr=True)

    cat_cmap: Optional[str] = field(default=None, repr=False)
    cont_cmap: str = field(default="viridis", repr=False)
    blending: str = field(default="opaque", repr=False)
    key_added: Optional[str] = None

    def __post_init__(self) -> None:
        self.alayer = ALayer(self.adata, is_raw=False, palette=self.cat_cmap)
        self.coordinates = self.adata.obsm[self.spatial_key][:, ::-1]

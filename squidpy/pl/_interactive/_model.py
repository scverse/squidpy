from __future__ import annotations

from typing import Sequence, TYPE_CHECKING
from dataclasses import field, dataclass

from anndata import AnnData

import numpy as np

from squidpy.im import ImageContainer  # type: ignore[attr-defined]
from squidpy._utils import NDArrayA, _unique_order_preserving
from squidpy.gr._utils import _assert_spatial_basis, _assert_categorical_obs
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
    library_key: str | None = None
    library_id: str | Sequence[str] | None = None
    spot_diameter: NDArrayA | float = field(default=0, init=False)
    coordinates: NDArrayA = field(init=False, repr=False)
    alayer: ALayer = field(init=False, repr=True)

    palette: str | None = field(default=None, repr=False)
    cmap: str = field(default="viridis", repr=False)
    blending: str = field(default="opaque", repr=False)
    key_added: str = "shapes"
    symbol: Symbol = Symbol.DISC

    def __post_init__(self) -> None:
        _assert_spatial_basis(self.adata, self.spatial_key)

        self.symbol = Symbol(self.symbol)
        self.adata = self.container._subset(self.adata, spatial_key=self.spatial_key, adjust_interactive=True)
        if not self.adata.n_obs:
            raise ValueError("No spots were selected. Please ensure that the image contains at least 1 spot.")
        self.coordinates = self.adata.obsm[self.spatial_key][:, ::-1][:, :2].copy()
        self.scale = self.container.data.attrs.get(Key.img.scale, 1)
        self._update_coords()

        if TYPE_CHECKING:
            assert isinstance(self.library_id, Sequence)

        self.alayer = ALayer(
            self.adata,
            self.library_id,
            is_raw=False,
            palette=self.palette,
        )

        try:
            self.container = ImageContainer._from_dataset(self.container.data.sel(z=self.library_id), deep=None)
        except KeyError:
            raise KeyError(
                f"Unable to subset the image container with library ids `{self.library_id}`. "
                f"Valid library ids are `{self.container.library_ids}`."
            ) from None

    def _update_coords(self) -> None:
        if self.library_key is None:
            if len(self.container.library_ids) > 1:
                raise KeyError(
                    f"ImageContainer has `{len(self.container.library_ids)}` Z-dimensions. "
                    f"Please specify `library_key` that maps observations to library ids."
                )
            self.coordinates = np.insert(self.coordinates, 0, values=0, axis=1)
            self.library_id = self.container.library_ids
            if TYPE_CHECKING:
                assert isinstance(self.library_id, Sequence)
            self.spot_diameter = Key.uns.spot_diameter(self.adata, self.spatial_key, self.library_id[0]) * self.scale
            return

        _assert_categorical_obs(self.adata, self.library_key)
        if self.library_id is None:
            self.library_id = self.adata.obs[self.library_key].cat.categories
        elif isinstance(self.library_id, str):
            self.library_id = [self.library_id]
        self.library_id, _ = _unique_order_preserving(self.library_id)  # type: ignore[assignment]

        if not len(self.library_id):
            raise ValueError("No library ids have been selected.")
        # invalid library ids from adata are filtered below
        # invalid library ids from container raise KeyError in __post_init__

        libraries = self.adata.obs[self.library_key]
        mask = libraries.isin(self.library_id)
        libraries = libraries[mask]
        self.library_id = list(libraries.cat.categories)

        self.coordinates = np.c_[libraries.cat.codes.values, self.coordinates[mask]]
        self.spot_diameter = np.array(
            [
                np.array([0.0] + [Key.uns.spot_diameter(self.adata, self.spatial_key, lid)] * 2) * self.scale
                for lid in libraries
            ]
        )

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import numpy as np
from anndata import AnnData

from squidpy._constants._constants import Symbol
from squidpy._constants._pkg_constants import Key
from squidpy._utils import NDArrayA, _unique_order_preserving
from squidpy.gr._utils import _assert_categorical_obs, _assert_spatial_basis
from squidpy.im import ImageContainer  # type: ignore[attr-defined]
from squidpy.im._coords import _NULL_COORDS, _NULL_PADDING, CropCoords, CropPadding
from squidpy.pl._utils import ALayer

__all__ = ["ImageModel"]


@dataclass
class ImageModel:
    """Model which holds the data for interactive visualization."""

    adata: AnnData
    container: ImageContainer
    spatial_key: str = field(default=Key.obsm.spatial, repr=False)
    library_key: str | None = None
    library_id: str | Sequence[str] | None = None
    spot_diameter_key: str = "spot_diameter_fullres"
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
        self.adata = self.container.subset(self.adata, spatial_key=self.spatial_key)
        if not self.adata.n_obs:
            raise ValueError("Please ensure that the image contains at least 1 spot.")
        self._set_scale_coords()
        self._set_library()

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
                f"Valid container library ids are `{self.container.library_ids}`. Please specify a valid `library_id`."
            ) from None

    def _set_scale_coords(self) -> None:
        self.scale = self.container.data.attrs.get(Key.img.scale, 1)
        coordinates = self.adata.obsm[self.spatial_key][:, :2] * self.scale

        c: CropCoords = self.container.data.attrs.get(Key.img.coords, _NULL_COORDS)
        p: CropPadding = self.container.data.attrs.get(Key.img.padding, _NULL_PADDING)
        if c != _NULL_COORDS:
            coordinates -= c.x0 - p.x_pre
            coordinates -= c.y0 - p.y_pre

        self.coordinates = coordinates[:, ::-1]

    def _set_library(self) -> None:
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
            self.spot_diameter = (
                Key.uns.spot_diameter(self.adata, self.spatial_key, self.library_id[0], self.spot_diameter_key)
                * self.scale
            )
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
        # invalid library ids from container raise KeyError in `__post_init__` after this call

        libraries = self.adata.obs[self.library_key]
        mask = libraries.isin(self.library_id)
        libraries = libraries[mask].cat.remove_unused_categories()
        self.library_id = list(libraries.cat.categories)

        self.coordinates = np.c_[libraries.cat.codes.values, self.coordinates[mask]]
        self.spot_diameter = np.array(
            [
                np.array([0.0] + [Key.uns.spot_diameter(self.adata, self.spatial_key, lid, self.spot_diameter_key)] * 2)
                * self.scale
                for lid in libraries
            ]
        )
        self.adata = self.adata[mask]

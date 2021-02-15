from abc import ABC, abstractmethod
from typing import Any, Tuple
from dataclasses import dataclass
import warnings

import numpy as np
import xarray as xr

import tifffile

from squidpy.gr._utils import _assert_non_negative


def _circular_mask(arr: np.ndarray, y: int, x: int, radius: float) -> np.ndarray:
    Y, X = np.ogrid[: arr.shape[0], : arr.shape[1]]
    return np.asarray(((Y - y) ** 2 + (X - x) ** 2) <= radius ** 2)


def _num_pages(fname: str) -> int:
    """Use tifffile to get the number of pages in the tif."""
    with tifffile.TiffFile(fname) as img:
        num_pages = len(img.pages)
    return num_pages


def _open_rasterio(path: str, **kwargs: Any) -> xr.DataArray:
    from rasterio.errors import NotGeoreferencedWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        return xr.open_rasterio(path, **kwargs)


class TupleSerializer(ABC):  # noqa: D101
    @abstractmethod
    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Return self as a :class:`tuple`."""

    @classmethod
    def from_tuple(cls, value: Tuple[float, float, float, float]) -> "TupleSerializer":
        """Create self from a :class:`tuple`."""
        return cls(*value)  # type: ignore[call-arg]


@dataclass(frozen=True)
class CropCoords(TupleSerializer):
    """Top-left and bottom right-corners of a crop."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.x0 > self.x1:
            raise ValueError(f"Expected `x0` <= `x1`, found `{self.x0}` > `{self.x1}`.")
        if self.y0 > self.y1:
            raise ValueError(f"Expected `y0` <= `y1`, found `{self.y0}` > `{self.y1}`.")

    @property
    def dx(self) -> float:
        """Width."""
        return self.x1 - self.x0

    @property
    def dy(self) -> float:
        """Height."""
        return self.y1 - self.y0

    @property
    def center_x(self) -> float:
        """Center of height."""
        return self.x0 + self.dx / 2.0

    @property
    def center_y(self) -> float:
        """Width of height."""
        return self.x0 + self.dy / 2.0

    def to_image_coordinates(self, padding: "CropPadding") -> "CropCoords":
        """
        Convert global image coordinates to local.

        Parameters
        ----------
        padding
            Padding for which to adjust.

        Returns
        -------
        Padding-adjusted image coordinates.
        """
        adj = self + padding
        return CropCoords(x0=padding.x_pre, y0=padding.y_pre, x1=adj.dx - padding.x_post, y1=adj.dy - padding.y_post)

    @property
    def slice(self) -> Tuple[slice, slice]:  # noqa: A003
        """Return the ``(height, width)`` slice."""
        return slice(self.y0, self.y1), slice(self.x0, self.x1)

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Return self as a :class:`tuple`."""
        return self.x0, self.y0, self.x1, self.y1

    def __add__(self, other: "CropPadding") -> "CropCoords":
        if not isinstance(other, CropPadding):
            return NotImplemented  # type: ignore[unreachable]

        return CropCoords(
            x0=self.x0 - other.x_pre, y0=self.y0 - other.y_pre, x1=self.x1 + other.x_post, y1=self.y1 + other.y_post
        )

    def __sub__(self, other: "CropCoords") -> "CropPadding":
        if not isinstance(other, CropCoords):
            return NotImplemented  # type: ignore[unreachable]

        return CropPadding(
            x_pre=abs(self.x0 - other.x0),
            y_pre=abs(self.y0 - other.y0),
            x_post=abs(self.x1 - other.x1),
            y_post=abs(self.y1 - other.y1),
        )


@dataclass(frozen=True)
class CropPadding(TupleSerializer):
    """Padding of a crop."""

    x_pre: float
    y_pre: float
    x_post: float
    y_post: float

    def __post_init__(self) -> None:
        _assert_non_negative(self.x_pre, name="x_pre")
        _assert_non_negative(self.y_pre, name="y_pre")
        _assert_non_negative(self.x_post, name="x_post")
        _assert_non_negative(self.y_post, name="y_post")

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Return self as a :class:`tuple`."""
        return self.x_pre, self.x_post, self.y_pre, self.y_post


_NULL_COORDS = CropCoords(0, 0, 0, 0)
_NULL_PADDING = CropPadding(0, 0, 0, 0)

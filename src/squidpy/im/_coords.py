from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Hashable,
    Union,  # noqa: F401
)

import numpy as np

from squidpy._constants._pkg_constants import Key
from squidpy._utils import NDArrayA
from squidpy.gr._utils import _assert_non_negative


def _circular_mask(arr: NDArrayA, y: int, x: int, radius: float) -> NDArrayA:
    Y, X = np.ogrid[: arr.shape[0], : arr.shape[1]]
    return np.asarray(((Y - y) ** 2 + (X - x) ** 2) <= radius**2)


class TupleSerializer(ABC):  # noqa: D101
    @abstractmethod
    def to_tuple(self) -> tuple[float, float, float, float]:
        """Return self as a :class:`tuple`."""

    @classmethod
    def from_tuple(cls, value: tuple[float, float, float, float]) -> TupleSerializer:
        """Create self from a :class:`tuple`."""
        return cls(*value)  # type: ignore[call-arg]

    @property
    @abstractmethod
    def T(self) -> TupleSerializer:
        """Transpose self."""  # currently unused

    def __mul__(self, other: int | float) -> TupleSerializer:
        if not isinstance(other, (int, float)):
            return NotImplemented

        a, b, c, d = self.to_tuple()
        res = type(self)(a * other, b * other, c * other, d * other)  # type: ignore[call-arg]
        return res

    def __rmul__(self, other: int | float) -> TupleSerializer:
        return self * other


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
    def T(self) -> CropCoords:
        """Transpose self."""
        return CropCoords(x0=self.y0, y0=self.x0, x1=self.y1, y1=self.x1)

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

    def to_image_coordinates(self, padding: CropPadding) -> CropCoords:
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
    def slice(self) -> tuple[slice, slice]:  # noqa: A003
        """Return the ``(height, width)`` int slice."""
        # has to convert to int, because of scaling, coords can also be floats
        return slice(int(self.y0), int(self.y1)), slice(int(self.x0), int(self.x1))

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Return self as a :class:`tuple`."""
        return self.x0, self.y0, self.x1, self.y1

    def __add__(self, other: CropPadding) -> CropCoords:
        if not isinstance(other, CropPadding):
            return NotImplemented

        return CropCoords(
            x0=self.x0 - other.x_pre, y0=self.y0 - other.y_pre, x1=self.x1 + other.x_post, y1=self.y1 + other.y_post
        )

    def __sub__(self, other: CropCoords) -> CropPadding:
        if not isinstance(other, CropCoords):
            return NotImplemented

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
    x_post: float
    y_pre: float
    y_post: float

    def __post_init__(self) -> None:
        _assert_non_negative(self.x_pre, name="x_pre")
        _assert_non_negative(self.y_pre, name="y_pre")
        _assert_non_negative(self.x_post, name="x_post")
        _assert_non_negative(self.y_post, name="y_post")

    @property
    def T(self) -> CropPadding:
        """Transpose self."""
        return CropPadding(x_pre=self.y_pre, y_pre=self.x_pre, x_post=self.y_post, y_post=self.x_post)

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Return self as a :class:`tuple`."""
        return self.x_pre, self.x_post, self.y_pre, self.y_post


_NULL_COORDS = CropCoords(0, 0, 0, 0)
_NULL_PADDING = CropPadding(0, 0, 0, 0)


# functions for updating attributes with new scaling, CropCoords, CropPadding
def _update_attrs_coords(attrs: dict[Hashable, Any], coords: CropCoords) -> dict[Hashable, Any]:
    old_coords = attrs.get(Key.img.coords, _NULL_COORDS)
    if old_coords != _NULL_COORDS:
        new_coords = CropCoords(
            x0=old_coords.x0 + coords.x0,
            y0=old_coords.y0 + coords.y0,
            x1=old_coords.x0 + coords.x1,
            y1=old_coords.y0 + coords.y1,
        )
        attrs[Key.img.coords] = new_coords
    else:
        attrs[Key.img.coords] = coords
    return attrs


def _update_attrs_scale(attrs: dict[Hashable, Any], scale: int | float) -> dict[Hashable, Any]:
    old_scale = attrs[Key.img.scale]
    attrs[Key.img.scale] = old_scale * scale
    attrs[Key.img.padding] = attrs[Key.img.padding] * scale
    attrs[Key.img.coords] = attrs[Key.img.coords] * scale
    return attrs

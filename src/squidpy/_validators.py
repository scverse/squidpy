"""Generic validation utilities for squidpy."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

from squidpy._utils import _unique_order_preserving

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData


def check_tuple_needles(
    needles: Sequence[tuple[Any, Any]],
    haystack: Sequence[Any],
    msg: str,
    reraise: bool = True,
) -> Sequence[tuple[Any, Any]]:
    filtered = []

    for needle in needles:
        if not isinstance(needle, Sequence):
            raise TypeError(f"Expected a `Sequence`, found `{type(needle).__name__}`.")
        if len(needle) != 2:
            raise ValueError(f"Expected a `tuple` of length `2`, found `{len(needle)}`.")
        a, b = needle

        if a not in haystack:
            if reraise:
                raise ValueError(msg.format(a))
            else:
                continue
        if b not in haystack:
            if reraise:
                raise ValueError(msg.format(b))
            else:
                continue

        filtered.append((a, b))

    return filtered


def assert_non_empty_sequence(
    seq: Hashable | Iterable[Hashable], *, name: str, convert_scalar: bool = True
) -> list[Any]:
    if isinstance(seq, str) or not isinstance(seq, Iterable):
        if not convert_scalar:
            raise TypeError(f"Expected a sequence, found `{type(seq)}`.")
        seq = (seq,)

    res, _ = _unique_order_preserving(seq)
    if not len(res):
        raise ValueError(f"No {name} have been selected.")

    return res


def get_valid_values(needle: Sequence[Any], haystack: Sequence[Any]) -> Sequence[Any]:
    res = [n for n in needle if n in haystack]
    if not len(res):
        raise ValueError(f"No valid values were found. Valid values are `{sorted(set(haystack))}`.")
    return res


def assert_positive(value: float, *, name: str) -> None:
    if value <= 0:
        raise ValueError(f"Expected `{name}` to be positive, found `{value}`.")


def assert_non_negative(value: float, *, name: str) -> None:
    if value < 0:
        raise ValueError(f"Expected `{name}` to be non-negative, found `{value}`.")


def assert_in_range(value: float, minn: float, maxx: float, *, name: str) -> None:
    if not (minn <= value <= maxx):
        raise ValueError(f"Expected `{name}` to be in interval `[{minn}, {maxx}]`, found `{value}`.")


def assert_isinstance(value: Any, expected_type: type | tuple[type, ...], *, name: str) -> None:
    """Raise TypeError if *value* is not an instance of *expected_type*."""
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            type_names = " or ".join(t.__name__ for t in expected_type)
        else:
            type_names = expected_type.__name__
        raise TypeError(f"Expected `{name}` to be of type `{type_names}`, got `{type(value).__name__}`.")


def assert_one_of(value: Any, options: Sequence[Any], *, name: str) -> None:
    """Raise ValueError if *value* is not in *options*."""
    if value not in options:
        raise ValueError(f"Expected `{name}` to be one of `{list(options)}`, got `{value!r}`.")


def assert_key_in(obj: Any, key: str, *, attr: str, obj_name: str, extra_msg: str = "") -> None:
    """Raise KeyError if *key* not in ``getattr(obj, attr)``."""
    container = getattr(obj, attr)
    if key not in container:
        available = list(container.keys()) if hasattr(container, "keys") else list(container)
        msg = f"Key `{key!r}` not found in `{obj_name}.{attr}`. Available keys: {available}."
        if extra_msg:
            msg = f"{msg} {extra_msg}"
        raise KeyError(msg)


def assert_key_in_adata(adata: AnnData, key: str, *, attr: str, extra_msg: str = "") -> None:
    """Raise KeyError if *key* not in ``getattr(adata, attr)``."""
    assert_key_in(adata, key, attr=attr, obj_name="adata", extra_msg=extra_msg)


def assert_key_in_sdata(sdata: SpatialData, key: str, *, attr: str, extra_msg: str = "") -> None:
    """Raise KeyError if *key* not in ``getattr(sdata, attr)``."""
    assert_key_in(sdata, key, attr=attr, obj_name="sdata", extra_msg=extra_msg)

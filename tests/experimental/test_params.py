from __future__ import annotations

from types import UnionType
from typing import Annotated, Any, TypedDict, Union, get_args, get_origin, get_type_hints

import pytest

from squidpy.experimental import types
from squidpy.experimental.utils._params import Default, defaults_of

#: Only the `*Params` types carry per-key defaults; the result types do not.
PARAMS_TYPES = [name for name in types.__all__ if name.endswith("Params")]


def _matches(value: object, hint: Any) -> bool:
    """Whether ``value`` satisfies ``hint``.

    A stand-in for the static check a type checker cannot do: nothing relates
    ``Annotated`` metadata to the type it annotates, so the defaults are only
    verifiable at runtime. Follows the PEP 484 numeric tower (an ``int`` default
    satisfies a ``float`` key) and keeps ``bool`` distinct from ``int``.
    """
    if get_origin(hint) in (Union, UnionType):
        return any(_matches(value, arg) for arg in get_args(hint))
    origin = get_origin(hint) or hint
    if origin is bool:
        return isinstance(value, bool)
    if origin is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if origin is float:
        return isinstance(value, int | float) and not isinstance(value, bool)
    return isinstance(value, origin)


class TestDefaultsOf:
    @pytest.mark.parametrize("name", PARAMS_TYPES)
    def test_every_key_has_a_default(self, name: str) -> None:
        cls = getattr(types, name)
        assert set(defaults_of(cls)) == set(cls.__annotations__)

    @pytest.mark.parametrize("name", PARAMS_TYPES)
    def test_every_default_matches_its_declared_type(self, name: str) -> None:
        cls = getattr(types, name)
        for key, hint in get_type_hints(cls, include_extras=True).items():
            (marker,) = (m for m in hint.__metadata__ if isinstance(m, Default))
            assert _matches(marker.value, hint.__origin__), f"{name}.{key} = {marker.value!r} is not {hint.__origin__}"

    def test_matches_rejects_the_wrong_type(self) -> None:
        # guards the guard: a `_matches` that waved everything through would make
        # the test above vacuous
        assert not _matches("0.5", float)
        assert not _matches(1, bool)
        assert not _matches(0.5, int)
        assert not _matches(None, int | str)

    def test_missing_default_raises(self) -> None:
        class Incomplete(TypedDict, total=False):
            a: Annotated[int, Default(1)]
            b: float

        with pytest.raises(TypeError, match="Incomplete.b` is missing a `Default"):
            defaults_of(Incomplete)

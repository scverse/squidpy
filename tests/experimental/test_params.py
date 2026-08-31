from __future__ import annotations

from types import UnionType
from typing import Annotated, Any, TypedDict, Union, get_args, get_origin, get_type_hints

import pytest

from squidpy import types
from squidpy.experimental.utils._params import Default, defaults_of, resolve_params

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
        # A member may not be `isinstance`-checkable at all -- `npt.ArrayLike` unions in
        # non-runtime-checkable protocols -- so an unverifiable arm is skipped rather than
        # failing the whole union.
        return any(_safe_matches(value, arg) for arg in get_args(hint))
    origin = get_origin(hint) or hint
    if origin is bool:
        return isinstance(value, bool)
    if origin is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if origin is float:
        return isinstance(value, int | float) and not isinstance(value, bool)
    return isinstance(value, origin)


def _safe_matches(value: object, hint: Any) -> bool:
    """`_matches`, treating an un-checkable hint as "not this arm" instead of an error."""
    try:
        return _matches(value, hint)
    except TypeError:
        return False


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


@pytest.mark.parametrize("name", PARAMS_TYPES)
class TestResolveContract:
    """`resolve_params` behaves the same for every params type.

    The per-module test files cover what differs -- each validator's ranges,
    coercions and `arg_name` -- and leave the shared contract to these.
    """

    @staticmethod
    def _defaults(name: str) -> dict[str, Any]:
        return dict(defaults_of(getattr(types, name)))

    def test_none_returns_defaults(self, name: str) -> None:
        assert resolve_params(None, defaults=self._defaults(name)) == self._defaults(name)

    def test_partial_fills_the_rest(self, name: str) -> None:
        defaults = self._defaults(name)
        first, *rest = defaults
        resolved = resolve_params({first: defaults[first]}, defaults=defaults)
        assert set(resolved) == set(defaults)
        assert all(resolved[key] == defaults[key] for key in rest)

    def test_defaults_not_mutated(self, name: str) -> None:
        defaults = self._defaults(name)
        resolve_params({key: defaults[key] for key in defaults}, defaults=defaults)
        assert defaults == self._defaults(name)

    def test_unknown_key_raises(self, name: str) -> None:
        with pytest.raises(ValueError, match="Unknown .* field"):
            resolve_params({"definitely_not_a_key": 1}, defaults=self._defaults(name))

    def test_non_mapping_raises(self, name: str) -> None:
        with pytest.raises(TypeError, match="must be a Mapping or None"):
            resolve_params(5, defaults=self._defaults(name))  # type: ignore[arg-type]

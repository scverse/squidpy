from __future__ import annotations

from types import UnionType
from typing import Annotated, Any, TypedDict, Union, get_args, get_origin, get_type_hints

import pytest

from squidpy import types
from squidpy._params import Default, defaults_of, resolve_params

PARAMS = [name for name in types.__all__ if name.endswith("Params")]  # the result types carry no defaults
SPECS = pytest.mark.parametrize("spec", [getattr(types, name) for name in PARAMS], ids=PARAMS)


def _matches(value: object, hint: Any) -> bool:
    """Whether ``value`` fits ``hint``; an ``int`` fits ``float``, a ``bool`` fits only ``bool``."""
    if get_origin(hint) in (Union, UnionType):
        # an arm may not be `isinstance`-checkable (`npt.ArrayLike` holds non-runtime protocols)
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


@SPECS
def test_every_default_matches_its_type(spec: type) -> None:
    hints = get_type_hints(spec, include_extras=True)
    assert set(defaults_of(spec)) == set(hints)
    for key, hint in hints.items():
        (marker,) = (m for m in hint.__metadata__ if isinstance(m, Default))
        assert _matches(marker.value, hint.__origin__), f"{spec.__name__}.{key} = {marker.value!r}"


@pytest.mark.parametrize(("value", "hint"), [("0.5", float), (1, bool), (0.5, int), (None, int | str)])
def test_matches_rejects_the_wrong_type(value: object, hint: Any) -> None:
    assert not _matches(value, hint)


def test_missing_default_raises() -> None:
    class Incomplete(TypedDict, total=False):
        a: Annotated[int, Default(1)]
        b: float

    with pytest.raises(TypeError, match="Incomplete.b` is missing a `Default"):
        defaults_of(Incomplete)


@SPECS
def test_resolve_fills_defaults_without_leaking_the_cache(spec: type) -> None:
    defaults = defaults_of(spec)
    key = next(k for k, v in defaults.items() if isinstance(v, int | float) and not isinstance(v, bool))
    override = defaults[key] / 2  # stays inside every validator's range
    assert resolve_params({key: override}, spec) == {**defaults, key: override}
    resolve_params(None, spec).clear()
    assert resolve_params(None, spec) == defaults


@SPECS
@pytest.mark.parametrize(
    ("params", "error", "match"),
    [({"definitely_not_a_key": 1}, ValueError, "Unknown `method_params` field"), (5, TypeError, "must be a Mapping")],
    ids=["unknown_key", "not_a_mapping"],
)
def test_resolve_rejects(spec: type, params: Any, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        resolve_params(params, spec)


@pytest.mark.parametrize(
    ("spec", "params"),
    [
        (types.MacenkoParams, {"alpha": 0.0}),
        (types.MacenkoParams, {"alpha": 50.0}),
        (types.MacenkoParams, {"beta": -1.0}),
        (types.VahadaneParams, {"beta": -1.0}),
        (types.VahadaneParams, {"lambda1": -1.0}),
        (types.VahadaneParams, {"n_iter": 0}),
        (types.ReinhardParams, {"luminosity_threshold": 0.0}),
        (types.ReinhardParams, {"luminosity_threshold": 1.5}),
    ],
)
def test_validator_rejects_out_of_range(spec: type, params: dict[str, Any]) -> None:
    (key,) = params
    with pytest.raises(ValueError, match=key):
        resolve_params(params, spec)


@pytest.mark.parametrize(
    ("spec", "params", "expected"),
    [
        (types.MacenkoParams, {"alpha": 2}, {"alpha": 2.0}),
        (types.VahadaneParams, {"n_iter": 5.0}, {"n_iter": 5}),
        (
            types.ReinhardParams,
            {"luminosity_threshold": 1, "mask_background": 0},
            {"luminosity_threshold": 1.0, "mask_background": False},
        ),
    ],
)
def test_validator_coerces(spec: type, params: dict[str, Any], expected: dict[str, Any]) -> None:
    resolved = resolve_params(params, spec)
    for key, value in expected.items():
        assert resolved[key] == value
        assert type(resolved[key]) is type(value)

from abc import ABC, ABCMeta
from enum import Enum, EnumMeta
from typing import Any, Type, Tuple, Mapping, Callable
from functools import wraps


def _pretty_raise_enum(cls: Type["ModeEnum"], fun: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(fun)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fun(*args, **kwargs)
        except ValueError as e:
            _cls, value, *_ = args
            e.args = (cls._format(value),)
            raise e

    if not issubclass(cls, ErrorFormatterABC):
        raise TypeError(f"Class `{cls}` must be subtype of `ErrorFormatterABC`.")
    elif not len(cls.__members__):
        # empty enum, for class hierarchy
        return fun

    return wrapper


class ErrorFormatterABC(ABC):
    """Mixin class that formats invalid value when constructing an enum."""

    __error_format__ = "Invalid option `{0}` for `{1}`. Valid options are: `{2}`."

    @classmethod
    def _format(cls, value: Enum) -> str:
        return cls.__error_format__.format(
            value, cls.__name__, [m.value for m in cls.__members__.values()]  # type: ignore[attr-defined]
        )


class PrettyEnum(Enum):
    """Enum with a pretty __str__ and __repr__."""

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self.value)


class ABCEnumMeta(EnumMeta, ABCMeta):
    """Metaclass which injects."""

    def __new__(  # noqa: D102
        cls, clsname: str, bases: Tuple[EnumMeta, ...], namespace: Mapping[str, Any]
    ) -> "ABCEnumMeta":
        res = super().__new__(cls, clsname, bases, namespace)  # type: ignore[arg-type]
        res.__new__ = _pretty_raise_enum(res, res.__new__)  # type: ignore[assignment,arg-type]
        return res


# TODO(michalk8): subclass string; remove .s?
class ModeEnum(ErrorFormatterABC, PrettyEnum, metaclass=ABCEnumMeta):
    """Enum which prints available values when invalid value has been passed."""

    @property
    def s(self) -> str:
        """Return the :attr:`value` as :class:`str`."""
        return str(self.value)

    @property
    def v(self) -> Any:
        """Alias for :attr:`value`."""
        return self.value

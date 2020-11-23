from abc import ABC, ABCMeta
from enum import Enum, EnumMeta
from functools import wraps


def _pretty_raise_enum(cls, fun):
    @wraps(fun)
    def wrapper(*args, **kwargs):
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


class ErrorFormatterABC(ABC):  # noqa: D101
    __error_format__ = "Invalid option `{!r}` for `{}`. Valid options are: `{}`."

    @classmethod
    def _format(cls, value):
        return cls.__error_format__.format(value, cls.__name__, [m.value for m in cls.__members__.values()])


class PrettyEnum(Enum):
    """Enum with a pretty __str__ and __repr__."""

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.value)


class ABCEnumMeta(EnumMeta, ABCMeta):  # noqa: D101
    def __call__(cls, *args, **kw):  # noqa: D102
        if getattr(cls, "__error_format__", None) is None:
            raise TypeError(f"Can't instantiate class `{cls.__name__}` " f"without `__error_format__` class attribute.")
        return super().__call__(*args, **kw)

    def __new__(cls, clsname, superclasses, attributedict):  # noqa: D102
        res = super().__new__(cls, clsname, superclasses, attributedict)
        res.__new__ = _pretty_raise_enum(res, res.__new__)
        return res


class ModeEnum(ErrorFormatterABC, PrettyEnum, metaclass=ABCEnumMeta):  # noqa: D101
    pass

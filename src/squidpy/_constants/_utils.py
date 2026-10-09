from __future__ import annotations

from enum import Enum
from typing import Any


class PrettyEnum(Enum):
    """Enum with a pretty __str__ and __repr__."""

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self.value)


# TODO(michalk8): members now compare equal to their value; `.s` is redundant at its 63 call sites.
class ModeEnum(str, PrettyEnum):
    """Enum which prints available values when invalid value has been passed."""

    @classmethod
    def _missing_(cls, value: object) -> None:
        raise ValueError(
            f"Invalid option `{value}` for `{cls.__name__}`. Valid options are: `{[m.value for m in cls]}`."
        )

    @property
    def s(self) -> str:
        """Return the :attr:`value` as :class:`str`."""
        return str(self.value)

    @property
    def v(self) -> Any:
        """Alias for :attr:`value`."""
        return self.value

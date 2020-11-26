"""Internal constants not exposed to the user."""
from typing import Callable, Optional

# rough proposal:
# _M -> obsm
# _U -> uns
# _O -> obs
# _V -> var

_SEP = "_"


class cprop:  # noqa: D101
    def __init__(self, f: Callable):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


class Key:  # noqa: D101
    class obs:  # noqa: D106
        pass

    class obsm:  # noqa: D106
        @cprop
        def spatial(cls) -> str:
            return "spatial"

    class uns:  # noqa: D106
        @cprop
        def spatial(cls) -> str:
            return Key.obsm.spatial

        @classmethod
        def spatial_neighs(cls, value: Optional[str] = None) -> str:
            return Key.uns.spatial if value is None else value

        @classmethod
        def spatial_dist(cls, value: Optional[str] = None) -> str:
            return f"{Key.uns.spatial_neighs(value)}_distances"

        @classmethod
        def spatial_conn(cls, value: Optional[str] = None) -> str:
            return f"{Key.uns.spatial_neighs(value)}_connectivities"

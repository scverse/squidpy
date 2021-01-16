"""Internal constants not exposed to the user."""
from typing import Any, Callable, Optional

_SEP = "_"


class cprop:  # noqa: D101
    def __init__(self, f: Callable[..., str]):
        self.f = f

    def __get__(self, obj: Any, owner: Any) -> str:
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
            return Key.obsp.spatial if value is None else value

        @classmethod
        def ligrec(cls, cluster: str, value: Optional[str] = None) -> str:
            return f"{cluster}_ligrec" if value is None else value

        @classmethod
        def nhood_enrichment(cls, cluster: str) -> str:
            return f"{cluster}_nhood_enrichment"

        @classmethod
        def centrality_scores(cls, cluster: str) -> str:
            return f"{cluster}_centrality_scores"

        @classmethod
        def interaction_matrix(cls, cluster: str) -> str:
            return f"{cluster}_interactions"

        @classmethod
        def co_occurrence(cls, cluster: str) -> str:
            return f"{cluster}_co_occurrence"

        @classmethod
        def ripley_k(cls, cluster: str) -> str:
            return f"{cluster}_ripley_k"

        @classmethod
        def colors(cls, cluster: str) -> str:
            return f"{cluster}_colors"

    class obsp:  # noqa: D106
        @cprop
        def spatial(cls) -> str:
            return f"{Key.obsm.spatial}_neighbors"

        @classmethod
        def spatial_dist(cls, value: Optional[str] = None) -> str:
            return f"{Key.obsp.spatial}_distances" if value is None else f"{value}_distances"

        @classmethod
        def spatial_conn(cls, value: Optional[str] = None) -> str:
            return f"{Key.obsp.spatial}_connectivities" if value is None else f"{value}_connectivities"

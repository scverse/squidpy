"""Internal constants not exposed to the user."""
from typing import Any, Callable, Optional

from anndata import AnnData

from squidpy._constants._constants import SegmentationBackend

_SEP = "_"


class cprop:  # noqa: D101
    def __init__(self, f: Callable[..., str]):
        self.f = f

    def __get__(self, obj: Any, owner: Any) -> str:
        return self.f(owner)


class Key:  # noqa: D101
    class img:  # noqa: D106
        @classmethod
        def segment(cls, backend: SegmentationBackend, key_added: Optional[str] = None) -> str:
            return f"segmented_{backend.s}" if key_added is None else key_added

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
            return f"{Key.obsm.spatial}_neighbors" if value is None else f"{value}_neighbors"

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

        @classmethod
        def library_id(cls, adata: AnnData, spatial_key: str, library_id: Optional[str] = None) -> str:
            if spatial_key not in adata.uns:
                raise KeyError(f"Spatial key `{spatial_key}` not found in `adata.uns`.")
            haystack = list(adata.uns[spatial_key].keys())
            if library_id is None:
                if len(haystack) > 1:
                    raise ValueError(f"Unable to select `library_id`. Please specify one from: `{sorted(haystack)}`.")
                library_id = haystack[0]

            if library_id not in haystack:
                raise KeyError()

            return library_id

    class obsp:  # noqa: D106
        @classmethod
        def spatial_dist(cls, value: Optional[str] = None) -> str:
            return f"{Key.obsm.spatial}_distances" if value is None else f"{value}_distances"

        @classmethod
        def spatial_conn(cls, value: Optional[str] = None) -> str:
            return f"{Key.obsm.spatial}_connectivities" if value is None else f"{value}_connectivities"

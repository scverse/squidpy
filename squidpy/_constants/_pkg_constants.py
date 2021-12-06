"""Internal constants not exposed to the user."""
from __future__ import annotations

from typing import Any, Union, Mapping, Callable, Optional, Sequence

from anndata import AnnData

from squidpy._constants._constants import Processing, SegmentationBackend

_SEP = "_"


class cprop:
    def __init__(self, f: Callable[..., str]):
        self.f = f

    def __get__(self, obj: Any, owner: Any) -> str:
        return self.f(owner)


class Key:
    class img:
        @classmethod
        def segment(cls, backend: Union[str, SegmentationBackend], layer_added: Optional[str] = None) -> str:
            return f"segmented_{SegmentationBackend(backend).s}" if layer_added is None else layer_added

        @classmethod
        def process(
            cls, method: Union[str, Processing, Callable[[Any], Any]], img_id: str, layer_added: Optional[str] = None
        ) -> str:
            if layer_added is not None:
                return layer_added
            if isinstance(method, Processing):
                method = method.s
            elif callable(method):
                method = getattr(method, "__name__", "custom")

            return f"{img_id}_{method}"

        @cprop
        def coords(cls) -> str:
            return "coords"

        @cprop
        def padding(cls) -> str:
            return "padding"

        @cprop
        def scale(self) -> str:
            return "scale"

        @cprop
        def mask_circle(cls) -> str:
            return "mask_circle"

        @cprop
        def obs(cls) -> str:
            return "cell"

    class obs:
        pass

    class obsm:
        @cprop
        def spatial(cls) -> str:
            return "spatial"

    class uns:
        @cprop
        def spatial(cls) -> str:
            return Key.obsm.spatial

        @cprop
        def image_key(cls) -> str:
            return "images"

        @cprop
        def scalefactors_key(cls) -> str:
            return "scalefactors"

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
        def ripley(cls, cluster: str, mode: str) -> str:
            return f"{cluster}_ripley_{mode}"

        @classmethod
        def colors(cls, cluster: str) -> str:
            return f"{cluster}_colors"

        @classmethod
        def spot_diameter(cls, adata: AnnData, spatial_key: str, library_id: Optional[str] = None) -> float:
            try:
                return float(adata.uns[spatial_key][library_id]["scalefactors"]["spot_diameter_fullres"])
            except KeyError:
                raise KeyError(
                    f"Unable to get the spot diameter from "
                    f"`adata.uns[{spatial_key!r}][{library_id!r}]['scalefactors']['spot_diameter_fullres']]`"
                ) from None

        @classmethod
        def library_id(
            cls,
            adata: AnnData,
            spatial_key: str,
            library_id: Optional[Sequence[str] | str] = None,
            unique_id: bool = True,
        ) -> Sequence[str] | str | None:

            haystack = cls._check_haystack(adata, spatial_key, library_id, sub_key=None)
            if len(haystack) == 0:
                library_id = None
            if library_id is None and unique_id:
                if len(haystack) > 1:
                    raise ValueError(
                        f"Unable to determine which library id to use. "
                        f"Please specify one from: `{sorted(haystack)}`."
                    )
                library_id = haystack[0]
            elif library_id is None and not unique_id:
                library_id = haystack

            return library_id

        @classmethod
        def image_id(
            cls,
            adata: AnnData,
            spatial_key: str,
            image_key: str,
            library_id: Optional[Sequence[str] | str] = None,
        ) -> Mapping[str, Sequence[str]]:
            haystack = cls._check_haystack(adata, spatial_key, library_id, sub_key=image_key)
            if library_id is None:
                library_id = haystack
            image_mapping = {i: list(adata.uns[spatial_key][i][image_key].keys()) for i in library_id}

            return image_mapping

        @classmethod
        def scalefactors_id(
            cls,
            adata: AnnData,
            spatial_key: str,
            scalefactors_key: str,
            library_id: Optional[Sequence[str] | str] = None,
        ) -> Mapping[str, Sequence[str]]:
            haystack = cls._check_haystack(adata, spatial_key, library_id, sub_key=scalefactors_key)
            if library_id is None:
                library_id = haystack
            scalefactors_mapping = {i: list(adata.uns[spatial_key][i][scalefactors_key].keys()) for i in library_id}

            return scalefactors_mapping

        @classmethod
        def _check_haystack(
            cls,
            adata: AnnData,
            spatial_key: str,
            library_id: Optional[Sequence[str] | str] = None,
            sub_key: Optional[str] = None,
        ) -> Sequence[str]:
            if spatial_key not in adata.uns:
                raise KeyError(f"Spatial key `{spatial_key}` not found in `adata.uns`.")
            haystack = list(adata.uns[spatial_key].keys())
            if library_id is not None:
                if not any(i in library_id for i in haystack):
                    raise KeyError(f"`library_id`: {library_id}` not found in `{sorted(haystack)}`.")
                if sub_key is not None:
                    if not all(sub_key in i for i in [adata.uns["spatial"][i].keys() for i in library_id]):
                        raise KeyError(f"`{sub_key}` not found in `adata.uns[{spatial_key}]['library_id'].keys()`.")

            return haystack

    class obsp:
        @classmethod
        def spatial_dist(cls, value: Optional[str] = None) -> str:
            return f"{Key.obsm.spatial}_distances" if value is None else f"{value}_distances"

        @classmethod
        def spatial_conn(cls, value: Optional[str] = None) -> str:
            return f"{Key.obsm.spatial}_connectivities" if value is None else f"{value}_connectivities"

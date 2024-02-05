"""Internal constants not exposed to the user."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence, Union

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
        def segment(cls, backend: str | SegmentationBackend, layer_added: str | None = None) -> str:
            return f"segmented_{SegmentationBackend(backend).s}" if layer_added is None else layer_added

        @classmethod
        def process(
            cls, method: str | Processing | Callable[[Any], Any], img_id: str, layer_added: str | None = None
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
        def image_res_key(cls) -> str:
            return "hires"

        @cprop
        def image_seg_key(cls) -> str:
            return "segmentation"

        @cprop
        def scalefactor_key(cls) -> str:
            return "scalefactors"

        @cprop
        def size_key(cls) -> str:
            return "spot_diameter_fullres"

        @classmethod
        def spatial_neighs(cls, value: str | None = None) -> str:
            return f"{Key.obsm.spatial}_neighbors" if value is None else f"{value}_neighbors"

        @classmethod
        def ligrec(cls, cluster: str, value: str | None = None) -> str:
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
        def spot_diameter(
            cls,
            adata: AnnData,
            spatial_key: str,
            library_id: str | None = None,
            spot_diameter_key: str = "spot_diameter_fullres",
        ) -> float:
            try:
                return float(adata.uns[spatial_key][library_id]["scalefactors"][spot_diameter_key])
            except KeyError:
                raise KeyError(
                    f"Unable to get the spot diameter from "
                    f"`adata.uns[{spatial_key!r}][{library_id!r}]['scalefactors'][{spot_diameter_key!r}].`"
                ) from None

        @classmethod
        def library_id(
            cls,
            adata: AnnData,
            spatial_key: str,
            library_id: Sequence[str] | str | None = None,
            return_all: bool = False,
        ) -> Sequence[str] | str | None:
            library_id = cls._sort_haystack(adata, spatial_key, library_id, sub_key=None)
            if return_all or library_id is None:
                return library_id
            if len(library_id) != 1:
                raise ValueError(
                    f"Unable to determine which library id to use. Please specify one from: `{sorted(library_id)}`."
                )
            return library_id[0]

        @classmethod
        def library_mapping(
            cls,
            adata: AnnData,
            spatial_key: str,
            sub_key: str,
            library_id: Sequence[str] | str | None = None,
        ) -> Mapping[str, Sequence[str]]:
            library_id = cls._sort_haystack(adata, spatial_key, library_id, sub_key)
            if library_id is None:
                raise ValueError("Invalid `library_id=None`")
            return {i: list(adata.uns[spatial_key][i][sub_key]) for i in library_id}

        @classmethod
        def _sort_haystack(
            cls,
            adata: AnnData,
            spatial_key: str,
            library_id: Sequence[str] | str | None = None,
            sub_key: str | None = None,
        ) -> Sequence[str] | None:
            if spatial_key not in adata.uns:
                raise KeyError(f"Spatial key {spatial_key!r} not found in `adata.uns`.")
            haystack = list(adata.uns[spatial_key])
            if library_id is not None:
                if isinstance(library_id, str):
                    library_id = [library_id]
                if not any(i in library_id for i in haystack):
                    raise KeyError(f"`library_id`: {library_id}` not found in `{sorted(haystack)}`.")
                if sub_key is not None:
                    if not all(sub_key in lib for lib in [adata.uns[spatial_key][lib] for lib in library_id]):
                        raise KeyError(
                            f"`{sub_key}` not found in `adata.uns[{spatial_key!r}]['library_id'])` "
                            f"with following `library_id`: {library_id}."
                        )
                return library_id
            return haystack

    class obsp:
        @classmethod
        def spatial_dist(cls, value: str | None = None) -> str:
            return f"{Key.obsm.spatial}_distances" if value is None else f"{value}_distances"

        @classmethod
        def spatial_conn(cls, value: str | None = None) -> str:
            return f"{Key.obsm.spatial}_connectivities" if value is None else f"{value}_connectivities"

from typing import Any, Callable

from anndata import AnnData

from squidpy.datasets._utils import PathLike

four_i: Callable[[PathLike, Any], AnnData]
imc: Callable[[PathLike, Any], AnnData]
seqfish: Callable[[PathLike, Any], AnnData]
visium_hne_adata: Callable[[PathLike, Any], AnnData]
visium_hne_adata_crop: Callable[[PathLike, Any], AnnData]
visium_fluo_adata: Callable[[PathLike, Any], AnnData]
visium_fluo_adata_crop: Callable[[PathLike, Any], AnnData]
sc_mouse_cortex: Callable[[PathLike, Any], AnnData]
mibitof: Callable[[PathLike, Any], AnnData]
merfish: Callable[[PathLike, Any], AnnData]
slideseqv2: Callable[[PathLike, Any], AnnData]

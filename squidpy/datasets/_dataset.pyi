from typing import Callable

from anndata import AnnData

four_i: Callable[[], AnnData]
imc: Callable[[], AnnData]
seqfish: Callable[[], AnnData]
visium_hne_adata: Callable[[], AnnData]
visium_hne_adata_crop: Callable[[], AnnData]
visium_fluo_adata: Callable[[], AnnData]
visium_fluo_adata_crop: Callable[[], AnnData]
sc_mouse_cortex: Callable[[], AnnData]
mibitof: Callable[[], AnnData]
merfish: Callable[[], AnnData]
slideseqv2: Callable[[], AnnData]

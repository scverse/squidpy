from typing import Any, Union, Protocol

from anndata import AnnData

from squidpy.datasets._utils import PathLike

class Dataset(Protocol):
    def __call__(self, path: Union[PathLike, None] = ..., **kwargs: Any) -> AnnData: ...

four_i: Dataset
imc: Dataset
seqfish: Dataset
visium_hne_adata: Dataset
visium_hne_adata_crop: Dataset
visium_fluo_adata: Dataset
visium_fluo_adata_crop: Dataset
sc_mouse_cortex: Dataset
mibitof: Dataset
merfish: Dataset
slideseqv2: Dataset

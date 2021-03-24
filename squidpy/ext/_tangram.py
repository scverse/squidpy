""" Deconvolution of spatial transcriptomics data with Tangram. """

import warnings

from scanpy import logging as logg
from anndata import AnnData

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        import torch
        import tangram
except ImportError:
    torch = None
    tangram = None


@d.dedent
def tangram(
    adata_sc: AnnData,
    adata_sp: AnnData,
    markers: Union[Sequence, str],
    copy: bool = False,  # TODO add all args for Tangram
):
    """
    Deconvolution of spatial Transcriptomics data with Tangram :cite:`tangram`.

    Tangram... # TODO add description

    Parameters
    ----------
    adata_sc
        Reference scRNA-seq data, as :class:`anndata.AnnData`.
    adata_sp
        Query spatial transcritpomics data, as :class:`anndata.AnnData`.
    markers
        Gene list to subset the query and reference data.
    %(copy)s
    # TODO add rest of parameters

    Returns
    -------
    If ``copy = True``, returns a :class:`pandas.DataFrame` with keys as celltypes from the query data:

    Otherwise, modifies the ``adata`` with the following key:

        - :attr:`anndata.AnnData.obsm` ``['tangram_deconvolution']`` - the above mentioned dataframe. # TODO settle on key for obsm
    """

    if torch is None or tangram is None:
        raise ImportError(
            "Please install `torch` and `tangram` as `pip install tangram`.\n"
            "Check out the original repo https://github.com/broadinstitute/Tangram\n"
            "For installation instructions"
        )

    ad_sc, ad_sp = tangram.pp_adatas(adata_sc, adata_sp, genes=markers)  # modify in place ?

    start = logg.info("Starting Tangram deconvolution run")
    # TODO fix deconv call
    ad_map = tg.map_cells_to_space(
        adata_cells=ad_sc,
        adata_space=ad_sp,
        device="cpu",
        # device='cuda:0',
    )

    if copy:
        logg.info("Finish", time=start)
        return ad_map

    # _save_data(adata, attr="uns", key="moranI", data=df, time=start)

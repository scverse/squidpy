import anndata as ad

import numpy as np
from scipy.sparse import csr_matrix
from pandas.api.types import is_categorical_dtype

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

from skimage.draw import disk

from squidpy.im.object import ImageContainer


class AnnData2Napari:
    """
    Class to interact with Napari.

    Parameters
    ----------
    adata
        the adata object

    """

    def __init__(
        self,
        adata: ad.AnnData,
        img: ImageContainer,
        obsm_key: str = "spatial",
        palette: str = "Set1",
    ):

        self.genes = adata.var_names.sort_values().values
        self.obs = adata.obs.columns.sort_values().values

        self.coords = adata.obsm[obsm_key]
        self.palette = palette

        self.adata = adata
        self.img = img

    def _get_image(
        self,
    ):

        img_library_id = list(self.img.data.keys())
        image = self.img.data[img_library_id[0]].transpose("y", "x", ...).values

        return image

    def _get_gene(self, name):

        idx = np.where(name == self.adata.var_names)[0][0]

        if isinstance(self.adata.X, csr_matrix):
            vec = self.adata.X[:, idx].todense()
        else:
            vec = self.adata.X[:, idx]

        vec = (vec - vec.min()) / (vec.max() - vec.min())
        return vec

    def _get_obs(self, name):

        if is_categorical_dtype(self.adata.obs[name]):
            vec = _get_col_categorical(self.adata, name, self.palette)
        else:
            vec = self.adata.obs[name]
            vec = (vec - vec.min()) / (vec.max() - vec.min())

        return vec


def _get_mask(adata, key: str):
    mask_lst = []

    spot_radius = _get_radius(adata)
    coords = adata.obsm[key]
    for i in np.arange(coords.shape[0]):
        y, x = coords[i, :]
        rr, cc = disk((x, y), spot_radius)
        mask_lst.append((rr, cc))

    return mask_lst


def _get_radius(adata):
    library_id = list(adata.uns["spatial"].keys())
    spot_radius = round(adata.uns["spatial"][library_id[0]]["scalefactors"]["spot_diameter_fullres"]) * 0.5
    return spot_radius


def _get_col_categorical(adata, c, palette):

    colors_key = f"{c}_colors"
    if colors_key in adata.uns.keys():
        cols = [to_rgba(i) for i in adata.uns[colors_key]]
    else:
        ncat = adata.obs[c].cat.categories.shape[0]
        cmap = plt.get_cmap(palette, ncat)
        cols = cmap(np.arange(ncat))

    cols_dict = {k: v for v, k in zip(cols, adata.obs[c].cat.categories)}
    cols = np.array([cols_dict[k] for k in adata.obs[c]])

    return cols

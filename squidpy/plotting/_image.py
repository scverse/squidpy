from typing import Union, Optional, Sequence

from anndata import AnnData

import numpy as np
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

from ..image.object import ImageContainer  # noqa: F401


def interactive(
    adata: AnnData,
    img: ImageContainer,
    color: Union[str, Sequence[str], None] = None,
    n_polygon: Union[int, Sequence[str]] = 6,
    with_qt: Optional[bool] = False,
    color_map: str = "viridis",
    palette: str = "Dark2",
) -> None:
    """
    Interactive Napari session for spot visualization.

    Params
    ------
    adata
        The AnnData object.
    img
        The ImgContainer object.
    color
        variables to visualize (either in adata.obs or adata.var_names)
    n_polygon
        number of sides of the polygon to plot (6=hexagon)
    with_qt
        whether start napari with context. Don't use in notebook, instead use %gui qt in cell
    color_map
        string for matplotlib colormap (for continuous variables)
    palette
        string for matplotlib palette (for categorical variables)

    Returns
    -------
    None

    """
    try:
        import napari
    except ImportError:
        raise ImportError("please install napari: pip install 'napari[all]'")

    color = [color] if isinstance(color, str) or color is None else list(color)
    assert set(color).issubset(
        adata.obs_keys() + adata.var_names.values.tolist()
    ), "color is not a subset of `adata.obs_keys()+adata.var_names`"

    library_id = list(adata.uns["spatial"].keys())
    spot_diameter = adata.uns["spatial"][library_id[0]]["scalefactors"]["spot_diameter_fullres"]

    color_lst = _get_colors(adata, color, color_map, palette)

    shapes = _create_shape(adata.obsm["spatial"], lt=spot_diameter * 0.5, n=n_polygon)
    shape_lst = [shapes] * len(color_lst)

    name_lst = color

    img = img.data[library_id[0]].transpose("y", "x", "channels")
    image = img.values

    if with_qt:  # context for script based run
        with napari.gui_qt():
            viewer = _open_napari(image, shape_lst, color_lst, name_lst)
    else:
        viewer = _open_napari(image, shape_lst, color_lst, name_lst)

    return viewer


def _open_napari(img: np.ndarray, shapes: list, colors: list, names: list):

    try:
        import napari
    except ImportError:
        raise ImportError("please install napari: pip install 'napari[all]'")

    viewer = napari.view_image(img, rgb=True)
    # add all colors as layers
    for i in range(len(names)):
        layer = viewer.add_shapes(  # noqa: F841
            shapes[i], shape_type="polygon", edge_width=1, edge_color="black", face_color=colors[i], name=names[i]
        )

    return viewer


def _get_colors(adata: AnnData, colors: list, color_map: str, palette: str):

    color_lst = []
    for c in colors:

        if c in adata.obs_keys():
            colors_key = f"{c}_colors"
            if colors_key in adata.uns.keys():
                cols = adata.uns[colors_key]
            else:
                ncat = adata.obs[c].unique().shape[0]
                cmap = plt.get_cmap(palette, ncat)
                cols = cmap(np.arange(ncat))

            cols_dict = {k: v for v, k in zip(cols, adata.obs[c].cat.categories)}
            cols = np.array([cols_dict[k] for k in adata.obs[c]])

        elif c in adata.var_names.values.tolist():
            if isinstance(adata.X, csr_matrix):
                gexp = adata[:, c].X.todense().flatten()
            else:
                gexp = adata[:, c].X.flatten()

            cmap = plt.get_cmap(color_map)
            gexp = (gexp - gexp.min()) / (gexp.max() - gexp.min())
            cols = cmap(gexp)
            cols = [to_hex(i, keep_alpha=False) for i in cols[-1]]

        color_lst.append(cols)

    return color_lst


def _create_shape(arr: np.ndarray, lt: float, n: int):
    coord = [np.array(_get_poly(x, y, lt, n)) for y, x in zip(arr[:, 0], arr[:, 1])]
    return coord


def _get_poly(x, y, lt, n):
    deg = int(360 / n)
    coord = [[x + np.cos(np.radians(angle)) * lt, y + np.sin(np.radians(angle)) * lt] for angle in range(0, 360, deg)]
    return coord


def plot_segmentation(img, key: str):
    """
    Plot segmentation on entire image.

    Params
    ------
    img: ImageContainer
        High-resolution image.
    key: str
        Name of layer that contains segmentation in img.
    """
    arr = img[key]
    plt.imshow(arr)
    # todo add other channels in background

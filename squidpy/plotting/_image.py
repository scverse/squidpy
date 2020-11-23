from anndata import AnnData

import numpy as np

import matplotlib.pyplot as plt

from ..image.object import ImageContainer  # noqa: F401


def interactive(adata: AnnData, img: np.ndarray, *args):
    """
    Launch Napari for spot visualization.

    Params
    ______
    adata:
        anndata

    """
    try:
        import napari
    except ImportError:
        raise ImportError("\nplease install napari: \n\n" "\tpip install napari\n")

    # colors_key = f"{cluster_key}_colors"
    # if colors_key in adata.uns.keys():
    #     ad.uns[colors_key] = adata.uns[colors_key]

    spot_diameter = adata.uns["spatial"]["V1_Adult_Mouse_Brain"]["scalefactors"]["spot_diameter_fullres"]

    new_shape = _create_shape(adata.obsm["spatial"], spot_diameter * 0.5)

    viewer = napari.view_image(img, rgb=True)
    layer = viewer.add_shapes(  # noqa: F841
        new_shape,
        shape_type="polygon",
        edge_width=1,
        edge_color="white",
        face_color=None,
    )

    return viewer


def _create_shape(arr: np.ndarray, lt: float, n: int):
    napari_coord = [np.array(_get_poly(x, y, lt, n)) for y, x in zip(arr[:, 0], arr[:, 1])]
    return napari_coord


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

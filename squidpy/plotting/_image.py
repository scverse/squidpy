import matplotlib.pyplot as plt
from anndata as AnnData

from ..image.object import ImageContainer

def interactive(adata: AnnData, img: ImageContainer, *args):

    viewer = napari.view_image(img, rgb=True)
    layer = viewer.add_shapes(
        new_shape,
        shape_type="polygon",
        edge_width=1,
        edge_color="white",
        face_color=colors,
    )


    return viewer


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

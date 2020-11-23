import matplotlib.pyplot as plt

from squidpy.image import ImageContainer


def plot_segmentation(img: ImageContainer, key: str) -> None:
    """
    Plot segmentation on entire image.

    Parameters
    ----------
    img
        High-resolution image.
    key
        Name of layer that contains segmentation in img.

    Returns
    -------
    None
        TODO.
    """
    arr = img[key]
    plt.imshow(arr)
    # todo add other channels in background

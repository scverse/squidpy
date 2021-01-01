import matplotlib.pyplot as plt

from squidpy.im import ImageContainer
from squidpy._docs import d


@d.dedent
def plot_segmentation(img: ImageContainer, key: str) -> None:
    """
    Plot segmentation on entire image.

    Parameters
    ----------
    %(img_container)s
    key
        Name of layer that contains segmentation in img.
    %(plotting)s

    Returns
    -------
    %(plotting_returns)s
    """
    # TODO: remove the type: ignore once IC is indexable
    arr = img[key]  # type: ignore
    plt.imshow(arr)
    # todo add other channels in background

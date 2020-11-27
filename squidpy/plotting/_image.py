import matplotlib.pyplot as plt

from squidpy._docs import d
from squidpy.image import ImageContainer


@d.dedent
def plot_segmentation(img: ImageContainer, key: str) -> None:
    """
    Plot segmentation on entire image.

    Parameters
    ----------
    %(img_container)s
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

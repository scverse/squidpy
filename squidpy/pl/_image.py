import matplotlib.pyplot as plt

# TODO: not sure why this happens, we might want to restructure a bit the directories
from squidpy.im import ImageContainer  # type: ignore[attr-defined]
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
    arr = img[key]
    plt.imshow(arr)
    # todo add other channels in background

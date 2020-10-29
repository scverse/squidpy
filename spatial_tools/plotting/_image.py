import matplotlib.pyplot as plt


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

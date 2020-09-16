import numpy as np
from tifffile import imread
import os
from skimage.util import img_as_ubyte

def read_tif(dataset_folder, dataset_name, rescale=True):
    """
    Args:
        rescale (bool): scale the image to uint8
    """
    # switch to tiffile to read images
    img_path = os.path.join(dataset_folder, f"{dataset_name}_image.tif")
    img = imread(img_path)
    if len(img.shape) > 2:
        if img.shape[0] in (2,3,4):
            # is the channel dimension the first dimension?
            img = np.transpose(img, (1,2,0))
    if rescale:
        img = img_as_ubyte(img)
    return img

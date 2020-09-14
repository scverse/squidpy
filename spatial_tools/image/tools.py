import numpy as np
import imageio
import os

def read_tif(dataset_folder, dataset_name):
    img = imageio.imread(os.path.join(dataset_folder, f"{dataset_name}_image.tif"))
    return img

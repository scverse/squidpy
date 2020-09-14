import numpy as np

def crop_img(img, x, y, scalef=1.0, sizef=1.0, spot_diameter=89.44476048022638, mask_circle=False):
    """
    extract a crop from `img` centered at `x` and `y`. 
    Attrs:
        x (int): x coord of crop in `img`
        y (int): y coord of crop in `img`
        scalef (float): resolution of the crop (smaller -> smaller image)
        sizef (float): amount of context (1.0 means size of spot, larger -> more context)
        spot_diameter (float): standard size of crop, with sizef == 1.0, taken from 
            `adata.uns['spatial'][dataset_name]['scalefactors']['spot_diameter_fullres']`
        mask_circle (bool): mask crop to a circle
    """
    img_size = np.round(spot_diameter*sizef*scalef).astype(int)
    return np.zeros((img_size, img_size))
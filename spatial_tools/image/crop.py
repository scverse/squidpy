import numpy as np
from skimage.transform import rescale
from skimage.draw import disk
import skimage.util

def crop_iterator(adata, img, **kwargs):
    """
    iterator over all obs_ids defined in adata. Yielding crops from high-resolution img 
    centered around coords defined in adata.obsm['spatial']
    
    :param adata: anndata object
    :param img: ImageContainer object
    :param dataset_name: name of the data in adata (if not specified, take first one)
    :param sizef: amount of context (1.0 means size of spot, larger -> more context)
    :param scale: resolution of the crop (smaller -> smaller image)
    mask_circle (bool): mask crop to a circle
    cval (float): the value outside image boundaries or the mask
    dtype (str): optional, type to which the output should be (safely cast)
    
    yields: (obs_id, crop)
    """
    dataset_name = kwargs.get('dataset_name', None)
    if dataset_name is None:
        dataset_name = list(adata.uns['spatial'].keys())[0]
    xcoord = adata.obsm["spatial"][:, 0]
    ycoord = adata.obsm["spatial"][:, 1]
    spot_diameter = adata.uns['spatial'][dataset_name]['scalefactors']['spot_diameter_fullres']
    sizef = kwargs.get('sizef', 1)
    s = np.round(spot_diameter*sizef).astype(int)
    
    obs_ids = adata.obs.index.tolist()
    
    for i, obs_id  in enumerate(obs_ids):
        yield (obs_id, img.crop(x=xcoord[i], y=ycoord[i], s=s, **kwargs))


def uncrop_img(crops, x, y, shape):
    """
    Re-assemble image from crops and their centres.

    Fills remaining positions with zeros.

    Attrs:
        crops (List[np.ndarray]): List of image crops.
        x (int): x coord of crop in `img`
        y (int): y coord of crop in `img`
        shape (int): Shape of full image
    """
    assert y < shape[0], f"y ({y}) is outsize of image range ({shape[0]})"
    assert x < shape[1], f"x ({x}) is outsize of image range ({shape[1]})"

    img = np.zeros(shape)
    if len(crops) > 1:
        for c, x, y in zip(crops, x, y):
            x0 = x - c.shape[0] // 2
            x1 = x + c.shape[0] - c.shape[0] // 2
            y0 = y - c.shape[1] // 2
            y1 = y + c.shape[1] - c.shape[1] // 2
            assert x0 >= 0, f"x ({x0}) is outsize of image range ({0})"
            assert y0 >= 0, f"x ({x0}) is outsize of image range ({0})"
            assert x1 < shape[0], f"x ({x1}) is outsize of image range ({shape[0]})"
            assert y1 < shape[1], f"x ({y1}) is outsize of image range ({shape[1]})"
            img[x0:x1, y0:y1] = c
        return img
    else:
        assert crops[0].shape == shape, "single crop is not of the target shape %s" % str(crops[0].shape)
        return crops[0]

def crop_img(img, x, y, s=100, **kwargs):
    """
    extract a crop from xarray `img` centered at `x` and `y`. 
    Attrs:
        img (xarray): image to crop from
        x (int): x coord of crop in `img`
        y (int): y coord of crop in `img`
        s (int): width and heigh of the crop in pixels
        scale (float): resolution of the crop (smaller -> smaller image)
        mask_circle (bool): mask crop to a circle
        cval (float): the value outside image boundaries or the mask
        dtype (str): optional, type to which the output should be (safely cast)
        
    Returns:
        np.ndarray with dimensions: y, x, channels
    """
    scale = kwargs.get("scale", 1.0)
    mask_circle = kwargs.get("mask_circle", False)
    cval = kwargs.get("cval", 0.0)
    dtype = kwargs.get('dtype', None)
    # get conversion function
    if dtype is not None:
        if dtype == 'uint8':
            convert = skimage.util.img_as_ubyte
        else:
            raise NotImplementedError(dtype)
    
    assert y < img.y.shape[0], f"y ({y}) is outsize of image range ({img.y.shape[0]})"
    assert x < img.x.shape[0], f"x ({x}) is outsize of image range ({img.x.shape[0]})"
    assert s > 0, f"image size cannot be 0"
    
    if len(img.shape) == 3:
        crop = (np.zeros((s,s,img.channels.shape[0]))+cval).astype(img.dtype)
    else:
        crop = (np.zeros((s,s))+cval).astype(img.dtype)
        
    # get crop coords
    x0 = x - s//2
    x1 = x + s - s//2
    y0 = y - s//2
    y1 = y + s - s//2
    
    # crop image and put in already prepared `crop`
    crop_x0 = min(x0, 0)*-1
    crop_y0 = min(y0, 0)*-1
    crop_x1 = s - max(x1 - img.x.shape[0], 0)
    crop_y1 = s - max(y1 - img.y.shape[0], 0)
    
    crop[crop_y0:crop_y1, crop_x0:crop_x1] = img[{'y': slice(max(y0,0),y1), 'x': slice(max(x0,0),x1)}].transpose('y', 'x', 'channels')
    # scale crop
    if scale != 1:
        multichannel = len(img.shape) > 2
        crop = rescale(crop, scale, preserve_range=True, multichannel=multichannel)
        crop = crop.astype(img.dtype)
        
    # mask crop
    if mask_circle:
        # get coords inside circle
        rr, cc = disk(center=(crop.shape[0]//2, crop.shape[1]//2), 
                                   radius=crop.shape[0]//2, shape=crop.shape)
        circle = np.zeros_like(crop)
        circle[rr, cc] = 1
        # set everything outside circle to cval
        crop[circle==0] = cval
        
    if dtype is not None:
        crop = convert(crop)
    return crop
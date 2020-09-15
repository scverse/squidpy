import numpy as np
import imageio
import os
import pandas as pd
from spatial_tools.image.manipulate import crop_img
import skimage.feature as sk_image

def read_tif(dataset_folder, dataset_name):
    img = imageio.imread(os.path.join(dataset_folder, f"{dataset_name}_image.tif"))
    return img

def features_abt(adata, dataset_folder, dataset_name, features=["hog"], scalef=1, sizef=1, key_added='features', inplace=True):
    """
    Calculate features table from high resolution h&e / fluorescence image for each obs (cell/spot)
    Args:
        adata: annotated data matrix
        dataset_folder: folder containing the tif image
        dataset_name: name of the dataset (used to read the tif image)
        features: list of feature names to add to dataframe, default to hog
        scalef: scale of image crop from which features are calculated
        sizef: size (neighborhood) of image crop from which features are calculated
        key_added (string): key under which to add the features in adata.obsm. (default: ``'features'``)
        inplace (bool): add features matrix to adata, or return features matrix
    Returns:
        None or features_log (depending on `inplace` argument)
    """
    features_log = pd.DataFrame()
    
    img = read_tif(dataset_folder, dataset_name)
    
    xcoord = adata.obsm["spatial"][:, 0]
    ycoord = adata.obsm["spatial"][:, 1]
    spot_diameter = adata.uns['spatial'][dataset_name]['scalefactors']['spot_diameter_fullres']
    
    for spot_id, cell_name  in enumerate(adata.obs.index.tolist()):
        crop_ = crop_img(img, xcoord[spot_id], ycoord[spot_id], scalef=scalef, 
                          sizef=sizef, spot_diameter=spot_diameter)
        
        features_pd = get_features_statistics(crop_, cell_name, features=features)
        features_log = pd.concat([features_log, features_pd], axis=0)
        
    features_log.set_index(["cell_name"], inplace=True)
    if inplace:
        adata.obsm[key_added] = features_log
    else:
        return features_log

def get_features_statistics(im, cell_name, features=["hog"]):
    """
    im: image (numpy array)
    spot_id: the spot id of the image element, int
    features: features to calculate (str), List
    output: pandas Data frame with all features for a image or crop
    """
    features_pd = pd.DataFrame([cell_name], columns=["cell_name"])
    
    for feature in features:
        if feature == "hog":
            features_pd = pd.concat([features_pd, get_hog_features(im)], axis=1)
    return features_pd

def get_hog_features(im):
    """
    im: image or image crop, numpy array
    spot_id: the spot id of the image element, int
    output: numpy array with hog features
    """
    features = sk_image.hog(im)
    hog_pd = pd.DataFrame(features).T
    hog_pd.columns = [str(col) + '_hog' for col in hog_pd.columns]
    return hog_pd


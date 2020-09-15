import numpy as np
import imageio
import os
import pandas as pd
from spatial_tools.image.manipulate import crop_img
import skimage.feature as sk_image

def read_tif(dataset_folder, dataset_name):
    img = imageio.imread(os.path.join(dataset_folder, f"{dataset_name}_image.tif"))
    return imgtures_abt(adata, dataset_folder, dataset_name, features=["hog"]):


def get_features_abt(adata, dataset_folder, dataset_name, features=["hog"]):

    """
    image: array of whole image to crop and calc features from
    spot_ids: array of integers of the spot_id to analyze
    xccord, ycoord: array of ints
    spot_diameter: float
    features: list of feature names to add to dataframe, default to hog
    """
    features_list = []
    
    img = read_tif(dataset_folder, dataset_name)
    
    xcoord = adata.obsm["spatial"][:, 0]
    ycoord = adata.obsm["spatial"][:, 1]
    spot_diameter = adata.uns['spatial'][dataset_name]['scalefactors']['spot_diameter_fullres']
    
    cell_names = adata.obs.index.tolist()
    
    for spot_id, cell_name  in enumerate(cell_names):
        crop_ = crop_img(img, xcoord[spot_id], ycoord[spot_id], scalef=1, 
                          sizef=1, spot_diameter=spot_diameter)
        
        features_dict = get_features_statistics(crop_, cell_name, features=features)        
        features_list.append(features_dict)
    
    features_log = pd.DataFrame(features_list)
    features_log["cell_name"] = cell_names
    features_log.set_index(["cell_name"], inplace=True)
    return features_log

def get_features_statistics(im, cell_name, features=["hog"]):
    """
    im: image (numpy array)
    spot_id: the spot id of the image element, int
    features: features to calculate (str), List
    output: pandas Data frame with all features for a image or crop
    """
    stat_dict = {}
    for feature in features:
        if feature == "hog":
            stat_dict.update(get_hog_features(im, feature))
    return stat_dict

def get_hog_features(im, feature_name):
    """
    im: image or image crop, numpy array
    spot_id: the spot id of the image element, int
    output: numpy array with hog features
    """
    hog_dict = {}
    hog_features = sk_image.hog(im)
    for k, hog_feature in enumerate(hog_features):
        hog_dict[f"{feature_name}_{k}"] = hog_feature
    return hog_dict


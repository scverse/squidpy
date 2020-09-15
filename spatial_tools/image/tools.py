import numpy as np
import imageio
import os

def read_tif(dataset_folder, dataset_name):
    img = imageio.imread(os.path.join(dataset_folder, f"{dataset_name}_image.tif"))
    return img

def features_abt(adata, dataset_folder, dataset_name, features=["hog"]):
    """
    image: array of whole image to crop and calc features from
    spot_ids: array of integers of the spot_id to analyze
    xccord, ycoord: array of ints
    spot_diameter: float
    features: list of feature names to add to dataframe, default to hog
    """
    features_log = pd.DataFrame()
    
    img = read_tif(dataset_folder, dataset_name)
    
    xcoord = adata.obsm["spatial"][:, 0]
    ycoord = adata.obsm["spatial"][:, 1]
    spot_diameter = adata.uns['spatial'][dataset_name]['scalefactors']['spot_diameter_fullres']
    
    for spot_id, cell_name  in enumerate(adata.obs.index.tolist()):
        crop_ = crop_img(img, xcoord[spot_id], ycoord[spot_id], scalef=1, 
                          sizef=1, spot_diameter=spot_diameter)
        
        features_pd = get_features_statistics(crop_, cell_name, features=features)
        features_log = pd.concat([features_log, features_pd], axis=0)
        
    features_log.set_index(["cell_name"], inplace=True)
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


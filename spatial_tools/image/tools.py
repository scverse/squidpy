import numpy as np
import imageio
import os
import pandas as pd
from spatial_tools.image.manipulate import crop_img
import skimage.feature as sk_image
from skimage.feature import greycoprops
from skimage.feature import greycomatrix

def read_tif(dataset_folder, dataset_name):
    img = imageio.imread(os.path.join(dataset_folder, f"{dataset_name}_image.tif"))
    return img

def get_image_features(adata, dataset_folder, dataset_name, features=["summary"]):

    """
    image: array of whole image to crop and calc features from
    spot_ids: array of integers of the spot_id to analyze
    xccord, ycoord: array of ints
    spot_diameter: float
    features: list of feature names to add to dataframe, default to hog
    """
    available_features = ["hog", "texture", "summary", "color_hist"]
    
    for feature in features:
            assert feature in available_features, f"feature: {feature} not a valid feature, select on of {available_features}"

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

def get_features_statistics(im, cell_name, features):

    '''
    im: image (numpy array)
    spot_id: the spot id of the image element, int
    features: features to calculate (str), List
    output: pandas Data frame with all features for a image or crop
    '''
    stat_dict = {}
    for feature in features:
        if feature == "hog":
            stat_dict.update(get_hog_features(im, feature))
        if feature == "texture":
            stat_dict.update(get_grey_texture_features(im, feature))
        if feature == "color_hist":
            stat_dict.update(get_color_hist(im, feature))
        if feature == "summary":
            stat_dict.update(get_summary_stats(im, feature))
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


 
def get_summary_stats(img, feature, quantiles=[0.9,0.5,0.1],mean=False,std=False,channels=[0,1,2]):
    """Calculate summary statistics of color channels
    
    Arguments
    ---------
    img: np.array 
        rgb image in uint8 format.
    qunatiles: list of floats
        Quantiles that are computed
    mean: bool
        Compute mean
    std: bool
        Compute std
    channels: list of ints
        define for which channels histograms are computed
        
    Returns
    -------
    dict of feature values
    
    """
    stats = {}
    for c in channels:
        for q in quantiles:
            stats[f'{feature}_quantile_{q}_ch_{c}'] = np.quantile(img[:,:,c], q)
        if mean:
            stats[f'{feature}_mean_ch_{c}'] = np.mean(img[:,:,c], q)
        if std:
            stats[f'{feature}_std_ch_{c}'] = np.std(img[:,:,c], q)
    return stats

def get_color_hist(img, feature, bins=10,channels=[0,1,2],v_range=(0,255)):
    """Compute histogram counts of color channel values 
    
    Arguments
    ---------
    img: np.array 
        rgb image in uint8 format.
    bins: int
        number of binned value intervals
    channels: list of ints
        define for which channels histograms are computed
    v_range: tuple of two ints
        Range on which values are binned.
        
    Returns
    -------
    dict of feature values
    
    """
    features = {}
    for c in channels:
        hist = np.histogram(img[:,:,c], bins=10, range=[0,255], weights=None, density=False)
        for i,count in enumerate(hist[0]):
            features[f'{feature}_ch_{c}_bin_{i}'] = count
    return features
    
    
def get_grey_texture_features(img, feature, props=['contrast', 'dissimilarity', 'homogeneity', 'correlation', 'ASM'], distances=[1],angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """Calculate texture features
    
    A grey level co-occurence matrix (GLCM) is computed for different combinations of distance and angle. 
    The distance defines the pixel difference of co occurence. The angle define the direction along which 
    we check for co-occurence. The GLCM includes the number of times that grey-level j occurs at a distance 
    d and at an angle theta from grey-level i.
    From a given GLCM texture features are infered.
    
    Arguments
    ---------
    img: np.array 
        rgb image in uint8 format.
    props: list of strs
        texture features that are calculated. See `prop` in skimage.feature.greycoprops
    distances: list of ints
        See `distances` in skimage.feature.greycomatrix 
    angles: list of floats
        See `angles` in skimage.feature.greycomatrix 
        
    Returns
    -------
    dict of feature values
    
    """
    features = {}    
    # get grey scale image
    multiplier = [0.299, 0.587, 0.114]
    grey_img = np.dot(img, multiplier).astype(np.uint8)
    
    comatrix = greycomatrix(grey_img, distances=distances, angles=angles, levels=256)
    for p in props:
        tmp_features = greycoprops(comatrix, prop=p)
        for d_idx, d in enumerate(distances):
            for a_idx, a in enumerate(angles):
                features[f'{feature}_{p}_dist_{d}_angle_{a:.2f}'] = tmp_features[d_idx,a_idx]
    return features

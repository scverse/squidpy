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

def summary_stats(img,quantiles=[0.9,0.5,0.1],mean=False,std=False,channels=[0,1,2]):
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
            stats[f'quantile_{q}_ch_{c}'] = np.quantile(img[:,:,c], q)
        if mean:
            stats[f'mean_ch_{c}'] = np.mean(img[:,:,c], q)
        if std:
            stats[f'std_ch_{c}'] = np.std(img[:,:,c], q)
    return stats

def color_hist(img,bins=10,channels=[0,1,2],v_range=(0,255)):
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
            features[f'ch_{c}_bin_{i}'] = count
    return features
    
    
def grey_texture_features(img, props=['contrast', 'dissimilarity', 'homogeneity', 'correlation', 'ASM'], distances=[1],angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
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
                features[f'{p}_dist_{d}_angle_{a:.2f}'] = tmp_features[d_idx,a_idx]
    return features
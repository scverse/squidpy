from __future__ import annotations

import typing
from typing import Protocol

import centrosome.cpmorphology
import centrosome.propagate
import centrosome.zernike
import numpy as np
import scipy.ndimage
import skimage.morphology
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte

from squidpy.im import ImageContainer


def _all_regionprops_names() -> list[str]:
    names = [
        "area",
        "area_bbox",
        "area_convex",
        "area_filled",
        "axis_major_length",
        "axis_minor_length",
        "centroid",  # TODO might drop centroids
        "centroid_local",
        "centroid_weighted_local",
        "eccentricity",
        "equivalent_diameter_area",
        "euler_number",
        "extent",
        "feret_diameter_max",
        # "inertia_tensor",
        "inertia_tensor_eigvals",
        "intensity_max",
        "intensity_min",
        "intensity_mean",
        # "intensity_std",  # TODO either bump version for skimage to >=0.23 for intensity_std to be included in regionprops or drop std
        # "moments",  # TODO evaluate if more moments necessary
        "moments_hu",
        # "moments_normalized",
        "num_pixels",
        "orientation",
        "perimeter",
        "perimeter_crofton",
        "solidity",
        "border_occupied_factor",
        "granularity",
        "zernike",
        "radial_distribution",
        "calculate_image_texture",
        "calculate_histogram",
        "calculate_quantiles",
    ]
    return names


# class CalcImageFeatureCallable(Protocol):
#     def __call__(self, mask: np.ndarray, pixels: np.ndarray, **kwargs: dict[str, typing.Any]) -> dict: ...


def calculate_image_feature(feature: typing.Callable, mask: np.ndarray, pixels: np.ndarray, *args, **kwargs) -> dict:
    result = {}
    for label in np.unique(mask):
        if label == 0:
            continue
        if _is_multichannel(pixels):
            result[label] = {}
            for channel in range(_get_n_channels(pixels)):
                result[label][channel] = feature(
                    mask=mask, pixels=pixels[..., channel][mask[..., 0] == label], **kwargs
                )
        else:
            result[label] = feature(mask=mask, pixels=pixels[mask == label], **kwargs)

    return result


def calculate_image_texture(mask: np.ndarray, pixels: np.ndarray, **kwargs) -> dict:
    distances = kwargs.get("image_texture_distances", (1,))
    angles = kwargs.get("image_texture_angles", (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4))
    props = kwargs.get(
        "image_texture_graycoprops",
        (
            "contrast",
            "dissimilarity",
            "homogeneity",
            "correlation",
            "ASM",
        ),
    )
    if not np.issubdtype(pixels.dtype, np.uint8):
        pixels = img_as_ubyte(pixels, force_copy=False)  # values must be in [0, 255]

    features = {}
    comatrix = graycomatrix(pixels, distances=distances, angles=angles, levels=256)
    for prop in props:
        tmp_features = graycoprops(comatrix, prop=prop)
        for distance_idx, dist in enumerate(distances):
            for angle_idx, a in enumerate(angles):
                features[f"{prop}_dist-{dist}_angle-{a:.2f}"] = tmp_features[distance_idx, angle_idx]
    return features


def calculate_histogram(mask: np.ndarray, pixels: np.ndarray, **kwargs) -> dict:
    """
    This is way too slow of an implementation


    Parameters
    ----------
    mask
    pixels
    kwargs

    Returns
    -------

    """
    bins = kwargs.get("bins", 10)
    v_range = kwargs.get("v_range", None)
    # if v_range is None, use whole-image range
    if v_range is None:
        v_range = np.min(pixels), np.max(pixels)
    hist, _ = np.histogram(
        pixels,
        bins=bins,
        range=v_range,
        weights=None,
        density=False,
    )
    result = {str(i): count for i, count in enumerate(hist)}
    return result


def calculate_quantiles(mask: np.ndarray, pixels: np.ndarray, **kwargs) -> dict[str, float]:
    quantiles = kwargs.get("quantiles", [0.1, 0.5, 0.9])
    result = {str(quantile): np.quantile(pixels, quantile) for quantile in quantiles}
    return result


# def calculate_quantiles(mask: np.ndarray, pixels: np.ndarray, *args, **kwargs) -> dict:
#     quantiles = kwargs.get("quantiles", [0.1, 0.5, 0.9])
#
#     result = {}
#     for label in np.unique(mask):
#         if label == 0:
#             continue
#         if _is_multichannel(pixels):
#             result[label] = {}
#             for channel in range(_get_n_channels(pixels)):
#                 # result[label][channel] = _get_quantile(label, mask, pixels[..., channel], quantiles)
#                 result[label][channel] = {quantile: np.quantile(pixels[..., channel][mask[..., 0] == label], quantile) for quantile in quantiles}
#                 # current_pixels = pixels[mask == label]
#                 # result = {}
#                 # for quantile in quantiles:
#                 #     result[quantile] = np.quantile(current_pixels, quantile)
#                 # return result
#         else:
#             result[label] = {quantile: np.quantile(pixels[mask == label], quantile) for quantile in quantiles}
#             # result[label] = _get_quantile(label, mask, pixels, quantiles)


def _get_n_channels(img: np.ndarray | ImageContainer) -> int:
    """
    Returns the number of channels in the image.
    Right now, ImageContainer's layer organization is assumed meaning
    (y, x, z, channels)

    Parameters
    ----------
    img : np.ndarray or ImageContainer The image to test

    Returns
    -------
    The number of channels in the image
    """
    if len(img.shape) == 4:
        return img.shape[3]
    else:
        raise NotImplementedError(f"Unexpected image shape (got {img.shape}, but expected (y, x, z, channels))")


def _is_multichannel(img: np.ndarray) -> bool:
    """
    Determines if the image is multichannel, i.e. if it has more than 1 intensity layer.
    Right now, ImageContainer's layer organization is assumed meaning
    (y, x, z, channels)

    Parameters
    ----------
    img : np.ndarray The image to test

    Returns
    -------
    True if img has more than one channel

    """
    return _get_n_channels(img) > 1


def border_occupied_factor(mask: np.ndarray, *args, **kwargs) -> dict[int, float]:
    """
    Calculates the percentage of border pixels that are in a 4-connected neighborhood of another label
    Takes ~1.7s/megapixels to calculate

    Parameters
    ----------
    mask: np.ndarray integer grayscale image with the labels
    args None
    kwargs None

    Returns
    -------
    Dict of percentages for each label key

    """
    n_border_pixels = {}
    n_border_occupied_pixels = {}

    adjacent_indices = [
        [-1, 0],
        [0, -1],
        [1, 0],
        [0, 1],
    ]

    for coordinates, pixel in np.ndenumerate(mask):
        if pixel == 0:
            continue
        is_border = False
        is_occupied = False

        for adjacent_index in adjacent_indices:
            try:
                adjacent_pixel = mask[coordinates[0] + adjacent_index[0], coordinates[1] + adjacent_index[1]]
            except IndexError:
                # At image border, don't count image borders
                continue
            if adjacent_pixel == pixel:
                continue

            is_border = True
            if adjacent_pixel != 0:
                is_occupied = True

        if is_border:
            try:
                n_border_pixels[pixel] += 1
            except KeyError:
                n_border_pixels[pixel] = 1
                n_border_occupied_pixels[pixel] = 0
        if is_occupied:
            n_border_occupied_pixels[pixel] += 1

    result = {key: n_border_occupied_pixels[key] / n_border_pixels[key] for key in n_border_pixels.keys()}

    return result


# Copied from https://github.com/afermg/cp_measurements/blob/main/src/cp_measure/minimal/measuregranularity.py
__doc__ = """\
MeasureGranularity
==================
**MeasureGranularity** outputs spectra of size measurements of the
textures in the image.

Image granularity is a texture measurement that tries to fit a series of
structure elements of increasing size into the texture of the image and outputs a spectrum of measures
based on how well they fit.
Granularity is measured as described by Ilya Ravkin (references below).

Basically, MeasureGranularity:
1 - Downsamples the image (if you tell it to). This is set in
**Subsampling factor for granularity measurements** or **Subsampling factor for background reduction**.
2 - Background subtracts anything larger than the radius in pixels set in
**Radius of structuring element.**
3 - For as many times as you set in **Range of the granular spectrum**, it gets rid of bright areas
that are only 1 pixel across, reports how much signal was lost by doing that, then repeats.
i.e. The first time it removes one pixel from all bright areas in the image,
(effectively deleting those that are only 1 pixel in size) and then reports what % of the signal was lost.
It then takes the first-iteration image and repeats the removal and reporting (effectively reporting
the amount of signal that is two pixels in size). etc.

|MeasureGranularity_example|

As of **CellProfiler 4.0** the settings for this module have been changed to simplify
configuration. A single set of parameters is now applied to all images and objects within the module,
rather than each image needing individual configuration.
Pipelines from older versions will be converted to match this format. If multiple sets of parameters
were defined CellProfiler will apply the first set from the older pipeline version.
Specifying multiple sets of parameters can still be achieved by running multiple copies of this module.


|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *Granularity:* The module returns one measurement for each instance
   of the granularity spectrum set in **Range of the granular spectrum**.

References
^^^^^^^^^^

-  Serra J. (1989) *Image Analysis and Mathematical Morphology*, Vol. 1.
   Academic Press, London
-  Maragos P. “Pattern spectrum and multiscale shape representation”,
   *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 11,
   N 7, pp. 701-716, 1989
-  Vincent L. (2000) “Granulometries and Opening Trees”, *Fundamenta
   Informaticae*, 41, No. 1-2, pp. 57-90, IOS Press, 2000.
-  Vincent L. (1992) “Morphological Area Opening and Closing for
   Grayscale Images”, *Proc. NATO Shape in Picture Workshop*,
   Driebergen, The Netherlands, pp. 197-208.
-  Ravkin I, Temov V. (1988) “Bit representation techniques and image
   processing”, *Applied Informatics*, v.14, pp. 41-90, Finances and
   Statistics, Moskow, (in Russian)
"""


def granularity(
    mask: np.ndarray,
    pixels: np.ndarray,
    subsample_size: float = 0.25,
    image_sample_size: float = 0.25,
    element_size: int = 10,
    granular_spectrum_length: int = 16,  # default 16
):
    """
    Parameters
    ----------
    subsample_size : float, optional
        Subsampling factor for granularity measurements.
        If the textures of interest are larger than a few pixels, we recommend
        you subsample the image with a factor <1 to speed up the processing.
        Downsampling the image will let you detect larger structures with a
        smaller sized structure element. A factor >1 might increase the accuracy
        but also require more processing time. Images are typically of higher
        resolution than is required for granularity measurements, so the default
        value is 0.25. For low-resolution images, increase the subsampling
        fraction; for high-resolution images, decrease the subsampling fraction.
        Subsampling by 1/4 reduces computation time by (1/4) :sup:`3` because the
        size of the image is (1/4) :sup:`2` of original and the range of granular
        spectrum can be 1/4 of original. Moreover, the results are sometimes
        actually a little better with subsampling, which is probably because
        with subsampling the individual granular spectrum components can be used
        as features, whereas without subsampling a feature should be a sum of
        several adjacent granular spectrum components. The recommendation on the
        numerical value cannot be determined in advance; an analysis as in this
        reference may be required before running the whole set. See this `pdf`_,
        slides 27-31, 49-50.

        .. _pdf:     http://www.ravkin.net/presentations/Statistical%20properties%20of%20algorithms%20for%20analysis%20of%20cell%20images.pdf"

    image_sample_size : float, optional
        Subsampling factor for background reduction.
        It is important to remove low frequency image background variations as
        they will affect the final granularity measurement. Any method can be
        used as a pre-processing step prior to this module; we have chosen to
        simply subtract a highly open image. To do it quickly, we subsample the
        image first. The subsampling factor for background reduction is usually
        [0.125 – 0.25]. This is highly empirical, but a small factor should be
        used if the structures of interest are large. The significance of
        background removal in the context of granulometry is that image volume
        at certain granular size is normalized by total image volume, which
        depends on how the background was removed.

    element_size : int, optional
        Radius of structuring element.
        This radius should correspond to the radius of the textures of interest
        *after* subsampling; i.e., if textures in the original image scale have
        a radius of 40 pixels, and a subsampling factor of 0.25 is used, the
        structuring element size should be 10 or slightly smaller, and the range
        of the spectrum defined below will cover more sizes.

    granular_spectrum_length : int, optional
        Range of the granular spectrum.
        You may need a trial run to see which granular
        spectrum range yields informative measurements. Start by using a wide
        spectrum and narrow it down to the informative range to save time.

    Returns
    -------
    None

    """
    #
    # Downsample the image and mask
    #
    new_shape = np.array(pixels.shape)
    if subsample_size < 1:
        new_shape = new_shape * subsample_size
        if pixels.ndim == 2:
            i, j = np.mgrid[0 : new_shape[0], 0 : new_shape[1]].astype(float) / subsample_size
            pixels = scipy.ndimage.map_coordinates(pixels, (i, j), order=1)
            mask = scipy.ndimage.map_coordinates(mask.astype(float), (i, j)) > 0.9
        else:
            k, i, j = np.mgrid[0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]].astype(float) / subsample_size
            pixels = scipy.ndimage.map_coordinates(pixels, (k, i, j), order=1)
            mask = scipy.ndimage.map_coordinates(mask.astype(float), (k, i, j)) > 0.9
    else:
        pixels = pixels.copy()
        mask = mask.copy()
    #
    # Remove background pixels using a greyscale tophat filter
    #
    if image_sample_size < 1:
        back_shape = new_shape * image_sample_size
        if pixels.ndim == 2:
            i, j = np.mgrid[0 : back_shape[0], 0 : back_shape[1]].astype(float) / image_sample_size
            back_pixels = scipy.ndimage.map_coordinates(pixels, (i, j), order=1)
            back_mask = scipy.ndimage.map_coordinates(mask.astype(float), (i, j)) > 0.9
        else:
            k, i, j = np.mgrid[0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]].astype(float) / subsample_size
            back_pixels = scipy.ndimage.map_coordinates(pixels, (k, i, j), order=1)
            back_mask = scipy.ndimage.map_coordinates(mask.astype(float), (k, i, j)) > 0.9
    else:
        back_pixels = pixels
        back_mask = mask
        back_shape = new_shape
    radius = element_size
    if pixels.ndim == 2:
        footprint = skimage.morphology.disk(radius, dtype=bool)
    else:
        footprint = skimage.morphology.ball(radius, dtype=bool)
    back_pixels_mask = np.zeros_like(back_pixels)
    back_pixels_mask[back_mask == True] = back_pixels[back_mask == True]
    back_pixels = skimage.morphology.erosion(back_pixels_mask, footprint=footprint)
    back_pixels_mask = np.zeros_like(back_pixels)
    back_pixels_mask[back_mask == True] = back_pixels[back_mask == True]
    back_pixels = skimage.morphology.dilation(back_pixels_mask, footprint=footprint)
    try:
        if image_sample_size < 1:
            if pixels.ndim == 2:
                i, j = np.mgrid[0 : new_shape[0], 0 : new_shape[1]].astype(float)
                #
                # Make sure the mapping only references the index range of
                # back_pixels.
                #
                i *= float(back_shape[0] - 1) / float(new_shape[0] - 1)
                j *= float(back_shape[1] - 1) / float(new_shape[1] - 1)
                back_pixels = scipy.ndimage.map_coordinates(back_pixels, (i, j), order=1)
            else:
                k, i, j = np.mgrid[0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]].astype(float)
                k *= float(back_shape[0] - 1) / float(new_shape[0] - 1)
                i *= float(back_shape[1] - 1) / float(new_shape[1] - 1)
                j *= float(back_shape[2] - 1) / float(new_shape[2] - 1)
                back_pixels = scipy.ndimage.map_coordinates(back_pixels, (k, i, j), order=1)
    # TODO Debug the reason for the ZeroDivisionError when using MIBI-TOF dataeset
    # from https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks/examples/technology_mibitof.html
    except ZeroDivisionError:
        return _handle_granularity_error(granular_spectrum_length)
    pixels -= back_pixels
    pixels[pixels < 0] = 0

    # Transcribed from the Matlab module: granspectr function
    #
    # CALCULATES GRANULAR SPECTRUM, ALSO KNOWN AS SIZE DISTRIBUTION,
    # GRANULOMETRY, AND PATTERN SPECTRUM, SEE REF.:
    # J.Serra, Image Analysis and Mathematical Morphology, Vol. 1. Academic Press, London, 1989
    # Maragos,P. "Pattern spectrum and multiscale shape representation", IEEE Transactions on Pattern Analysis and Machine Intelligence, 11, N 7, pp. 701-716, 1989
    # L.Vincent "Granulometries and Opening Trees", Fundamenta Informaticae, 41, No. 1-2, pp. 57-90, IOS Press, 2000.
    # L.Vincent "Morphological Area Opening and Closing for Grayscale Images", Proc. NATO Shape in Picture Workshop, Driebergen, The Netherlands, pp. 197-208, 1992.
    # I.Ravkin, V.Temov "Bit representation techniques and image processing", Applied Informatics, v.14, pp. 41-90, Finances and Statistics, Moskow, 1988 (in Russian)
    # THIS IMPLEMENTATION INSTEAD OF OPENING USES EROSION FOLLOWED BY RECONSTRUCTION
    #
    ng = granular_spectrum_length
    startmean = np.mean(pixels[mask])
    ero = pixels.copy()
    # Mask the test image so that masked pixels will have no effect
    # during reconstruction
    #
    ero[~mask] = 0
    currentmean = startmean
    startmean = max(startmean, np.finfo(float).eps)

    if pixels.ndim == 2:
        footprint = skimage.morphology.disk(1, dtype=bool)
    else:
        footprint = skimage.morphology.ball(1, dtype=bool)
    results = []
    for i in range(1, ng + 1):
        prevmean = currentmean
        ero_mask = np.zeros_like(ero)
        ero_mask[mask == True] = ero[mask == True]
        ero = skimage.morphology.erosion(ero_mask, footprint=footprint)
        rec = skimage.morphology.reconstruction(ero, pixels, footprint=footprint)
        currentmean = np.mean(rec[mask])
        gs = (prevmean - currentmean) * 100 / startmean

        # TODO find better solution for the return
        # results[f"Granularity_{str(i)}"] = gs
        results.append(gs)
        # Restore the reconstructed image to the shape of the
        # original image so we can match against object labels
        #
        orig_shape = pixels.shape
        try:
            if pixels.ndim == 2:
                i, j = np.mgrid[0 : orig_shape[0], 0 : orig_shape[1]].astype(float)
                #
                # Make sure the mapping only references the index range of
                # back_pixels.
                #
                i *= float(new_shape[0] - 1) / float(orig_shape[0] - 1)
                j *= float(new_shape[1] - 1) / float(orig_shape[1] - 1)
                rec = scipy.ndimage.map_coordinates(rec, (i, j), order=1)
            else:
                k, i, j = np.mgrid[0 : orig_shape[0], 0 : orig_shape[1], 0 : orig_shape[2]].astype(float)
                k *= float(new_shape[0] - 1) / float(orig_shape[0] - 1)
                i *= float(new_shape[1] - 1) / float(orig_shape[1] - 1)
                j *= float(new_shape[2] - 1) / float(orig_shape[2] - 1)
                rec = scipy.ndimage.map_coordinates(rec, (k, i, j), order=1)

        # TODO Debug the reason for the ZeroDivisionError when using MIBI-TOF dataeset
        # from https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks/examples/technology_mibitof.html
        except ZeroDivisionError:
            return _handle_granularity_error(granular_spectrum_length)

        # TODO check if this is necessary
        # Calculate the means for the objects
        #
        # for object_record in object_records:
        #     assert isinstance(object_record, ObjectRecord)
        #     if object_record.nobjects > 0:
        #         new_mean = fix(
        #             scipy.ndimage.mean(
        #                 rec, object_record.labels, object_record.range
        #             )
        #         )
        #         gss = (
        #             (object_record.current_mean - new_mean)
        #             * 100
        #             / object_record.start_mean
        #         )
        #         object_record.current_mean = new_mean
        #     else:
        #         gss = np.zeros((0,))
        #     measurements.add_measurement(object_record.name, feature, gss)

    if len(results) == 1:
        return results[0]
    else:
        return results


def _handle_granularity_error(granular_spectrum_length: int):
    return [None for _ in range(granular_spectrum_length)]


# Inspired by https://github.com/afermg/cp_measure/blob/main/src/cp_measure/fast/measureobjectsizeshape.py
def zernike(masks: np.ndarray, pixels: np.ndarray, zernike_numbers: int = 9) -> dict[int, np.ndarray]:
    unique_indices = np.unique(masks)
    unique_indices = unique_indices[unique_indices != 0]
    res = centrosome.zernike.zernike(
        zernike_indexes=centrosome.zernike.get_zernike_indexes(zernike_numbers + 1),
        labels=masks,
        indexes=unique_indices,
    )
    return {orig_idx: res[res_idx] for orig_idx, res_idx in zip(unique_indices, range(res.shape[0]))}


# Copied from https://github.com/afermg/cp_measure/blob/main/src/cp_measure/fast/measureobjectintensitydistribution.py
def radial_distribution(
    labels: np.ndarray,
    pixels: np.ndarray,
    scaled: bool = True,
    bin_count: int = 4,
    maximum_radius: int = 100,
):
    """
    zernike_degree : int
        Maximum zernike moment.

        This is the maximum radial moment that will be calculated. There are
        increasing numbers of azimuthal moments as you increase the radial
        moment, so higher values are increasingly expensive to calculate.

    scaled : bool
        Scale the bins?

        When True divide the object radially into the number of bins
        that you specify. Otherwise create the number of bins you specify
        based on distance. If True, it will use a maximum distance so
        that each object will have the same measurements (which might be zero
        for small objects) and so that the measurements can be taken without
        knowing the maximum object radius before the run starts.

    bin_count : int
        Number of bins

        Specify the number of bins that you want to use to measure the distribution.
        Radial distribution is measured with respect to a series of concentric
        rings starting from the object center (or more generally, between contours
        at a normalized distance from the object center). This number specifies
        the number of rings into which the distribution is to be divided.

    maximum_radius : int
        Maximum radius

        Specify the maximum radius for the unscaled bins. The unscaled binning method
        creates the number of bins that you specify and creates equally spaced bin
        boundaries up to the maximum radius. Parts of the object that are beyond this
        radius will be counted in an overflow bin. The radius is measured in pixels.
    """

    if labels.dtype == bool:
        labels = labels.astype(np.integer)

    M_CATEGORY = "RadialDistribution"
    F_FRAC_AT_D = "FracAtD"
    F_MEAN_FRAC = "MeanFrac"
    F_RADIAL_CV = "RadialCV"

    FF_SCALE = "%dof%d"
    FF_OVERFLOW = "Overflow"
    FF_GENERIC = FF_SCALE

    MF_FRAC_AT_D = "_".join((M_CATEGORY, F_FRAC_AT_D, FF_GENERIC))
    MF_MEAN_FRAC = "_".join((M_CATEGORY, F_MEAN_FRAC, FF_GENERIC))
    MF_RADIAL_CV = "_".join((M_CATEGORY, F_RADIAL_CV, FF_GENERIC))
    OF_FRAC_AT_D = "_".join((M_CATEGORY, F_FRAC_AT_D, FF_OVERFLOW))
    OF_MEAN_FRAC = "_".join((M_CATEGORY, F_MEAN_FRAC, FF_OVERFLOW))
    OF_RADIAL_CV = "_".join((M_CATEGORY, F_RADIAL_CV, FF_OVERFLOW))

    unique_labels = np.unique(labels)
    n_objects = len(unique_labels[unique_labels > 0])
    d_to_edge = centrosome.cpmorphology.distance_to_edge(labels)

    # Find the point in each object farthest away from the edge.
    # This does better than the centroid:
    # * The center is within the object
    # * The center tends to be an interesting point, like the
    #   center of the nucleus or the center of one or the other
    #   of two touching cells.
    #
    # MODIFICATION: Delegated label indices to maximum_position_of_labels
    # This should not affect this one-mask/object function
    i, j = centrosome.cpmorphology.maximum_position_of_labels(
        # d_to_edge, labels, indices=[1]
        d_to_edge,
        labels,
        indices=[1],
    )

    center_labels = np.zeros(labels.shape, int)

    center_labels[i, j] = labels[i, j]

    #
    # Use the coloring trick here to process touching objects
    # in separate operations
    #
    colors = centrosome.cpmorphology.color_labels(labels)

    n_colors = np.max(colors)

    d_from_center = np.zeros(labels.shape)

    cl = np.zeros(labels.shape, int)

    for color in range(1, n_colors + 1):
        mask = colors == color
        l, d = centrosome.propagate.propagate(np.zeros(center_labels.shape), center_labels, mask, 1)

        d_from_center[mask] = d[mask]

        cl[mask] = l[mask]

    good_mask = cl > 0

    i_center = np.zeros(cl.shape)

    i_center[good_mask] = i[cl[good_mask] - 1]

    j_center = np.zeros(cl.shape)

    j_center[good_mask] = j[cl[good_mask] - 1]

    normalized_distance = np.zeros(labels.shape)

    if scaled:
        total_distance = d_from_center + d_to_edge

        normalized_distance[good_mask] = d_from_center[good_mask] / (total_distance[good_mask] + 0.001)
    else:
        normalized_distance[good_mask] = d_from_center[good_mask] / maximum_radius

    n_good_pixels = np.sum(good_mask)

    good_labels = labels[good_mask]

    bin_indexes = (normalized_distance * bin_count).astype(int)

    bin_indexes[bin_indexes > bin_count] = bin_count

    labels_and_bins = (good_labels - 1, bin_indexes[good_mask])

    histogram = scipy.sparse.coo_matrix((pixels[good_mask], labels_and_bins), (n_objects, bin_count + 1)).toarray()

    sum_by_object = np.sum(histogram, 1)

    sum_by_object_per_bin = np.dstack([sum_by_object] * (bin_count + 1))[0]

    fraction_at_distance = histogram / sum_by_object_per_bin

    number_at_distance = scipy.sparse.coo_matrix(
        (np.ones(n_good_pixels), labels_and_bins), (n_objects, bin_count + 1)
    ).toarray()

    sum_by_object = np.sum(number_at_distance, 1)

    sum_by_object_per_bin = np.dstack([sum_by_object] * (bin_count + 1))[0]

    fraction_at_bin = number_at_distance / sum_by_object_per_bin

    mean_pixel_fraction = fraction_at_distance / (fraction_at_bin + np.finfo(float).eps)

    # Anisotropy calculation.  Split each cell into eight wedges, then
    # compute coefficient of variation of the wedges' mean intensities
    # in each ring.
    #
    # Compute each pixel's delta from the center object's centroid
    i, j = np.mgrid[0 : labels.shape[0], 0 : labels.shape[1]]

    i_mask = i[good_mask] > i_center[good_mask]

    j_mask = j[good_mask] > j_center[good_mask]

    abs_mask = abs(i[good_mask] - i_center[good_mask]) > abs(j[good_mask] - j_center[good_mask])

    radial_index = i_mask.astype(int) + j_mask.astype(int) * 2 + abs_mask.astype(int) * 4

    results = {}

    for bin_idx in range(bin_count + (0 if scaled else 1)):
        bin_mask = good_mask & (bin_indexes == bin_idx)

        bin_pixels = np.sum(bin_mask)

        bin_labels = labels[bin_mask]

        bin_radial_index = radial_index[bin_indexes[good_mask] == bin_idx]

        labels_and_radii = (bin_labels - 1, bin_radial_index)

        radial_values = scipy.sparse.coo_matrix((pixels[bin_mask], labels_and_radii), (n_objects, 8)).toarray()

        pixel_count = scipy.sparse.coo_matrix((np.ones(bin_pixels), labels_and_radii), (n_objects, 8)).toarray()

        mask = pixel_count == 0

        radial_means = np.ma.masked_array(radial_values / pixel_count, mask)

        radial_cv = np.std(radial_means, 1) / np.mean(radial_means, 1)

        radial_cv[np.sum(~mask, 1) == 0] = 0

        for measurement, feature, overflow_feature in (
            (fraction_at_distance[:, bin_idx], MF_FRAC_AT_D, OF_FRAC_AT_D),
            (mean_pixel_fraction[:, bin_idx], MF_MEAN_FRAC, OF_MEAN_FRAC),
            (np.array(radial_cv), MF_RADIAL_CV, OF_RADIAL_CV),
        ):
            if bin_idx == bin_count:
                measurement_name = overflow_feature
            else:
                measurement_name = feature % (bin_idx + 1, bin_count)

            results[measurement_name] = measurement

    return results

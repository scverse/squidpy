from __future__ import annotations

import centrosome.zernike
import numpy
import numpy as np
import scipy.ndimage
import skimage.morphology


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
    ]
    return names


def border_occupied_factor(mask: numpy.ndarray, *args, **kwargs) -> dict[int, float]:
    """
    Calculates the percentage of border pixels that are in a 4-connected neighborhood of another label
    Takes ~1.7s/megapixels to calculate

    Parameters
    ----------
    mask: numpy.ndarray integer grayscale image with the labels
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


if __name__ == "__main__":
    import time

    from spatialdata.datasets import blobs

    # label_image = np.array([
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 1, 1, 1, 0, 0, 0, 0],
    #     [0, 1, 1, 1, 2, 2, 2, 0],
    #     [0, 1, 1, 1, 2, 2, 2, 0],
    #     [0, 0, 3, 3, 2, 2, 2, 0],
    #     [0, 0, 3, 3, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    # ])
    label_image = blobs(length=512).labels["blobs_labels"].values

    start_time = time.perf_counter()
    actual = border_occupied_factor(label_image)
    end_time = time.perf_counter()
    print(end_time - start_time)
    assert len(actual) == len(np.unique(label_image)) - 1
    # for idx, actual_value in enumerate(actual):
    #     assert actual_value == expected[idx]

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
    mask: numpy.ndarray,
    pixels: numpy.ndarray,
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
    new_shape = numpy.array(pixels.shape)
    if subsample_size < 1:
        new_shape = new_shape * subsample_size
        if pixels.ndim == 2:
            i, j = numpy.mgrid[0 : new_shape[0], 0 : new_shape[1]].astype(float) / subsample_size
            pixels = scipy.ndimage.map_coordinates(pixels, (i, j), order=1)
            mask = scipy.ndimage.map_coordinates(mask.astype(float), (i, j)) > 0.9
        else:
            k, i, j = numpy.mgrid[0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]].astype(float) / subsample_size
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
            i, j = numpy.mgrid[0 : back_shape[0], 0 : back_shape[1]].astype(float) / image_sample_size
            back_pixels = scipy.ndimage.map_coordinates(pixels, (i, j), order=1)
            back_mask = scipy.ndimage.map_coordinates(mask.astype(float), (i, j)) > 0.9
        else:
            k, i, j = numpy.mgrid[0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]].astype(float) / subsample_size
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
    back_pixels_mask = numpy.zeros_like(back_pixels)
    back_pixels_mask[back_mask == True] = back_pixels[back_mask == True]
    back_pixels = skimage.morphology.erosion(back_pixels_mask, footprint=footprint)
    back_pixels_mask = numpy.zeros_like(back_pixels)
    back_pixels_mask[back_mask == True] = back_pixels[back_mask == True]
    back_pixels = skimage.morphology.dilation(back_pixels_mask, footprint=footprint)
    try:
        if image_sample_size < 1:
            if pixels.ndim == 2:
                i, j = numpy.mgrid[0 : new_shape[0], 0 : new_shape[1]].astype(float)
                #
                # Make sure the mapping only references the index range of
                # back_pixels.
                #
                i *= float(back_shape[0] - 1) / float(new_shape[0] - 1)
                j *= float(back_shape[1] - 1) / float(new_shape[1] - 1)
                back_pixels = scipy.ndimage.map_coordinates(back_pixels, (i, j), order=1)
            else:
                k, i, j = numpy.mgrid[0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]].astype(float)
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
    startmean = numpy.mean(pixels[mask])
    ero = pixels.copy()
    # Mask the test image so that masked pixels will have no effect
    # during reconstruction
    #
    ero[~mask] = 0
    currentmean = startmean
    startmean = max(startmean, numpy.finfo(float).eps)

    if pixels.ndim == 2:
        footprint = skimage.morphology.disk(1, dtype=bool)
    else:
        footprint = skimage.morphology.ball(1, dtype=bool)
    results = []
    for i in range(1, ng + 1):
        prevmean = currentmean
        ero_mask = numpy.zeros_like(ero)
        ero_mask[mask == True] = ero[mask == True]
        ero = skimage.morphology.erosion(ero_mask, footprint=footprint)
        rec = skimage.morphology.reconstruction(ero, pixels, footprint=footprint)
        currentmean = numpy.mean(rec[mask])
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
                i, j = numpy.mgrid[0 : orig_shape[0], 0 : orig_shape[1]].astype(float)
                #
                # Make sure the mapping only references the index range of
                # back_pixels.
                #
                i *= float(new_shape[0] - 1) / float(orig_shape[0] - 1)
                j *= float(new_shape[1] - 1) / float(orig_shape[1] - 1)
                rec = scipy.ndimage.map_coordinates(rec, (i, j), order=1)
            else:
                k, i, j = numpy.mgrid[0 : orig_shape[0], 0 : orig_shape[1], 0 : orig_shape[2]].astype(float)
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
        #         gss = numpy.zeros((0,))
        #     measurements.add_measurement(object_record.name, feature, gss)

    if len(results) == 1:
        return results[0]
    else:
        return results


def _handle_granularity_error(granular_spectrum_length: int):
    return [None for _ in range(granular_spectrum_length)]


# Copied from https://github.com/afermg/cp_measure/blob/main/src/cp_measure/fast/measureobjectsizeshape.py
def zernike(masks: numpy.ndarray, pixels: numpy.ndarray, zernike_numbers: int = 9) -> dict[int, numpy.ndarray]:
    unique_indices = numpy.unique(masks)
    unique_indices = unique_indices[unique_indices != 0]
    res = centrosome.zernike.zernike(
        zernike_indexes=centrosome.zernike.get_zernike_indexes(zernike_numbers + 1),
        labels=masks,
        indexes=unique_indices,
    )
    return {orig_idx: res[res_idx] for orig_idx, res_idx in zip(unique_indices, range(res.shape[0]))}

    # #
    # # Zernike features
    # #
    # unique_indices = numpy.unique(masks)
    # unique_indices = unique_indices[unique_indices>0]
    # indices = list(range(1,len(unique_indices) + 1))
    # labels = masks
    # zernike_numbers = centrosome.zernike.get_zernike_indexes(zernike_numbers + 1)
    #
    # zf_l = centrosome.zernike.zernike(zernike_numbers, labels, indices)
    # results = {}
    # for (n, m), z in zip(zernike_numbers, zf_l.transpose()):
    #     results[f"Zernike_{n}_{m}"] = z

    # return results

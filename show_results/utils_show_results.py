import os
import SimpleITK as sitk
import numpy as np
from difflib import SequenceMatcher
import warnings


def get_result_filename(dirname):
    """Find the filename of the result coordinate file.

    This should be result.txt  If this file is not present,
    it tries to find the closest filename."""

    files = os.listdir(dirname)

    if not files:
        raise Exception("No results in " + dirname)

    # Find the filename that is closest to 'result.txt'
    ratios = [SequenceMatcher(a=f, b='result.txt').ratio() for f in files]
    result_filename = files[int(np.argmax(ratios))]

    # Return the full path to the file.
    return os.path.join(dirname, result_filename)


def get_locations(test_filename):
    """Return the locations and radius of actual aneurysms as a NumPy array"""

    # Read comma-separated coordinates from a text file.
    with warnings.catch_warnings():
        # Suppress empty file warning from genfromtxt.
        warnings.filterwarnings("ignore", message=".*Empty input file.*")

        # atleast_2d() makes sure that test_locations is a 2D array, even if there is just a single location.
        # genfromtxt() raises a ValueError if the number of columns is inconsistent.
        test_locations = np.atleast_2d(np.genfromtxt(test_filename, delimiter=',', encoding='utf-8-sig'))

    # Reshape an empty result into a 0x4 array.
    if test_locations.size == 0:
        test_locations = test_locations.reshape(0, 4)

    # DEBUG: verify that the inner dimension size is 4.
    assert test_locations.shape[1] == 4

    return test_locations


def get_result(result_filename):
    """Read Result file and extract coordinates as a NumPy array"""

    # Read comma-separated coordinates from a text file.
    with warnings.catch_warnings():
        # Suppress empty file warning from genfromtxt.
        warnings.filterwarnings("ignore", message=".*Empty input file.*")

        # atleast_2d() makes sure that test_locations is a 2D array, even if there is just a single location.
        # genfromtxt() raises a ValueError if the number of columns is inconsistent.
        result_locations = np.atleast_2d(np.genfromtxt(result_filename, delimiter=',', encoding='utf-8-sig'))

    # Reshape an empty result into a 0x3 array.
    if result_locations.size == 0:
        result_locations = result_locations.reshape(0, 3)

    # DEBUG: verify that the inner dimension size is 3.
    assert result_locations.shape[1] == 3

    return result_locations


def get_treated_locations(test_image):
    """Return an array with a list of locations of treated aneurysms(based on aneurysms.nii.gz)"""
    treated_image = test_image > 1.5
    treated_array = sitk.GetArrayFromImage(treated_image)

    if np.sum(treated_array) == 0:
        # no treated aneurysms
        return np.array([])

    # flip so (x,y,z)
    treated_coords = np.flip(np.nonzero(treated_array))

    return np.array(list(zip(*treated_coords)))


def get_detection_metrics_for_conf_int(test_locations, result_locations, test_image):
    """Calculate sensitivity and false positive count for each image.

    The distance between every result-location and test-locations must be less
    than the radius."""

    test_radii = test_locations[:, -1]

    # Transform the voxel coordinates into physical coordinates. TransformContinuousIndexToPhysicalPoint handles
    # sub-voxel (i.e. floating point) indices.
    test_coords = np.array([
        test_image.TransformContinuousIndexToPhysicalPoint(coord[:3]) for coord in test_locations.astype(float)])
    pred_coords = np.array([
        test_image.TransformContinuousIndexToPhysicalPoint(coord) for coord in result_locations.astype(float)])
    treated_locations = get_treated_locations(test_image)
    treated_coords = np.array([
        test_image.TransformContinuousIndexToPhysicalPoint(coord.astype(float)) for coord in treated_locations.astype(float)])

    # Reshape empty arrays into 0x3 arrays.
    if test_coords.size == 0:
        test_coords = test_coords.reshape(0, 3)
    if pred_coords.size == 0:
        pred_coords = pred_coords.reshape(0, 3)

    # True positives lie within radius of true aneurysm. Only count one true positive per aneurysm.
    true_positives = 0
    for location, radius in zip(test_coords, test_radii):
        detected = False
        for detection in pred_coords:
            distance = np.linalg.norm(detection - location)
            if distance <= radius:
                detected = True
        if detected:
            true_positives += 1

    false_positives = 0
    for detection in pred_coords:
        found = False
        if detection in treated_coords:
            continue
        for location, radius in zip(test_coords, test_radii):
            distance = np.linalg.norm(location - detection)
            if distance <= radius:
                found = True
        if not found:
            false_positives += 1

    if len(test_locations) == 0:
        sensitivity = np.nan
        cnt = np.nan
        true_positives = np.nan
    else:
        cnt = len(test_locations)
        sensitivity = true_positives / cnt

    return sensitivity, false_positives, true_positives, cnt


def detection_one_sub_for_conf_int(result_filename, ground_truth_folder):
    assert os.path.exists(result_filename), "Path {0} does not exist".format(result_filename)
    assert os.path.exists(ground_truth_folder), "Path {0} does not exist".format(ground_truth_folder)

    test_locations = get_locations(os.path.join(ground_truth_folder, 'location.txt'))
    result_locations = get_result(result_filename)
    test_image = sitk.ReadImage(os.path.join(ground_truth_folder, 'aneurysms.nii.gz'))

    sensitivity, false_positive_count, tp, cnt = get_detection_metrics_for_conf_int(test_locations, result_locations, test_image)

    print("Sens: {}; FP = {}".format(sensitivity, false_positive_count))

    return np.asarray([false_positive_count, sensitivity, tp, cnt])


import os
import sys
import SimpleITK as sitk
import numpy as np
from difflib import SequenceMatcher
import warnings
from joblib import Parallel, delayed
import re
import cc3d
from scipy import ndimage
from sklearn.metrics import auc
from statsmodels.stats.proportion import proportion_confint
from scipy.ndimage.measurements import center_of_mass
from inference.utils_inference import round_half_up


__author__ = "Tommaso Di Noto"
__version__ = "0.0.1"
__email__ = "tommydino@hotmail.it"
__status__ = "Prototype"


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


def extract_unique_elements(lst: list, ordered=True) -> list:
    """This function extracts the unique elements of the input list and returns them as an output list; by defualt, the returned list is ordered.
    Args:
        lst (list): input list from which we want to extract the unique elements
        ordered (bool): whether the output list of unique values is sorted or not; True by default
    Returns:
        out_list (list): list containing unique values
    """
    out_list = list(set(lst))  # type: list

    if ordered:  # if we want to sort the list of unique values
        out_list.sort()  # type: list

    return out_list


def get_detection_metrics(test_locations, result_locations, test_image):
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
    else:
        sensitivity = true_positives / len(test_locations)

    return sensitivity, false_positives


def get_center_of_mass(result_image):
    """Based on result segmentation, find coordinate of centre of mass of predicted aneurysms."""
    result_array = sitk.GetArrayFromImage(result_image)
    if np.sum(result_array) == 0:
        # no detections
        return np.ndarray((0, 3))

    structure = ndimage.generate_binary_structure(rank=result_array.ndim, connectivity=result_array.ndim)

    label_array = ndimage.label(result_array, structure)[0]
    index = np.unique(label_array)[1:]

    # Get locations in x, y, z order.
    locations = np.fliplr(ndimage.measurements.center_of_mass(result_array, label_array, index))
    return locations


def sanity_checks_images(test_image, result_image):
    assert test_image.GetSize() == result_image.GetSize()

    # Get meta data from the test-image, needed for some sitk methods that check this
    result_image.CopyInformation(test_image)

    # Remove treated aneurysms from the test and result images, since we do not evaluate on this
    treated_image = test_image != 2  # treated aneurysms == 2
    masked_result_image = sitk.Mask(result_image, treated_image)
    masked_test_image = sitk.Mask(test_image, treated_image)

    # Return two binary masks
    return masked_test_image > 0.5, masked_result_image > 0.5


def sanity_checks_partial_binary_mask(binary_segm_map, idx):

    # check that mask in binary and non-empty
    if not np.array_equal(binary_segm_map, binary_segm_map.astype(bool)) or np.sum(binary_segm_map) == 0:
        raise ValueError("Mask must be binary and non-empty because we are looping only over patients with aneurysms")

    # check that there are exactly idx+1 connected components
    labels_out = cc3d.connected_components(binary_segm_map)
    numb_labels = np.max(labels_out)  # extract number of different connected components found
    assert numb_labels == idx+1


def sort_dict_by_value(d, reverse=False):
    """This function sorts the input dictionary by value; the argument bool is used to decide between ascending or descending order.
    Args:
        d (dict): input dictionary that we want to sort
        reverse (bool): if True, the dict is sorted in descending value order; if False, the dict is sorted in ascending value order
    Returns:
        sorted_dict = sorted dictionary
    """
    assert sys.version_info >= (3, 6), "This function only works with python >= 3.6"
    sorted_dict = dict(sorted(d.items(), key=lambda x: x[1], reverse=reverse))
    return sorted_dict


def sort_dict_by_list_of_keys(input_dict, list_of_keys):
    """This function sorts the input dict according to a list of keys (also given as input).
    Args:
        input_dict (dict): input dictionary that we want to sort
        list_of_keys (list): list of keys; the order of the keys in the list is the exact same order of (key, value) that we want in the sorted dict
    Returns:
        reordered_dict (dict): sorted dict according to the key values in list_of_keys
    """
    reordered_dict = {k: input_dict[k] for k in list_of_keys if k in input_dict.keys()}
    return reordered_dict


def create_mapping_centers_avg_brightness(numb_labels, probabilistic_prediction, labels_out):

    mappings_centers_avg_brightness = {}  # type: dict # it contains the centers of the connected components as keys and the average birghtness of the candidate prediction as value
    mappings_centers_nonzero_coords = {}
    # loop over different conn. components
    for seg_id in range(1, numb_labels + 1):
        # extract connected component from the probabilistic predicted volume
        extracted_image = probabilistic_prediction * (labels_out == seg_id)

        # extract voxel coordinates of the center of the predicted lesion
        predicted_center = center_of_mass(extracted_image)  # type: tuple
        # round to closest int
        pred_center_tof_space = [round_half_up(x) for x in predicted_center]  # type: list
        pred_center_tof_space_int = np.asarray(pred_center_tof_space, dtype=int)  # type: np.ndarray # cast to int

        # compute average intensity of this predicted aneurysm
        avg_brightness_prob_conn_comp = np.mean(extracted_image[np.nonzero(extracted_image)])

        mappings_centers_avg_brightness[tuple(pred_center_tof_space_int)] = avg_brightness_prob_conn_comp
        mappings_centers_nonzero_coords[tuple(pred_center_tof_space_int)] = np.nonzero(extracted_image)

    return mappings_centers_avg_brightness, mappings_centers_nonzero_coords


def get_ground_truth_image(test_filename):
    """Return the test and result images"""
    test_image = sitk.ReadImage(test_filename)

    # Return two binary masks
    return test_image


def compute_fp_and_sens(prediction_dir_path, ground_truth_dir_path, max_nb_fp_per_subject):
    probabilistic_prediction_path = os.path.join(prediction_dir_path, "probabilistic_result.nii.gz")
    probabilistic_prediction_sitk = sitk.ReadImage(probabilistic_prediction_path)
    probabilistic_prediction = sitk.GetArrayFromImage(probabilistic_prediction_sitk)
    probabilistic_prediction_binary = np.asarray(np.where(probabilistic_prediction > 0, 1, 0), dtype=int)

    test_image = get_ground_truth_image(os.path.join(ground_truth_dir_path, "aneurysms.nii.gz"))
    test_locations = get_locations(os.path.join(ground_truth_dir_path, 'location.txt'))

    # extract 3D connected components
    labels_out = cc3d.connected_components(np.asarray(probabilistic_prediction_binary, dtype=int))
    numb_labels = np.max(labels_out)  # extract number of different connected components found

    # if there is at least one connected component in the prediction
    if numb_labels >= 1:
        mappings_centers_avg_brightness, mappings_centers_nonzero_coords = create_mapping_centers_avg_brightness(numb_labels, probabilistic_prediction, labels_out)

        # sort dictionary such that the most probable connected component is the first one, and so on
        sorted_mappings_centers_avg_brightness = sort_dict_by_value(mappings_centers_avg_brightness, reverse=True)
        desired_key_order = list(sorted_mappings_centers_avg_brightness.keys())
        sorted_mappings_centers_nonzero_coords = sort_dict_by_list_of_keys(mappings_centers_nonzero_coords, desired_key_order)

        sensitivities_one_sub = []
        false_positives_one_sub = []

        if len(sorted_mappings_centers_nonzero_coords) > 5:  # if there are more than 5 predicted connected components
            fp_range = max_nb_fp_per_subject + 1
        else:
            fp_range = len(sorted_mappings_centers_nonzero_coords)

        # loop over connected components; at each iteration, we retain one more FP wrt the previous iteration (e.g., iter1: we retain 1; iter 2: we retain 2, etc)
        for idx in range(fp_range):
            trimmed_connected_components_nonzero_coords = dict(list(sorted_mappings_centers_nonzero_coords.items())[:idx + 1])

            # initialize empty binary mask; this will increasingly include more connected components (e.g. 1 cc in the first iteration, 2 cc in the second, etc)
            binary_segm_map = np.zeros(probabilistic_prediction_binary.shape, dtype=int)

            # loop over retained connected components
            for centers, nonzero_coords in trimmed_connected_components_nonzero_coords.items():
                binary_segm_map[nonzero_coords] = 1  # fill with 1 the retained connected components

            sanity_checks_partial_binary_mask(binary_segm_map, idx)  # make sure that partial binary mask is correct

            # begin computation of sensitivity and FP as performed by the ADAM challenge organizers
            result_image = sitk.GetImageFromArray(binary_segm_map)
            test_image, result_image = sanity_checks_images(test_image, result_image)
            result_locations = get_center_of_mass(result_image)
            sensitivity, false_positives = get_detection_metrics(test_locations, result_locations, test_image)
            sensitivities_one_sub.append(sensitivity)
            false_positives_one_sub.append(false_positives)

    # if instead there is no connected component in the prediction but the subject has aneurysms
    elif numb_labels == 0 and not os.stat(os.path.join(ground_truth_dir_path, 'location.txt')).st_size == 0:
        sensitivities_one_sub = [0.]
        false_positives_one_sub = [0]
    else:
        raise ValueError("Check this subject. Something is wrong")

    return [false_positives_one_sub, sensitivities_one_sub]


def froc_param_one_sub(prediction_dir_path, ground_truth_dir_path, max_nb_fp_per_subject, cnt):
    sub = re.findall(r"sub-\d+", ground_truth_dir_path)[0]  # extract sub
    ses = re.findall(r"ses-\w{6}\d+", ground_truth_dir_path)[0]  # extract ses

    output_list = compute_fp_and_sens(prediction_dir_path, ground_truth_dir_path, max_nb_fp_per_subject)
    print("{}) {}_{} Increasing connected components [FP, Sens] = {}".format(cnt+1, sub, ses, output_list))

    return output_list


def extract_aufrocs(out_metrics_list, max_nb_fp_per_subject):

    aufrocs_distribution = []
    for sub_ses_metrics in out_metrics_list:  # loop over every sub-ses
        sens_vector = np.asarray(sub_ses_metrics[1])  # type: np.ndarray # extract sensitivities

        # pad vectors with last value (i.e., with maximum achieved sensitivity) only to the right
        sens_vector_padded = np.pad(np.asarray(sens_vector, dtype=float), (0, max_nb_fp_per_subject), 'constant', constant_values=np.asarray(sens_vector, dtype=float)[-1])

        # only take the first 5 values (we trim the FROC curves at 5 FP per subject)
        sens_vector_padded_trimmed = sens_vector_padded[:max_nb_fp_per_subject]

        # add zero because the froc curve starts from (0,0)
        sens_vector_padded_trimmed = np.insert(sens_vector_padded_trimmed, 0, 0)

        # create fp x-axis
        fp_axis = np.arange(0, len(sens_vector_padded_trimmed), 1)

        if np.isnan(sens_vector_padded_trimmed).any():  # if there are nans
            nan_idxs = list(np.argwhere(np.isnan(sens_vector_padded_trimmed)).flatten())  # find idxs of nans
            sensitivities_without_nans = np.delete(sens_vector_padded_trimmed, nan_idxs)  # remove nans
            fp_without_nans = np.delete(fp_axis, nan_idxs)  # also remove nans from fp vector
            aufroc_model = auc(fp_without_nans, sensitivities_without_nans)
        else:  # if instead there are no nans
            aufroc_model = auc(fp_axis, sens_vector_padded_trimmed)

        aufrocs_distribution.append(aufroc_model)

    return aufrocs_distribution


def compute_areas_under_froc_curve(prediction_dir_model, ground_truth_dir, nb_parallel_jobs, max_nb_fp_per_subject):
    prediction_paths = []
    ground_truth_paths = []

    for fold in sorted(os.listdir(prediction_dir_model)):  # loop over all files
        if os.path.isdir(os.path.join(prediction_dir_model, fold)) and "fold" in fold:  # if it is a folder
            for sub in sorted(os.listdir(os.path.join(prediction_dir_model, fold))):  # loop over all subjects in output folder
                if os.path.isdir(os.path.join(prediction_dir_model, fold, sub)):
                    for ses in sorted(os.listdir(os.path.join(prediction_dir_model, fold, sub))):
                        if os.path.isdir(os.path.join(prediction_dir_model, fold, sub, ses)):
                            location_txt_path = os.path.join(ground_truth_dir, sub, ses, "location.txt")
                            assert os.path.exists(location_txt_path), "Path to location ground truth file doesn't exist for {}_{}".format(sub, ses)

                            # if the txt file is not empty (i.e. if the subject has one or more aneurysm)
                            if not os.stat(location_txt_path).st_size == 0:
                                prediction_paths.append(os.path.join(prediction_dir_model, fold, sub, ses))
                                ground_truth_paths.append(os.path.join(ground_truth_dir, sub, ses))

    out_metrics_list = Parallel(n_jobs=nb_parallel_jobs, backend='threading')(delayed(froc_param_one_sub)(prediction_paths[idx],
                                                                                                          ground_truth_paths[idx],
                                                                                                          max_nb_fp_per_subject,
                                                                                                          idx) for idx in range(len(prediction_paths)))

    # compute a distribution of aufrocs (one for every sub-ses)
    aufrocs_distribution = extract_aufrocs(out_metrics_list, max_nb_fp_per_subject)

    return aufrocs_distribution


def get_detection_metrics_multiple_frocs_conf_int(test_locations, result_locations, test_image):
    """Calculate sensitivity and false positive count for each image.

    The distance between every result-location and test-locations must be less
    than the radius."""

    test_radii = test_locations[:, -1]

    # Transform the voxel coordinates into physical coordinates. TransformContinuousIndexToPhysicalPoint handles
    # sub-voxel (i.e. floating point) indices.
    test_coords = np.array([test_image.TransformContinuousIndexToPhysicalPoint(coord[:3]) for coord in test_locations.astype(float)])
    pred_coords = np.array([test_image.TransformContinuousIndexToPhysicalPoint(coord) for coord in result_locations.astype(float)])
    treated_locations = get_treated_locations(test_image)
    treated_coords = np.array([test_image.TransformContinuousIndexToPhysicalPoint(coord.astype(float)) for coord in treated_locations.astype(float)])

    # Reshape empty arrays into 0x3 arrays.
    if test_coords.size == 0:
        test_coords = test_coords.reshape(0, 3)
    if pred_coords.size == 0:
        pred_coords = pred_coords.reshape(0, 3)

    # True positives lie within radius of true aneurysm. Only count one true positive per aneurysm.
    cnt = 0

    true_positives = 0
    for location, radius in zip(test_coords, test_radii):
        cnt += 1
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
    else:
        sensitivity = true_positives / len(test_locations)

    return sensitivity, false_positives, cnt, true_positives


def froc_param_one_sub_with_conf_int(prediction_dir_path, ground_truth_dir_path, max_nb_fp_per_subject, cnt):
    sub = re.findall(r"sub-\d+", ground_truth_dir_path)[0]  # extract sub
    ses = re.findall(r"ses-\w{6}\d+", ground_truth_dir_path)[0]  # extract ses
    print("{}) {}_{}".format(cnt+1, sub, ses))

    probabilistic_prediction_path = os.path.join(prediction_dir_path, "probabilistic_result.nii.gz")
    probabilistic_prediction_sitk = sitk.ReadImage(probabilistic_prediction_path)
    probabilistic_prediction = sitk.GetArrayFromImage(probabilistic_prediction_sitk)
    probabilistic_prediction_binary = np.asarray(np.where(probabilistic_prediction > 0, 1, 0), dtype=int)

    test_image = get_ground_truth_image(os.path.join(ground_truth_dir_path, "aneurysms.nii.gz"))
    test_locations = get_locations(os.path.join(ground_truth_dir_path, 'location.txt'))

    # extract 3D connected components
    labels_out = cc3d.connected_components(np.asarray(probabilistic_prediction_binary, dtype=int))
    numb_labels = np.max(labels_out)  # extract number of different connected components found

    # if there is at least one connected component in the prediction
    if numb_labels >= 1:
        mappings_centers_avg_brightness, mappings_centers_nonzero_coords = create_mapping_centers_avg_brightness(numb_labels, probabilistic_prediction, labels_out)

        # sort dictionary such that the most probable connected component is the first one, and so on
        sorted_mappings_centers_avg_brightness = sort_dict_by_value(mappings_centers_avg_brightness, reverse=True)
        desired_key_order = list(sorted_mappings_centers_avg_brightness.keys())
        sorted_mappings_centers_nonzero_coords = sort_dict_by_list_of_keys(mappings_centers_nonzero_coords, desired_key_order)

        sensitivities_one_sub = []
        false_positives_one_sub = []
        cnts_one_sub = []
        tps_one_sub = []

        # set a maximum cap for fp_range, cause in any case we are not interested in more than 5 FP per subject
        if len(sorted_mappings_centers_nonzero_coords) > 5:
            fp_range = max_nb_fp_per_subject + 1
        else:
            fp_range = len(sorted_mappings_centers_nonzero_coords)

        # loop over connected components; at each iteration, we retain one more FP wrt the previous iteration (e.g., iter1: we retain 1; iter 2: we retain 2, etc)
        for idx in range(fp_range):
            trimmed_connected_components_nonzero_coords = dict(list(sorted_mappings_centers_nonzero_coords.items())[:idx + 1])

            # initialize empty binary mask; this will increasingly include more connected components (e.g. 1 cc in the first iteration, 2 cc in the second, etc)
            binary_segm_map = np.zeros(probabilistic_prediction_binary.shape, dtype=int)

            # loop over retained connected components
            for centers, nonzero_coords in trimmed_connected_components_nonzero_coords.items():
                binary_segm_map[nonzero_coords] = 1  # fill with 1 the retained connected components

            sanity_checks_partial_binary_mask(binary_segm_map, idx)  # make sure that partial binary mask is correct

            # begin ADAM computation of sensitivity and FP
            result_image = sitk.GetImageFromArray(binary_segm_map)
            test_image, result_image = sanity_checks_images(test_image, result_image)
            result_locations = get_center_of_mass(result_image)
            sensitivity, false_positives, cnt, tp = get_detection_metrics_multiple_frocs_conf_int(test_locations, result_locations, test_image)
            sensitivities_one_sub.append(sensitivity)
            false_positives_one_sub.append(false_positives)
            cnts_one_sub.append(cnt)
            tps_one_sub.append(tp)

    # if instead there is no connected component in the prediction but the subject has aneurysms
    elif numb_labels == 0 and not os.stat(os.path.join(ground_truth_dir_path, 'location.txt')).st_size == 0:
        sensitivities_one_sub = [0.]
        false_positives_one_sub = [0]
        cnts_one_sub = [test_locations.shape[0]]
        tps_one_sub = [0]
    else:
        raise ValueError("Check this subject. Something is wrong")

    return false_positives_one_sub, sensitivities_one_sub, cnts_one_sub, tps_one_sub


def extract_mean_sensitivities_modified(out_metrics_list, max_nb_fp_per_subject):
    all_sensitivities_list = []
    all_cnts_list = []
    all_tps_list = []
    for sub_ses_metrics in out_metrics_list:  # loop over every sub-ses
        sens_vector = np.asarray(sub_ses_metrics[1])  # type: np.ndarray # extract sensitivities
        cnts_vector = np.asarray(sub_ses_metrics[2])  # type: np.ndarray # extract cnts
        tps_vector = np.asarray(sub_ses_metrics[3])  # type: np.ndarray # extract tps

        # pad vectors with last value (i.e., with maximum achieved sensitivity) only to the right
        sens_vector_padded = np.pad(np.asarray(sens_vector, dtype=float), (0, max_nb_fp_per_subject), 'constant', constant_values=np.asarray(sens_vector, dtype=float)[-1])
        cnts_vector_padded = np.pad(np.asarray(cnts_vector, dtype=float), (0, max_nb_fp_per_subject), 'constant', constant_values=np.asarray(cnts_vector, dtype=float)[-1])
        tps_vector_padded = np.pad(np.asarray(tps_vector, dtype=float), (0, max_nb_fp_per_subject), 'constant', constant_values=np.asarray(tps_vector, dtype=float)[-1])

        # only take the first 5 values (we trim the FROC curves at 5 FP per subject)
        sens_vector_padded_trimmed = sens_vector_padded[:max_nb_fp_per_subject]
        cnts_vector_padded_trimmed = cnts_vector_padded[:max_nb_fp_per_subject]
        tps_vector_padded_trimmed = tps_vector_padded[:max_nb_fp_per_subject]
        all_sensitivities_list.append(sens_vector_padded_trimmed)
        all_cnts_list.append(cnts_vector_padded_trimmed)
        all_tps_list.append(tps_vector_padded_trimmed)

    all_sensitivities_np = np.asarray(all_sensitivities_list)
    mean_sensitivities = np.nanmean(all_sensitivities_np, axis=0)

    all_cnts_np = np.asarray(all_cnts_list)
    all_tps_np = np.asarray(all_tps_list)

    lower_bound_list = []
    upper_bound_list = []

    for column in range(all_cnts_np.shape[1]):
        cnts = int(np.sum(all_cnts_np[:, column]))
        tps = int(np.sum(all_tps_np[:, column]))

        conf_int = proportion_confint(count=tps,
                                      nobs=cnts,
                                      alpha=0.05,
                                      method='wilson')  # compute 95% Wilson CI

        lower_bound = conf_int[0]
        upper_bound = conf_int[1]

        lower_bound_list.append(lower_bound)
        upper_bound_list.append(upper_bound)

    return mean_sensitivities, np.asarray(lower_bound_list), np.asarray(upper_bound_list)


def compute_sensitivities_froc_curve_with_conf_int(prediction_dir_model, ground_truth_dir, nb_parallel_jobs, max_nb_fp_per_subject):
    prediction_paths = []
    ground_truth_paths = []

    for fold in sorted(os.listdir(prediction_dir_model)):  # loop over all files
        if os.path.isdir(os.path.join(prediction_dir_model, fold)) and "fold" in fold:  # if it is a folder
            for sub in sorted(os.listdir(os.path.join(prediction_dir_model, fold))):  # loop over all subjects in output folder
                if os.path.isdir(os.path.join(prediction_dir_model, fold, sub)):
                    for ses in sorted(os.listdir(os.path.join(prediction_dir_model, fold, sub))):
                        if os.path.isdir(os.path.join(prediction_dir_model, fold, sub, ses)):
                            location_txt_path = os.path.join(ground_truth_dir, sub, ses, "location.txt")

                            # if the txt file is not empty (i.e. if the subject has one or more aneurysm)
                            if not os.stat(location_txt_path).st_size == 0:
                                prediction_paths.append(os.path.join(prediction_dir_model, fold, sub, ses))
                                ground_truth_paths.append(os.path.join(ground_truth_dir, sub, ses))

    out_metrics_list = Parallel(n_jobs=nb_parallel_jobs, backend='threading')(delayed(froc_param_one_sub_with_conf_int)(prediction_paths[idx],
                                                                                                                        ground_truth_paths[idx],
                                                                                                                        max_nb_fp_per_subject,
                                                                                                                        idx) for idx in range(len(prediction_paths)))

    mean_sensitivities, lower_bounds, upper_bounds = extract_mean_sensitivities_modified(out_metrics_list, max_nb_fp_per_subject)

    return mean_sensitivities, lower_bounds, upper_bounds

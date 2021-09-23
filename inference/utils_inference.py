import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import csv
from ants import apply_transforms_to_points
import cc3d
from scipy.ndimage.measurements import center_of_mass
import math
import tensorflow as tf
import warnings
from difflib import SequenceMatcher
import re
import scipy.spatial
from joblib import Parallel, delayed
import shutil
from skimage.measure import regionprops
import cv2
import pickle
import argparse
import json
import time


def create_input_lists(bids_dir):
    """This function creates the input lists that are used to run the patient-wise analysis in parallel
    Args:
        bids_dir (str): path to BIDS dataset folder
    Returns:
        all_subdirs (list): list of subdirs (i.e. path to parent dir of n4bfc angio volumes)
        all_files (list): list of filenames of n4bfc angio volumes
    """
    regexp_sub = re.compile(r'sub')  # create a substring template to match
    ext_gz = '.gz'  # type: str # set zipped files extension
    all_subdirs = []  # type: list # it will be the input list for the parallel computation
    all_files = []  # type: list # it will be the second input list for the parallel computation
    for subdir, dirs, files in os.walk(bids_dir):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()  # get the file extension
            # only retain paths of skull-stripped N4 bias field corrected volumes
            if regexp_sub.search(file) and ext in ext_gz and "N4bfc_brain_mask" in file and "N4_bias_field_corrected" in subdir:
                all_subdirs.append(subdir)
                all_files.append(file)

    assert all_subdirs and all_files, "Input lists must be non-empty"

    return all_subdirs, all_files


def extract_registration_quality_metrics(bids_ds_path, sub_ses_test):
    """This function checks the registration quality metrics for all patients and saves some thresholds. If during the patient-wise one subject has registration quality
     values which are outliers, the sliding-window approach for that subject will not be anatomically-informed. Instead, the whole brain will be scanned.
     In general, the problematic registration is the struct_2_tof one (and not the mni_2_struct), because the TOF volume sometimes is cut.
     Args:
         bids_ds_path (str): path to BIDS dataset
         sub_ses_test (list): sub_ses of the test set; we use it to take only the sub_ses of the training set
     Returns:
        p3_neigh_corr_struct_2_tof (float): the 3th percentile of the Neighborhood Correlation quality metric distribution (of the struct_2_tof registration)
        p97_mut_inf_struct_2_tof (float): the 97th percentile of the Mattes Mutual Information quality metric distribution (of the struct_2_tof registration)
     """
    reg_metrics_dir = os.path.join(bids_ds_path, "derivatives", "registrations", "reg_metrics")
    all_struct_2_tof_neigh_corr = []
    all_struct_2_tof_mutual_inf = []
    for sub in os.listdir(os.path.join(reg_metrics_dir)):
        for ses in os.listdir(os.path.join(reg_metrics_dir, sub)):
            sub_ses = "{}_{}".format(sub, ses)
            if sub_ses not in sub_ses_test:  # only use training ones
                for files in os.listdir(os.path.join(reg_metrics_dir, sub, ses)):
                    if "mni2struct" in files:
                        pass
                    elif "struct2tof" in files:
                        struct_2_tof_metrics = pd.read_csv(os.path.join(reg_metrics_dir, sub, ses, files))
                        all_struct_2_tof_neigh_corr.append(struct_2_tof_metrics.iloc[0]['neigh_corr'])
                        all_struct_2_tof_mutual_inf.append(struct_2_tof_metrics.iloc[0]['mut_inf'])
                    else:
                        raise ValueError("Unknown filename")

    p3_neigh_corr_struct_2_tof = np.percentile(all_struct_2_tof_neigh_corr, [3])[0]
    p97_mut_inf_struct_2_tof = np.percentile(all_struct_2_tof_mutual_inf, [97])[0]

    return p3_neigh_corr_struct_2_tof, p97_mut_inf_struct_2_tof


def extract_lesion_info(path_to_lesion: str, prints=False):
    """This function takes as input the path of a binary volumetric mask, loops through the slices which have some non-zero pixels and returns some information about the entire lesion: i.e.
    number of slices enclosing the lesion, index of slice with more white pixels, the equivalent diameter of the lesion width in that specific slice, and the coordinates of the
    centroid of the lesion. If "prints" is set to True, the method also prints this information."""

    lesion_info = {}  # initialize empty dict; this will be the output of the function
    lesion_obj = nib.load(path_to_lesion)  # load lesion volume as nibabel object
    lesion_volume = lesion_obj.get_fdata()  # convert from object to image (i.e. numpy array)
    if len(lesion_volume.shape) != 3:  # if the numpy array is not 3D
        lesion_volume = np.squeeze(lesion_volume, axis=3)  # we drop the fourth dimension (time dimension) which is useless in our case

    assert np.array_equal(lesion_volume, lesion_volume.astype(bool)), "WATCH OUT: mask is not binary for {0}".format(path_to_lesion)
    assert np.count_nonzero(lesion_volume) > 0, "WATCH OUT: mask is empty (i.e. all zero-voxels)"

    labels_out = cc3d.connected_components(np.asarray(lesion_volume, dtype=int))
    numb_labels = np.max(labels_out)  # extract number of different connected components found
    assert numb_labels == 1, "This function is intended for binary masks that only contain ONE lesion."

    slices_enclosing_aneurysms = 0  # it's gonna be the number of slices that enclose the aneurysm
    idx = 0  # it's gonna be the index of the slice with the biggest lesion (the biggest number of white pixels)
    nb_white_pixels = 0
    tot_nb_white_pixels = []  # type: list # will contain the number of white pixels for each non-empty slice
    for z in range(0, lesion_volume.shape[2]):  # z will be the index of the slices
        if np.sum(lesion_volume[:, :, z]) != 0:  # if the sum of the pixels is different than zero (i.e. if there's at least one white pixel)
            slices_enclosing_aneurysms += 1  # increment
            tot_nb_white_pixels.append(np.count_nonzero(lesion_volume[:, :, z]))
            if np.count_nonzero(lesion_volume[:, :, z]) > nb_white_pixels:  # for the first iteration, we compare to 0, so the if is always verified if there's at least one non-zero pixel
                nb_white_pixels = np.count_nonzero(lesion_volume[:, :, z])  # update max number of white pixels if there are more than the previous slice
                idx = z  # update slice index if there are more white pixels than the previous one
    if prints:  # if prints is set to True when invoking the method
        print("\nThe aneurysms is present in {0} different slices.".format(slices_enclosing_aneurysms))
        print("\nThe slice with more white pixels has index {0} and contains {1} white pixels. \n".format(idx, np.count_nonzero(lesion_volume[:, :, idx])))

    properties = regionprops(lesion_volume[:, :, idx].astype(int))  # extract properties of slice with more white pixels

    for p in properties:
        equiv_diameter = np.array(p.equivalent_diameter).astype(int)  # we save the diameter of a circle with the same area as our ROI (we save it as int for simplicity)

    m = cv2.moments(lesion_volume[:, :, idx])  # calculate moments of binary image
    cX = int(m["m10"] / m["m00"])  # calculate x coordinate of center
    cY = int(m["m01"] / m["m00"])  # calculate y coordinate of center
    if prints:  # if prints is set to True when invoking the method
        print("The widest ROI has an equivalent diameter of {0} pixels and is approximately centered at x,y = [{1},{2}]\n".format(equiv_diameter, cX, cY))

    # create dict fields (keys) and fill them with values
    lesion_info["slices"] = slices_enclosing_aneurysms
    lesion_info["idx_slice_with_more_white_pixels"] = idx
    lesion_info["equivalent_diameter"] = equiv_diameter
    lesion_info["centroid_x_coord"] = cX
    lesion_info["centroid_y_coord"] = cY
    lesion_info["widest_dimension"] = slices_enclosing_aneurysms if slices_enclosing_aneurysms > equiv_diameter else equiv_diameter  # save biggest dimension between the two
    lesion_info["nb_non_zero_voxels"] = sum(tot_nb_white_pixels)  # sum all elements inside list

    return lesion_info  # returns the dictionary with the lesion information


def extract_lesion_info_modified(path_to_lesion, tmp_folder, new_spacing, sub_, ses_, prints=False):
    """This function takes as input the path of a binary volumetric mask, loops through the slices which have some non-zero pixels and returns
    some information about the entire lesion: i.e. number of slices enclosing the lesion, index of slice with more white pixels, the equivalent
    diameter of the lesion width in that specific slice, and the coordinates of the centroid of the lesion. If "prints" is set to True, the
    function also prints this information.
    Args:
        path_to_lesion (str): path to the binary mask of the aneurysm
        tmp_folder (str): path to folder where we temporarily save the resampled volumes
        new_spacing (list): desired voxel spacing
        sub_ (str): subject of interest
        ses_ (str): session of interest
        prints (bool): defaults to False. If set to True, it prints some information about the lesion
    Returns:
        lesion_info (dict): it contains some info/metrics about the lesion
    Raises:
        AssertionError: if the binary mask is either non-binary or full of zeros
        AssertionError: if the binary mask has more than 1 connected component
    """
    lesion_info = {}  # initialize empty dict; this will be the output of the function
    out_path_ = os.path.join(tmp_folder, "{0}_{1}_mask.nii.gz".format(sub_, ses_))
    # resample lesion mask. N.B. interpolator is set to NearestNeighbor so we are sure that we don't create new connected components
    _, _, lesion_volume = resample_volume(path_to_lesion, new_spacing, out_path_, interpolator=sitk.sitkNearestNeighbor)
    if len(lesion_volume.shape) == 4:  # if the numpy array is not 3D
        lesion_volume = np.squeeze(lesion_volume, axis=3)  # we drop the fourth dimension (time dimension) which is useless in our case

    assert np.array_equal(lesion_volume, lesion_volume.astype(bool)), "WATCH OUT: mask is not binary for {0}".format(path_to_lesion)
    assert np.count_nonzero(lesion_volume) > 0, "WATCH OUT: mask is empty (i.e. all zero-voxels)"

    labels_out = cc3d.connected_components(np.asarray(lesion_volume, dtype=int))
    numb_labels = np.max(labels_out)  # extract number of different connected components found
    assert numb_labels == 1, "This function is intended for binary masks that only contain ONE lesion."

    slices_enclosing_aneurysms = 0  # it's gonna be the number of slices that enclose the aneurysm
    idx = 0  # it's gonna be the index of the slice with the biggest lesion (the biggest number of white pixels)
    nb_white_pixels = 0
    tot_nb_white_pixels = []  # type: list # will contain the number of white pixels for each non-empty slice
    for z in range(0, lesion_volume.shape[2]):  # z will be the index of the slices
        if np.sum(lesion_volume[:, :, z]) != 0:  # if the sum of the pixels is different than zero (i.e. if there's at least one white pixel)
            slices_enclosing_aneurysms += 1  # increment
            tot_nb_white_pixels.append(np.count_nonzero(lesion_volume[:, :, z]))
            if np.count_nonzero(lesion_volume[:, :, z]) > nb_white_pixels:  # for the first iteration, we compare to 0, so the if is always verified if there's at least one non-zero pixel
                nb_white_pixels = np.count_nonzero(lesion_volume[:, :, z])  # update max number of white pixels if there are more than the previous slice
                idx = z  # update slice index if there are more white pixels than the previous one
    if prints:  # if prints is set to True when invoking the method
        print("\nThe aneurysms is present in {0} different slices.".format(slices_enclosing_aneurysms))
        print("\nThe slice with more white pixels has index {0} and contains {1} white pixels. \n".format(idx, np.count_nonzero(lesion_volume[:, :, idx])))

    properties = regionprops(lesion_volume[:, :, idx].astype(int))  # extract properties of slice with more white pixels

    for p in properties:
        equiv_diameter = np.array(p.equivalent_diameter).astype(int)  # we save the diameter of a circle with the same area as our ROI (we save it as int for simplicity)

    m = cv2.moments(lesion_volume[:, :, idx])  # calculate moments of binary image
    cx = int(m["m10"] / m["m00"])  # calculate x coordinate of center
    cy = int(m["m01"] / m["m00"])  # calculate y coordinate of center
    if prints:  # if prints is set to True when invoking the method
        print("The widest ROI has an equivalent diameter of {0} pixels and is approximately centered at x,y = [{1},{2}]\n".format(equiv_diameter, cx, cy))

    # create dict fields (keys) and fill them with values
    lesion_info["slices"] = slices_enclosing_aneurysms
    lesion_info["idx_slice_with_more_white_pixels"] = idx
    lesion_info["equivalent_diameter"] = equiv_diameter
    lesion_info["centroid_x_coord"] = cx
    lesion_info["centroid_y_coord"] = cy
    lesion_info["widest_dimension"] = slices_enclosing_aneurysms if slices_enclosing_aneurysms > equiv_diameter else equiv_diameter  # save biggest dimension between the two
    lesion_info["nb_non_zero_voxels"] = sum(tot_nb_white_pixels)  # sum all elements inside list

    return lesion_info  # returns the dictionary with the lesion information


def retrieve_intensity_conditions_one_sub(subdir, aneurysm_mask_path, data_path, new_spacing, patch_side, out_folder, overlapping):
    """This function computes the intensity thresholds for extracting the vessel-like negative patches
    Args:
        subdir (str): path to parent folder of aneurysm_mask_path
        aneurysm_mask_path (str): path to aneurysm mask
        data_path (str): path to BIDS dataset
        new_spacing (list): desired voxel spacing to which we want to resample
        patch_side (int): side of cubic patches that will be created
        out_folder (str): path to output folder; during ds creation it is where we create the output dataset; at inference, it is where we save segmentation outputs
        overlapping (float): amount of overlapping between patches in sliding-window approach
    Returns:
        out_list (list): it contains values of interest from which we will draw the final thresholds; if no positive patch was evaluated, we return None
    Raises:
        AssertionError: if the BIDS dataset path does not exist
        AssertionError: if the folder containing the vesselMNI deformed volume does not exist
        AssertionError: if the folder containing the bias-field-corrected volumes does not exist
        AssertionError: if the sub ID was not found
        AssertionError: if the session (i.e. exam date) was not found
        AssertionError: if the lesion name was not found
    """
    assert os.path.exists(data_path), "path {0} does not exist".format(data_path)
    vessel_mni_registration_dir = os.path.join(data_path, "derivatives", "registrations", "vesselMNI_2_angioTOF")
    assert os.path.exists(vessel_mni_registration_dir), "path {0} does not exist".format(vessel_mni_registration_dir)  # make sure that path exists
    bfc_derivatives_dir = os.path.join(data_path, "derivatives", "N4_bias_field_corrected")
    assert os.path.exists(bfc_derivatives_dir), "path {0} does not exist".format(bfc_derivatives_dir)  # make sure that path exists

    shift_scale_1 = patch_side // 2

    ratios_local_vessel_mni = []
    ratios_global_vessel_mni = []
    ratios_local_tof_bet_bfc = []
    ratios_global_tof_bet_bfc = []
    non_zeros_vessel_mni = []

    sub = re.findall(r"sub-\d+", subdir)[0]
    ses = re.findall(r"ses-\w{6}\d+", subdir)[0]  # extract ses

    if "Treated" in aneurysm_mask_path:
        lesion_name = re.findall(r"Treated_Lesion_\d+", aneurysm_mask_path)[0]  # type: str # extract lesion name
    else:
        lesion_name = re.findall(r"Lesion_\d+", aneurysm_mask_path)[0]  # type: str # extract lesion name

    assert len(sub) != 0, "Subject ID not found"
    assert len(ses) != 0, "Session number not found"
    assert len(lesion_name) != 0, "Lesion name not found"

    # if we are NOT dealing with a treated aneurysm
    if "Treated" not in lesion_name:

        # create unique tmp folder where we save temporary files (this folder will be deleted at the end)
        tmp_folder = os.path.join(out_folder, "tmp_{}_{}_{}_pos_patches".format(sub, ses, lesion_name))
        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)

        # uncomment line below for debugging
        # print("{}-{}-{}".format(sub, ses, lesion_name))

        # save path of corresponding vesselMNI co-registered volume
        if "ADAM" in subdir:
            vessel_mni_reg_volume_path = os.path.join(vessel_mni_registration_dir, sub, ses, "anat", "{0}_{1}_desc-vesselMNI2angio_deformed_ADAM.nii.gz".format(sub, ses))
        else:
            vessel_mni_reg_volume_path = os.path.join(vessel_mni_registration_dir, sub, ses, "anat", "{0}_{1}_desc-vesselMNI2angio_deformed.nii.gz".format(sub, ses))
        assert os.path.exists(vessel_mni_reg_volume_path), "Path {} does not exist".format(vessel_mni_reg_volume_path)

        if "ADAM" in subdir:
            bet_angio_bfc_path = os.path.join(bfc_derivatives_dir, sub, ses, "anat", "{}_{}_desc-angio_N4bfc_brain_mask_ADAM.nii.gz".format(sub, ses))  # type: str # save path of angio brain after Brain Extraction Tool (BET)
        else:
            bet_angio_bfc_path = os.path.join(bfc_derivatives_dir, sub, ses, "anat", "{}_{}_desc-angio_N4bfc_brain_mask.nii.gz".format(sub, ses))  # type: str # save path of angio brain after Brain Extraction Tool (BET)
        assert os.path.exists(bet_angio_bfc_path), "path {0} does not exist".format(bet_angio_bfc_path)  # make sure that path exists

        # Load N4 bias-field-corrected angio volume after BET and resample to new spacing
        out_path = os.path.join(tmp_folder, "resampled_bet_tof_bfc.nii.gz")
        _, _, nii_volume_resampled = resample_volume(bet_angio_bfc_path, new_spacing, out_path)
        rows_range, columns_range, slices_range = nii_volume_resampled.shape  # save dimensions of resampled angio-BET volume

        # Load corresponding vesselMNI volume and resample to new spacing
        out_path = os.path.join(tmp_folder, "resampled_vessel_atlas.nii.gz")
        _, _, vessel_mni_volume_resampled = resample_volume(vessel_mni_reg_volume_path, new_spacing, out_path)

        lesion = (extract_lesion_info_modified(os.path.join(subdir, aneurysm_mask_path), tmp_folder, new_spacing, sub, ses))  # invoke external function and save dict with lesion information
        sc_shift = lesion["widest_dimension"] // 2  # define Sanity Check shift (half side of widest lesion dimension)
        # N.B. I INVERT X and Y BECAUSE of OpenCV (see https://stackoverflow.com/a/56849032/9492673)
        x_center = lesion["centroid_y_coord"]  # extract y coordinate of lesion centroid
        y_center = lesion["centroid_x_coord"]  # extract x coordinate of lesion centroid
        z_central = lesion["idx_slice_with_more_white_pixels"]  # extract idx of slice with more non-zero pixels
        x_min, x_max = x_center - sc_shift, x_center + sc_shift  # compute safest (largest) min and max x of patch containing lesion
        y_min, y_max = y_center - sc_shift, y_center + sc_shift  # compute safest (largest) min and max y of patch containing lesion
        z_min, z_max = z_central - sc_shift, z_central + sc_shift  # compute safest (largest) min and max z of patch containing lesion
        # lesion_coord[aneur_path] = [x_min, x_max, y_min, y_max, z_min, z_max]  # save lesion information in external dict

        cnt_positive_patches = 0  # type: int # counter to keep track of how many pos patches are selected for each patient
        step = int(round_half_up((1 - overlapping) * patch_side))  # type: int
        for i in range(shift_scale_1, rows_range, step):  # loop over rows
            for j in range(shift_scale_1, columns_range, step):  # loop over columns
                for k in range(shift_scale_1, slices_range, step):  # loop over slices

                    flag = 0  # flag is just a dummy variable that we increment when a candidate patch overlaps with one aneurysm
                    # N.B. we check the overlap ONLY with the small-scale TOF patch cause the sequential scanning is performed with the small scale range
                    range_i = np.arange(i - shift_scale_1, i + shift_scale_1)  # create x range of the patch that will be evaluated
                    range_j = np.arange(j - shift_scale_1, j + shift_scale_1)  # create y range of the patch that will be evaluated
                    range_k = np.arange(k - shift_scale_1, k + shift_scale_1)  # create z range of the patch that will be evaluated

                    lesion_range_x = np.arange(x_min-1, x_max+1)  # create x range of the lesion (i.e. aneurysm)
                    lesion_range_y = np.arange(y_min-1, y_max+1)  # create y range of the lesion (i.e. aneurysm)
                    lesion_range_z = np.arange(z_min-1, z_max+1)  # create z range of the lesion (i.e. aneurysm)

                    # the boolean masks have all False if none of the voxels overlap (between candidate patch and lesion), and have True for the coordinates that do overlap
                    boolean_mask_x = (range_i >= np.min(lesion_range_x)) & (range_i <= np.max(lesion_range_x))
                    boolean_mask_y = (range_j >= np.min(lesion_range_y)) & (range_j <= np.max(lesion_range_y))
                    boolean_mask_z = (range_k >= np.min(lesion_range_z)) & (range_k <= np.max(lesion_range_z))

                    # if ALL the three boolean masks have at least one True value
                    if np.all(boolean_mask_y == False) == False and np.all(boolean_mask_x == False) == False and np.all(boolean_mask_z == False) == False:
                        flag += 1  # increment flag; if it gets incremented, it means that the current candidate patch overlaps with one aneurysm with at least one voxel

                    # ensure that the evaluated patch is not out of bound by using small scale
                    if i - shift_scale_1 >= 0 and i + shift_scale_1 < rows_range and j - shift_scale_1 >= 0 and j + shift_scale_1 < columns_range and k - shift_scale_1 >= 0 and k + shift_scale_1 < slices_range:
                        if flag != 0:  # if the patch contains an aneurysm
                            cnt_positive_patches += 1  # increment counter
                            # extract patch from angio after BET
                            angio_patch_after_bet_scale_1 = nii_volume_resampled[i - shift_scale_1:i + shift_scale_1,
                                                                                 j - shift_scale_1:j + shift_scale_1,
                                                                                 k - shift_scale_1:k + shift_scale_1]

                            # extract small-scale patch from vesselMNI volume; we don't need the big-scale vessel patch
                            vessel_mni_patch = vessel_mni_volume_resampled[i - shift_scale_1:i + shift_scale_1,
                                                                           j - shift_scale_1:j + shift_scale_1,
                                                                           k - shift_scale_1:k + shift_scale_1]

                            # if mean and max intensities are not "nan" and the translated patch doesn't have more zeros than the centered one
                            if not math.isnan(np.mean(vessel_mni_patch)) and not math.isnan(np.max(vessel_mni_patch)) and not math.isnan(np.max(vessel_mni_volume_resampled)) \
                                    and np.max(vessel_mni_patch) != 0 and not math.isnan(np.mean(angio_patch_after_bet_scale_1)) and np.max(angio_patch_after_bet_scale_1) != 0:
                                ratio_local_vessel_mni = np.mean(vessel_mni_patch) / np.max(vessel_mni_patch)  # compute intensity ratio (mean/max) only on vesselMNI patch
                                ratio_global_vessel_mni = np.mean(vessel_mni_patch) / np.max(vessel_mni_volume_resampled)  # compute intensity ratio (mean/max) on vesselMNI patch wrt entire volume
                                ratio_local_tof_bet = np.mean(angio_patch_after_bet_scale_1) / np.max(angio_patch_after_bet_scale_1)  # compute local intensity ratio (mean/max) on bet_tof
                                ratio_global_tof_bet = np.mean(angio_patch_after_bet_scale_1) / np.max(nii_volume_resampled)  # compute global intensity ratio (mean/max) on bet_tof

                                ratios_local_vessel_mni.append(ratio_local_vessel_mni)
                                ratios_global_vessel_mni.append(ratio_global_vessel_mni)
                                ratios_local_tof_bet_bfc.append(ratio_local_tof_bet)
                                ratios_global_tof_bet_bfc.append(ratio_global_tof_bet)
                                non_zeros_vessel_mni.append(np.count_nonzero(vessel_mni_patch))

        if cnt_positive_patches == 0:
            print("WARNING: no positive patch evaluated for {}_{}_{}; we are looping over patients with aneurysm(s)".format(sub, ses, lesion_name))
        # -------------------------------------------------------------------------------------
        # remove temporary folder for this subject
        if os.path.exists(tmp_folder) and os.path.isdir(tmp_folder):
            shutil.rmtree(tmp_folder)

        if cnt_positive_patches > 0:
            out_list = [np.median(ratios_local_vessel_mni), np.median(ratios_global_vessel_mni), np.median(ratios_local_tof_bet_bfc), np.median(ratios_global_tof_bet_bfc), np.median(non_zeros_vessel_mni)]
            return out_list
        else:
            return None  # if no positive patch was found, we return None (which we later remove), otherwise we degrade the precision of the distribution
    else:  # if it's a treated aneurysm
        return None


def extract_thresholds_of_intensity_criteria(data_path, sub_ses_test, patch_side, new_spacing, out_folder, n_parallel_jobs, overlapping, prints=True):
    """This function computes the threshold to use for the extraction of the vessel-like negative patches (i.e. the
    negative patches that roughly have the same average intensity of the positive patches and include vessels)
    Args:
        data_path (str): path to BIDS dataset
        sub_ses_test (list): sub_ses of the test set; we use it to take only the sub_ses of the training set
        patch_side (int): side of cubic patches
        new_spacing (list): desired voxel spacing
        out_folder (str): path to output folder; during ds creation it is where we create the output dataset; at inference, it is where we save segmentation outputs
        n_parallel_jobs (int): number of jobs to run in parallel
        overlapping (float): amount of overlapping between patches in sliding-window approach
        prints (bool): whether to print numerical thresholds that were found; defaults to True
    Returns:
        intensity_thresholds (tuple): it contains the values to use for the extraction of the vessel-like negative samples
    """
    regexp_sub = re.compile(r'sub')  # create a substring template to match
    ext_gz = '.gz'  # type: str # set zipped files extension

    # create new input lists to create positive patches in parallel
    all_subdirs = []
    all_files = []
    for subdir, dirs, files in os.walk(data_path):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()  # get the file extension
            if regexp_sub.search(file) and ext == ext_gz and "Lesion" in file and "registrations" not in subdir:
                sub = re.findall(r"sub-\d+", subdir)[0]
                ses = re.findall(r"ses-\w{6}\d+", subdir)[0]  # extract ses
                sub_ses = "{}_{}".format(sub, ses)
                if sub_ses not in sub_ses_test:  # only use training sub_ses otherwise we might introduce a bias towards the intensities of the aneurysms in the test set
                    all_subdirs.append(subdir)
                    all_files.append(file)

    assert all_subdirs and all_files, "Input lists must be non-empty"
    out_list = Parallel(n_jobs=n_parallel_jobs, backend='loky')(delayed(retrieve_intensity_conditions_one_sub)(all_subdirs[idx],
                                                                                                               all_files[idx],
                                                                                                               data_path,
                                                                                                               new_spacing,
                                                                                                               patch_side,
                                                                                                               out_folder,
                                                                                                               overlapping) for idx in range(len(all_subdirs)))

    out_list = [x for x in out_list if x]  # remove None values from list if present
    out_list_np = np.asarray(out_list)  # type: np.ndarray # convert from list to numpy array

    q5_local_vessel_mni, q7_local_vessel_mni = np.percentile(out_list_np[:, 0], [5, 7])
    q5_global_vessel_mni, q7_global_vessel_mni = np.percentile(out_list_np[:, 1], [5, 7])
    q5_local_tof_bet, q7_local_tof_bet = np.percentile(out_list_np[:, 2], [5, 7])
    q5_global_tof_bet, q7_global_tof_bet = np.percentile(out_list_np[:, 3], [5, 7])
    q5_nz_vessel_mni = np.percentile(out_list_np[:, 4], [5])[0]

    if prints:
        print("\nIntensity thresholds with patch_side={}".format(patch_side))
        print("\nMean-Max local intensity ratio in vesselMNI positive patches:")
        print("5th percentile = {0}".format(q5_local_vessel_mni))
        print("7th percentile = {0}".format(q7_local_vessel_mni))

        print("\nMean-Max global intensity ratio in vesselMNI positive patches:")
        print("5th percentile = {0}".format(q5_global_vessel_mni))
        print("7th percentile = {0}".format(q7_global_vessel_mni))

        print("\nMean-Max local intensity ratio in bet TOF positive patches:")
        print("5th percentile = {0}".format(q5_local_tof_bet))
        print("7th percentile = {0}".format(q7_local_tof_bet))

        print("\nMean-Max global intensity ratio in bet TOF positive patches:")
        print("5th percentile = {0}".format(q5_global_tof_bet))
        print("7th percentile = {0}".format(q7_global_tof_bet))

        print("\nNumber of non-zero voxels in vesselMNI positive patches:")
        print("5th percentile = {0}".format(q5_nz_vessel_mni))

    intensity_thresholds = (q5_local_vessel_mni, q5_global_vessel_mni, q5_local_tof_bet, q5_global_tof_bet, q5_nz_vessel_mni)

    return intensity_thresholds


def extract_reg_quality_metrics_one_sub(input_path):
    """This function extracts and returns the struct_2_tof registration quality metrics for one subject.
    Args:
        input_path (str): path to folder where the registration quality metrics are stored for this subject
    Returns:
        struct_2_tof_nc (float): neighborhood correlation metric for the struct_2_tof registration of this subject
        struct_2_tof_mi (float): mattes mutual information metric for the struct_2_tof registration of this subject
    """
    for files in os.listdir(input_path):
        if "struct2tof" in files:
            struct_2_tof_quality_metrics = pd.read_csv(os.path.join(input_path, files))
            struct_2_tof_nc = struct_2_tof_quality_metrics.iloc[0]['neigh_corr']
            struct_2_tof_mi = struct_2_tof_quality_metrics.iloc[0]['mut_inf']
        elif "mni2struct" in files:
            pass
        else:
            raise ValueError("Unknown values found")

    return struct_2_tof_nc, struct_2_tof_mi


def retrieve_registration_params(registration_dir_):
    """This function retrieves the registration parameters for each subject
    Args:
        registration_dir_ (str): path to folder containing the registration parameters of each subject
    Returns:
        mni_2_T1_mat_path_ (str): path to .mat file corresponding to the MNI --> struct registration
        struct_2_tof_mat_path_ (str): path to .mat file corresponding to the struct --> TOF registration
        mni_2_T1_warp_path_ (str): path to warp field corresponding to the MNI --> struct registration
        mni_2_T1_inverse_warp_path_ (str): path to inverse warp field corresponding to the MNI --> struct registration
    Raises:
        AssertionError: if input path does not exist
        AssertionError: if more (or less) than 4 registration paths are retrieved
    """
    assert os.path.exists(registration_dir_), "Path {0} does not exist".format(registration_dir_)
    extension_mat = '.mat'  # type: str # set file extension to be matched
    extension_gz = '.gz'  # type: str # set file extension to be matched
    cnt = 0  # type: int # counter variable that we use to ensure that exactly 4 registration parameters were retrieved

    for files_ in os.listdir(registration_dir_):
        ext_ = os.path.splitext(files_)[-1].lower()  # get the file extension
        if "ADAM" in registration_dir_:
            if ext_ in extension_mat and "MNI_2_struct" in files_:  # if the extensions matches and a specific substring is in the file path
                mni_2_t1_mat_path_ = os.path.join(registration_dir_, files_)  # type: str
                cnt += 1  # increment counter
            elif ext_ in extension_mat and "struct_2_TOF" in files_:  # if the extensions matches and a specific substring is in the file path
                struct_2_tof_mat_path_ = os.path.join(registration_dir_, files_)  # type: str
                cnt += 1  # increment counter
            elif ext_ in extension_gz and "MNI_2_struct_1Warp" in files_:  # if the extensions matches and a specific substring is in the file path
                mni_2_t1_warp_path_ = os.path.join(registration_dir_, files_)  # type: str
                cnt += 1  # increment counter
            elif ext_ in extension_gz and "MNI_2_struct_1InverseWarp" in files_:  # if the extensions matches and a specific substring is in the file path
                mni_2_t1_inverse_warp_path_ = os.path.join(registration_dir_, files_)  # type: str
                cnt += 1  # increment counter
        else:
            if ext_ in extension_mat and "MNI_2_T1" in files_:  # if the extensions matches and a specific substring is in the file path
                mni_2_t1_mat_path_ = os.path.join(registration_dir_, files_)  # type: str
                cnt += 1  # increment counter
            elif ext_ in extension_mat and "T1_2_TOF" in files_:  # if the extensions matches and a specific substring is in the file path
                struct_2_tof_mat_path_ = os.path.join(registration_dir_, files_)  # type: str
                cnt += 1  # increment counter
            elif ext_ in extension_gz and "MNI_2_T1_1Warp" in files_:  # if the extensions matches and a specific substring is in the file path
                mni_2_t1_warp_path_ = os.path.join(registration_dir_, files_)  # type: str
                cnt += 1  # increment counter
            elif ext_ in extension_gz and "MNI_2_T1_1InverseWarp" in files_:  # if the extensions matches and a specific substring is in the file path
                mni_2_t1_inverse_warp_path_ = os.path.join(registration_dir_, files_)  # type: str
                cnt += 1  # increment counter

    assert cnt == 4, "Exactly 4 registration parameters must be retrieved"

    return mni_2_t1_mat_path_, struct_2_tof_mat_path_, mni_2_t1_warp_path_, mni_2_t1_inverse_warp_path_


def resample_volume(volume_path, new_spacing, out_path, interpolator=sitk.sitkLinear):
    """This function resamples the input volume to a specified voxel spacing
    Args:
        volume_path (str): input volume path
        new_spacing (list): desired voxel spacing that we want
        out_path (str): path where we temporarily save the resampled output volume
        interpolator (int): interpolator that we want to use (e.g. 1= NearNeigh., 2=linear, ...)
    Returns:
        resampled_volume_sitk_obj (sitk.Image): resampled volume as sitk object
        resampled_volume_nii_obj (nib.Nifti1Image): resampled volume as nib object
        resampled_volume_nii (np.ndarray): resampled volume as numpy array
    """
    volume = sitk.ReadImage(volume_path)  # read volume
    original_size = volume.GetSize()  # extract size
    original_spacing = volume.GetSpacing()  # extract spacing
    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
    resampled_volume_sitk_obj = sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                                              volume.GetOrigin(), new_spacing, volume.GetDirection(), 0,
                                              volume.GetPixelID())
    sitk.WriteImage(resampled_volume_sitk_obj, out_path)  # write sitk volume object to disk
    resampled_volume_nii_obj = nib.load(out_path)  # type: nib.Nifti1Image # load volume as nibabel object
    resampled_volume_nii = np.asanyarray(resampled_volume_nii_obj.dataobj)  # type: np.ndarray # convert from nibabel object to np.array
    os.remove(out_path)  # remove volume from disk to save space

    return resampled_volume_sitk_obj, resampled_volume_nii_obj, resampled_volume_nii


def load_nifti_and_resample(volume_path, tmp_folder_, out_name, new_spacing_, binary_mask=False):
    """This function loads a nifti volume, resamples it to a specified voxel spacing, and returns both
    the resampled nifti object and the resampled volume as numpy array, together with the affine matrix
    Args:
        volume_path (str): path to nifti volume
        tmp_folder_ (str): path to folder where we temporarily save the resampled volume
        out_name (str): name of resampled volume temporarily saved to disk
        new_spacing_ (list): desired voxel spacing for output volume
        binary_mask (bool): defaults to False. If set to True, it means that the volume is a binary mask
    Returns:
        resampled_volume_obj_nib (Nifti1Image): nibabel object of resampled output volume
        resampled_volume (np.ndarray): resampled output volume
        aff_matrix (np.ndarray): affine matrix associated with resampled output volume
        resampled_volume_obj_sitk (sitk.Image): sitk object of resampled output volume
    """
    if not os.path.exists(tmp_folder_):
        os.makedirs(tmp_folder_)
    out_path = os.path.join(tmp_folder_, out_name)
    if binary_mask:  # if the volume is a mask, use near.neighbor interpolator in order not to create new connected components
        resampled_volume_obj_sitk, resampled_volume_obj_nib, resampled_volume = resample_volume(volume_path, new_spacing_, out_path, interpolator=sitk.sitkNearestNeighbor)
    else:  # instead, if it's a normal nifti volume, use linear interpolator
        resampled_volume_obj_sitk, resampled_volume_obj_nib, resampled_volume = resample_volume(volume_path, new_spacing_, out_path, interpolator=sitk.sitkLinear)
    assert len(resampled_volume.shape) == 3, "Nifti volume is not 3D"
    aff_matrix = resampled_volume_obj_nib.affine  # extract and save affine matrix
    return resampled_volume_obj_sitk, resampled_volume_obj_nib, resampled_volume, aff_matrix


def round_half_up(n, decimals=0):
    """This function rounds to the nearest integer number (e.g 2.4 becomes 2.0 and 2.6 becomes 3);
     in case of tie, it rounds up (e.g. 1.5 becomes 2.0 and not 1.0)
    Args:
        n (float): number to round
        decimals (int): number of decimal figures that we want to keep; defaults to zero
    """
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier


def convert_angio_to_mni(original_angio_volume_sitk_, voxel_space_angio_coords, mni_2_struct_mat_path_, struct_2_tof_mat_path_, mni_2_struct_warp_path_, tmp_path):
    """This function takes as input a point [i,j,k] in angio voxel space and returns the corresponding [x,y,z] point in mm but registered in mni space.
    Args:
        original_angio_volume_sitk_ (sitk.Image): original angio volume loaded with sitk
        voxel_space_angio_coords (list): [i,j,k] voxel coordinate that must be registered from angio space to mni space
        mni_2_struct_mat_path_ (str): path to MNI_2_struct .mat file
        struct_2_tof_mat_path_ (str): path to struct_2_TOF .mat file
        mni_2_struct_warp_path_ (str): path to mni_2_struct warp field
        tmp_path (str): path to folder where we save temporary registration files
    Returns:
        mm_coord_mni (np.ndarray): [x,y,z] mm coordinate in mni space
    """
    center_mm_coord = list(original_angio_volume_sitk_.TransformIndexToPhysicalPoint(voxel_space_angio_coords))  # convert coordinate from voxel space to physical space (mm)
    center_mm_coord.append(0)  # we put 0 in the time variable, because ANTs requires the coordinates to be in the shape [x, y, z, t] in physical space

    # WRITE original angio coordinate in physical space (mm) as csv file
    csv_folder = os.path.join(tmp_path, "tmp_points_CHUV")  # specify folder where we save the csv file
    if not os.path.exists(csv_folder):  # if path does not exist
        os.makedirs(csv_folder)  # create it
    csv_path = os.path.join(csv_folder, "Center_Coordinate_TOF_in_mm.csv")  # add filename to path

    # create csv file
    with open(csv_path, 'w') as myfile:
        wr = csv.writer(myfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        wr.writerow(['x', 'y', 'z', 't'])
        wr.writerow(center_mm_coord)

    # ------------------------------------------------------------------ TOF_2_struct -------------------------------------------------------------
    # output_path_struct = os.path.join(csv_folder, "Transformed_Point_mm_struct.csv")  # save output filename

    # load point as dataframe
    tof_df = pd.read_csv(csv_path)
    # duplicate first row (this is needed to run apply_transforms_to_points; it's a bug that they still have to fix)
    modified_df = pd.DataFrame(np.repeat(tof_df.values, 2, axis=0))
    modified_df.columns = tof_df.columns
    # apply registration to point
    transform_list = [struct_2_tof_mat_path_]
    which_to_invert = [False]
    struct_df = apply_transforms_to_points(dim=3,
                                           points=modified_df,
                                           transformlist=transform_list,
                                           whichtoinvert=which_to_invert)

    struct_df = struct_df.drop_duplicates()

    # ------------------------------------------------------------------ struct_2_MNI -------------------------------------------------------------
    output_path_mni = os.path.join(csv_folder, "Transformed_Point_mm_MNI.csv")  # save output filename

    modified_struct_df = pd.DataFrame(np.repeat(struct_df.values, 2, axis=0))
    modified_struct_df.columns = struct_df.columns
    # apply registration to point
    transform_list = [mni_2_struct_warp_path_, mni_2_struct_mat_path_]
    which_to_invert = [False, False]
    mni_df = apply_transforms_to_points(dim=3,
                                        points=modified_struct_df,
                                        transformlist=transform_list,
                                        whichtoinvert=which_to_invert)

    mni_df = mni_df.drop_duplicates()
    # save dataframe as csv file
    mni_df.to_csv(output_path_mni, index=False)

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # READ MNI center coordinates in physical space from csv created
    df_mni_center_coord = pd.read_csv(output_path_mni)
    center_mm_coord_mni = list(df_mni_center_coord.iloc[0])[:-1]  # extract first row of pd dataframe, convert to list and remove last item (t), cause we don't care about it

    # UNCOMMENT lines below for DEBUGGING
    # N.B. BE SURE TO LOAD THE VESSEL ATLAS FOR DEBUGGING AND NOT THE T1 or T2 ATLAS CAUSE THEY HAVE DIFFERENT SIZE AND AXES ORDER
    # vessel_mni_atlas = sitk.ReadImage("/home/newuser/Desktop/ADAM_release_subjs_unzipped/vessel_mni_atlas.nii.gz")
    # center_voxel_coord_mni = vessel_mni_atlas.TransformPhysicalPointToIndex(center_mm_coord_mni)
    # print("DEBUG: MNI voxel coord = {}".format(center_voxel_coord_mni))

    return center_mm_coord_mni


def distance_is_plausible(patch_center_coordinates_physical_space, df_landmarks_tof_space, distances_thresholds):
    """This function takes as input the center of a candidate patch in angio voxel space and computes the distances from this point to a series of landmarks points
    (also warped to tof space) where aneurysms are recurrent. Then it returns True if the min and mean distances are within plausible ranges, otherwise it returns False.
    Args:
        patch_center_coordinates_physical_space (list): [x,y,z] physical coordinate of the patch center
        df_landmarks_tof_space (pd.Dataframe): it contains the 3D coordinates of the landmark points where aneurysms are most recurrent (in physical tof space)
        distances_thresholds (list): it contains the thresholds to use for the distances to the landmark points
    Returns:
        True: if distances are within plausible ranges for aneurysms (ranges were computed empirically)
        False: if distances are not within plausible ranges for aneurysms (ranges were computed empirically)
        """
    assert len(distances_thresholds) == 2, "List must have len==2; instead, len=={}".format(len(distances_thresholds))

    distances = []
    for _, coords in df_landmarks_tof_space.iterrows():
        landmark_point_physical_space_tof = np.asarray([coords["x"], coords["y"], coords["z"]])
        eucl_distance = np.linalg.norm(patch_center_coordinates_physical_space - landmark_point_physical_space_tof)  # type: float # compute euclidean distance in TOF physical space
        distances.append(eucl_distance)

    min_dist = np.min(distances)
    mean_dist = np.mean(distances)

    # return True if (min_dist < XX and mean_dist < YY) else False, where XX and YY are computed empirically
    return True if (min_dist < distances_thresholds[0] and mean_dist < distances_thresholds[1]) else False


def extracting_conditions_are_met(angio_patch_after_bet_scale_1_, vessel_mni_patch_, vessel_mni_volume_, nii_volume_, patch_center_coordinates_physical_space,
                                  patch_side_scale_1, df_landmarks_tof_space, registration_accurate_enough, intensity_thresholds, distances_thresholds, anatomically_informed):
    """ This function checks whether the candidate patch fulfills specific extraction conditions. There are 6 main conditions checked:
        1) the corresponding vesselMNI patch must have intensity ratios (local and global) similar to positive patches (thresholds found empirically from pos. patches)
        2) the corresponding vesselMNI patch must have at least K non-zero voxels (K found empirically from pos. patches)
        3) patches must have both a local and global mean/max intensity ratio comparable to those of positive patches (thresholds found empirically from pos. patches)
        4) the center of the candidate patch must be within a reasonable distance from one of the landmark points recurrent for aneurysm occurrence
        5) the neigh_corr quality metric of this subject must be above the 10th percentile of the distribution of all neigh_corr in the dataset. If it's not, there was probably a registration mistake
        6) the mutual_inf quality metric of this subject must be below the 90th percentile of the distribution of all mutual_inf in the dataset. If it's not, there was probably a registration mistake
    Args:
        angio_patch_after_bet_scale_1_ (np.ndarray): small-scale angio patch
        vessel_mni_patch_ (np.ndarray): corresponding vessel mni patch of same dims of angio patch
        vessel_mni_volume_ (np.ndarray): vessel mni volume of this subject resampled to uniform voxel space
        nii_volume_ (np.ndarray): BET angio volume (i.e. after brain extraction) resampled to uniform voxel space
        patch_center_coordinates_physical_space (list): it contains the 3D physical (in mm) center coordinate of this candidate patch
        patch_side_scale_1 (int): side of sliding-window patches
        df_landmarks_tof_space (pd.Dataframe): it contains the 3D coordinates of the landmark points where aneurysms are most recurrent (in physical tof space)
        registration_accurate_enough (bool): it indicates whether the registration accuracy is high enough to perform the anatomically-informed sliding window
        intensity_thresholds (list): it contains the thresholds for the intensity conditions, namely [q5_local_vessel_mni, q5_global_vessel_mni, q5_local_tof_bet, q5_global_tof_bet, q5_nz_vessel_mni]
        distances_thresholds (list): it contains the thresholds for the distances from the patch centers to the landmark points
        anatomically_informed (bool) whether to conduct the sliding-window in an anatomically-informed fashion or not
    Returns:
        conditions_fulfilled (bool): initialized as False; if all extracting conditions are met, it becomes True
    """
    conditions_fulfilled = False  # type: bool # initialize as False; then, if all sliding-window conditions are met, this will become True

    if anatomically_informed:  # if we want to make an anatomically-informed sliding window (i.e. we check the extracting conditions)
        # if the registration is accurate enough (i.e. the registration metrics for this subject are not outliers wrt the entire distribution)
        if registration_accurate_enough:
            # if the small-scale angio patch is not empty and the vesselMNI patch is a cube
            if np.count_nonzero(angio_patch_after_bet_scale_1_) != 0 and vessel_mni_patch_.shape == (patch_side_scale_1, patch_side_scale_1, patch_side_scale_1):
                # to avoid division by 0, first check that denominator is != 0 and there are no NaNs
                if np.max(vessel_mni_patch_) != 0 and not math.isnan(np.mean(vessel_mni_patch_)) and not math.isnan(np.mean(angio_patch_after_bet_scale_1_)):
                    ratio_local_vessel_mni = np.mean(vessel_mni_patch_) / np.max(vessel_mni_patch_)  # compute intensity ratio (mean/max) only on vesselMNI patch
                    ratio_global_vessel_mni = np.mean(vessel_mni_patch_) / np.max(vessel_mni_volume_)  # compute intensity ratio (mean/max) on vesselMNI patch wrt entire volume
                    ratio_local_tof_bet = np.mean(angio_patch_after_bet_scale_1_) / np.max(angio_patch_after_bet_scale_1_)  # compute local intensity ratio (mean/max) on bet_tof
                    ratio_global_tof_bet = np.mean(angio_patch_after_bet_scale_1_) / np.max(nii_volume_)  # compute global intensity ratio (mean/max) on bet_tof

                    # UNCOMMENT lines below for debugging
                    # print("ratio_local_vessel_mni = {}".format(ratio_local_vessel_mni))
                    # print("ratio_global_vessel_mni = {}".format(ratio_global_vessel_mni))
                    # print("ratio_local_tof_bet = {}".format(ratio_local_tof_bet))
                    # print("ratio_global_tof_bet = {}".format(ratio_global_tof_bet))
                    # print("count_nonzero vessel_mni = {}\n".format(np.count_nonzero(vessel_mni_patch_)))

                    if ratio_local_vessel_mni > intensity_thresholds[0] and ratio_global_vessel_mni > intensity_thresholds[1] and \
                            ratio_local_tof_bet > intensity_thresholds[2] and ratio_global_tof_bet > intensity_thresholds[3] and np.count_nonzero(vessel_mni_patch_) > intensity_thresholds[4]:

                        if distance_is_plausible(patch_center_coordinates_physical_space, df_landmarks_tof_space, distances_thresholds):
                            conditions_fulfilled = True

        # if instead the registration is not accurate enough, disregard registration conditions and only check intensity conditions
        else:
            # if the small-scale angio patch is not empty
            if np.count_nonzero(angio_patch_after_bet_scale_1_) != 0:
                # to avoid division by 0, first check that denominator is != 0 and there are no NaNs
                if not math.isnan(np.mean(angio_patch_after_bet_scale_1_)):
                    ratio_local_tof_bet = np.mean(angio_patch_after_bet_scale_1_) / np.max(angio_patch_after_bet_scale_1_)  # compute local intensity ratio (mean/max) on bet_tof
                    ratio_global_tof_bet = np.mean(angio_patch_after_bet_scale_1_) / np.max(nii_volume_)  # compute global intensity ratio (mean/max) on bet_tof

                    # UNCOMMENT lines below for debugging
                    # print(ratio_local_tof_bet)
                    # print(ratio_global_tof_bet)
                    # print()

                    if ratio_local_tof_bet > intensity_thresholds[2] and ratio_global_tof_bet > intensity_thresholds[3]:
                        conditions_fulfilled = True

    else:  # if instead we just do a brute-force sliding-window (i.e. not anatomically-informed) --> just take all the patches that have non-zero voxels and the correct shape
        # if the small-scale angio patch is not empty
        if np.count_nonzero(angio_patch_after_bet_scale_1_) != 0 and angio_patch_after_bet_scale_1_.shape == (patch_side_scale_1, patch_side_scale_1, patch_side_scale_1):
            conditions_fulfilled = True

    return conditions_fulfilled


def create_tf_dataset(all_angio_patches_scale_1_numpy_list_, unet_batch_size):
    """This function creates a batched tf.data.Dataset from lists of numpy arrays containing small-scale tof and labels.
    Args:
        all_angio_patches_scale_1_numpy_list_ (list): list containing all small_scale angio patches
        unet_batch_size (int): batch size. Not really relevant (cause we're doing inference), but still needed
    Returns:
        batched_dataset_ (tf.data.Dataset): batched dataset containing all samples and labels
    """
    all_patch_volumes_tensors_ = tf.convert_to_tensor(all_angio_patches_scale_1_numpy_list_, dtype=tf.float32)  # convert from list of numpy arrays to tf.Tensor

    # CREATE tf.data.Dataset (since we are doing inference, there are no labels)
    dataset = tf.data.Dataset.from_tensor_slices(all_patch_volumes_tensors_)

    # DIVIDE dataset into batches (only needed for dim. compatibility, but not very relevant since we are doing just inference and not training again)
    batched_dataset_ = dataset.batch(unet_batch_size)

    # ADD channel dimension to datasets through the "map" method of tf.data.Dataset
    # This ensures that the patches are readable by the first conv layer and that labels have comparable shape with the output of the net.
    batched_dataset_ = batched_dataset_.map(lambda patches: (tf.expand_dims(patches, axis=-1)))

    return batched_dataset_


def create_unet(inputs_, conv_filters):
    """This function creates a 3D U-Net starting from an input with modifiable dimensions and it returns the compiled model
    Args:
        inputs_ (tf.keras.Input): input to the model; used to set the input dimensions
        conv_filters (list): optimal number of convolutional filters found with optuna on the validation set
    Returns:
        model (tf.keras.Model): the U-Net
    """
    # DOWNWARD PATH
    conv1 = tf.keras.layers.Conv3D(conv_filters[0], 3, activation='relu', padding='same', data_format="channels_last")(inputs_)
    conv1 = tf.keras.layers.Conv3D(conv_filters[0], 3, activation='relu', padding='same')(conv1)
    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    pool1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(bn1)
    conv2 = tf.keras.layers.Conv3D(conv_filters[1], 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv3D(conv_filters[1], 3, activation='relu', padding='same')(conv2)
    bn2 = tf.keras.layers.BatchNormalization()(conv2)
    pool2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(bn2)
    conv3 = tf.keras.layers.Conv3D(conv_filters[2], 3, activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv3D(conv_filters[2], 3, activation='relu', padding='same')(conv3)
    bn3 = tf.keras.layers.BatchNormalization()(conv3)
    pool3 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(bn3)

    conv4 = tf.keras.layers.Conv3D(conv_filters[3], 3, activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv3D(conv_filters[3], 3, activation='relu', padding='same')(conv4)
    bn4 = tf.keras.layers.BatchNormalization()(conv4)

    # UPWARD PATH
    up5 = tf.keras.layers.Conv3D(conv_filters[2], 2, activation='relu', padding='same')(tf.keras.layers.UpSampling3D(size=(2, 2, 2))(bn4))
    merge5 = tf.keras.layers.Concatenate(axis=-1)([bn3, up5])
    conv5 = tf.keras.layers.Conv3D(conv_filters[2], 3, activation='relu', padding='same')(merge5)
    conv5 = tf.keras.layers.Conv3D(conv_filters[2], 3, activation='relu', padding='same')(conv5)
    bn5 = tf.keras.layers.BatchNormalization()(conv5)
    up6 = tf.keras.layers.Conv3D(conv_filters[1], 2, activation='relu', padding='same')(tf.keras.layers.UpSampling3D(size=(2, 2, 2))(bn5))
    merge6 = tf.keras.layers.Concatenate(axis=-1)([bn2, up6])
    conv6 = tf.keras.layers.Conv3D(conv_filters[1], 3, activation='relu', padding='same')(merge6)
    conv6 = tf.keras.layers.Conv3D(conv_filters[1], 3, activation='relu', padding='same')(conv6)
    bn6 = tf.keras.layers.BatchNormalization()(conv6)
    up7 = tf.keras.layers.Conv3D(conv_filters[0], 2, activation='relu', padding='same')(tf.keras.layers.UpSampling3D(size=(2, 2, 2))(bn6))
    merge7 = tf.keras.layers.Concatenate(axis=-1)([bn1, up7])
    conv7 = tf.keras.layers.Conv3D(conv_filters[0], 3, activation='relu', padding='same')(merge7)
    conv7 = tf.keras.layers.Conv3D(conv_filters[0], 3, activation='relu', padding='same')(conv7)
    bn7 = tf.keras.layers.BatchNormalization()(conv7)
    outputs_ = tf.keras.layers.Conv3D(1, 1, activation='sigmoid')(bn7)

    model = tf.keras.Model(inputs=inputs_, outputs=outputs_)

    return model


def save_volume_mask_to_disk(input_volume, out_path, nii_aff, output_filename):
    """This function saves "input_volume" to disk
    Args:
        input_volume (np.ndarray): volume to be saved
        out_path (str): path where the volume will be saved
        nii_aff (np.ndarray): affine matrix of volume that we want to save
        output_filename (str): name to assign to the volume we save to disk
    Returns:
        None
    """
    volume_obj = nib.Nifti1Image(input_volume, nii_aff)  # convert from numpy to nib object
    if not os.path.exists(out_path):  # if output path does not exist
        os.makedirs(out_path)  # create it

    nib.save(volume_obj, os.path.join(out_path, output_filename))  # save mask


def reduce_fp_in_segm_map(txt_file_path, output_folder_path, out_filename, mapping_centers_masks):
    """This function removes the less probable predictions (i.e. connected components) basing on
    the corresponding txt file that contains the already-reduced predicted aneurysm centers
    Args:
        txt_file_path (str): path to txt output file
        output_folder_path (str): path to output folder for this subject
        out_filename (str): filename of the output segmentation mask
        mapping_centers_masks (dict): it contains the centers of the predictions as keys and the coordinates of non-zero voxels as values
    Raises:
        AssertionError: if the created segmentation mask is not binary
    """
    binary_segm_map_obj = nib.load(os.path.join(output_folder_path, "result.nii.gz"))  # type: nib.nifti1.Nifti1Image
    binary_segm_map = np.asanyarray(binary_segm_map_obj.dataobj)  # type: np.ndarray # load output segmentation map as np.array
    new_segm_map = np.zeros(binary_segm_map.shape, dtype=int)  # type: np.ndarray # create new output segmentation map, initially filled with zeros

    if not os.stat(txt_file_path).st_size == 0:  # if the output file is not empty (i.e. there's at least one predicted aneurysm location)
        df_txt_file = pd.read_csv(txt_file_path, header=None)  # type: pd.DataFrame # load txt file with pandas

        # loop over dataframe's rows
        for row_idxs in range(df_txt_file.shape[0]):
            pred_center = tuple(df_txt_file.iloc[row_idxs].values)  # extract row
            non_zero_coords = mapping_centers_masks[pred_center]
            new_segm_map[non_zero_coords] = 1

        # make sure that the new output mask is binary
        assert np.array_equal(new_segm_map, new_segm_map.astype(bool)), "WATCH OUT: mask is not binary"

        # sanity check: control that we have same number of connected components
        labels_out = cc3d.connected_components(np.asarray(new_segm_map, dtype=int))  # extract 3D connected components
        numb_labels = np.max(labels_out)  # extract number of different connected components found
        assert numb_labels == df_txt_file.shape[0], "Mismatch between output files: {} connected components and {} centers".format(numb_labels, df_txt_file.shape[0])

        # OVERWRITE previous binary mask
        save_volume_mask_to_disk(new_segm_map, output_folder_path, binary_segm_map_obj.affine, out_filename)

    else:  # if instead the txt file is empty, we just save an empty binary mask
        # OVERWRITE previous binary mask
        save_volume_mask_to_disk(new_segm_map, output_folder_path, binary_segm_map_obj.affine, out_filename)


def resample_volume_inverse(volume_path, new_spacing, new_size, out_path, interpolator=sitk.sitkLinear):
    """This function resamples the input volume to a specified voxel spacing and to a new size
    Args:
        volume_path (str): input volume path
        new_spacing (list): desired voxel spacing that we want
        new_size (list): size of output volume
        out_path (str): path where we temporarily save the resampled output volume
        interpolator (int): interpolator that we want to use (e.g. 1=NearNeigh., 2=linear, ...)
    Returns:
        resampled_volume_sitk_obj (sitk.Image): resampled volume as sitk object
        resampled_volume_nii_obj (nib.Nifti1Image): resampled volume as nib object
        resampled_volume_nii (np.ndarray): resampled volume as numpy array
    """
    volume = sitk.ReadImage(volume_path)  # read volume
    resampled_volume_sitk_obj = sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                                              volume.GetOrigin(), new_spacing, volume.GetDirection(), 0,
                                              volume.GetPixelID())
    sitk.WriteImage(resampled_volume_sitk_obj, out_path)  # write sitk volume object to disk
    resampled_volume_nii_obj = nib.load(out_path)  # type: nib.Nifti1Image # load volume as nibabel object
    resampled_volume_nii = np.asanyarray(resampled_volume_nii_obj.dataobj)  # type: np.ndarray # convert from nibabel object to np.array
    os.remove(out_path)  # remove volume from disk to save space

    return resampled_volume_sitk_obj, resampled_volume_nii_obj, resampled_volume_nii


def create_txt_output_file_and_remove_dark_fp(output_volume_path, txt_file_path, original_bfc_bet_tof, remove_dark_fp, dark_fp_threshold):
    """This function creates the output txt file used for the detection task and returns the binarized segmentation output volume
    Args:
        output_volume_path (str): path to output segmentation volume
        txt_file_path (str): path of detection output file
        original_bfc_bet_tof (np.ndarray): original bias-field-corrected volume after brain extraction before resampling
        remove_dark_fp (bool): if set to True, candidate aneurysms that are not brighter than a certain threshold (on average) are discarded
        dark_fp_threshold (float): threshold used to remove predictions which are on average too dark for being a true aneurysm
    Returns:
        binary_output_segm_map (np.ndarray): binary output segmentation mask
        avg_brightness_predicted_aneurysms (list): it contains the mean brightness of non-zero-voxels of the retained candidate aneurysms
        mapping_centers_masks (dict): it contains the centers of the predictions as keys and the coordinates of non-zero voxels as values; used later for FP reduction
    """
    probabilistic_out_segm_map_obj = nib.load(output_volume_path)  # type: nib.Nifti1Image # load output segmentation with nibabel
    probabilistic_out_segm_map = np.asanyarray(probabilistic_out_segm_map_obj.dataobj)  # type: np.ndarray # extract numpy array

    # assign 1 to voxels that are non-zero
    binary_output_segm_map = np.asarray(np.where(probabilistic_out_segm_map != 0, 1, 0), dtype=int)
    # binary_output_segm_map = np.asarray(np.where(probabilistic_out_segm_map >= unet_threshold, 1, 0), dtype=int)

    labels_out = cc3d.connected_components(binary_output_segm_map)
    numb_labels = np.max(labels_out)  # extract number of different connected components found

    # create output txt file
    with open(txt_file_path, "w") as txt_file:
        wr = csv.writer(txt_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        avg_brightness_predicted_aneurysms = []  # type: list # will contain the mean brightness of non-zero-voxels of the retained candidate aneurysms
        mapping_centers_masks = {}  # type: dict # create a map between a predicted center and the corresponding connected components; will be used later for the FP reduction
        # loop over different conn. components
        for seg_id in range(1, numb_labels + 1):
            # extract connected component from the probabilistic predicted volume
            extracted_image = probabilistic_out_segm_map * (labels_out == seg_id)
            # extract same connected component but from original bias-field-corrected tof volume after brain extraction
            extracted_original_bet_tof = original_bfc_bet_tof * (labels_out == seg_id)
            assert extracted_image.shape == extracted_original_bet_tof.shape
            assert np.max(extracted_image) <= 1, "This should be a probabilistic map with values between 0 and 1"

            if remove_dark_fp:
                p90 = np.percentile(original_bfc_bet_tof, [90])[0]  # find xxth intensity percentile for this subject
                # only retain non-zero voxels (i.e. voxels belonging to the aneurysm mask)
                non_zero_voxels_intensities = extracted_original_bet_tof[np.nonzero(extracted_original_bet_tof)]  # type: np.ndarray
                # compute ratio between mean intensity of predicted aneurysm voxels and the xxth intensity percentile of the entire skull-stripped image
                intensity_ratio = np.mean(non_zero_voxels_intensities) / p90
                if intensity_ratio > dark_fp_threshold:
                    # extract voxel coordinates of the center of the predicted lesion
                    predicted_center = center_of_mass(extracted_image)  # type: tuple
                    # round to closest int
                    pred_center_tof_space = [round_half_up(x) for x in predicted_center]  # type: list
                    pred_center_tof_space_int = np.asarray(pred_center_tof_space, dtype=int)  # type: np.ndarray # cast to int
                    wr.writerow(pred_center_tof_space_int)  # write aneurysm center as a row

                    # update mapping; it contains the centers of the predictions as keys and the coordinates of non-zero voxels as values
                    mapping_centers_masks[tuple(pred_center_tof_space_int)] = np.nonzero(extracted_image)

                    # compute average intensity of this predicted aneurysm
                    avg_brightness_prob_larg_conn_comp = np.mean(extracted_image[np.nonzero(extracted_image)])
                    avg_brightness_predicted_aneurysms.append(avg_brightness_prob_larg_conn_comp)

            # if instead we do not discard dark fp predictions
            else:
                # extract voxel coordinates of the center of the predicted lesion
                predicted_center = center_of_mass(extracted_image)  # type: tuple
                # round to closest int
                pred_center_tof_space = [round_half_up(x) for x in predicted_center]  # type: list
                pred_center_tof_space_int = np.asarray(pred_center_tof_space, dtype=int)  # type: np.ndarray # cast to int
                wr.writerow(pred_center_tof_space_int)  # write aneurysm center as a row

                # update mapping; it contains the centers of the predictions as keys and the coordinates of non-zero voxels as values
                mapping_centers_masks[tuple(pred_center_tof_space_int)] = np.nonzero(extracted_image)

                # compute average intensity of this predicted aneurysm
                avg_brightness_prob_larg_conn_comp = np.mean(extracted_image[np.nonzero(extracted_image)])
                avg_brightness_predicted_aneurysms.append(avg_brightness_prob_larg_conn_comp)

    return binary_output_segm_map, avg_brightness_predicted_aneurysms, mapping_centers_masks


def extract_largest_conn_comp(predicted_patch, predicted_patch_thresholded_binary):
    """This function takes as input the predicted patch and its binary version and returns the same patches but only with the largest connected component
    Args:
        predicted_patch (np.ndarray): probabilistic predicted patch
        predicted_patch_thresholded_binary (np.ndarray): thresholded binary version of predicted patch
    Returns:
        probabilistic_largest_conn_comp (np.ndarray): probabilistic predicted patch with only the largest connected component
        binary_largest_conn_comp (np.ndarray): thresholded binary version of predicted patch with only the largest connected component
    """
    # find connected components in 3D numpy array; each connected component will be assigned a label starting from 1 and then increasing, 2, 3, etc.
    labels_out = cc3d.connected_components(np.asarray(predicted_patch_thresholded_binary, dtype=int))
    numb_labels = np.max(labels_out)  # extract number of different connected components found

    non_zero_voxels = []  # type: list # will contain the number of non-zero voxels in each connected component
    for seg_id in range(1, numb_labels + 1):  # loop over different conn. components
        extracted_image = labels_out * (labels_out == seg_id)  # extract connected component
        non_zero_voxels.append(np.count_nonzero(extracted_image))  # append nb. of non-zero voxels for this component

    # extract intensity value of the connected component with more non-zero voxels
    largest_conn_comp_value = np.argmax(non_zero_voxels) + 1
    # compute the probabilistic largest connected component
    probabilistic_largest_conn_comp = predicted_patch * (labels_out == largest_conn_comp_value)
    # compute binary largest connected component
    binary_largest_conn_comp = predicted_patch_thresholded_binary * (labels_out == largest_conn_comp_value)

    return probabilistic_largest_conn_comp, binary_largest_conn_comp


def reduce_false_positives(txt_file_path, mean_brightness_retained_patches, max_fp, output_folder_path, out_filename, mapping_centers_masks):
    """If more than "max_fp" candide centers were predicted, this function reduces this number to exactly "max_fp", retaining only the most probable
    (i.e. brightest) candidate aneurysm centers. The rationale behind this choice is that it's extremely unlikely that a subject has more than "max_fp"
    aneurysms in one scan.
    Args:
        txt_file_path (str): path to the output txt file
        mean_brightness_retained_patches (list): contains the average intensity of the patches corresponding to the predicted centers
        max_fp (int): maximum allowed number of aneurysms per patient
        output_folder_path (str): path to output folder for this subject
        out_filename (str): filename of the output segmentation mask
        mapping_centers_masks (dict): it contains the centers of the predictions as keys and the coordinates of non-zero voxels as values
    Returns:
        None
    Raises:
        AssertionError: if input path does not exist
    """
    assert os.path.exists(txt_file_path), "Path {} does not exist".format(txt_file_path)
    if not os.stat(txt_file_path).st_size == 0:  # if the output file is not empty (i.e. there's at least one predicted aneurysm location)
        df_txt_file = pd.read_csv(txt_file_path, header=None)  # type: pd.DataFrame # load txt file with pandas
        new_rows = []  # type: list # will contain the new rows of the output file
        if df_txt_file.shape[0] > max_fp:  # if the dataframe has more than max_fp rows (i.e. if there are more than max_fp predicted aneurysms)
            # find indexes corresponding to the "max_fp" most probable aneurysms (i.e. the brightest)
            idxs_brightest_patches = sorted(range(len(mean_brightness_retained_patches)), key=lambda k: mean_brightness_retained_patches[k])[-max_fp:]
            for idx in idxs_brightest_patches:
                lesion_center = np.asarray(df_txt_file.iloc[idx])
                new_rows.append(lesion_center)

            # save modified txt file (it overwrites the previous one by default)
            np.savetxt(txt_file_path, np.asarray(new_rows), delimiter=",", fmt='%i')

            # also remove less probable connected components from segmentation map
            reduce_fp_in_segm_map(txt_file_path, output_folder_path, out_filename, mapping_centers_masks)


def save_output_mask_and_output_location(segm_map_resampled, output_folder_path_, aff_mat_resampled, orig_bfc_angio_sitk, tmp_path,
                                         txt_file_path, original_bet_bfc_angio, remove_dark_fp, reduce_fp, max_fp, dark_fp_threshold):
    """This function saves the output mask (resampled back to original space) to disk. Plus, it also saves the corresponding location file (with the candidate aneurysm(s) center(s))
    Args:
        segm_map_resampled (np.ndarray): resampled segmentation map to save
        output_folder_path_ (str): folder path where we'll save the output file
        aff_mat_resampled (np.ndarray): affine matrix of resampled space
        orig_bfc_angio_sitk (sitk.Image): sitk volume of the bias-field-corrected, non-resampled angio
        tmp_path (str): path where we save temporary files
        txt_file_path (str): path to txt file containing the aneurysm(s) center location(s)
        original_bet_bfc_angio (np.ndarray): bias-field-corrected angio volume after BET before resampling
        remove_dark_fp (bool): if set to True, candidate aneurysms that are not brighter than a certain threshold (on average) are discarded
        reduce_fp (bool): if set to true, we only retain the "max_fp" most probable aneurysm candidates
        max_fp (int): max numb. of aneurysms retained
        dark_fp_threshold (float): threshold used to remove predictions which are on average too dark for being a true aneurysm
    Returns:
        None
    """
    # save resampled probabilistic segmentation mask to disk
    out_filename = "result.nii.gz"  # type: str # define filename of predicted binary segmentation volume
    out_filename_probabilistic = "probabilistic_result.nii.gz"   # type: str # define filename of predicted probabilistic segmentation volume
    save_volume_mask_to_disk(segm_map_resampled, output_folder_path_, aff_mat_resampled, out_filename)

    # extract voxel spacing of original (i.e. non-resampled) angio volume
    original_spacing = list(orig_bfc_angio_sitk.GetSpacing())
    # extract size of original (i.e. non-resampled) angio volume
    original_size = list(orig_bfc_angio_sitk.GetSize())
    # create output file for registration
    out_path = os.path.join(tmp_path, "result_mask_original_space.nii.gz")
    # resample output volume to original spacing
    _, segm_map_nib_obj, segm_map = resample_volume_inverse(os.path.join(output_folder_path_, out_filename),
                                                            original_spacing,
                                                            original_size,
                                                            out_path,
                                                            interpolator=sitk.sitkNearestNeighbor)  # set near-neighb. interpolator to avoid holes in the mask

    # SAVE probabilistic segmentation output map in original space
    save_volume_mask_to_disk(segm_map, output_folder_path_, segm_map_nib_obj.affine, out_filename_probabilistic)

    # create output txt file with probabilistic segmentation mask in original (i.e. non-resampled) space
    segm_map_binary, avg_brightness_predicted_aneurysms, mapping_centers_masks = create_txt_output_file_and_remove_dark_fp(os.path.join(output_folder_path_, out_filename_probabilistic),
                                                                                                                           txt_file_path,
                                                                                                                           original_bet_bfc_angio,
                                                                                                                           remove_dark_fp,
                                                                                                                           dark_fp_threshold)

    # make sure mask is binary
    assert np.array_equal(segm_map_binary, segm_map_binary.astype(bool)), "WATCH OUT: mask is not binary in original space"

    # save binary output mask, OVERWRITING previous probabilistic one
    save_volume_mask_to_disk(segm_map_binary, output_folder_path_, segm_map_nib_obj.affine, out_filename)

    # also remove dark FP from segmentation map
    if remove_dark_fp:
        reduce_fp_in_segm_map(txt_file_path, output_folder_path_, out_filename, mapping_centers_masks)

    if reduce_fp:
        # reduce FPs to a maximum of max_fp, only retaining the most probable patch centers
        reduce_false_positives(txt_file_path, avg_brightness_predicted_aneurysms, max_fp, output_folder_path_, out_filename, mapping_centers_masks)


def create_output_folder(batched_ds, output_folder_path_, unet_threshold, unet, resampled_nii_volume_after_bet,
                         reduce_fp_with_volume, min_aneurysm_volume, nii_volume_obj_after_bet_resampled, patch_center_coords, shift_scale_1, orig_bfc_angio_sitk,
                         aff_mat_resampled, tmp_path, reduce_fp, max_fp, remove_dark_fp, dark_fp_threshold, original_bet_bfc_angio, sub_ses, test_time_augmentation, unet_batch_size):
    """This function creates the output file "result.txt" containing the voxel coordinates of the center of the predicted aneurysm(s);
    also, it creates the output file result.nii.gz which is the predicted segmentation mask for the subject being analyzed.
    Args:
        batched_ds (tf.data.Dataset): dataset to evaluate
        output_folder_path_ (str): folder path where we'll save the output file
        unet_threshold (float): threshold used for obtaining binary volume from the probabilistic output of the UNet
        unet (tf.keras.Model): trained network that we use for inference
        resampled_nii_volume_after_bet (np.ndarray): resampled bias-field-corrected angio volume after BET
        reduce_fp_with_volume (bool): if set to True, only the candidate lesions that have a volume (mm^3) > than a specific threshold are retained
        min_aneurysm_volume (float): minimum aneurysm volume; if the model predicts an aneurysm with volume smaller than this, this prediction is removed
        nii_volume_obj_after_bet_resampled ((Nifti1Image): nibabel object of the resampled bias-field-corrected BET angio volume
        patch_center_coords (list): it contains the center coordinates of the retained patches (i.e. patches that fulfilled the extraction criteria)
        shift_scale_1 (int): half patch side
        orig_bfc_angio_sitk (sitk.Image): sitk volume of the bias-field-corrected, non-resampled angio
        aff_mat_resampled (np.ndarray): affine matrix of resampled space
        tmp_path (str): path where we save temporary files
        reduce_fp (bool): if set to true, we only retain the "max_fp" most probable aneurysm candidates
        max_fp (int): max numb. of aneurysms retained
        remove_dark_fp (bool): if set to True, candidate aneurysms that are not brighter than a certain threshold (on average) are discarded
        dark_fp_threshold (float): threshold used to remove predictions which are on average too dark for being a true aneurysm
        original_bet_bfc_angio (np.ndarray): bias-field-corrected angio volume after BET before resampling
        sub_ses (str): subject id and session id
        test_time_augmentation (bool): whether to perform test-time augmentation
        unet_batch_size (int): batch size. Not really relevant (cause we're doing inference), but still needed
    """
    # start_out_dir = time.time()

    if not os.path.exists(output_folder_path_):  # if output path does not exist
        os.makedirs(output_folder_path_)  # create it

    # if at least one patch was retained
    if len(patch_center_coords) > 0:

        if test_time_augmentation:  # if we want to apply test-time augmentation
            # start_tta = time.time()
            pred_patches = compute_test_time_augmentation(batched_ds, unet, unet_batch_size, aff_mat_resampled)  # type: np.ndarray # compute the average dataset across augmentations
            # end_tta = time.time()
            # print_running_time(start_tta, end_tta, "Test-time augmentation {}".format(sub_ses))
        else:  # if instead we don't want to perform test-time augmentation
            # compute U-Net's predictions
            pred_patches = unet.predict(batched_ds)  # type: np.ndarray
            # remove redundant dimension (i.e. extra channel dim)
            pred_patches = np.squeeze(pred_patches, axis=-1)  # type: np.ndarray

        # create empty segmentation map of entire volume
        segm_map_resampled = np.zeros(resampled_nii_volume_after_bet.shape, dtype=np.float32)  # type: np.ndarray
        # create empty overlap map to keep track of the voxels where there is overlapping
        overlap_map_resampled = np.zeros(segm_map_resampled.shape, dtype=np.float32)  # type: np.ndarray

        # compute volume of one voxel
        if reduce_fp_with_volume:
            voxel_dimensions = (nii_volume_obj_after_bet_resampled.header["pixdim"])[1:4]  # extract voxel size to later compute the volume
            voxel_volume = np.prod(voxel_dimensions)

        txt_file_path = os.path.join(output_folder_path_, "result.txt")
        # loop over all the anatomically-plausible patches that were retained in the sliding-window approach
        for index, pred_patch in enumerate(pred_patches):

            # set to 1 all voxels > threshold and to 0 all those < threshold; then, cast to int
            pred_patch_thresholded_binary = np.asarray(np.where(pred_patch > unet_threshold, 1, 0), dtype=int)

            if reduce_fp_with_volume:
                # if after thresholding there are still non-zero voxels, we consider it an aneurysm
                if np.count_nonzero(pred_patch_thresholded_binary) > 0:
                    # only retain largest connected component of the patch (both probabilistic and binary)
                    probabilistic_largest_conn_comp, binary_largest_conn_comp = extract_largest_conn_comp(pred_patch, pred_patch_thresholded_binary)

                    # compute volume of predicted aneurysm
                    nonzero_voxel_count = np.count_nonzero(binary_largest_conn_comp)  # compute number of non-zero voxels in largest conn. comp. volume
                    aneur_volume = nonzero_voxel_count * voxel_volume

                    # impose a minimum threshold for the aneurysm volume in mm^3. This threshold is the 5th percentile of training ADAM masks' volumes
                    if aneur_volume > min_aneurysm_volume:
                        center_tof_space = patch_center_coords[index]  # find corresponding patch center in TOF space
                        segm_map_resampled[center_tof_space[0]-shift_scale_1:center_tof_space[0]+shift_scale_1,
                                           center_tof_space[1]-shift_scale_1:center_tof_space[1]+shift_scale_1,
                                           center_tof_space[2]-shift_scale_1:center_tof_space[2]+shift_scale_1] += probabilistic_largest_conn_comp
                        overlap_map_resampled[center_tof_space[0]-shift_scale_1:center_tof_space[0]+shift_scale_1,
                                              center_tof_space[1]-shift_scale_1:center_tof_space[1]+shift_scale_1,
                                              center_tof_space[2]-shift_scale_1:center_tof_space[2]+shift_scale_1] += binary_largest_conn_comp

            # if we don't apply a FP reduction by volume
            else:
                # if after thresholding there are still non-zero voxels, we consider it an aneurysm
                if np.count_nonzero(pred_patch_thresholded_binary) > 0:
                    # only retain largest connected component of the patch (both probabilistic and binary)
                    probabilistic_largest_conn_comp, binary_largest_conn_comp = extract_largest_conn_comp(pred_patch, pred_patch_thresholded_binary)

                    center_tof_space = patch_center_coords[index]  # find corresponding patch center in TOF space
                    segm_map_resampled[center_tof_space[0] - shift_scale_1:center_tof_space[0] + shift_scale_1,
                                       center_tof_space[1] - shift_scale_1:center_tof_space[1] + shift_scale_1,
                                       center_tof_space[2] - shift_scale_1:center_tof_space[2] + shift_scale_1] += probabilistic_largest_conn_comp
                    overlap_map_resampled[center_tof_space[0] - shift_scale_1:center_tof_space[0] + shift_scale_1,
                                          center_tof_space[1] - shift_scale_1:center_tof_space[1] + shift_scale_1,
                                          center_tof_space[2] - shift_scale_1:center_tof_space[2] + shift_scale_1] += binary_largest_conn_comp

        # average overlapping predictions (if there were any); divide only for non-zero voxels in order to avoid divisions by 0
        segm_map_resampled_avg = np.divide(segm_map_resampled, overlap_map_resampled, out=np.zeros_like(segm_map_resampled), where=overlap_map_resampled != 0)

        save_output_mask_and_output_location(segm_map_resampled_avg, output_folder_path_, aff_mat_resampled, orig_bfc_angio_sitk, tmp_path,
                                             txt_file_path, original_bet_bfc_angio, remove_dark_fp, reduce_fp, max_fp, dark_fp_threshold)

        # end_out_dir = time.time()
        # print_running_time(start_out_dir, end_out_dir, "Output dir creation {}".format(sub_ses))

    elif len(patch_center_coords) == 0:
        print("WARNING: no patch was retained for {}; predicting empty segmentation volume".format(sub_ses))
        # create empty segmentation map of entire volume
        segm_map_resampled = np.zeros(resampled_nii_volume_after_bet.shape, dtype=np.float32)  # type: np.ndarray

        txt_file_path = os.path.join(output_folder_path_, "result.txt")
        save_output_mask_and_output_location(segm_map_resampled, output_folder_path_, aff_mat_resampled, orig_bfc_angio_sitk, tmp_path,
                                             txt_file_path, original_bet_bfc_angio, remove_dark_fp, reduce_fp, max_fp, dark_fp_threshold)
    else:
        raise ValueError("Unexpected value for len(patch_center_coords): {}".format(len(patch_center_coords)))


def check_registration_quality(quality_metrics_thresholds, sub_quality_metrics):
    """This function checks whether the registration was correct or not for the subject being evaluated.
    Args:
        quality_metrics_thresholds (tuple): registration quality metrics; specifically, it contains (p3_neigh_corr_struct_2_tof, p97_mut_inf_struct_2_tof)
        sub_quality_metrics (tuple): it contains the registration quality metrics for the subject being evaluated; specifically, (struct_2_tof_nc, struct_2_tof_mi)
    Returns:
        registration_accurate_enough (bool): it indicates whether the registration accuracy is high enough to perform the anatomically-informed sliding window
    """
    registration_accurate_enough = False

    # if the reg_metrics of this subject are not outliers, then the registration was most probably correct
    if sub_quality_metrics[0] > quality_metrics_thresholds[0] and sub_quality_metrics[1] < quality_metrics_thresholds[1]:
        registration_accurate_enough = True

    return registration_accurate_enough


def get_result_filename_coord_file(dirname):
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


def get_images(test_filename, result_filename):
    """Return the test and result images, thresholded and treated aneurysms removed."""
    test_image = sitk.ReadImage(test_filename)
    result_image = sitk.ReadImage(result_filename)

    assert test_image.GetSize() == result_image.GetSize()

    # Get meta data from the test-image, needed for some sitk methods that check this
    result_image.CopyInformation(test_image)

    # Remove treated aneurysms from the test and result images, since we do not evaluate on this
    treated_image = test_image != 2  # treated aneurysms == 2
    masked_result_image = sitk.Mask(result_image, treated_image)
    masked_test_image = sitk.Mask(test_image, treated_image)

    # Return two binary masks
    return masked_test_image > 0.5, masked_result_image > 0.5


def get_result_filename_volume(dirname):
    """Find the filename of the result image.

    This should be result.nii.gz or result.nii. If these files are not present,
    it tries to find the closest filename."""

    files = os.listdir(dirname)

    if not files:
        raise Exception("No results in " + dirname)

    # Find the filename that is closest to either 'result.nii.gz' or 'result.nii'.
    ratios = [[SequenceMatcher(a=a, b=b).ratio() for b in ['result.nii.gz', 'result.nii']] for a in files]
    result_filename = files[int(np.argmax(np.max(ratios, axis=1)))]

    # Return the full path to the file.
    return os.path.join(dirname, result_filename)


def get_dsc(test_image, result_image):
    """Compute the Dice Similarity Coefficient."""
    test_array = sitk.GetArrayFromImage(test_image).flatten()
    result_array = sitk.GetArrayFromImage(result_image).flatten()

    test_sum = np.sum(test_array)
    result_sum = np.sum(result_array)

    if test_sum == 0 and result_sum == 0:
        # Perfect result in case of no aneurysm
        return np.nan
    elif test_sum == 0 and not result_sum == 0:
        # Some segmentations, while there is no aneurysm
        return 0
    else:
        # There is an aneurysm, return similarity = 1.0 - dissimilarity
        return 1.0 - scipy.spatial.distance.dice(test_array, result_array)


def get_hausdorff(test_image, result_image):
    """Compute the Hausdorff distance."""

    def get_distances_from_a_to_b(a, b):
        kd_tree = scipy.spatial.KDTree(a, leafsize=100)
        return kd_tree.query(b, k=1, eps=0, p=2)[0]

    result_statistics = sitk.StatisticsImageFilter()
    result_statistics.Execute(result_image)

    test_statistics = sitk.StatisticsImageFilter()
    test_statistics.Execute(test_image)

    if result_statistics.GetSum() == 0 or test_statistics.GetSum() == 0:
        hd = np.nan
        return hd

    # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 3D
    e_test_image = sitk.BinaryErode(test_image, (1, 1, 1))
    e_result_image = sitk.BinaryErode(result_image, (1, 1, 1))

    # save eroded volumes to disk (for debugging)
    # sitk.WriteImage(e_test_image, "/home/newuser/Desktop/eroded_ground_truth.nii.gz")
    # sitk.WriteImage(e_result_image, "/home/newuser/Desktop/eroded_prediction.nii.gz")

    h_test_image = sitk.Subtract(test_image, e_test_image)
    h_result_image = sitk.Subtract(result_image, e_result_image)

    # save difference volumes to disk (for debugging)
    # sitk.WriteImage(h_test_image, "/home/newuser/Desktop/difference_ground_truth.nii.gz")
    # sitk.WriteImage(h_result_image, "/home/newuser/Desktop/difference_prediction.nii.gz")

    # save indexes of non-zero voxels of the edge detection/subtraction images
    h_test_indices = np.flip(np.argwhere(sitk.GetArrayFromImage(h_test_image))).tolist()
    h_result_indices = np.flip(np.argwhere(sitk.GetArrayFromImage(h_result_image))).tolist()

    # save indexes in physical space
    test_coordinates = [test_image.TransformIndexToPhysicalPoint(x) for x in h_test_indices]
    result_coordinates = [test_image.TransformIndexToPhysicalPoint(x) for x in h_result_indices]

    d_test_to_result = get_distances_from_a_to_b(test_coordinates, result_coordinates)
    d_result_to_test = get_distances_from_a_to_b(result_coordinates, test_coordinates)

    hd = max(np.percentile(d_test_to_result, 95), np.percentile(d_result_to_test, 95))

    return hd


def get_vs(test_image, result_image):
    """Volumetric Similarity.

    VS = 1 -abs(A-B)/(A+B)

    A = ground truth
    B = predicted
    """

    test_statistics = sitk.StatisticsImageFilter()
    result_statistics = sitk.StatisticsImageFilter()

    test_statistics.Execute(test_image)
    result_statistics.Execute(result_image)

    numerator = abs(test_statistics.GetSum() - result_statistics.GetSum())
    denominator = test_statistics.GetSum() + result_statistics.GetSum()

    if denominator > 0:
        vs = 1 - float(numerator) / denominator
    else:
        vs = np.nan

    return vs


def compute_patient_wise_metrics(out_dir, ground_truth_dir, sub, ses):
    """This function computes detection and segmentation metrics for one subject with the scripts of the ADAM challenge
    Args:
        out_dir (str): path to folder where the output files of this subject are stored
        ground_truth_dir (str): path to folder containing ground truth segmentation volume and location file of this subject
        sub (str): ipp of subject being evaluated
        ses (str): session of subject being evaluated (i.e. exam date)
    Returns:
        out_metrics (pd.Dataframe): it contains detection and segmentation metrics in the order (sensitivity, FP rate, DSC, HD95, VS)
    """
    # --------------- DETECTION ---------------
    result_filename_coord_file = get_result_filename_coord_file(out_dir)
    test_locations = get_locations(os.path.join(ground_truth_dir, 'location.txt'))
    result_locations = get_result(result_filename_coord_file)
    test_image = sitk.ReadImage(os.path.join(ground_truth_dir, 'aneurysms.nii.gz'))
    sensitivity, false_positive_count = get_detection_metrics(test_locations, result_locations, test_image)

    # -------------- SEGMENTATION --------------
    result_filename_volume = get_result_filename_volume(out_dir)
    test_image, result_image = get_images(os.path.join(ground_truth_dir, 'aneurysms.nii.gz'), result_filename_volume)
    dsc = get_dsc(test_image, result_image)
    h95 = get_hausdorff(test_image, result_image)
    vs = get_vs(test_image, result_image)

    # ----------- merge metrics together -----------
    out_list = [sensitivity, false_positive_count, dsc, h95, vs]  # type: list
    out_metrics = pd.DataFrame([out_list], columns=["Sens", "FP count", "DSC", "HD95", "VS"], index=["{}_{}".format(sub, ses)])  # type: pd.DataFrame

    return out_metrics


def save_and_print_results(metrics_cv_fold, inference_outputs_path, out_date):
    """This function saves output results to disk and prints them.
    Args:
        metrics_cv_fold (list): it contains the output metrics of all the subjects for each fold
        inference_outputs_path (str): path to folder where we save all the outputs of the patient-wise analysis
        out_date (str): today's date
    """
    # create unique dataframe with the metrics of all subjects
    out_metrics_df = pd.concat(metrics_cv_fold)
    # save output dataframe to disk
    out_metrics_df.to_csv(os.path.join(inference_outputs_path, "all_folds_out_metrics_{}.csv".format(out_date)))

    # uncomment print below for debugging
    # print(out_metrics_df)

    print("\n---------------------------")
    print("CUMULATIVE PERFORMANCES OVER {} subjects".format(out_metrics_df.shape[0]))
    print("Average Sensitivity: {:.3f}".format(out_metrics_df["Sens"].mean()))
    print("False Positives: {} (avg: {:.1f} per subject)".format(out_metrics_df["FP count"].sum(), out_metrics_df["FP count"].mean()))
    print("Average DSC: {:.3f}".format(out_metrics_df["DSC"].mean()))
    print("Average HD95: {:.3f}".format(out_metrics_df["HD95"].mean()))
    print("Average VS: {:.3f}".format(out_metrics_df["VS"].mean()))


def convert_mni_to_angio(df_landmarks, bfc_angio_volume_sitk, tmp_path, mni_2_struct_mat_path, struct_2_tof_mat_path, mni_2_struct_inverse_warp_path):
    """This function registers the landmark points to subject space (i.e. angio-TOF space)
    Args:
        df_landmarks (pd.Dataframe): it contains the 3D coordinates of the MNI landmark points in mm
        bfc_angio_volume_sitk (sitk.Image): original bfc tof volume loaded with sitk; used for debugging
        tmp_path (str): path to folder where we save temporary registration files
        mni_2_struct_mat_path (str): path to MNI_2_struct .mat file
        struct_2_tof_mat_path (str): path to struct_2_TOF .mat file
        mni_2_struct_inverse_warp_path (str): path to mni_2_struct inverse warp field
    Returns:
        df_landmark_points_tof_space (pd.Dataframe): it contains the physical (mm) coordinates of the landmark points warped to tof space
    """
    landmark_points_tof_space = []  # type: list # will contain the physical (mm) coordinates of the landmark points warped to tof space
    # loop over landmark points
    for _, coords in df_landmarks.iterrows():
        landmark_point_physical_space_mni = [coords["x"], coords["y"], coords["z"], 0]  # add 0 in the time variable, because ANTs requires the coordinates to be in the shape [x, y, z, t] in physical space

        # WRITE landmark physical coordinate as csv file
        csv_folder = os.path.join(tmp_path, "tmp_points_CHUV_mni2tof")  # specify folder where we save the csv file
        if not os.path.exists(csv_folder):  # if path does not exist
            os.makedirs(csv_folder)  # create it
        csv_path = os.path.join(csv_folder, "Landmark_Coordinate_MNI_in_mm.csv")  # add filename to path

        # create csv file
        with open(csv_path, 'w') as myfile:
            wr = csv.writer(myfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            wr.writerow(['x', 'y', 'z', 't'])
            wr.writerow(landmark_point_physical_space_mni)
        # --------------------------------------------------------------- MNI_2_struct ------------------------------------------------------------
        # load landmark point as dataframe
        mni_df = pd.read_csv(csv_path)
        # duplicate first row (this is needed to run apply_transforms_to_points; it's a bug that they still have to fix)
        modified_df = pd.DataFrame(np.repeat(mni_df.values, 2, axis=0))
        modified_df.columns = mni_df.columns

        # apply registration to point
        transform_list = [mni_2_struct_mat_path, mni_2_struct_inverse_warp_path]
        which_to_invert = [True, False]
        struct_df = apply_transforms_to_points(dim=3,
                                               points=modified_df,
                                               transformlist=transform_list,
                                               whichtoinvert=which_to_invert)

        struct_df = struct_df.drop_duplicates()

        # -------------------------------------------------------------- struct_2_TOF -------------------------------------------------------------
        output_path_tof = os.path.join(csv_folder, "Landmark_Coordinate_TOF_in_mm.csv")  # save output filename

        modified_struct_df = pd.DataFrame(np.repeat(struct_df.values, 2, axis=0))
        modified_struct_df.columns = struct_df.columns
        # apply registration to point
        transform_list = [struct_2_tof_mat_path]
        which_to_invert = [True]
        tof_df = apply_transforms_to_points(dim=3,
                                            points=modified_struct_df,
                                            transformlist=transform_list,
                                            whichtoinvert=which_to_invert)

        tof_df = tof_df.drop_duplicates()
        # save dataframe as csv file
        tof_df.to_csv(output_path_tof, index=False)

        # ---------------------------------------------------------------------------------------------------------------------------------------------
        # read TOF landmark coordinates in physical space from csv created
        df_tof_landmark_coords = pd.read_csv(output_path_tof)
        landmark_mm_coord_tof = list(df_tof_landmark_coords.iloc[0])[:-1]  # extract first row of pd dataframe, convert to list and remove last item (t), cause we don't care about it

        landmark_points_tof_space.append(landmark_mm_coord_tof)

        # UNCOMMENT lines below for DEBUGGING
        # landmark_voxel_coord_tof = bfc_angio_volume_sitk.TransformPhysicalPointToIndex(landmark_mm_coord_tof)
        # print("DEBUG: landmark TOF voxel coord = {}".format(landmark_voxel_coord_tof))

    df_landmark_points_tof_space = pd.DataFrame(landmark_points_tof_space, columns=["x", "y", "z"])  # type: pd.DataFrame # convert from list to dataframe

    return df_landmark_points_tof_space


def extract_distance_one_aneurysm(subdir, aneur_path, bids_path, overlapping, patch_side, landmarks_physical_space_path, out_dir):
    """This function computes the distances from the positive patches (with aneurysm) of one patient to the landmark points where aneurysm occurrence is most frequent
    Args:
        subdir (str): parent directory to aneurysm nifti file
        aneur_path (str): filename of aneurysm nifti file
        bids_path (str): path to BIDS dataset
        overlapping (float): amount of overlapping between patches in sliding-window approach
        patch_side (int): side of cubic patches
        landmarks_physical_space_path (str): path to file containing the landmark points in MNI physical space
        out_dir (str): path to folder where output files will be saved
    Returns:
        min_and_mean_distances (list): it contains the min and mean distances from the positive patches to the landmark points for one patient
    """
    shift_scale_1 = patch_side // 2
    sub = re.findall(r"sub-\d+", subdir)[0]
    ses = re.findall(r"ses-\w{6}\d+", subdir)[0]  # extract ses
    if "Treated" in aneur_path:
        lesion_name = re.findall(r"Treated_Lesion_\d+", aneur_path)[0]  # type: str # extract lesion name
    else:
        lesion_name = re.findall(r"Lesion_\d+", aneur_path)[0]  # type: str # extract lesion name
    assert len(sub) != 0, "Subject ID not found"
    assert len(ses) != 0, "Session number not found"
    assert len(lesion_name) != 0, "Lesion name not found"

    # print("{}_{}_{}".format(sub, ses, lesion_name))

    # if we are NOT dealing with a treated aneurysm
    if "Treated" not in lesion_name:

        registration_params_dir = os.path.join(bids_path, "derivatives", "registrations", "reg_params")
        assert os.path.exists(registration_params_dir), "Path {} does not exist".format(registration_params_dir)
        bfc_derivatives_dir = os.path.join(bids_path, "derivatives", "N4_bias_field_corrected")
        assert os.path.exists(bfc_derivatives_dir), "path {0} does not exist".format(bfc_derivatives_dir)  # make sure that path exists

        if "ADAM" in bfc_derivatives_dir:
            bet_angio_bfc_path = os.path.join(bfc_derivatives_dir, sub, ses, "anat", "{}_{}_desc-angio_N4bfc_brain_mask_ADAM.nii.gz".format(sub, ses))  # type: str # save path of angio brain after Brain Extraction Tool (BET)
        else:
            bet_angio_bfc_path = os.path.join(bfc_derivatives_dir, sub, ses, "anat", "{}_{}_desc-angio_N4bfc_brain_mask.nii.gz".format(sub, ses))  # type: str # save path of angio brain after Brain Extraction Tool (BET)
        assert os.path.exists(bet_angio_bfc_path), "path {0} does not exist".format(bet_angio_bfc_path)  # make sure that path exists

        # since we are running patients in parallel, we must create separate tmp folders, otherwise we risk to overwrite/overload files of other subjects
        tmp_folder = os.path.join(out_dir, "tmp_{}_{}_{}_pos_patches".format(sub, ses, lesion_name))
        if not os.path.exists(tmp_folder):  # if directory does not exist
            os.makedirs(tmp_folder)  # create it

        # retrieve useful registration parameters by invoking dedicated function
        mni_2_struct_mat_path, struct_2_tof_mat_path, _, mni_2_struct_inverse_warp_path = retrieve_registration_params(os.path.join(registration_params_dir, sub, ses))

        bet_angio_bfc_obj = nib.load(bet_angio_bfc_path)  # type: nib.Nifti1Image
        bet_angio_bfc_volume = np.asanyarray(bet_angio_bfc_obj.dataobj)  # type: np.ndarray
        rows_range, columns_range, slices_range = bet_angio_bfc_volume.shape  # save dimensions of resampled angio-BET volume
        bet_angio_bfc_sitk = sitk.ReadImage(bet_angio_bfc_path)

        df_landmarks = pd.read_csv(landmarks_physical_space_path)  # type: pd.DataFrame
        df_landmarks_tof_physical_space = convert_mni_to_angio(df_landmarks, bet_angio_bfc_sitk, tmp_folder, mni_2_struct_mat_path,
                                                               struct_2_tof_mat_path, mni_2_struct_inverse_warp_path)  # type: pd.DataFrame

        lesion = (extract_lesion_info(os.path.join(subdir, aneur_path)))  # invoke external method and save dict with lesion information
        sc_shift = lesion["widest_dimension"] // 2  # define Sanity Check shift (half side of widest lesion dimension)
        # N.B. I INVERT X and Y BECAUSE of OpenCV (see https://stackoverflow.com/a/56849032/9492673)
        x_center = lesion["centroid_y_coord"]  # extract x coordinate of lesion centroid
        y_center = lesion["centroid_x_coord"]  # extract y coordinate of lesion centroid
        z_central = lesion["idx_slice_with_more_white_pixels"]  # extract idx of slice with more non-zero pixels
        x_min, x_max = x_center - sc_shift, x_center + sc_shift  # compute safest (largest) min and max x of patch containing lesion
        y_min, y_max = y_center - sc_shift, y_center + sc_shift  # compute safest (largest) min and max y of patch containing lesion
        z_min, z_max = z_central - sc_shift, z_central + sc_shift  # compute safest (largest) min and max z of patch containing lesion

        cnt_positive_patches = 0  # type: int # counter to keep track of how many pos patches are selected for each patient
        step = int(round_half_up((1 - overlapping) * patch_side))  # type: int
        all_distances = []
        for i in range(shift_scale_1, rows_range, step):  # loop over rows
            for j in range(shift_scale_1, columns_range, step):  # loop over columns
                for k in range(shift_scale_1, slices_range, step):  # loop over slices
                    flag = 0  # flag is just a dummy variable that we increment when a candidate patch overlaps with one aneurysm
                    # N.B. we check the overlap ONLY with the small-scale TOF patch cause the sequential scanning is performed with the small scale range
                    range_i = np.arange(i - shift_scale_1, i + shift_scale_1)  # create i range of the patch that will be evaluated
                    range_j = np.arange(j - shift_scale_1, j + shift_scale_1)  # create j range of the patch that will be evaluated
                    range_k = np.arange(k - shift_scale_1, k + shift_scale_1)  # create k range of the patch that will be evaluated

                    lesion_range_x = np.arange(x_min-1, x_max+1)  # create i range of the lesion (i.e. aneurysm)
                    lesion_range_y = np.arange(y_min-1, y_max+1)  # create j range of the lesion (i.e. aneurysm)
                    lesion_range_z = np.arange(z_min-1, z_max+1)  # create k range of the lesion (i.e. aneurysm)

                    # the boolean masks have all False if none of the voxels overlap (between candidate patch and lesion), and have True for the coordinates that do overlap
                    boolean_mask_x = (range_i >= np.min(lesion_range_x)) & (range_i <= np.max(lesion_range_x))
                    boolean_mask_y = (range_j >= np.min(lesion_range_y)) & (range_j <= np.max(lesion_range_y))
                    boolean_mask_z = (range_k >= np.min(lesion_range_z)) & (range_k <= np.max(lesion_range_z))

                    # if ALL the three boolean masks have at least one True value
                    if np.all(boolean_mask_y == False) == False and np.all(boolean_mask_x == False) == False and np.all(boolean_mask_z == False) == False:
                        flag += 1  # increment flag; if it gets incremented, it means that the current candidate patch overlaps with one aneurysm with at least one voxel

                    # ensure that the evaluated patch is not out of bound by using small scale
                    if i - shift_scale_1 >= 0 and i + shift_scale_1 < rows_range and j - shift_scale_1 >= 0 and j + shift_scale_1 < columns_range and k - shift_scale_1 >= 0 and k + shift_scale_1 < slices_range:
                        if flag != 0:  # if the patch contains an aneurysm
                            cnt_positive_patches += 1  # increment counter

                            # convert patch center from voxel to physical space (mm)
                            patch_center_tof_physical_space = list(bet_angio_bfc_sitk.TransformIndexToPhysicalPoint([i, j, k]))

                            # loop over all landmark points
                            for _, coords in df_landmarks_tof_physical_space.iterrows():
                                landmark_point_tof_physical_space = np.asarray([coords["x"], coords["y"], coords["z"]])
                                eucl_distance = np.linalg.norm(patch_center_tof_physical_space - landmark_point_tof_physical_space)  # type: float # compute euclidean distance in TOF physical space
                                all_distances.append(eucl_distance)
        if cnt_positive_patches == 0:
            print("WARNING: There should be at least one patch containing the aneurysm; found 0 for {}_{}_{}".format(sub, ses, lesion_name))
        # -------------------------------------------------------------------------------------
        # remove temporary folder for this subject
        if os.path.exists(tmp_folder) and os.path.isdir(tmp_folder):
            shutil.rmtree(tmp_folder)

        if all_distances:  # if list is non-empty
            min_and_mean_distances = [np.min(all_distances), np.mean(all_distances)]
            return min_and_mean_distances
        else:  # if instead the list is empty
            min_and_mean_distances = None
            return min_and_mean_distances
    else:  # if instead we are dealing with a treated aneurysm
        return None # in any case the Nones will be removed later


def extract_distance_thresholds(bids_ds_path, reg_quality_metrics_threshold, sub_ses_test, n_parallel_jobs, overlapping, patch_side, landmarks_physical_space_path, out_dir):
    """This function computes the distances from the true aneurysms of the train input dataset to the landmark points where aneurysm occurrence is most frequent.
    From the distribution of these distances, it extracts the min and the mean value which will be used in the sliding-window to extract anatomically-plausible
    patches (i.e. patches which are "not-too-far" from the landmark points).
    Args:
        bids_ds_path (str): path to BIDS dataset
        reg_quality_metrics_threshold (tuple): it contains some registration quality metrics to assess whether the registration was accurate or not for each subject
        sub_ses_test (list): sub_ses of the test set; we use it to take only the sub_ses of the training set
        n_parallel_jobs (int): number of jobs to run in parallel with joblib
        overlapping (float): amount of overlapping between patches in sliding-window approach
        patch_side (int): side of cubic patches
        landmarks_physical_space_path (str): path to file containing the landmark points in MNI physical space
        out_dir (str): path to folder where output files will be saved
    Returns:
        distances_thresholds (tuple): it contains min and mean values extracted from the distribution of distances; we'll use these in the anatomically-informed sliding-window
    """
    all_subdirs = []  # type: list
    all_files = []  # type: list
    ext_gz = ".gz"  # type: str # set extension to match

    registration_metrics_dir = os.path.join(bids_ds_path, "derivatives", "registrations", "reg_metrics")

    for subdir, dirs, files in os.walk(bids_ds_path):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()  # get the file extension
            # save path of every positive patch
            if "Lesion" in file and ext == ext_gz and "registrations" not in subdir:  # if we're dealing with the aneurysm volume
                sub = re.findall(r"sub-\d+", subdir)[0]
                ses = re.findall(r"ses-\w{6}\d+", subdir)[0]  # extract ses
                sub_ses = "{}_{}".format(sub, ses)
                if sub_ses not in sub_ses_test:  # only use training sub_ses otherwise we might introduce a bias towards the locations of the aneurysms in the test set
                    sub_quality_metrics = extract_reg_quality_metrics_one_sub(os.path.join(registration_metrics_dir, sub, ses))  # type: tuple
                    registration_accurate_enough = check_registration_quality(reg_quality_metrics_threshold, sub_quality_metrics)  # type: bool
                    # only compute distances for subjects with correct registration
                    if registration_accurate_enough:
                        all_subdirs.append(subdir)
                        all_files.append(file)

    assert all_subdirs and all_files, "Input lists must be non-empty"
    out_list = Parallel(n_jobs=n_parallel_jobs, backend='loky')(delayed(extract_distance_one_aneurysm)(all_subdirs[idx],
                                                                                                       all_files[idx],
                                                                                                       bids_ds_path,
                                                                                                       overlapping,
                                                                                                       patch_side,
                                                                                                       landmarks_physical_space_path,
                                                                                                       out_dir) for idx in range(len(all_subdirs)))
    out_list = [x for x in out_list if x]  # remove None values from list if present

    out_list_np = np.asanyarray(out_list)
    assert out_list_np.shape == (len(out_list), 2), "Shape mismatch"
    min_distances = out_list_np[:, 0]
    mean_distances = out_list_np[:, 1]

    p97_min_distances = np.percentile(min_distances, [97])[0]
    p97_mean_distances = np.percentile(mean_distances, [97])[0]

    distances_thresholds = (p97_min_distances, p97_mean_distances)

    return distances_thresholds


def extract_dark_fp_threshold_one_aneurysm(subdir, aneur_path, bids_dir):
    """This function computes the intensity ratio of one aneurysm
    Args:
        subdir (str): parent directory to aneurysm nifti file
        aneur_path (str): filename of aneurysm nifti file
        bids_dir (str): path to BIDS dataset
    Returns:
        intensity_ratio (float): computed as mean_aneurysm_intensity/p90 where p90 is the 90th percentile of the original bias-field-corrected TOF volume after brain extraction before resampling
    """
    sub = re.findall(r"sub-\d+", subdir)[0]
    ses = re.findall(r"ses-\w{6}\d+", subdir)[0]  # extract ses
    # print("{}_{}".format(sub, ses))
    bfc_derivatives_dir = os.path.join(bids_dir, "derivatives", "N4_bias_field_corrected")
    assert os.path.exists(bfc_derivatives_dir), "path {0} does not exist".format(bfc_derivatives_dir)  # make sure that path exists

    if "ADAM" in bfc_derivatives_dir:
        bet_angio_bfc_path = os.path.join(bfc_derivatives_dir, sub, ses, "anat", "{}_{}_desc-angio_N4bfc_brain_mask_ADAM.nii.gz".format(sub, ses))  # type: str # save path of angio brain after Brain Extraction Tool (BET)
    else:
        bet_angio_bfc_path = os.path.join(bfc_derivatives_dir, sub, ses, "anat", "{}_{}_desc-angio_N4bfc_brain_mask.nii.gz".format(sub, ses))  # type: str # save path of angio brain after Brain Extraction Tool (BET)
    assert os.path.exists(bet_angio_bfc_path), "path {0} does not exist".format(bet_angio_bfc_path)  # make sure that path exists

    bet_angio_bfc_obj = nib.load(bet_angio_bfc_path)  # type: nib.Nifti1Image
    bet_angio_bfc_volume = np.asanyarray(bet_angio_bfc_obj.dataobj)  # type: np.ndarray
    p90 = np.percentile(bet_angio_bfc_volume, [90])[0]  # find xxth intensity percentile for this subject

    binary_mask_obj = nib.load(os.path.join(subdir, aneur_path))  # type: nib.Nifti1Image
    binary_mask = np.asanyarray(binary_mask_obj.dataobj)  # type: np.ndarray
    assert binary_mask.dtype == 'uint8', "Aneurysm binary mask must have type uint8"
    labels_out = cc3d.connected_components(binary_mask)
    numb_labels = np.max(labels_out)  # extract number of different connected components found
    assert numb_labels == 1, "This function is intended to work with one aneurysm per mask volume"

    # loop over different conn. components
    for seg_id in range(1, numb_labels + 1):
        # extract connected component from original bias-field-corrected tof volume after brain extraction
        extracted_original_bet_tof = bet_angio_bfc_volume * (labels_out == seg_id)
        assert extracted_original_bet_tof.shape == bet_angio_bfc_volume.shape == binary_mask.shape, "The shapes of the volumes should be identical"
        # only retain non-zero voxels (i.e. voxels belonging to the aneurysm mask)
        non_zero_voxels_intensities = extracted_original_bet_tof[np.nonzero(extracted_original_bet_tof)]  # type: np.ndarray
        # compute ratio between mean intensity of predicted aneurysm voxels and the xxth intensity percentile of the entire skull-stripped image
        intensity_ratio = np.mean(non_zero_voxels_intensities) / p90

    return intensity_ratio


def extract_dark_fp_threshold(bids_dir, sub_ses_test, nb_parallel_jobs):
    """This function computes the threshold that we use to discard false positive predictions which are too dark. If a prediction is correct,
    it is usually bright because aneurysms tend to be bright.
    Args:
        bids_dir (str): path to BIDS dataset
        sub_ses_test (list): sub_ses of the test set; we use it to take only the sub_ses of the training set
        nb_parallel_jobs (int): number of jobs to run in parallel with joblib
    Returns:
        dark_fp_threshold (float): desired threshold
    """
    all_subdirs = []  # type: list
    all_files = []  # type: list
    ext_gz = ".gz"  # type: str # set extension to match
    regexp_sub = re.compile(r'sub')  # create a substring template to match

    for subdir, dirs, files in os.walk(bids_dir):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()  # get the file extension
            # if file name matches the specified template and extension is correct and we're dealing with a BET volume (original volume after skull-stripping)
            if regexp_sub.search(file) and ext == ext_gz and "Lesion" in file and "N4" not in subdir and "registrations" not in subdir:
                sub = re.findall(r"sub-\d+", subdir)[0]
                ses = re.findall(r"ses-\w{6}\d+", subdir)[0]  # extract ses
                sub_ses = "{}_{}".format(sub, ses)
                if sub_ses not in sub_ses_test:  # only use training sub_ses otherwise we might introduce a bias towards the locations of the aneurysms in the test set
                    all_subdirs.append(subdir)
                    all_files.append(file)

    assert all_subdirs and all_files, "Input lists must be non-empty"
    out_list = Parallel(n_jobs=nb_parallel_jobs, backend='loky')(delayed(extract_dark_fp_threshold_one_aneurysm)(all_subdirs[idx],
                                                                                                                 all_files[idx],
                                                                                                                 bids_dir) for idx in range(len(all_subdirs)))
    out_list = [x for x in out_list if x]  # remove None values from list if present
    dark_fp_threshold = min(out_list)  # in order to be conservative, take the min of the list
    return dark_fp_threshold


def extract_thresholds_for_anatomically_informed(bids_dir, sub_ses_test, unet_patch_side, new_spacing, inference_outputs_path, nb_parallel_jobs, overlapping, landmarks_physical_space_path, out_dir):
    """This function computes some thresholds that are needed for the anatomically-informed sliding-window approach. Specifically, it computes the registration quality metrics
    thresholds, some intensity thresholds and the distance thresholds.
    Args:
        bids_dir (str): path to BIDS dataset
        sub_ses_test (list): sub_ses of the test set; we use it to take only the sub_ses of the training set
        unet_patch_side (int): patch side of cubic patches
        new_spacing (list): desired voxel spacing that we want
        inference_outputs_path (str): path to folder where we save all the outputs of the patient-wise analysis
        nb_parallel_jobs (int): number of jobs to run in parallel with joblib
        overlapping (float): amount of overlapping between patches in sliding-window approach
        landmarks_physical_space_path (str): path to file containing the landmark points in MNI physical space
        out_dir (str): path to folder where output files will be saved
    Returns:
        reg_quality_metrics_threshold (tuple): thresholds to use to check whether the registration of a patient was correct or not
        intensity_thresholds (tuple): thresholds to use for extracting bright negative patches (that ideally contain vessels)
        distances_thresholds (tuple): thresholds to use for selecting patches which are "not-too-far" from the landmark points
        dark_fp_threshold (float): threshold used to remove predictions which are on average too dark for being a true aneurysm
    """
    # extract registration quality metrics; they will be used to decide whether the sliding-window is anatomically-informed or not
    print("\nComputing registration quality thresholds...")
    reg_quality_metrics_threshold = extract_registration_quality_metrics(bids_dir,
                                                                         sub_ses_test)  # type: tuple
    # we must extract some numerical thresholds to use for extracting vessel-like negative patches (i.e. neg patches similar to positive ones)
    print("\nComputing intensity thresholds...")
    intensity_thresholds = extract_thresholds_of_intensity_criteria(bids_dir,
                                                                    sub_ses_test,
                                                                    unet_patch_side,
                                                                    new_spacing,
                                                                    inference_outputs_path,
                                                                    nb_parallel_jobs,
                                                                    overlapping,
                                                                    prints=False)
    # UNCOMMENT for fast debugging
    # reg_quality_metrics_threshold = (-3.7482000589370728, -0.3016154289245605)
    # ODL_intensity_thresholds = [0.012, 0.0008, 0.098, 0.065, 5272]
    # intensity_thresholds = (0.043146080218234, 0.017068370725809585, 0.11095124781131743, 0.0831613838672638, 18371.2)
    # we must also extract some numerical thresholds to use for computing the distances from the patch centers to the landmark points
    print("\nComputing distance thresholds...")
    distances_thresholds = extract_distance_thresholds(bids_dir,
                                                       reg_quality_metrics_threshold,
                                                       sub_ses_test,
                                                       nb_parallel_jobs,
                                                       overlapping,
                                                       unet_patch_side,
                                                       landmarks_physical_space_path,
                                                       out_dir)

    # we must also extract a numerical threshold to use for removing the false positives which are too dark
    print("\nComputing threshold for dark FP reduction...")
    dark_fp_threshold = extract_dark_fp_threshold(bids_dir,
                                                  sub_ses_test,
                                                  nb_parallel_jobs)

    return reg_quality_metrics_threshold, intensity_thresholds, distances_thresholds, dark_fp_threshold


def load_file_from_disk(file_path):
    """This function loads a file from disk and returns it
    Args:
        file_path (str): path to file
    Returns:
        sub_ses_test (*): loaded file; it can be a list, a dict, etc.
    """
    assert os.path.exists(file_path), "Path {} does not exist".format(file_path)
    open_file = open(file_path, "rb")
    sub_ses_test = pickle.load(open_file)
    open_file.close()
    return sub_ses_test


def save_sliding_window_mask_to_disk(sliding_window_mask_volume, aff_mat_resampled, output_folder_path_, orig_bfc_angio_sitk, tmp_path, out_filename="mask_sliding_window.nii.gz"):
    """This function saves the sliding-window mask to disk; this serves just for visual purposes. We want to check which are the patches that were
    retained during inference and see if they are actually
    Args:
        sliding_window_mask_volume (np.ndarray): resampled volume to save to disk; will be first brought back to original space and then saved
        aff_mat_resampled (np.ndarray): affine matrix of resampled space
        output_folder_path_ (str): path to output folder
        orig_bfc_angio_sitk (sitk.Image): sitk volume of the bias-field-corrected, non-resampled angio
        tmp_path (str): path where we save temporary files
        out_filename (str): filename of output file
    """
    save_volume_mask_to_disk(sliding_window_mask_volume, output_folder_path_, aff_mat_resampled, out_filename)

    # extract voxel spacing of original (i.e. non-resampled) angio volume
    original_spacing = list(orig_bfc_angio_sitk.GetSpacing())
    # extract size of original (i.e. non-resampled) angio volume
    original_size = list(orig_bfc_angio_sitk.GetSize())
    # create output file for registration
    out_path = os.path.join(tmp_path, "mask_sliding_window_original_space.nii.gz")

    # resample volume to original spacing
    _, resampled_sliding_window_mask_volume_obj, resampled_sliding_window_mask_volume = resample_volume_inverse(os.path.join(output_folder_path_, out_filename),
                                                                                                                original_spacing,
                                                                                                                original_size,
                                                                                                                out_path,
                                                                                                                interpolator=sitk.sitkNearestNeighbor)  # set near-neighb. interpolator to avoid holes in the mask
    save_volume_mask_to_disk(resampled_sliding_window_mask_volume, output_folder_path_, resampled_sliding_window_mask_volume_obj.affine, out_filename)


def check_output_consistency_between_detection_and_segmentation(output_folder, sub, ses):
    """This function checks that the output files (txt and .nii.gz) correspond. In other words, if the txt file contains 3 aneurysms,
    the corresponding binary mask should have 3 connected components.
    Args:
        output_folder (str): path to dir containing result files
        sub (str): subject being analyzed
        ses (str): session (i.e. exam date)
    Returns:
        None
    Raises:
        AssertionError: if there is a mismatch between the two files
    """
    binary_segm_map_obj = nib.load(os.path.join(output_folder, "result.nii.gz"))  # type: nib.nifti1.Nifti1Image
    binary_segm_map = np.asanyarray(binary_segm_map_obj.dataobj)  # type: np.ndarray # load output segmentation map as np.array
    # extract 3D connected components
    labels_out = cc3d.connected_components(np.asarray(binary_segm_map, dtype=int))
    numb_labels = np.max(labels_out)  # extract number of different connected components found

    txt_file_path = os.path.join(output_folder, "result.txt")

    if not os.stat(txt_file_path).st_size == 0:  # if the output file is not empty (i.e. there's at least one predicted aneurysm location)
        df_txt_file = pd.read_csv(txt_file_path, header=None)  # type: pd.DataFrame # load txt file with pandas
        if df_txt_file.shape[0] != numb_labels:
            print("\nWARNING for {}_{}: mismatch between output files: {} centers vs. {} connected components".format(sub, ses, df_txt_file.shape[0], numb_labels))
    else:  # if instead the txt file is empty
        if numb_labels != 0:
            print("\nWARNING for {}_{}: there shouldn't be any connected component; found {} instead".format(sub, ses, numb_labels))


def sanity_check_inputs(unet_patch_side, unet_batch_size, unet_threshold, overlapping, new_spacing, conv_filters, cv_folds, anatomically_informed_sliding_window,
                        test_time_augmentation, reduce_fp, max_fp, reduce_fp_with_volume, min_aneurysm_volume, remove_dark_fp, bids_dir, training_outputs_path,
                        landmarks_physical_space_path, ground_truth_dir):
    """This function runs some sanity checks on the inputs of the sliding-window.
    Args:
        unet_patch_side (int): patch side of cubic patches
        unet_batch_size (int): batch size to use (not really relevant since we are doing inference, but still needed)
        unet_threshold (float): # threshold used to binarize the probabilistic U-Net's prediction
        overlapping (float): rate of overlapping to use during the sliding-window; defaults to 0
        new_spacing (list): it contains the desired voxel spacing to which we will resample
        conv_filters (list): it contains the number of filters in the convolutional layers
        cv_folds (int): number of cross-validation folds
        anatomically_informed_sliding_window (bool): whether to perform the anatomically-informed sliding-window
        test_time_augmentation (bool): whether to perform test time augmentation
        reduce_fp (bool): if set to True, only the "max_fp" most probable candidate aneurysm are retained; defaults to True
        max_fp (int): maximum number of allowed FPs per subject. If the U-Net predicts more than these, only the MAX_FP most probable are retained
        reduce_fp_with_volume (bool): if set to True, only the candidate lesions that have a volume (mm^3) > than a specific threshold are retained; defaults to True
        min_aneurysm_volume (float): minimum aneurysm volume; below this value we discard the candidate predictions
        remove_dark_fp (bool): if set to True, candidate aneurysms that are not brighter than a certain threshold (on average) are discarded; defaults to True
        bids_dir (str): path to BIDS dataset
        training_outputs_path (str): path to folder where we stored the weights of the network at the end of training
        landmarks_physical_space_path (str): path to file containing the coordinates of the landmark points
        ground_truth_dir (str): path to directory containing the ground truth masks and location files
    Returns:
        None
    """
    assert isinstance(unet_patch_side, int), "Patch side must be of type int; got {} instead".format(type(unet_patch_side))
    assert unet_patch_side > 0, "Patch side must be > 0; got {} instead".format(unet_patch_side)
    assert isinstance(unet_batch_size, int), "Batch size must be of type int; got {} instead".format(type(unet_batch_size))
    assert unet_batch_size > 0, "Batch size must be > 0; got {} instead".format(unet_batch_size)
    assert isinstance(unet_threshold, float), "UNET threshold must be of type float; got {} instead".format(type(unet_threshold))
    assert 0 < unet_threshold < 1, "UNET threshold must be in the range (0,1)"
    assert isinstance(overlapping, float), "Overlapping must be of type float; got {} instead".format(type(overlapping))
    assert 0 < overlapping < 1, "Overlapping must be in the range (0,1)"
    assert isinstance(new_spacing, list), "new_spacing must be of type list; got {} instead".format(type(new_spacing))
    assert all(isinstance(x, float) for x in new_spacing), "All elements inside new_spacing list must be of type float"
    assert isinstance(conv_filters, list), "conv_filters must be of type list; got {} instead".format(type(conv_filters))
    assert all(isinstance(x, int) for x in conv_filters), "All elements inside conv_filters list must be of type int"
    assert isinstance(cv_folds, int), "cv_folds must be of type int; got {} instead".format(type(cv_folds))
    assert cv_folds > 0, "cv_folds must be > 0"
    assert isinstance(anatomically_informed_sliding_window, bool), "anatomically_informed_sliding_window must be of type bool; got {} instead".format(type(anatomically_informed_sliding_window))
    assert isinstance(test_time_augmentation, bool), "test_time_augmentation must be of type bool; got {} instead".format(type(test_time_augmentation))
    assert isinstance(reduce_fp, bool), "reduce_fp must be of type bool; got {} instead".format(type(reduce_fp))
    assert isinstance(max_fp, int), "max_fp must be of type int; got {} instead".format(type(max_fp))
    assert isinstance(min_aneurysm_volume, float), "min_aneurysm_volume must be of type float; got {} instead".format(type(min_aneurysm_volume))
    assert min_aneurysm_volume > 0, "min_aneurysm_volume must be > 0, otherwise no prediction is retained"
    assert isinstance(reduce_fp_with_volume, bool), "reduce_fp must be of type bool; got {} instead".format(type(reduce_fp_with_volume))
    assert isinstance(remove_dark_fp, bool), "reduce_fp must be of type bool; got {} instead".format(type(remove_dark_fp))
    assert os.path.exists(bids_dir), "Path {} does not exist".format(bids_dir)
    assert os.path.exists(training_outputs_path), "Path {} does not exist".format(training_outputs_path)
    assert os.path.exists(landmarks_physical_space_path), "Path {} does not exist".format(landmarks_physical_space_path)
    assert os.path.exists(ground_truth_dir), "Path {} does not exist".format(ground_truth_dir)


def str2bool(v: str) -> bool:
    """This function converts the input parameter into a boolean
    Args:
        v (*): input argument
    Returns:
        True: if the input argument is 'yes', 'true', 't', 'y', '1'
        False: if the input argument is 'no', 'false', 'f', 'n', '0'
    Raises:
        ValueError: if the input argument is none of the above
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def get_parser():
    """This function creates a parser for handling input arguments"""
    p = argparse.ArgumentParser(description='Aneurysm_Net')
    p.add_argument('--config', type=str, required=True, help='Path to json configuration file.')
    return p


def invert_augmentations(augm_patch_hor_flip, augm_patch_ver_flip, augm_patch_270_rot, augm_patch_180_rot, augm_patch_90_rot, idx, aff_mat_resampled):
    """This function inverts the data augmentation in order to go back to the orientation of the original patch (i.e. the non-augmented one)
    Args:
        augm_patch_hor_flip (EagerTensor): patch augmented with horizontal flipping
        augm_patch_ver_flip (EagerTensor): patch augmented with vertical flipping
        augm_patch_270_rot (EagerTensor): patch augmented with 270° rotation clockwise
        augm_patch_180_rot (EagerTensor): patch augmented with 180° rotation clockwise
        augm_patch_90_rot (EagerTensor): patch augmented with 90° rotation clockwise
    Returns:
        augm_patch_hor_flip_orig (EagerTensor): the augmented patch brought back to the orientation of the original patch
        augm_patch_ver_flip_orig (EagerTensor): the augmented patch brought back to the orientation of the original patch
        augm_patch_270_rot_orig (EagerTensor): the augmented patch brought back to the orientation of the original patch
        augm_patch_180_rot_orig (EagerTensor): the augmented patch brought back to the orientation of the original patch
        augm_patch_90_rot_orig (EagerTensor): the augmented patch brought back to the orientation of the original patch
    """
    def horizontal_flipping(sample):
        hor_flip_patch_angio = tf.image.flip_left_right(sample)
        return hor_flip_patch_angio

    def vertical_flipping(sample):
        ver_flip_patch_angio = tf.image.flip_up_down(sample)
        return ver_flip_patch_angio

    # Rotations (!counter-clockwise!); k indicates the number of times rotation occurs
    def rotate_270(sample):
        rot_270_patch_angio = tf.image.rot90(sample, k=1, name="270_CC")
        return rot_270_patch_angio

    def rotate_180(sample):
        rot_180_patch_angio = tf.image.rot90(sample, k=2, name="180_CC")
        return rot_180_patch_angio

    def rotate_90(sample):
        rot_90_patch_angio = tf.image.rot90(sample, k=3, name="90_CC")
        return rot_90_patch_angio

    # we now have to invert the augmentation and go back to the orientation of the original patch
    augm_patch_hor_flip_orig = horizontal_flipping(augm_patch_hor_flip)  # for horizontal flipping we just apply it a second time
    augm_patch_ver_flip_orig = vertical_flipping(augm_patch_ver_flip)  # for vertical flipping we just apply it a second time
    augm_patch_270_rot_orig = rotate_90(augm_patch_270_rot)  # here, 1 rotation of 90° is sufficient
    augm_patch_180_rot_orig = rotate_180(augm_patch_180_rot)  # here, 2 rotations of 90° are needed
    augm_patch_90_rot_orig = rotate_270(augm_patch_90_rot)  # here, 3 rotations of 90° are needed

    # uncomment lines below for debugging and visualize the predictions
    # if idx == 28:
    #     pred_augm_patch_hor_flip_np = np.squeeze(augm_patch_hor_flip_orig, axis=-1)
    #     pred_augm_patch_ver_flip_np = np.squeeze(augm_patch_ver_flip_orig, axis=-1)
    #     pred_augm_patch_270_rot_np = np.squeeze(augm_patch_270_rot_orig, axis=-1)
    #     pred_augm_patch_180_rot_np = np.squeeze(augm_patch_180_rot_orig, axis=-1)
    #     pred_augm_patch_90_rot_np = np.squeeze(augm_patch_90_rot_orig, axis=-1)
#
    #     pred_augm_patch_hor_flip_np_obj = nib.Nifti1Image(pred_augm_patch_hor_flip_np, affine=aff_mat_resampled)
    #     pred_augm_patch_ver_flip_np_obj = nib.Nifti1Image(pred_augm_patch_ver_flip_np, affine=aff_mat_resampled)
    #     pred_augm_patch_270_rot_np_obj = nib.Nifti1Image(pred_augm_patch_270_rot_np, affine=aff_mat_resampled)
    #     pred_augm_patch_180_rot_np_obj = nib.Nifti1Image(pred_augm_patch_180_rot_np, affine=aff_mat_resampled)
    #     pred_augm_patch_90_rot_np_obj = nib.Nifti1Image(pred_augm_patch_90_rot_np, affine=aff_mat_resampled)
#
    #     trial_path = "/home/newuser/Desktop/MICCAI_Aneurysms/Trial_test_time_augmentation/"
    #     nib.save(pred_augm_patch_hor_flip_np_obj, os.path.join(trial_path, "pred_hor_flip_orig.nii.gz"))
    #     nib.save(pred_augm_patch_ver_flip_np_obj, os.path.join(trial_path, "pred_ver_flip_orig.nii.gz"))
    #     nib.save(pred_augm_patch_270_rot_np_obj, os.path.join(trial_path, "pred_270_orig.nii.gz"))
    #     nib.save(pred_augm_patch_180_rot_np_obj, os.path.join(trial_path, "pred_180_orig.nii.gz"))
    #     nib.save(pred_augm_patch_90_rot_np_obj, os.path.join(trial_path, "pred_90_orig.nii.gz"))

    return augm_patch_hor_flip_orig, augm_patch_ver_flip_orig, augm_patch_270_rot_orig, augm_patch_180_rot_orig, augm_patch_90_rot_orig


def apply_augmentations(patch):
    """This function applies the data augmentations to the patches retained in the sliding-window approach
    Args:
        patch (EagerTensor): patch that will be augmented
    Returns:
        augm_patch_hor_flip (EagerTensor): patch augmented with horizontal flipping
        augm_patch_ver_flip (EagerTensor): patch augmented with vertical flipping
        augm_patch_270_rot (EagerTensor): patch augmented with 270° rotation clockwise
        augm_patch_180_rot (EagerTensor): patch augmented with 180° rotation clockwise
        augm_patch_90_rot (EagerTensor): patch augmented with 90° rotation clockwise
        augm_patch_adj_contr (EagerTensor): patch augmented with contrast adjustment
        augm_patch_gamma_corr (EagerTensor): patch augmented with gamma correction
        augm_patch_gauss_noise (EagerTensor): patch augmented with gaussian noise
    """
    def horizontal_flipping(sample):
        hor_flip_patch_angio = tf.image.flip_left_right(sample)
        return hor_flip_patch_angio

    def vertical_flipping(sample):
        ver_flip_patch_angio = tf.image.flip_up_down(sample)
        return ver_flip_patch_angio

    # Rotations (!counter-clockwise!); k indicates the number of times rotation occurs
    def rotate_270(sample):
        rot_270_patch_angio = tf.image.rot90(sample, k=1, name="270_CC")
        return rot_270_patch_angio

    def rotate_180(sample):
        rot_180_patch_angio = tf.image.rot90(sample, k=2, name="180_CC")
        return rot_180_patch_angio

    def rotate_90(sample):
        rot_90_patch_angio = tf.image.rot90(sample, k=3, name="90_CC")
        return rot_90_patch_angio

    def adjust_contrast(sample):
        contr_adj_patch_angio = tf.image.adjust_contrast(sample, contrast_factor=2)  # adjust contrast
        return contr_adj_patch_angio

    def gamma_correction(sample):
        if (sample.numpy() > 0).all():  # if the patch has all non-negative values
            gamma_adj_patch_angio = tf.image.adjust_gamma(sample, gamma=0.2, gain=1)  # apply the correction Out = gain * In**gamma
            return gamma_adj_patch_angio
        else:
            return sample

    def gaussian_noise(sample):
        noise = tf.random.normal(shape=tf.shape(sample), mean=0.0, stddev=1, dtype=tf.float32)
        gauss_noise_patch_angio = tf.math.add(sample, noise)
        return gauss_noise_patch_angio

    augm_patch_hor_flip = horizontal_flipping(patch)
    augm_patch_ver_flip = vertical_flipping(patch)
    augm_patch_270_rot = rotate_270(patch)
    augm_patch_180_rot = rotate_180(patch)
    augm_patch_90_rot = rotate_90(patch)
    augm_patch_adj_contr = adjust_contrast(patch)
    augm_patch_gamma_corr = gamma_correction(patch)
    augm_patch_gauss_noise = gaussian_noise(patch)

    return augm_patch_hor_flip, augm_patch_ver_flip, augm_patch_270_rot, augm_patch_180_rot,\
            augm_patch_90_rot, augm_patch_adj_contr, augm_patch_gamma_corr, augm_patch_gauss_noise


def predict_augmented_patches(patch, augm_patch_hor_flip, augm_patch_ver_flip, augm_patch_270_rot, augm_patch_180_rot,
                              augm_patch_90_rot, augm_patch_adj_contr, augm_patch_gamma_corr, augm_patch_gauss_noise, unet, unet_batch_size, idx, aff_mat_resampled):
    """This function computes the predictions for the augmented patches
    Args:
        patch (EagerTensor): original patch
        augm_patch_hor_flip (EagerTensor): patch augmented with horizontal flipping
        augm_patch_ver_flip (EagerTensor): patch augmented with vertical flipping
        augm_patch_270_rot (EagerTensor): patch augmented with 270° rotation clockwise
        augm_patch_180_rot (EagerTensor): patch augmented with 180° rotation clockwise
        augm_patch_90_rot (EagerTensor): patch augmented with 90° rotation clockwise
        augm_patch_adj_contr (EagerTensor): patch augmented with contrast adjustment
        augm_patch_gamma_corr (EagerTensor): patch augmented with gamma correction
        augm_patch_gauss_noise (EagerTensor): patch augmented with gaussian noise
        unet (tf.keras.Model): trained model that we use for inference (i.e. for computing the predictions)
        unet_batch_size (int): batch size. Not really relevant (cause we're doing inference), but still needed
    """
    # first, we need to re-create the tf.data.Dataset for compatibility with the network structure
    patch_ds = create_tf_dataset([patch], unet_batch_size)
    augm_patch_hor_flip_ds = create_tf_dataset([augm_patch_hor_flip], unet_batch_size)
    augm_patch_ver_flip_ds = create_tf_dataset([augm_patch_ver_flip], unet_batch_size)
    augm_patch_270_rot_ds = create_tf_dataset([augm_patch_270_rot], unet_batch_size)
    augm_patch_180_rot_ds = create_tf_dataset([augm_patch_180_rot], unet_batch_size)
    augm_patch_90_rot_ds = create_tf_dataset([augm_patch_90_rot], unet_batch_size)
    augm_patch_adj_contr_ds = create_tf_dataset([augm_patch_adj_contr], unet_batch_size)
    augm_patch_gamma_corr_ds = create_tf_dataset([augm_patch_gamma_corr], unet_batch_size)
    augm_patch_gauss_noise_ds = create_tf_dataset([augm_patch_gauss_noise], unet_batch_size)

    # compute predictions and remove batch axis (i.e. axis=0)
    pred_orig_patch = tf.squeeze(unet.predict(patch_ds), axis=0)
    pred_augm_patch_hor_flip = tf.squeeze(unet.predict(augm_patch_hor_flip_ds), axis=0)
    pred_augm_patch_ver_flip = tf.squeeze(unet.predict(augm_patch_ver_flip_ds), axis=0)
    pred_augm_patch_270_rot = tf.squeeze(unet.predict(augm_patch_270_rot_ds), axis=0)
    pred_augm_patch_180_rot = tf.squeeze(unet.predict(augm_patch_180_rot_ds), axis=0)
    pred_augm_patch_90_rot = tf.squeeze(unet.predict(augm_patch_90_rot_ds), axis=0)
    pred_augm_patch_adj_contr = tf.squeeze(unet.predict(augm_patch_adj_contr_ds), axis=0)
    pred_augm_patch_gamma_corr = tf.squeeze(unet.predict(augm_patch_gamma_corr_ds), axis=0)
    pred_augm_patch_gauss_noise = tf.squeeze(unet.predict(augm_patch_gauss_noise_ds), axis=0)

    # uncomment lines below for debugging and visualization
    # if idx == 28:
    #     pred_orig_patch_np = np.squeeze(pred_orig_patch, axis=-1)
    #     pred_augm_patch_hor_flip_np = np.squeeze(pred_augm_patch_hor_flip, axis=-1)
    #     pred_augm_patch_ver_flip_np = np.squeeze(pred_augm_patch_ver_flip, axis=-1)
    #     pred_augm_patch_270_rot_np = np.squeeze(pred_augm_patch_270_rot, axis=-1)
    #     pred_augm_patch_180_rot_np = np.squeeze(pred_augm_patch_180_rot, axis=-1)
    #     pred_augm_patch_90_rot_np = np.squeeze(pred_augm_patch_90_rot, axis=-1)
    #     pred_augm_patch_adj_contr_np = np.squeeze(pred_augm_patch_adj_contr, axis=-1)
    #     pred_augm_patch_gamma_corr_np = np.squeeze(pred_augm_patch_gamma_corr, axis=-1)
    #     pred_augm_patch_gauss_noise_np = np.squeeze(pred_augm_patch_gauss_noise, axis=-1)
#
    #     pred_orig_patch_np_obj = nib.Nifti1Image(pred_orig_patch_np, affine=aff_mat_resampled)
    #     pred_augm_patch_hor_flip_np_obj = nib.Nifti1Image(pred_augm_patch_hor_flip_np, affine=aff_mat_resampled)
    #     pred_augm_patch_ver_flip_np_obj = nib.Nifti1Image(pred_augm_patch_ver_flip_np, affine=aff_mat_resampled)
    #     pred_augm_patch_270_rot_np_obj = nib.Nifti1Image(pred_augm_patch_270_rot_np, affine=aff_mat_resampled)
    #     pred_augm_patch_180_rot_np_obj = nib.Nifti1Image(pred_augm_patch_180_rot_np, affine=aff_mat_resampled)
    #     pred_augm_patch_90_rot_np_obj = nib.Nifti1Image(pred_augm_patch_90_rot_np, affine=aff_mat_resampled)
    #     pred_augm_patch_adj_contr_np_obj = nib.Nifti1Image(pred_augm_patch_adj_contr_np, affine=aff_mat_resampled)
    #     pred_augm_patch_gamma_corr_np_obj = nib.Nifti1Image(pred_augm_patch_gamma_corr_np, affine=aff_mat_resampled)
    #     pred_augm_patch_gauss_noise_np_obj = nib.Nifti1Image(pred_augm_patch_gauss_noise_np, affine=aff_mat_resampled)
#
    #     trial_path = "/home/newuser/Desktop/MICCAI_Aneurysms/Trial_test_time_augmentation/"
    #     nib.save(pred_orig_patch_np_obj, os.path.join(trial_path, "pred_orig.nii.gz"))
    #     nib.save(pred_augm_patch_hor_flip_np_obj, os.path.join(trial_path, "pred_hor_flip.nii.gz"))
    #     nib.save(pred_augm_patch_ver_flip_np_obj, os.path.join(trial_path, "pred_ver_flip.nii.gz"))
    #     nib.save(pred_augm_patch_270_rot_np_obj, os.path.join(trial_path, "pred_270.nii.gz"))
    #     nib.save(pred_augm_patch_180_rot_np_obj, os.path.join(trial_path, "pred_180.nii.gz"))
    #     nib.save(pred_augm_patch_90_rot_np_obj, os.path.join(trial_path, "pred_90.nii.gz"))
    #     nib.save(pred_augm_patch_adj_contr_np_obj, os.path.join(trial_path, "pred_contr_adj.nii.gz"))
    #     nib.save(pred_augm_patch_gamma_corr_np_obj, os.path.join(trial_path, "pred_gamma_corr.nii.gz"))
    #     nib.save(pred_augm_patch_gauss_noise_np_obj, os.path.join(trial_path, "pred_gauss_noise.nii.gz"))

    return pred_orig_patch, pred_augm_patch_hor_flip, pred_augm_patch_ver_flip, pred_augm_patch_270_rot, pred_augm_patch_180_rot,\
           pred_augm_patch_90_rot, pred_augm_patch_adj_contr, pred_augm_patch_gamma_corr, pred_augm_patch_gauss_noise


def namestr(obj, namespace):
    """This function extracts the name of the variable in a list
    Args:
        obj: variable whose name we want to extract
        namespace: namespace of interest
    """
    return [name for name in namespace if namespace[name] is obj]


def compute_test_time_augmentation(batched_dataset, unet, unet_batch_size, aff_mat_resampled):
    """This function performs test time data augmentation on the retained patches of the sliding-window approach
    Args:
        batched_dataset (tf.data.Dataset): input dataset to augment
        unet (tf.keras.Model): trained network with which we perform inference
        unet_batch_size (int): batch size. Not really relevant (cause we're doing inference), but still needed
    Returns:
        tta_pred_patches (np.ndarray): average predictions (across augmentations) for all the retained patches of the sliding-window
    """
    unbatched_dataset = batched_dataset.unbatch()  # unbatch dataset because we want to loop over every single retained patch

    mean_predictions_all_retained_patches = []  # type: list
    for idx, patch in enumerate(unbatched_dataset.take(-1)):  # loop over all retained patches
        # Apply augmentations to a single patch
        augm_patch_hor_flip, augm_patch_ver_flip, augm_patch_270_rot, augm_patch_180_rot, \
            augm_patch_90_rot, augm_patch_adj_contr, augm_patch_gamma_corr, augm_patch_gauss_noise = apply_augmentations(patch)

        # compute predictions for original and augmented patches
        pred_patch, pred_augm_patch_hor_flip, pred_augm_patch_ver_flip, pred_augm_patch_270_rot, pred_augm_patch_180_rot, pred_augm_patch_90_rot, \
            pred_augm_patch_adj_contr, pred_augm_patch_gamma_corr, pred_augm_patch_gauss_noise = predict_augmented_patches(patch,
                                                                                                                           augm_patch_hor_flip,
                                                                                                                           augm_patch_ver_flip,
                                                                                                                           augm_patch_270_rot,
                                                                                                                           augm_patch_180_rot,
                                                                                                                           augm_patch_90_rot,
                                                                                                                           augm_patch_adj_contr,
                                                                                                                           augm_patch_gamma_corr,
                                                                                                                           augm_patch_gauss_noise,
                                                                                                                           unet,
                                                                                                                           unet_batch_size,
                                                                                                                           idx,
                                                                                                                           aff_mat_resampled)

        # invert (some) augmentations to overlay all the predictions with the original orientation
        pred_augm_patch_hor_flip_orig, pred_augm_patch_ver_flip_orig, pred_augm_patch_270_rot_orig, \
            pred_augm_patch_180_rot_orig, pred_augm_patch_90_rot_orig = invert_augmentations(pred_augm_patch_hor_flip,
                                                                                             pred_augm_patch_ver_flip,
                                                                                             pred_augm_patch_270_rot,
                                                                                             pred_augm_patch_180_rot,
                                                                                             pred_augm_patch_90_rot,
                                                                                             idx,
                                                                                             aff_mat_resampled)

        # group predictions in list
        augmented_prediction_inverted = [pred_patch, pred_augm_patch_hor_flip_orig, pred_augm_patch_ver_flip_orig, pred_augm_patch_270_rot_orig, pred_augm_patch_180_rot_orig,
                                         pred_augm_patch_90_rot_orig, pred_augm_patch_adj_contr, pred_augm_patch_gamma_corr, pred_augm_patch_gauss_noise]

        # remove redundant dimension (i.e. extra channel dim)
        augmented_prediction_inverted = [np.squeeze(pred_patch_orig, axis=-1) for pred_patch_orig in augmented_prediction_inverted]

        # compute the voxelwise mean across all augmented predictions
        pred_mean_patch = np.mean(augmented_prediction_inverted, axis=0)  # type: np.ndarray

        # uncomment lines below for debugging
        # if idx == 28:  # check mean prediction for one specific retained patch
        #     trial_path = "/home/newuser/Desktop/MICCAI_Aneurysms/Trial_test_time_augmentation/"
        #     patch_obj = nib.Nifti1Image(pred_mean_patch, affine=aff_mat_resampled)
        #     nib.save(patch_obj, os.path.join(trial_path, "trial_mean_patch.nii.gz"))

        # append mean prediction to external list
        mean_predictions_all_retained_patches.append(pred_mean_patch)

    # convert list to numpy array
    tta_pred_patches = np.asarray(mean_predictions_all_retained_patches)  # type: np.ndarray

    return tta_pred_patches


def print_running_time(start_time, end_time, process_name):
    """This function takes as input the start and the end time of a process and prints to console the time elapsed for this process
    Args:
        start_time (float): instant when the timer is started
        end_time (float): instant when the timer was stopped
        process_name (string): name of the process
    Returns:
        None
    """
    sentence = str(process_name)  # convert to string whatever the user inputs as third argument
    temp = end_time - start_time  # compute time difference
    hours = temp // 3600  # compute hours
    temp = temp - 3600 * hours  # if hours is not zero, remove equivalent amount of seconds
    minutes = temp // 60  # compute minutes
    seconds = temp - 60 * minutes  # compute minutes
    print('\n%s time: %d hh %d mm %d ss' % (sentence, hours, minutes, seconds))


def load_config_file():
    """This function loads the input config file
    Returns:
        config_dictionary (dict): it contains the input arguments
    """
    parser = get_parser()  # create parser
    args = parser.parse_args()  # convert argument strings to objects
    with open(args.config, 'r') as f:
        config_dictionary = json.load(f)

    return config_dictionary

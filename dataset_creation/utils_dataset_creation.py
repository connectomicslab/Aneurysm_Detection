"""Utility scripts for aneurysm detection pipeline."""

import numpy as np
import SimpleITK as sitk
import nibabel as nib
import os
import cc3d
from skimage.measure import regionprops
import cv2
import math
import csv
from itertools import islice
import pandas as pd
from ants import apply_transforms_to_points
import random
import re
from joblib import Parallel, delayed
import shutil
import pickle
from typing import List


def round_half_up(n, decimals=0):
    """This function rounds to the nearest integer number (e.g 2.4 becomes 2.0 and 2.6 becomes 3);
     in case of tie, it rounds up (e.g. 1.5 becomes 2.0 and not 1.0)
    Args:
        n (float): number to round
        decimals (int): number of decimal figures that we want to keep; defaults to zero
    """
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier


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
    return


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


def load_resampled_vol_and_boundaries(volume_path, new_spacing_, tmp_folder_, sub_, ses_):
    """ This function loads a 3D nifti volume, converts it to numpy, resamples it to "new_spacing_" and computes the boundaries
    of non-empty rows, columns and slices. It returns these boundaries, the numpy volume and the affine transformation of the nifti object.
    Args:
        volume_path (str): path where nii.gz file is stored
        new_spacing_ (list): voxel spacing to which we want to resample
        tmp_folder_ (str): path where we temporarily store the resampled volumes
        sub_ (str): current subject
        ses_ (str): current session
    Returns:
        resampled_volume (np.array): nifti volume as numpy array
        affine_transf (np_array): affine matrix of nifti object
        resampled_volume_obj_sitk (sitk.Image): sitk resampled volume
        min_x_nii_volume (int): min non-zero row
        max_x_nii_volume (int): max non-zero row
        min_y_nii_volume (int): min non-zero column
        max_y_nii_volume (int): max non-zero column
        min_z_nii_volume (int): min non-zero slice of numpy volume
        max_z_nii_volume (int): max non-zero slice of numpy volume
    """
    out_path = os.path.join(tmp_folder_, "{0}_{1}_resampled_bet_tof_bfc.nii.gz".format(sub_, ses_))
    resampled_volume_obj_sitk, resampled_volume_obj_nib, resampled_volume = resample_volume(volume_path, new_spacing_, out_path)
    assert len(resampled_volume.shape) == 3, "Nifti volume is not 3D"
    non_zero_array = np.asarray(np.argwhere(resampled_volume != 0))  # save all zero voxels
    min_x_nii_volume, max_x_nii_volume = np.min(non_zero_array[:, 0]), np.max(non_zero_array[:, 0])  # extract min and max non-zero columns from BET volume
    min_y_nii_volume, max_y_nii_volume = np.min(non_zero_array[:, 1]), np.max(non_zero_array[:, 1])  # extract min and max non-zero rows from BET volume
    min_z_nii_volume, max_z_nii_volume = np.min(non_zero_array[:, 2]), np.max(non_zero_array[:, 2])  # extract min and max non-zero slices from BET volume

    affine_transf = resampled_volume_obj_nib.affine  # save affine matrix of nifti object

    return resampled_volume, affine_transf, resampled_volume_obj_sitk, min_x_nii_volume, max_x_nii_volume, min_y_nii_volume, max_y_nii_volume, min_z_nii_volume, max_z_nii_volume


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
    """
    out_path = os.path.join(tmp_folder_, out_name)
    if binary_mask:  # if the volume is a mask, use near.neighbor interpolator in order not to create new connected components
        resampled_volume_obj_sitk, resampled_volume_obj_nib, resampled_volume = resample_volume(volume_path, new_spacing_, out_path, interpolator=sitk.sitkNearestNeighbor)
    else:  # instead, if it's a normal nifti volume, use linear interpolator
        resampled_volume_obj_sitk, resampled_volume_obj_nib, resampled_volume = resample_volume(volume_path, new_spacing_, out_path, interpolator=sitk.sitkLinear)
    assert len(resampled_volume.shape) == 3, "Nifti volume is not 3D"
    aff_matrix = resampled_volume_obj_nib.affine  # extract and save affine matrix
    return resampled_volume_obj_nib, resampled_volume, aff_matrix


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


def extracting_conditions_are_met(seed_, seed_ext_, lesion_coord_, x_coord, y_coord, z_coord, patch_side,
                                  random_vessel_patch, vessel_nii_volume_resampled, random_patch_tof, resampled_tof_volume, intensity_thresholds):
    """This function checks if the current candidate patch fulfills several intensity conditions.
    Args:
        seed_ (int): random seed
        seed_ext_ (list): it contains the already-used seeds
        lesion_coord_ (dict): it contains the x,y,z min and max limits of the patient's lesion. If sub is a control, lesion_coord_ is defined as empty
        x_coord (int): x center coordinate of candidate patch
        y_coord (int): y center coordinate of candidate patch
        z_coord (int): z center coordinate of candidate patch
        patch_side (int): side of tof patch
        random_vessel_patch (np.ndarray): corresponding vesselMNI patch
        vessel_nii_volume_resampled (np.ndarray): vesselMNI volume registered in subject space
        random_patch_tof (np.ndarray): candidate tof patch
        resampled_tof_volume (np.ndarray): bet tof volume resampled to desired voxel spacing
        intensity_thresholds (list): it contains the values to use for the extraction of the vessel-like negative samples
    Returns:
        conditions_fulfilled (bool): initialized to False. If it turns True, all extracting conditions were met
    """
    conditions_fulfilled = False

    if seed_ not in seed_ext_:  # if we haven't already used this seed
        flag = 0  # flag is just a dummy variable that we increment if the random center of the candidate negative patch is inside/close to one of the lesions of the patient
        for key in lesion_coord_.keys():
            if lesion_coord_[key][0] < x_coord < lesion_coord_[key][1] and \
                    lesion_coord_[key][2] < y_coord < lesion_coord_[key][3] and \
                    lesion_coord_[key][4] < z_coord < lesion_coord_[key][5]:
                flag += 1  # increment; if flag gets incremented, it means that the candidate negative patch overlaps with one lesion

        if flag == 0:  # if the candidate neg patch does not overlap with any aneurysm
            # if at least more than half of the voxels are non-zeros
            if np.count_nonzero(random_patch_tof) > (random_patch_tof.size // 2):
                # if the size is different than 0 (e.g. patch is out-of-bound)
                if random_patch_tof.size != 0:
                    # if the shape is correct
                    if random_patch_tof.shape == (patch_side, patch_side, patch_side) and random_vessel_patch.shape == (patch_side, patch_side, patch_side):
                        # if intensity_threshold list is not empty --> i.e. if the extracted sample is not random (i.e. it is vessel-like), then check for intensity conditions
                        if intensity_thresholds:
                            # check now for the intensity conditions on the small-scale candidate patch
                            if not math.isnan(np.mean(random_vessel_patch)) and not math.isnan(np.max(random_vessel_patch)) and not math.isnan(np.max(vessel_nii_volume_resampled)) and np.max(
                                    random_vessel_patch) != 0 and np.max(vessel_nii_volume_resampled) != 0:
                                ratio_local_vessel_mni = np.mean(random_vessel_patch) / np.max(random_vessel_patch)  # compute in vesselMNI patch the local ratio between mean and max intensity
                                ratio_global_vessel_mni = np.mean(random_vessel_patch) / np.max(vessel_nii_volume_resampled)  # compute in vesselMNI patch the global ratio between mean and max intensities wrt the whole volume
                                ratio_local_tof_bet = np.mean(random_patch_tof) / np.max(random_patch_tof)  # compute local intensity ratio (mean/max) on bet_tof
                                ratio_global_tof_bet = np.mean(random_patch_tof) / np.max(resampled_tof_volume)  # compute global intensity ratio (mean/max) on bet_tof
                                if ratio_local_vessel_mni > intensity_thresholds[0] and ratio_global_vessel_mni > intensity_thresholds[1] and \
                                        ratio_local_tof_bet > intensity_thresholds[2] and ratio_global_tof_bet > intensity_thresholds[3] and\
                                        np.count_nonzero(random_vessel_patch) > intensity_thresholds[4]:
                                    conditions_fulfilled = True
                        # if instead we want to extract a random patch, then we can neglect the intensity conditions
                        else:
                            conditions_fulfilled = True
    return conditions_fulfilled


def retrieve_registration_params(registration_dir_, sub_, ses_):
    """This function retrieves the registration parameters for each subject
    Args:
        registration_dir_ (str): root path where all subject folders containing the registration parameters are stored
        sub_ (str): subject ID
        ses_ (str): session date (YYYYMMDD)
    Returns:
        mni_2_struct_mat_path (str): path to .mat file corresponding to the MNI --> T1 registration
        struct_2_tof_mat_path (str): path to .mat file corresponding to the T1 --> TOF registration
        mni_2_struct_warp_path (str): path to warp field corresponding to the MNI --> T1 registration
        mni_2_struct_inverse_warp_path (str): path to inverse warp field corresponding to the MNI --> T1 registration
    Raises:
        AssertionError: if more (or less) than 4 registration paths are retrieved
    """
    extension_mat = '.mat'  # type: str # set file extension to be matched
    extension_gz = '.gz'  # type: str # set file extension to be matched
    cnt = 0  # type: int # counter variable that we use to ensure that exactly 4 registration parameters were retrieved
    assert os.path.exists(os.path.join(registration_dir_, sub_, ses_)), "Path {} does not exist".format(os.path.join(registration_dir_, sub_, ses_))

    for files_ in os.listdir(os.path.join(registration_dir_, sub_, ses_)):
        ext_ = os.path.splitext(files_)[-1].lower()  # get the file extension
        if "ADAM" in registration_dir_:
            if ext_ in extension_mat and "MNI_2_struct" in files_:  # if the extensions matches and a specific substring is in the file path
                mni_2_struct_mat_path = os.path.join(registration_dir_, sub_, ses_, files_)  # type: str
                cnt += 1  # increment counter
            elif ext_ in extension_mat and "struct_2_TOF" in files_:  # if the extensions matches and a specific substring is in the file path
                struct_2_tof_mat_path = os.path.join(registration_dir_, sub_, ses_, files_)  # type: str
                cnt += 1  # increment counter
            elif ext_ in extension_gz and "MNI_2_struct_1Warp" in files_:  # if the extensions matches and a specific substring is in the file path
                mni_2_struct_warp_path = os.path.join(registration_dir_, sub_, ses_, files_)  # type: str
                cnt += 1  # increment counter
            elif ext_ in extension_gz and "MNI_2_struct_1InverseWarp" in files_:  # if the extensions matches and a specific substring is in the file path
                mni_2_struct_inverse_warp_path = os.path.join(registration_dir_, sub_, ses_, files_)  # type: str
                cnt += 1  # increment counter
        else:
            if ext_ in extension_mat and "MNI_2_T1" in files_:  # if the extensions matches and a specific substring is in the file path
                mni_2_struct_mat_path = os.path.join(registration_dir_, sub_, ses_, files_)  # type: str
                cnt += 1  # increment counter
            elif ext_ in extension_mat and "T1_2_TOF" in files_:  # if the extensions matches and a specific substring is in the file path
                struct_2_tof_mat_path = os.path.join(registration_dir_, sub_, ses_, files_)  # type: str
                cnt += 1  # increment counter
            elif ext_ in extension_gz and "MNI_2_T1_1Warp" in files_:  # if the extensions matches and a specific substring is in the file path
                mni_2_struct_warp_path = os.path.join(registration_dir_, sub_, ses_, files_)  # type: str
                cnt += 1  # increment counter
            elif ext_ in extension_gz and "MNI_2_T1_1InverseWarp" in files_:  # if the extensions matches and a specific substring is in the file path
                mni_2_struct_inverse_warp_path = os.path.join(registration_dir_, sub_, ses_, files_)  # type: str
                cnt += 1  # increment counter

    assert cnt == 4, "Exactly 4 registration parameters must be retrieved"

    return mni_2_struct_mat_path, struct_2_tof_mat_path, mni_2_struct_warp_path, mni_2_struct_inverse_warp_path


def generate_candidate_patches(seed, angio_min_x, angio_max_x, angio_min_y, angio_max_y, angio_min_z, angio_max_z, shift_scale_1, vessel_mni_volume_resampled, resampled_tof_volume):
    """This function generates a random angio-tof candidate patch and the corresponding vessel patch
    Args:
        seed (int): random seed used to select a center coordinate
        angio_min_x (int): min non-zero row
        angio_max_x (int): max non-zero row
        angio_min_y (int): min non-zero column
        angio_max_y (int): max non-zero column
        angio_min_z (int): min non-zero slice of numpy volume
        angio_max_z (int): max non-zero slice of numpy volume
        shift_scale_1 (int): half patch side
        vessel_mni_volume_resampled (np.ndarray): vessel mni volume resampled to new spacing
        resampled_tof_volume (np.ndarray): tof volume resampled to new spacing
    Returns:
        random_x_coord (int): x coordinate of random patch center
        random_y_coord (int): y coordinate of random patch center
        random_z_coord (int): z coordinate of random patch center
        random_neg_patch_vessel (np.ndarray): candidate random vessel patch
        random_neg_patch_angio (np.ndarray): candidate random tof patch
    """
    random.seed(seed)  # set fixed random seed; by doing this, the random numbers calculated in the following lines will always be the same
    # randomly pick the [i,j,k] coordinate of the new negative patch that will be created
    random_x_coord = random.randint(angio_min_x, angio_max_x)  # select random column from non-zero columns of BET volume
    random_y_coord = random.randint(angio_min_y, angio_max_y)  # select random row from non-zero rows of BET volume
    random_z_coord = random.randint(angio_min_z, angio_max_z)  # select random slice from non-zero slices of BET volume

    # crop sub-volume from resampled vesselMNI volume
    random_neg_patch_vessel = vessel_mni_volume_resampled[random_x_coord - shift_scale_1:random_x_coord + shift_scale_1,
                                                          random_y_coord - shift_scale_1:random_y_coord + shift_scale_1,
                                                          random_z_coord - shift_scale_1:random_z_coord + shift_scale_1]

    # crop sub-volumes from main volumes with the candidate random center
    random_neg_patch_angio = resampled_tof_volume[random_x_coord - shift_scale_1:random_x_coord + shift_scale_1,
                                                  random_y_coord - shift_scale_1:random_y_coord + shift_scale_1,
                                                  random_z_coord - shift_scale_1:random_z_coord + shift_scale_1]

    return random_x_coord, random_y_coord, random_z_coord, random_neg_patch_vessel, random_neg_patch_angio


def extract_neg_patches_from_anatomical_landmarks(lesion_coords, resampled_original_angio_volume, bet_volume_affine,
                                                  registration_dir_, sub_, ses_, patch_pair_, mni_landmark_points_path,
                                                  shift_scale_1, neg_patches_path, resampled_bfc_tof_volume_sitk, csv_folder, orig_angio_path):
    """This function extracts negative patches (i.e. without aneurysm) centered around some anatomical locations that are recurrent for aneurysm occurrence.
    For patients, it extracts the patches only if the created patch do not overlap with any of the aneurysms of the patient.
    Args:
        lesion_coords (dict): it contains the x,y,z min and max limits of the patient's lesion. If sub is a control, lesion_coords is defined as empty
        resampled_original_angio_volume (np.ndarray): resampled original angio-TOF volume (before brain extraction tool)
        bet_volume_affine (np.ndarray): affine matrix of nibabel volume after brain extraction tool (bet)
        registration_dir_ (str): root path where all subject folders containing the registration parameters are stored
        sub_ (str): subject ID
        ses_ (str): session date (YYYYMMDD)
        patch_pair_ (int): last number of patch pair created before invoking this function
        mni_landmark_points_path (str): path to csv file containing MNI landmark points in LPS convention
        shift_scale_1 (int): half side of patches
        neg_patches_path (str): path of folder containing negative patches
        resampled_bfc_tof_volume_sitk (sitk.Image): sitk resampled bias-field-corrected tof
        csv_folder (str): path to folder where we save temporary files deriving from the points registrations
        orig_angio_path (str): path to original tof volume
    Returns:
        nb_landmarks (int): number of landmark points used for the patch extraction
    Raises:
        AssertionError: if bet_volume is not three-dimensional
        AssertionError: if original angio volume is not three-dimensional
        AssertionError: if shape of small-scale patch is wrong
        AssertionError: if shape of big-scale patch is wrong
    """
    assert len(resampled_original_angio_volume.shape) == 3, "Original angio volume must be 3D"
    mni_landmark_points_df = pd.read_csv(mni_landmark_points_path)  # load landmark points as pandas Dataframe

    # retrieve useful registration parameters by invoking dedicated function
    mni_2_struct_mat_path, struct_2_tof_mat_path, _, mni_2_struct_inverse_warp_path = retrieve_registration_params(registration_dir_, sub_, ses_)
    nb_landmarks = mni_landmark_points_df.shape[0]

    for _, row in islice(mni_landmark_points_df.iterrows(), nb_landmarks):  # loop over rows of pd Dataframe
        center_mm_coord_mni = [row["x"], row["y"], row["z"], 0]  # type: list # save center coordinate in list

        # WRITE original angio coordinate in physical space (mm) as csv file
        if not os.path.exists(csv_folder):  # if path does not exist
            os.makedirs(csv_folder)  # create it
        csv_path = os.path.join(csv_folder, "Original_Point_mm_MNI.csv")  # add filename to path

        # create csv file
        with open(csv_path, 'w') as myfile:
            wr = csv.writer(myfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            wr.writerow(['x', 'y', 'z', 't'])
            wr.writerow(center_mm_coord_mni)

        # ------------------------------------------------------------------ MNI_2_T1 -------------------------------------------------------------
        # load landmark point as dataframe
        mni_df = pd.read_csv(csv_path)
        # duplicate first row (this is needed to run apply_transforms_to_points; it's a bug that they still have to fix)
        modified_df = pd.DataFrame(np.repeat(mni_df.values, 2, axis=0))
        modified_df.columns = mni_df.columns
        # apply registration to point
        transform_list = [mni_2_struct_mat_path, mni_2_struct_inverse_warp_path]
        which_to_invert = [True, False]
        t1_df = apply_transforms_to_points(dim=3,
                                           points=modified_df,
                                           transformlist=transform_list,
                                           whichtoinvert=which_to_invert)

        t1_df = t1_df.drop_duplicates()
        # ------------------------------------------------------------------ T1_2_TOF -------------------------------------------------------------
        output_path_tof = os.path.join(csv_folder, "Transformed_Point_mm_TOF.csv")  # save output filename
        # inverse_t1_2_tof_mat_file_path = "[{0}, 1]".format(struct_2_tof_mat_path)  # create inverted .mat file as requested from

        modified_t1_df = pd.DataFrame(np.repeat(t1_df.values, 2, axis=0))
        modified_t1_df.columns = t1_df.columns
        # apply registration to point
        transform_list = [struct_2_tof_mat_path]
        which_to_invert = [True]
        tof_df = apply_transforms_to_points(dim=3,
                                            points=modified_t1_df,
                                            transformlist=transform_list,
                                            whichtoinvert=which_to_invert)

        # remove duplicates
        tof_df = tof_df.drop_duplicates()
        # save dataframe as csv file
        tof_df.to_csv(output_path_tof, index=False)
        # ------------------------------------------------------------------------------------------------------------------------------------------

        # READ TOF center coordinates in physical space from csv created with pandas
        df_t1_2_tof = pd.read_csv(output_path_tof)
        center_mm_coord_tof = list(df_t1_2_tof.iloc[0])[:-1]  # extract first row of pd.Dataframe and remove fourth component (t) that we don't need

        # UNCOMMENT lines below to check correspondence between original and resampled volume
        # load original angio TOF with sitk
        angio_volume_sitk = sitk.ReadImage(orig_angio_path)
        # convert patch center coordinate from physical space (mm) to voxel space
        center_voxel_coord_tof = list(angio_volume_sitk.TransformPhysicalPointToIndex(center_mm_coord_tof))

        # convert point from original space to resampled space
        center_voxel_coord_tof_resampled = resampled_bfc_tof_volume_sitk.TransformPhysicalPointToIndex(center_mm_coord_tof)

        # check that there is no overlap with the positive patches
        flag_ = 0  # flag is just a dummy variable that we increment if the random center of the candidate negative patch is inside/close to one of the lesions of the patient
        for key_ in lesion_coords.keys():
            if lesion_coords[key_][0] < center_voxel_coord_tof_resampled[0] < lesion_coords[key_][1] and\
                    lesion_coords[key_][2] < center_voxel_coord_tof_resampled[1] < lesion_coords[key_][3] and\
                    lesion_coords[key_][4] < center_voxel_coord_tof_resampled[2] < lesion_coords[key_][5]:
                flag_ += 1  # increment; if flag gets incremented, it means that the negative patch might overlap with one lesion

        if flag_ == 0:  # if there's no possible overlap with any aneurysm
            # extract patches from original tof, cause bet might have cut out parts of the landmark point area
            small_scale_bet_patch = resampled_original_angio_volume[center_voxel_coord_tof_resampled[0] - shift_scale_1: center_voxel_coord_tof_resampled[0] + shift_scale_1,
                                                                    center_voxel_coord_tof_resampled[1] - shift_scale_1: center_voxel_coord_tof_resampled[1] + shift_scale_1,
                                                                    center_voxel_coord_tof_resampled[2] - shift_scale_1: center_voxel_coord_tof_resampled[2] + shift_scale_1]

            # check that the shape of the patch is correct; sometimes the volume is just too small. If it's one of the shapes is not correct, skip this anatomical landmark
            if small_scale_bet_patch.shape == (shift_scale_1*2, shift_scale_1*2, shift_scale_1*2):
                print("------------ patch_pair_{0} anatomical landmark".format(patch_pair_))
                # create nibabel objects
                small_scale_bet_patch_obj = nib.Nifti1Image(small_scale_bet_patch, affine=bet_volume_affine)

                # save nibabel objects
                out_path = os.path.join(neg_patches_path, "{0}_{1}".format(sub_, ses_), "patch_pair_{0}_landmark".format(patch_pair_))
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                    bet_angio_patch_name_scale_1 = '{0}_{1}_patch_pair_{2}_neg_patch_angio.nii.gz'.format(sub_, ses_, str(patch_pair_))

                    nib.save(small_scale_bet_patch_obj, os.path.join(out_path, bet_angio_patch_name_scale_1))
                    print("Negative patches {0} created at center coordinates [{1}, {2}, {3}]".format(patch_pair_, center_voxel_coord_tof_resampled[0], center_voxel_coord_tof_resampled[1], center_voxel_coord_tof_resampled[2]))

                    patch_pair_ += 1  # increment patch pair such that the name of the following pair will change
    return nb_landmarks


def create_negative_patch(output_path, sub, ses, n, seed, x_coord, y_coord, z_coord, random_neg_patch_angio, resampled_bfc_tof_aff_mat, neg_patches_path, vessel_like=False):
    """This function saves the extracted patch to disk
    Args:
        output_path (str): path where we will save the extracted patch
        sub (str): subject number
        ses (str): session (i.e. exam date)
        n (int): number of patch created
        seed (int): random seed
        x_coord (int): x coordinate of patch center
        y_coord (int): y coordinate of patch center
        z_coord (int): z coordinate of patch center
        random_neg_patch_angio (np.ndarray): candidate patch that will be created
        resampled_bfc_tof_aff_mat (np.ndarray): affine matrix of original resampled volume
        neg_patches_path (str): path where the negative patches will be saved
        vessel_like (bool): whether the created patch is vessel-like or random
    """
    if not os.path.exists(neg_patches_path):  # if folder doesn't exist
        print("\nCreating Negative Patch folder")
        os.makedirs(neg_patches_path)  # create folder

    os.makedirs(output_path)
    angio_patch_name = '{0}_{1}_patch_pair_{2}_neg_patch_angio.nii.gz'.format(sub, ses, str(n))
    # convert patches from numpy arrays to nibabel objects, preserving original affine, and casting to float32
    neg_patch_angio_obj = nib.Nifti1Image(random_neg_patch_angio, affine=resampled_bfc_tof_aff_mat)

    if not os.path.exists(os.path.join(output_path, angio_patch_name)):  # if file does not exist
        nib.save(neg_patch_angio_obj, os.path.join(output_path, angio_patch_name))  # create it

    if vessel_like:
        print("------------ patch_pair_{0} intensity-matched, seed={1}".format(n, seed))
    else:
        print("------------ patch_pair_{0} random, seed={1}".format(n, seed))
    print("Negative patch {0} created at center coordinates [i, j, k] = [{1}, {2}, {3}]".format(n, x_coord, y_coord, z_coord))


def nb_last_created_patch(input_path):
    """This function returns the maximum number contained in the folder names of a specific path
    Args:
        input_path (str): path where we search for the folder names
    Returns:
        max_numb (int): highest number across folder names
        folder_list_only_landmark_patches (list): it contains the path to landmark patches (if any)
        folder_list_only_random_patches (list): it contains the path to the random patches (if any)
    """
    if not os.path.exists(input_path):  # if path does not exist
        os.makedirs(input_path)  # create it
    folder_list = os.listdir(input_path)  # save in list all the folder names

    folder_list_only_landmark_patches = []  # initialize as empty list; if it will be changed, it means that there are already landmark negative patches in the folder
    folder_list_only_random_patches = []  # initialize as empty list; if it will be changed, it means that there are already random negative patches in the folder

    # if list is non-emtpy
    if folder_list:
        folder_lists_only_numbers = [int(re.sub("[^0-9]", "", item)) for item in folder_list]  # retain from each folder name only the numbers
        folder_list_only_landmark_patches = [item for item in folder_list if "landmark" in item]
        folder_list_only_random_patches = [item for item in folder_list if "random" in item]
        max_numb = max(folder_lists_only_numbers)  # extract highest number
    # instead, if there are already other folders
    else:
        max_numb = 1
    return max_numb, folder_list_only_landmark_patches, folder_list_only_random_patches


def extract_vessel_like_neg_patches(nb_vessel_like_patches_per_sub, angio_min_x, angio_max_x, angio_min_y, angio_max_y, angio_min_z, angio_max_z, shift_scale_1,
                                    vessel_mni_volume_resampled, resampled_bfc_tof_volume, lesion_coord, patch_side, neg_patches_path, sub, ses, resampled_bfc_tof_aff_mat, intensity_thresholds):
    """This function extracts the negative patches located in correspondence of brain vessels
    Args:
        nb_vessel_like_patches_per_sub (int): number of vessel-like patches to extract
        angio_min_x (int): min non-zero row
        angio_max_x (int): max non-zero row
        angio_min_y (int): min non-zero column
        angio_max_y (int): max non-zero column
        angio_min_z (int): min non-zero slice
        angio_max_z (int): max non-zero slice
        shift_scale_1 (int): half side of patches
        vessel_mni_volume_resampled (np.ndarray): vessel mni volume resampled to new spacing
        resampled_bfc_tof_volume (np.ndarray): bias field corrected tof volume resampled to new spacing
        lesion_coord (dict): it contains the x,y,z min and max limits of the patient's lesion. If sub is a control, lesion_coords is defined as empty
        patch_side (int): side of patches that will be created
        neg_patches_path (str): path to folder where the negative patches will be saved
        sub (str): subject number
        ses (str): session (i.e. exam date)
        resampled_bfc_tof_aff_mat (np.ndarray): affine matrix of resampled bias field corrected tof volume
        intensity_thresholds (list): it contains the values to use for the extraction of the vessel-like negative samples
    Returns:
        seed_ext (list): it contains the random seeds used for creating the vessel-like patches; if none was created, list is empty
    """
    seed_ext = []  # initialize empty list where we'll store the seed used for good negative patches (i.e. the ones that fulfill all extraction criteria)
    for n in range(nb_vessel_like_patches_per_sub):  # repeat the extraction k times
        seed = 1  # start with a set random seed; this will be modified if the proposed negative patch doesn't fulfill specific requirements (see "if" statement below)
        while True:
            random_x_coord, random_y_coord, random_z_coord, random_neg_patch_vessel, \
                random_neg_patch_angio = generate_candidate_patches(seed, angio_min_x, angio_max_x, angio_min_y, angio_max_y, angio_min_z, angio_max_z, shift_scale_1, vessel_mni_volume_resampled, resampled_bfc_tof_volume)

            if seed < 7000:  # we try changing the seed XX times
                if extracting_conditions_are_met(seed, seed_ext, lesion_coord, random_x_coord,
                                                 random_y_coord, random_z_coord, patch_side, random_neg_patch_vessel,
                                                 vessel_mni_volume_resampled, random_neg_patch_angio, resampled_bfc_tof_volume, intensity_thresholds):
                    seed_ext.append(seed)  # append seed to external list so that it won't be re-used afterwards

                    output_path = os.path.join(neg_patches_path, "{0}_{1}".format(sub, ses), "patch_pair_{0}_vessel_like".format(n + 1))
                    if not os.path.exists(output_path):
                        create_negative_patch(output_path, sub, ses, n + 1, seed, random_x_coord, random_y_coord, random_z_coord, random_neg_patch_angio, resampled_bfc_tof_aff_mat, neg_patches_path, vessel_like=True)
                        break  # stop while loop if the negative patch is good
                    else:
                        print("{0}_{1}/patch_pair_{2} already exists".format(sub, ses, n + 1))
                        break  # stop while loop
                else:  # if the extracting conditions are not met
                    seed += 1  # increment seed, and thus try another patch center
            else:  # if however, the XX seed changes are not enough, discard subject
                print("\nWARNING: Couldn't create negative patch for {0}_{1}".format(sub, ses))
                break  # stop while loop

    return seed_ext


def extract_random_neg_patches(n, nb_random_patches_per_sub, angio_min_x, angio_max_x, angio_min_y, angio_max_y, angio_min_z, angio_max_z, shift_scale_1,
                               vessel_mni_volume_resampled, resampled_bfc_tof_volume, seed_ext, lesion_coord, patch_side, neg_patches_path, sub, ses,
                               resampled_bfc_tof_aff_mat, random_patches_list, intensity_thresholds):
    """This function extracts some random negative patches located in the brain
    Args:
        n (int): number of last negative patch extracted for this subject
        nb_random_patches_per_sub (int): number of random negative patches to extract
        angio_min_x (int): min non-zero row
        angio_max_x (int): max non-zero row
        angio_min_y (int): min non-zero column
        angio_max_y (int): max non-zero column
        angio_min_z (int): min non-zero slice
        angio_max_z (int): max non-zero slice
        shift_scale_1 (int): half side of patches
        vessel_mni_volume_resampled (np.ndarray): vessel mni volume resampled to new spacing
        resampled_bfc_tof_volume (np.ndarray): bias field corrected tof volume resampled to new spacing
        seed_ext (list): list of random seeds already used in the extraction
        lesion_coord (dict): it contains the x,y,z min and max limits of the patient's lesion. If sub is a control, lesion_coords is defined as empty
        patch_side (int): side of patches that will be created
        neg_patches_path (str): path to folder where the negative patches will be saved
        sub (str): subject number
        ses (str): session (i.e. exam date)
        resampled_bfc_tof_aff_mat (np.ndarray): affine matrix of resampled bias field corrected tof volume
        random_patches_list (list): it contains the path to the (already created) random patches. If no patch was yet created, the list is empty
        intensity_thresholds (list): it contains the values to use for the extraction of the vessel-like negative samples
    """
    # create list for iteration
    if n == 1:  # if this is the first negative patch created
        iter_list = list(range(n, n + nb_random_patches_per_sub))
    elif n > 1:  # if there are already other negative patches
        iter_list = list(range(n + 1, n + 1 + nb_random_patches_per_sub))
    else:
        raise ValueError("This line should never be reached; n should not be lower than 1")

    for r in iter_list:
        seed = 1  # start with a set random seed; this will be modified if the proposed negative patch doesn't fulfill specific requirements (see "if" statement below)
        while True:
            random_x_coord, random_y_coord, random_z_coord, random_neg_patch_vessel, \
                random_neg_patch_angio = generate_candidate_patches(seed, angio_min_x, angio_max_x, angio_min_y, angio_max_y, angio_min_z, angio_max_z, shift_scale_1, vessel_mni_volume_resampled, resampled_bfc_tof_volume)

            if seed < 7000:  # we try changing the seed XX times
                if extracting_conditions_are_met(seed, seed_ext, lesion_coord, random_x_coord,
                                                 random_y_coord, random_z_coord, patch_side, random_neg_patch_vessel,
                                                 vessel_mni_volume_resampled, random_neg_patch_angio, resampled_bfc_tof_volume, intensity_thresholds):
                    seed_ext.append(seed)  # append seed to external list so that it won't be re-used afterwards

                    output_path = os.path.join(neg_patches_path, "{0}_{1}".format(sub, ses), "patch_pair_{0}_random".format(r))
                    # if path does not exist and there are yet no random patches (i.e. random_patches_list is empty)
                    if not os.path.exists(output_path) and not random_patches_list:
                        create_negative_patch(output_path, sub, ses, r, seed, random_x_coord, random_y_coord, random_z_coord, random_neg_patch_angio, resampled_bfc_tof_aff_mat, neg_patches_path)
                        break  # stop while loop if the negative patch is good
                    else:
                        print("{0}_{1}/patch_pair_{2} already exists".format(sub, ses, r))
                        break  # stop while loop
                else:  # if the extracting conditions are not met
                    seed += 1  # increment seed, and thus try another patch center
            else:  # if however, the XX seed changes are not enough, discard subject
                print("\nWARNING: Couldn't create negative patch for {0}_{1}".format(sub, ses))
                break  # stop while loop


def extract_neg_landmark_patches(neg_patches_path, sub, ses, n, tmp_folder, original_angio_volume_path, desired_spacing, resampled_bfc_tof_aff_mat,
                                 registrations_dir, mni_landmark_points_path, shift_scale_1, resampled_bfc_tof_volume_sitk, lesion_coord, landmark_patches_list):
    if n == 1:
        patch_pair = 1
    elif n > 1:
        patch_pair = n + 1
    else:
        raise ValueError("n should not be lower than 1")
    new_samples_output_path = os.path.join(neg_patches_path, "{0}_{1}".format(sub, ses), "patch_pair_{0}".format(patch_pair))
    # if path does not exist and there are no landmark patches created (i.e. landmark_patches_list is empty)
    if not os.path.exists(new_samples_output_path) and not landmark_patches_list:
        out_path = os.path.join(tmp_folder, "{0}_{1}_orig_tof_bfc.nii.gz".format(sub, ses))
        _, _, resampled_orig_angio_volume = resample_volume(original_angio_volume_path, desired_spacing, out_path)  # extract numpy array of original angio TOF
        # invoke function to extract patches in correspondence to anatomical landmark points recurrent for aneurysm occurrence
        _ = extract_neg_patches_from_anatomical_landmarks(lesion_coord, resampled_orig_angio_volume, resampled_bfc_tof_aff_mat,
                                                          registrations_dir, sub, ses, patch_pair, mni_landmark_points_path,
                                                          shift_scale_1, neg_patches_path, resampled_bfc_tof_volume_sitk,
                                                          tmp_folder, original_angio_volume_path)
    else:
        print("Anatomical landmark patches already exist")


def randomly_translate_coordinates(shift_, center_x, center_y, center_z, seed_):
    """ This function takes as input the shift (i.e. number of voxels of which we want to translate our patch) and (x,y,z) which corresponds to the center of the patch.
    Then, if the lesion is not too big, it randomly shifts (x,y,z) of values which are in the range [-shift_; +shift_]. Then, it returns the shifted (x,y,z). If instead
    the lesion is too big, the function just returns the input coordinates untouched.
    Args:
        shift_ (int): number of voxels of which we want to shift/translate a coordinate
        center_x (int): x coordinate of the patch center before the shift/translation
        center_y (int): y coordinate of the patch center before the shift/translation
        center_z (int): z coordinate of the patch center before the shift/translation
        seed_ (int): random seed; set for reproducibility.
    Returns:
        shifted_x (int): x coordinate, shifted of a random value in the interval [-shift_, +shift_]
        shifted_y (int): y coordinate, shifted of a random value in the interval [-shift_, +shift_]
        shifted_z (int): z coordinate, shifted of a random value in the interval [-shift_, +shift_]
    """
    if shift_ > 0:  # if the lesion is not too big
        possible_shifts = np.arange(-shift_, +shift_+1, 1, dtype=int)  # define possible shifts
        random.seed(seed_)  # set fixed random seed
        shifted_x = int(center_x + random.choice(possible_shifts))  # type: int # compute shifted x
        shifted_y = int(center_y + random.choice(possible_shifts))  # type: int # compute shifted y
        shifted_z = int(center_z + random.choice(possible_shifts))  # type: int # compute shifted z
        return shifted_x, shifted_y, shifted_z
    else:  # if instead the lesion is too big
        return center_x, center_y, center_z  # just return non-translated coordinates


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

    # create unique tmp folder where we save temporary files (this folder will be deleted at the end)
    tmp_folder = os.path.join(out_folder, "tmp_{}_{}_{}_pos_patches".format(sub, ses, lesion_name))
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    print("{}-{}-{}".format(sub, ses, lesion_name))

    # save path of corresponding vesselMNI co-registered volume
    if "ADAM" in subdir:
        vessel_mni_reg_volume_path = os.path.join(vessel_mni_registration_dir, sub, ses, "anat", "{0}_{1}_desc-vesselMNI2angio_deformed_ADAM.nii.gz".format(sub, ses))
    else:
        vessel_mni_reg_volume_path = os.path.join(vessel_mni_registration_dir, sub, ses, "anat", "{0}_{1}_desc-vesselMNI2angio_deformed.nii.gz".format(sub, ses))
    assert os.path.exists(vessel_mni_reg_volume_path), "Path {0} does not exist".format(vessel_mni_reg_volume_path)  # make sure path exists

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
    if sc_shift == 0:  # if the aneurysm is extremely tiny
        sc_shift = 1  # set at least a margin of 1 pixel
    # N.B. I INVERT X and Y BECAUSE of OpenCV (see https://stackoverflow.com/a/56849032/9492673)
    x_center = lesion["centroid_y_coord"]  # extract y coordinate of lesion centroid
    y_center = lesion["centroid_x_coord"]  # extract x coordinate of lesion centroid
    z_central = lesion["idx_slice_with_more_white_pixels"]  # extract idx of slice with more non-zero pixels
    x_min, x_max = x_center - sc_shift, x_center + sc_shift  # compute safest (largest) min and max x of patch containing lesion
    y_min, y_max = y_center - sc_shift, y_center + sc_shift  # compute safest (largest) min and max y of patch containing lesion
    z_min, z_max = z_central - sc_shift, z_central + sc_shift  # compute safest (largest) min and max z of patch containing lesion

    lesion_range_x = np.arange(x_min, x_max)  # create x range of the lesion (i.e. aneurysm)
    lesion_range_y = np.arange(y_min, y_max)  # create y range of the lesion (i.e. aneurysm)
    lesion_range_z = np.arange(z_min, z_max)  # create z range of the lesion (i.e. aneurysm)

    cnt_positive_patches = 0  # type: int # counter to keep track of how many pos patches are selected for each patient
    step = int(round_half_up((1 - overlapping) * patch_side))  # type: int
    for x in range(shift_scale_1, rows_range, step):  # loop over rows
        for y in range(shift_scale_1, columns_range, step):  # loop over columns
            for z in range(shift_scale_1, slices_range, step):  # loop over slices

                flag = 0  # flag is just a dummy variable that we increment when a candidate patch overlaps with one aneurysm
                # N.B. we check the overlap ONLY with the small-scale TOF patch cause the sequential scanning is performed with the small scale range
                range_x = np.arange(x - shift_scale_1, x + shift_scale_1)  # create x range of the patch that will be evaluated
                range_y = np.arange(y - shift_scale_1, y + shift_scale_1)  # create y range of the patch that will be evaluated
                range_z = np.arange(z - shift_scale_1, z + shift_scale_1)  # create z range of the patch that will be evaluated

                # the boolean masks have all False if none of the voxels overlap (between candidate patch and lesion), and have True for the coordinates that do overlap
                boolean_mask_x = (range_x >= np.min(lesion_range_x)) & (range_x <= np.max(lesion_range_x))
                boolean_mask_y = (range_y >= np.min(lesion_range_y)) & (range_y <= np.max(lesion_range_y))
                boolean_mask_z = (range_z >= np.min(lesion_range_z)) & (range_z <= np.max(lesion_range_z))

                # if ALL the three boolean masks have at least one True value
                if np.all(boolean_mask_x == False) == False and\
                    np.all(boolean_mask_y == False) == False and\
                        np.all(boolean_mask_z == False) == False:
                            flag += 1  # increment flag; if it gets incremented, it means that the current candidate patch overlaps with one aneurysm with at least one voxel

                # ensure that the evaluated patch is not out of bound by using small scale
                if x - shift_scale_1 >= 0 and x + shift_scale_1 < rows_range and y - shift_scale_1 >= 0 and y + shift_scale_1 < columns_range and z - shift_scale_1 >= 0 and z + shift_scale_1 < slices_range:
                    if flag != 0:  # if the patch contains an aneurysm
                        cnt_positive_patches += 1  # increment counter
                        # extract patch from angio after BET
                        angio_patch_after_bet_scale_1 = nii_volume_resampled[x - shift_scale_1:x + shift_scale_1,
                                                                             y - shift_scale_1:y + shift_scale_1,
                                                                             z - shift_scale_1:z + shift_scale_1]

                        # extract small-scale patch from vesselMNI volume; we don't need the big-scale vessel patch
                        vessel_mni_patch = vessel_mni_volume_resampled[x - shift_scale_1:x + shift_scale_1,
                                                                       y - shift_scale_1:y + shift_scale_1,
                                                                       z - shift_scale_1:z + shift_scale_1]

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


def extract_thresholds_of_intensity_criteria(data_path, patch_side, new_spacing, out_folder, n_parallel_jobs, overlapping, prints=True):
    """This function computes the threshold to use for the extraction of the vessel-like negative patches (i.e. the
    negative patches that roughly have the same average intensity of the positive patches and include vessels)
    Args:
        data_path (str): path to BIDS dataset
        patch_side (int): side of cubic patches
        new_spacing (list): desired voxel spacing
        out_folder (str): path to output folder; during ds creation it is where we create the output dataset; at inference, it is where we save segmentation outputs
        n_parallel_jobs (int): number of jobs to run in parallel
        overlapping (float): amount of overlapping between patches in sliding-window approach
        prints (bool): whether to print numerical thresholds that were found; defaults to True
    Returns:
        intensity_thresholds (list): it contains the values to use for the extraction of the vessel-like negative samples
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

    out_list = [x for x in out_list if x]  # remove None values from list
    out_list_np = np.asarray(out_list)  # type: np.ndarray # convert from list to numpy array

    q5_local_vessel_mni, q7_local_vessel_mni = np.percentile(out_list_np[:, 0], [5, 7])
    q5_global_vessel_mni, q7_global_vessel_mni = np.percentile(out_list_np[:, 1], [5, 7])
    q5_local_tof_bet, q7_local_tof_bet = np.percentile(out_list_np[:, 2], [5, 7])
    q5_global_tof_bet, q7_global_tof_bet = np.percentile(out_list_np[:, 3], [5, 7])
    q5_nz_vessel_mni = np.percentile(out_list_np[:, 4], [5])

    if prints:
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

    intensity_thresholds = [q5_local_vessel_mni, q5_global_vessel_mni, q5_local_tof_bet, q5_global_tof_bet, q5_nz_vessel_mni]

    return intensity_thresholds


def refine_weak_label_one_sub(pos_path_path, masks_path):
    """ This function refines the mask of a positive patch: it removes the XX% percentile darkest voxels which most likely do not belong to the aneurysm
    Args:
        pos_path_path (str): path to the positive patch to be refined
        masks_path (str): path to the folder containing all the positive masks
    Returns:
        None
    Raises:
        ValueError: if any of the masks is either non-binary or empty
        ValueError: if the newly created refined mask is either non-binary or empty
    """
    patch_obj = nib.load(pos_path_path)
    aff_mat = patch_obj.affine  # type: np.ndarray
    pos_patch = np.asanyarray(patch_obj.dataobj)  # type: np.ndarray

    sub_ses_lesion = re.findall(r"sub-\d+_ses-\d+_Lesion_\d+", pos_path_path)[0]
    patch_pair = re.findall(r"patch_pair_\d+", pos_path_path)[0]
    last_part_of_path = os.path.basename(os.path.normpath(pos_path_path))
    filename_mask = last_part_of_path.replace("pos_patch_angio", "mask_patch")
    mask_obj = nib.load(os.path.join(masks_path, sub_ses_lesion, patch_pair, filename_mask))
    mask_patch = np.asanyarray(mask_obj.dataobj)  # type: np.ndarray

    if not np.array_equal(mask_patch, mask_patch.astype(bool)) and np.sum(mask_patch) != 0:
        raise ValueError("Mask of positive patches must be binary and non-empty")
    # create soft threshold volume (set to 0 all voxels outside the mask, but keep original values inside the mask)
    soft_threshold_patch = pos_patch * (mask_patch == 1)  # type: np.ndarray
    # only retain non-zero voxels
    non_zero_voxels_intensities = soft_threshold_patch[np.nonzero(soft_threshold_patch)]  # type: np.ndarray
    threshold = np.percentile(non_zero_voxels_intensities, [15])  # type: np.ndarray # find a specific percentile intensity value
    # set to 1 all voxels > threshold and to 0 all those < threshold (i.e. remove darker voxels); then, cast to int
    hard_threshold_patch = np.asarray(np.where(soft_threshold_patch > threshold, 1, 0), dtype=int)  # type: np.ndarray
    if np.count_nonzero(hard_threshold_patch) > 0:  # make sure there are some non-zero voxels left
        # find connected components in 3D numpy array; each connected component will be assigned a label starting from 1 and then increasing, 2, 3, etc.
        labels_out = cc3d.connected_components(np.asarray(hard_threshold_patch, dtype=int))
        numb_labels = np.max(labels_out)  # extract number of different connected components found
        numb_non_zero_voxels = []
        for seg_id in range(1, numb_labels + 1):  # loop over different labels (i.e. conn. components) found
            extracted_image = labels_out * (labels_out == seg_id)  # extract one conn. component volume
            numb_non_zero_voxels.append(np.count_nonzero(extracted_image))  # append number of non-zero voxels for this conn. component volume
        largest_conn_comp_value = np.argmax(numb_non_zero_voxels) + 1
        largest_conn_comp_binary = hard_threshold_patch * (labels_out == largest_conn_comp_value)
        if not np.array_equal(largest_conn_comp_binary, largest_conn_comp_binary.astype(bool)) and np.sum(largest_conn_comp_binary) != 0:
            raise ValueError("Created mask must be binary and non-empty")

        assert len(np.where(largest_conn_comp_binary + mask_patch == 2)[0]) > 0, "Watch out: there's no overlap among original and refined masks!"

        # save newly created mask (N.B: it OVERWRITES the existing one)
        largest_conn_comp_binary_obj = nib.Nifti1Image(largest_conn_comp_binary, affine=aff_mat)
        nib.save(largest_conn_comp_binary_obj, os.path.join(masks_path, sub_ses_lesion, patch_pair, filename_mask))
    else:
        raise ValueError("All white voxels were removed for {0}".format(filename_mask))


def load_pickle_list_from_disk(path_to_list: str) -> List:
    """This function loads a list from disk
    Args:
        path_to_list (str): path to where the list is saved
    Returns:
        loaded_list (list): loaded list
    Raises:
        AssertionError: if list path does not exist
        AssertionError: if extension is not .pkl
    """
    assert os.path.exists(path_to_list), "Path {} does not exist".format(path_to_list)
    ext = os.path.splitext(path_to_list)[-1].lower()  # get the file extension
    assert ext == ".pkl", "Expected .pkl file, got {} instead".format(ext)
    open_file = open(path_to_list, "rb")
    loaded_list = pickle.load(open_file)  # load from disk
    open_file.close()

    return loaded_list


def create_bin_sphere(arr_size, center, radius):
    """This function creates a 3D binary volume which contains a sphere. It has value 1 inside the sphere and 0 around it.
    To obtain over-labelled spheres, we set "X*radius" when creating the output volume instead of just "radius" (e.g. 2*radius).
    Args:
        arr_size (tuple): size of output 3D volume
        center (np.ndarray): 3D coordinate of sphere center
        radius (float): radius of the sphere in voxels
    Returns:
        binary_output (np.ndarray): binary volume with 1 inside the sphere and 0 outside of it
    """
    assert len(arr_size) == 3, "This function is intended to work for 3D arrays"
    coords = np.ogrid[:arr_size[0], :arr_size[1], :arr_size[2]]  # type: list
    distance = np.sqrt((coords[0] - center[0])**2 + (coords[1]-center[1])**2 + (coords[2]-center[2])**2)  # type: np.ndarray
    binary_output = 1 * (distance <= 2*radius)  # type: np.ndarray
    return binary_output


def extract_lesion_info(lesion_volume, prints=False):
    """This function extracts the lesion info for a binary mask (e.g. the voxel center, the equivalent diameter, etc.)
    Args:
        lesion_volume (np.ndarray): binary mask volume
        prints (bool): if set to True, some information about the lesion is printed
    Returns:
        lesion_info (dict): it contains the relevant lesion information
    """
    lesion_info = {}  # initialize empty dict; this will be the output of the function
    if len(lesion_volume.shape) == 4:  # if the numpy array is not 3D
        lesion_volume = np.squeeze(lesion_volume, axis=3)  # we drop the fourth dimension (time dimension) which is useless in our case

    assert np.array_equal(lesion_volume, lesion_volume.astype(bool)), "WATCH OUT: mask is not binary"
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

    m = cv2.moments(lesion_volume[:, :, idx].astype(np.uint8))  # calculate moments of binary image
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


def weakify_voxelwise_label_one_sub(pos_path_path, masks_path):
    """ This function converts the voxelwise mask of a positive patch into a weak mask: it creates a sphere around the aneurysm center
    Args:
        pos_path_path (str): path to the positive patch to be converted
        masks_path (str): path to the folder containing all the positive masks
    Returns:
        None
    Raises:
        ValueError: if the voxelwise mask is either non-binary or empty
        ValueError: if the newly created weak mask is either non-binary or empty
    """
    sub_ses_lesion = re.findall(r"sub-\d+_ses-\d+_Lesion_\d+", pos_path_path)[0]
    patch_pair = re.findall(r"patch_pair_\d+", pos_path_path)[0]
    last_part_of_path = os.path.basename(os.path.normpath(pos_path_path))
    filename_mask = last_part_of_path.replace("pos_patch_angio", "mask_patch")

    voxelwise_mask_obj = nib.load(os.path.join(masks_path, sub_ses_lesion, patch_pair, filename_mask))
    voxelwise_mask_patch = np.asanyarray(voxelwise_mask_obj.dataobj)  # type: np.ndarray
    if not np.array_equal(voxelwise_mask_patch, voxelwise_mask_patch.astype(bool)) and np.sum(voxelwise_mask_patch) != 0:
        raise ValueError("Voxelwise mask of positive patches must be binary and non-empty")

    lesion = extract_lesion_info(voxelwise_mask_patch)
    # N.B. I INVERT X and Y BECAUSE of OpenCV (see https://stackoverflow.com/a/56849032/9492673)
    x_center = lesion["centroid_y_coord"]  # extract y coordinate of lesion centroid
    y_center = lesion["centroid_x_coord"]  # extract x coordinate of lesion centroid
    z_central = lesion["idx_slice_with_more_white_pixels"]  # extract idx of slice with more non-zero pixels

    aneur_center, aneur_radius = np.asarray([x_center, y_center, z_central]), lesion["equivalent_diameter"] / 2

    weak_mask_with_sphere_patch = create_bin_sphere(voxelwise_mask_patch.shape, aneur_center, aneur_radius)  # type: np.ndarray
    weak_mask_with_sphere_patch = weak_mask_with_sphere_patch.astype(np.uint8)
    if not np.array_equal(voxelwise_mask_patch, voxelwise_mask_patch.astype(bool)) and np.sum(voxelwise_mask_patch) != 0:
        raise ValueError("Voxelwise mask of positive patches must be binary and non-empty")

    weak_mask_obj = nib.Nifti1Image(weak_mask_with_sphere_patch, affine=voxelwise_mask_obj.affine)

    # overwrite mask patch
    nib.save(weak_mask_obj, os.path.join(masks_path, sub_ses_lesion, patch_pair, filename_mask))
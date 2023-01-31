"""
Created on Apr 6, 2021

This script loops through the input BIDS dataset and creates a sub-dataset of patches. For controls (subject without aneurysms), only negative patches
are extracted. For patients (subjects with aneurysm(s)), both negative (without aneurysm) and positive (with aneurysm) patches are extracted.

"""

import time
import os
import sys
PROJECT_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # extract directory of PyCharm project
sys.path.append(PROJECT_HOME)  # this line is needed to recognize the dir as a python package
from datetime import datetime
import re
from joblib import Parallel, delayed
import shutil
import random
from random import randrange
import numpy as np
import nibabel as nib
from dataset_creation.utils_dataset_creation import load_resampled_vol_and_boundaries, resample_volume, extract_lesion_info_from_resampled_mask_volume, print_running_time, \
    nb_last_created_patch, extract_vessel_like_neg_patches, extract_random_neg_patches, extract_neg_landmark_patches, load_nifti_and_resample, \
    randomly_translate_coordinates, extract_thresholds_of_intensity_criteria, refine_weak_label_one_sub, load_pickle_list_from_disk, \
    weakify_voxelwise_label_one_sub
from inference.utils_inference import load_config_file, str2bool, create_dir_if_not_exist


__author__ = "Tommaso Di Noto"
__version__ = "0.0.1"
__email__ = "tommydino@hotmail.it"
__status__ = "Prototype"


def extract_negative_patches(subdir: str,
                             n4bfc_bet_angio_path: str,
                             bids_dataset_path: str,
                             desired_spacing: tuple,
                             out_dataset_path: str,
                             mni_landmark_points_path: str,
                             intensity_thresholds: list,
                             extract_landmark_patches: bool = True,
                             nb_vessel_like_patches_per_sub: int = 20,
                             nb_random_patches_per_sub: int = 0,
                             patch_side: int = 64) -> None:
    """This function extracts negative patches from all subjects. For patients (subjects with aneurysm(s)), negative patches are only extracted if there is no overlap with an aneurysm
    Args:
        subdir: path to parent folder of n4bfc_angio_bet
        n4bfc_bet_angio_path: path to n4bfc_angio_bet volume
        bids_dataset_path: path to BIDS dataset
        desired_spacing: it contains the desired voxel spacing for resampling the input volumes
        out_dataset_path: path to folder where we create the output dataset
        mni_landmark_points_path: path to the csv file containing the landmark points coordinates
        intensity_thresholds: it contains the threshold values to use in the extraction of the vessel-like negative patches
        extract_landmark_patches: if True, patches in correspondence of landmark points are extracted
        nb_vessel_like_patches_per_sub: number of vessel-like negative patches to extract for each subject
        nb_random_patches_per_sub: number of random negative patches to extract for each subject
        patch_side: side of cubic patches that will be extracted (both negative and positive)
    Raises:
        AssertionError: if bids_dataset_path does not exist
        AssertionError: if vessel_mni_registration_dir does not exist
        AssertionError: if registrations_dir does not exist
        AssertionError: if mni_landmark_points_path does not exist
        AssertionError: if the extension of the files containing the landmark points is not correct
    """
    assert os.path.exists(bids_dataset_path), f"Path {bids_dataset_path} does not exist"  # make sure that path exists

    try:
        os.makedirs(out_dataset_path, exist_ok=True)
        print(f"Directory {out_dataset_path} created successfully")
    except OSError as error:
        print(f"Directory {out_dataset_path} can not be created")

    neg_patches_path = os.path.join(out_dataset_path, "Negative_Patches")  # type: str # create path of folder that will contain the negative patches
    vessel_mni_registration_dir = os.path.join(bids_dataset_path, "derivatives/registrations/vesselMNI_2_angioTOF/")  # type: str
    assert os.path.exists(vessel_mni_registration_dir), f"Path {vessel_mni_registration_dir} does not exist"  # make sure that path exists
    registrations_dir = os.path.join(bids_dataset_path, "derivatives/registrations/reg_params/")
    assert os.path.exists(registrations_dir), f"Path {registrations_dir} does not exist"  # make sure that path exists
    assert os.path.exists(mni_landmark_points_path), f"Path {mni_landmark_points_path} does not exist"
    _, ext_ = os.path.splitext(mni_landmark_points_path)  # extract extension
    assert ext_ == ".csv", "File containing landmark points must have .csv extension"

    shift_scale_1 = patch_side // 2  # define shift of cubic patches
    sub = re.findall(r"sub-\d+", subdir)[0]  # extract sub
    ses = re.findall(r"ses-\w{6}\d+", subdir)[0]  # extract ses

    # create tmp folder where we save temporary files (this folder will be deleted at the end)
    tmp_folder = os.path.join(out_dataset_path, f"tmp_{sub}_{ses}_neg_patches")
    create_dir_if_not_exist(tmp_folder)

    # save path of original angio path before BET
    if "ADAM" in subdir:
        original_angio_volume_path = os.path.join(bids_dataset_path, sub, ses, "anat", f"{sub}_{ses}_angio_ADAM.nii.gz")
    else:
        original_angio_volume_path = os.path.join(bids_dataset_path, sub, ses, "anat", "{}_{}_angio.nii.gz".format(sub, ses))
    assert os.path.exists(original_angio_volume_path), "Path {} does not exist".format(original_angio_volume_path)

    # save path of corresponding vesselMNI co-registered volume
    if "ADAM" in subdir:
        vessel_mni_reg_volume_path = os.path.join(vessel_mni_registration_dir, sub, ses, "anat", "{}_{}_desc-vesselMNI2angio_deformed_ADAM.nii.gz".format(sub, ses))
    else:
        vessel_mni_reg_volume_path = os.path.join(vessel_mni_registration_dir, sub, ses, "anat", "{}_{}_desc-vesselMNI2angio_deformed.nii.gz".format(sub, ses))
    assert os.path.exists(vessel_mni_reg_volume_path), "Path {} does not exist".format(vessel_mni_reg_volume_path)

    # resample N4bfc angio to new spacing
    resampled_bfc_tof_volume, resampled_bfc_tof_aff_mat, resampled_bfc_tof_volume_sitk, angio_min_x, angio_max_x, \
        angio_min_y, angio_max_y, angio_min_z, angio_max_z = load_resampled_vol_and_boundaries(os.path.join(subdir, n4bfc_bet_angio_path),
                                                                                               desired_spacing,
                                                                                               tmp_folder,
                                                                                               sub,
                                                                                               ses)

    # Load corresponding vesselMNI volume and resample to new spacing
    out_path = os.path.join(tmp_folder, "{}_{}_resampled_vessel_atlas.nii.gz".format(sub, ses))
    _, _, vessel_mni_volume_resampled = resample_volume(vessel_mni_reg_volume_path, desired_spacing, out_path)

    lesions = []  # initialize empty list; this will remain empty for control subjects, while it will contain the path to the lesions for patients with one (or more) aneurysm(s)
    manual_masks_path = os.path.join(bids_dataset_path, "derivatives", "manual_masks", sub, ses, "anat")
    for item in os.listdir(manual_masks_path):
        item_ext = os.path.splitext(item)[-1].lower()  # get the file extension
        if "Lesion" in item and item_ext == '.gz':  # if this subject has a lesion and the file extension is correct
            lesions.append(item)  # append lesion path to external list

    if lesions:  # if list is not empty --> we're dealing with a patient with one or more aneurysm(s)
        print("\n-------------------------- {}_{} --------------------------".format(sub, ses))
        lesion_coord = {}  # initialize empty dict
        for aneur_path in lesions:  # loop over aneurysm(s) found for this patient
            # invoke external function and save dict with lesion information
            lesion = (extract_lesion_info_from_resampled_mask_volume(os.path.join(manual_masks_path, aneur_path), tmp_folder, desired_spacing, sub, ses))
            sc_shift = lesion["widest_dimension"] // 2  # define sanity check shift (half side of cube)
            # N.B. invert x and y because because of OpenCV (cv2); see https://stackoverflow.com/a/56849032/9492673
            x_center = lesion["centroid_y_coord"]  # extract y coordinate of lesion centroid
            y_center = lesion["centroid_x_coord"]  # extract x coordinate of lesion centroid
            z_central = lesion["idx_slice_with_more_white_pixels"]  # extract idx of slice with more non-zero pixels
            x_min, x_max = x_center - sc_shift - patch_side, x_center + sc_shift + patch_side  # compute safest (largest) min and max x of patch containing lesion
            y_min, y_max = y_center - sc_shift - patch_side, y_center + sc_shift + patch_side  # compute safest (largest) min and max y of patch containing lesion
            z_min, z_max = z_central - sc_shift - patch_side, z_central + sc_shift + patch_side  # compute safest (largest) min and max z of patch containing lesion
            lesion_coord[aneur_path] = [x_min, x_max, y_min, y_max, z_min, z_max]  # save lesion information in external dict

        # extract VESSEL-LIKE negative patches
        seed_ext = extract_vessel_like_neg_patches(nb_vessel_like_patches_per_sub, angio_min_x, angio_max_x, angio_min_y, angio_max_y, angio_min_z, angio_max_z,
                                                   shift_scale_1, vessel_mni_volume_resampled, resampled_bfc_tof_volume, lesion_coord, patch_side, neg_patches_path,
                                                   sub, ses, resampled_bfc_tof_aff_mat, intensity_thresholds)

        # extract LANDMARK negative patches
        if extract_landmark_patches:  # if we want to extract the negative patches in correspondence of the landmark points
            n, landmark_patches_list, _ = nb_last_created_patch(os.path.join(neg_patches_path, "{}_{}".format(sub, ses)))
            extract_neg_landmark_patches(neg_patches_path, sub, ses, n, tmp_folder, original_angio_volume_path, desired_spacing, resampled_bfc_tof_aff_mat,
                                         registrations_dir, mni_landmark_points_path, shift_scale_1, resampled_bfc_tof_volume_sitk, lesion_coord, landmark_patches_list)

        # extract RANDOM negative patches
        n, _, random_patches_list = nb_last_created_patch(os.path.join(neg_patches_path, "{}_{}".format(sub, ses)))
        extract_random_neg_patches(n, nb_random_patches_per_sub, angio_min_x, angio_max_x, angio_min_y, angio_max_y, angio_min_z, angio_max_z, shift_scale_1,
                                   vessel_mni_volume_resampled, resampled_bfc_tof_volume, seed_ext, lesion_coord, patch_side, neg_patches_path, sub, ses,
                                   resampled_bfc_tof_aff_mat, random_patches_list)

    else:  # if list is empty --> we're dealing with a control subject (i.e. subject without aneurysm(s))
        print("\n-------------------------- {}_{} --------------------------".format(sub, ses))
        lesion_coord = {}  # since it's a control subject, initialize dict as empty

        # extract VESSEL-LIKE negative patches
        seed_ext = extract_vessel_like_neg_patches(nb_vessel_like_patches_per_sub, angio_min_x, angio_max_x, angio_min_y, angio_max_y, angio_min_z, angio_max_z,
                                                   shift_scale_1, vessel_mni_volume_resampled, resampled_bfc_tof_volume, lesion_coord, patch_side, neg_patches_path,
                                                   sub, ses, resampled_bfc_tof_aff_mat, intensity_thresholds)

        # extract LANDMARK patches
        if extract_landmark_patches:  # if we want to extract the negative patches in correspondence of the landmark points
            n, landmark_patches_list, _ = nb_last_created_patch(os.path.join(neg_patches_path, "{}_{}".format(sub, ses)))
            extract_neg_landmark_patches(neg_patches_path, sub, ses, n, tmp_folder, original_angio_volume_path, desired_spacing, resampled_bfc_tof_aff_mat,
                                         registrations_dir, mni_landmark_points_path, shift_scale_1, resampled_bfc_tof_volume_sitk, lesion_coord, landmark_patches_list)

        # extract RANDOM negative patches
        n, _, random_patches_list = nb_last_created_patch(os.path.join(neg_patches_path, "{}_{}".format(sub, ses)))
        extract_random_neg_patches(n, nb_random_patches_per_sub, angio_min_x, angio_max_x, angio_min_y, angio_max_y, angio_min_z, angio_max_z, shift_scale_1,
                                   vessel_mni_volume_resampled, resampled_bfc_tof_volume, seed_ext, lesion_coord, patch_side, neg_patches_path, sub, ses,
                                   resampled_bfc_tof_aff_mat, random_patches_list)
    # -------------------------------------------------------------------------------------
    # remove temporary folder for this subject
    if os.path.exists(tmp_folder) and os.path.isdir(tmp_folder):
        shutil.rmtree(tmp_folder)


def extract_positive_patches(subdir: str,
                             aneurysm_mask_path: str,
                             bids_dataset_path: str,
                             desired_spacing: tuple,
                             out_dataset_path: str,
                             nb_pos_patches_per_sub: int = 5,
                             patch_side: int = 64) -> None:
    """This function extracts positive patches for each patient (subject with aneurysm(s)). Positive patches are extracted as random shifts around the aneurysm center
    Args:
        subdir: path to parent folder of aneurysm_mask_path
        aneurysm_mask_path: path to aneurysm mask
        bids_dataset_path: path to BIDS dataset
        desired_spacing: list containing the desired voxel spacing for resampling the input volumes
        out_dataset_path: path to folder where we create the output dataset
        nb_pos_patches_per_sub: number of positive patches to extract for each patient; defaults to 5
        patch_side: side of cubic patches that will be extracted; defaults to 64 (voxels)
    Raises:
        AssertionError: if BIDS dataset does not exits
        AssertionError: if folder containing bias-field-corrected volumes does not exist
    """

    if not os.path.exists(out_dataset_path):  # if folder doesn't exist
        os.makedirs(out_dataset_path)  # create data_set folder with today's date in the filename
        print("\nCreated Data Set Folder\n")

    assert os.path.exists(bids_dataset_path), f"Path {bids_dataset_path} does not exist"  # make sure that path exists
    bias_field_corrected_folders = os.path.join(bids_dataset_path, "derivatives", "N4_bias_field_corrected")
    assert os.path.exists(bias_field_corrected_folders), f"Path {bias_field_corrected_folders} does not exist"  # make sure that path exists

    shift_scale_1 = patch_side // 2  # define shift of cubic patches
    sub = re.findall(r"sub-\d+", subdir)[0]  # type: str # extract sub
    ses = re.findall(r"ses-\w{6}\d+", subdir)[0]  # extract ses

    if "Treated" in aneurysm_mask_path:
        lesion_name = re.findall(r"Treated_Lesion_\d+", aneurysm_mask_path)[0]  # type: str # extract lesion name
    else:
        lesion_name = re.findall(r"Lesion_\d+", aneurysm_mask_path)[0]  # type: str # extract lesion name
    center_coord_shift_scale_1 = shift_scale_1 - 1  # we only shift inside (within) the small patch, otherwise we risk that only the big patch includes the lesion

    # if we are NOT dealing with a treated aneurysm
    if "Treated" not in lesion_name:
        # create unique tmp folder where we save temporary files (this folder will be deleted at the end)
        tmp_folder = os.path.join(out_dataset_path, "tmp_{}_{}_{}_pos_patches".format(sub, ses, lesion_name))
        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)

        tmp_path_pos_patches = os.path.join(out_dataset_path, "Positive_Patches", "{}_{}_{}".format(sub, ses, lesion_name))
        tmp_path_pos_patches_masks = os.path.join(out_dataset_path, "Positive_Patches_Masks", "{}_{}_{}".format(sub, ses, lesion_name))
        print("\n-------------------------- {}_{}_{} --------------------------".format(sub, ses, lesion_name))
        if not os.path.exists(tmp_path_pos_patches) and not os.path.exists(tmp_path_pos_patches_masks):
            if "ADAM" in subdir:
                bet_tof_bfc_path = os.path.join(bias_field_corrected_folders, sub, ses, "anat",
                                                "{}_{}_desc-angio_N4bfc_brain_mask_ADAM.nii.gz".format(sub, ses))  # type: str # save path of angio brain after Brain Extraction Tool (BET)
            else:
                bet_tof_bfc_path = os.path.join(bias_field_corrected_folders, sub, ses, "anat", "{}_{}_desc-angio_N4bfc_brain_mask.nii.gz".format(sub, ses))  # type: str # save path of angio brain after Brain Extraction Tool (BET)
            assert os.path.exists(bet_tof_bfc_path), "Path {} does not exist".format(bet_tof_bfc_path)

            if "ADAM" in subdir:
                original_tof_bfc_path = os.path.join(bias_field_corrected_folders, sub, ses, "anat", "{}_{}_desc-angio_N4bfc_mask_ADAM.nii.gz".format(sub, ses))
            else:
                original_tof_bfc_path = os.path.join(bias_field_corrected_folders, sub, ses, "anat", "{}_{}_desc-angio_N4bfc_mask.nii.gz".format(sub, ses))
            assert os.path.exists(original_tof_bfc_path), "Path {} does not exist".format(original_tof_bfc_path)

            # invoke external method and save lesion information
            lesion = (extract_lesion_info_from_resampled_mask_volume(os.path.join(subdir, aneurysm_mask_path), tmp_folder, desired_spacing, sub, ses))
            # N.B. invert x and y because because of OpenCV (cv2); see https://stackoverflow.com/a/56849032/9492673
            x_center = lesion["centroid_y_coord"]  # extract x coordinate of lesion centroid
            y_center = lesion["centroid_x_coord"]  # extract y coordinate of lesion centroid
            z_central = lesion["idx_slice_with_more_white_pixels"]  # extract idx of slice with more non-zero pixels
            sc_shift = lesion["widest_dimension"] // 2  # extract half of the widest dimension of the mask
            nb_white_voxels = lesion["nb_non_zero_voxels"]

            # Load Mask Volume and resample it
            out_name = "{}_{}_binary_mask.nii.gz".format(sub, ses)
            nii_mask_volume_obj, nii_mask_volume, aff_mat_nii_mask = load_nifti_and_resample(os.path.join(subdir, aneurysm_mask_path), tmp_folder, out_name, desired_spacing, binary_mask=True)

            # Load angio Image Volume after BET and resample it
            out_name = "{}_{}_bet_tof_bfc.nii.gz".format(sub, ses)
            nii_volume_obj_after_bet, nii_volume_after_bet, aff_mat_after_bet = load_nifti_and_resample(bet_tof_bfc_path, tmp_folder, out_name, desired_spacing)

            # Load original angio Volume before BET and resample it
            out_name = "{}_{}_tof_bfc.nii.gz".format(sub, ses)
            nii_original_obj, nii_original, aff_mat_nii_original = load_nifti_and_resample(original_tof_bfc_path, tmp_folder, out_name, desired_spacing)

            # define flag that remains 0 if the lesion is small and we can create more samples around it; if instead shifting is impossible, flag will be incremented
            big_lesion_flag = 0  # type: int
            seed_ext = []  # initialize empty list where we'll store the seed used for good positive patches
            for n in range(nb_pos_patches_per_sub):
                pos_patches_masks_path = os.path.join(tmp_path_pos_patches_masks, "patch_pair_{}".format(n + 1))
                pos_patch_path = os.path.join(tmp_path_pos_patches, "patch_pair_{}".format(n + 1))
                seed = randrange(3000)  # generate random seed
                if not os.path.exists(pos_patches_masks_path) and not os.path.exists(pos_patch_path) and seed not in seed_ext:  # if folder doesn't exist
                    if big_lesion_flag == 0:  # this if condition will be satisfied at least once (first iteration)
                        # invoke function to shift center coordinates cause we don't want exactly-centered positive patches
                        x_transl, y_transl, z_transl = randomly_translate_coordinates(shift_=center_coord_shift_scale_1 - sc_shift,
                                                                                      center_x=x_center,
                                                                                      center_y=y_center,
                                                                                      center_z=z_central,
                                                                                      seed_=seed)
                        emergency_exit = 0  # emergency flag to avoid infinite while loop
                        while True:
                            emergency_exit += 1
                            if emergency_exit < 2000:  # we try to find a good patch for XX times
                                # Create translated Mask-Patches using lesion information
                                patch_mask_translated_scale_1 = nii_mask_volume[x_transl - shift_scale_1: x_transl + shift_scale_1,
                                                                                y_transl - shift_scale_1: y_transl + shift_scale_1,
                                                                                z_transl - shift_scale_1: z_transl + shift_scale_1]

                                # if the size of the bigger translated mask patch is not squared, or is empty, or does not include at least 80% of the mask
                                if patch_mask_translated_scale_1.size == 0 or patch_mask_translated_scale_1.shape != (patch_side, patch_side, patch_side) or np.count_nonzero(patch_mask_translated_scale_1) < 0.8 * nb_white_voxels:
                                    # check whether we are dealing with a huge lesion
                                    if center_coord_shift_scale_1 - sc_shift <= 0:
                                        big_lesion_flag += 1  # increment flag such that for this patient we only extract one sample, cause shifting is not feasible
                                        break  # stop while loop and only take central coord --> we can't shift. N.B. "randomly_translate_coordinates" returns the central coord if center_coord_shift_scale_1 - sc_shift <= 0

                                    seed = randrange(3000)  # change random seed
                                    # invoke function to shift coordinates with a new random seed
                                    x_transl, y_transl, z_transl = randomly_translate_coordinates(shift_=center_coord_shift_scale_1 - sc_shift,
                                                                                                  center_x=x_center,
                                                                                                  center_y=y_center,
                                                                                                  center_z=z_central,
                                                                                                  seed_=seed)

                                # if patch is correct
                                else:
                                    # check whether we are dealing with a huge lesion
                                    if center_coord_shift_scale_1 - sc_shift <= 0:
                                        big_lesion_flag += 1  # increment flag such that for this patient we only extract one sample, cause shifting is not feasible
                                        break  # stop while loop and only take central coord --> we can't shift. N.B. "randomly_translate_coordinates" returns the central coord if center_coord_shift_scale_1 - sc_shift <= 0
                                    break  # we found a good translated patch so we can go out of the while loop

                            else:  # if however, the XX seed changes are not enough, discard subject
                                break  # stop while loop

                        if emergency_exit < 2000:
                            # since there are no problems with the masks, create nib objects
                            seed_ext.append(seed)
                            patch_mask_obj_translated_scale_1 = nib.Nifti1Image(patch_mask_translated_scale_1.astype(np.int32), affine=aff_mat_nii_mask)  # convert patch from np array to nibabel object, preserving original affine

                            # Create TOF Patch using lesion information
                            patch_before_bet_transl_scale_1 = nii_original[x_transl - shift_scale_1: x_transl + shift_scale_1,
                                                                           y_transl - shift_scale_1: y_transl + shift_scale_1,
                                                                           z_transl - shift_scale_1: z_transl + shift_scale_1]  # crop volume

                            patch_after_bet_transl_scale_1 = nii_volume_after_bet[x_transl - shift_scale_1: x_transl + shift_scale_1,
                                                                                  y_transl - shift_scale_1: y_transl + shift_scale_1,
                                                                                  z_transl - shift_scale_1: z_transl + shift_scale_1]  # crop

                            assert patch_before_bet_transl_scale_1.size != 0 and patch_before_bet_transl_scale_1.shape == (patch_side, patch_side, patch_side)
                            assert patch_after_bet_transl_scale_1.size != 0 and patch_after_bet_transl_scale_1.shape == (patch_side, patch_side, patch_side)

                            # Sanity Check (SC) to see if the lesion was not cropped out by the Brain Extraction Tool (BET); performed with coords/volumes before padding
                            sc_patch_original = nii_original[x_center - sc_shift:x_center + sc_shift,
                                                             y_center - sc_shift:y_center + sc_shift,
                                                             z_central - sc_shift:z_central + sc_shift]  # crop volume with lesion info
                            sc_patch_after_bet = nii_volume_after_bet[x_center - sc_shift:x_center + sc_shift,
                                                                      y_center - sc_shift:y_center + sc_shift,
                                                                      z_central - sc_shift:z_central + sc_shift]  # crop volume with lesion info

                            # round decimal digits to 1 because sometimes the last digits are different between original and BET
                            sc_patch_original = np.around(sc_patch_original, decimals=1)
                            sc_patch_after_bet = np.around(sc_patch_after_bet, decimals=1)

                            entered = False  # type: bool # dummy boolean variable
                            if not np.array_equal(sc_patch_original, sc_patch_after_bet):  # if part of the lesion is different between original volume and BET volume
                                # if the lesion was completely excluded by the BET or if we lost more than 10% (i.e. we have less than 90% of the original left) of information due to BET
                                if np.count_nonzero(sc_patch_original) != 0:  # check to avoid division by 0
                                    if np.count_nonzero(sc_patch_after_bet) == 0 or (np.count_nonzero(sc_patch_after_bet) / np.count_nonzero(sc_patch_original)) < 0.9:
                                        entered = True
                                        print(f"Lesion voxels excluded by BET for {aneurysm_mask_path}; therefore, extract patch from original volume")

                            if entered is False:  # if entered is false, the lesion was not excluded by BET, thus we can use the volume after BET
                                patch_obj_transl_scale_1 = nib.Nifti1Image(patch_after_bet_transl_scale_1, affine=aff_mat_after_bet)  # convert translated patch from numpy array to nibabel object, preserving original affine array
                            else:  # if entered is true, the lesion was excluded by BET, thus we use the original volume (before BET)
                                patch_obj_transl_scale_1 = nib.Nifti1Image(patch_before_bet_transl_scale_1,
                                                                           affine=aff_mat_nii_original)  # convert translated patch from numpy array to nibabel object, preserving original affine array

                            # Save mask_patch
                            os.makedirs(pos_patches_masks_path)  # create folder containing mask patches
                            patch_mask_name_transl_scale_1 = '{}_{}_{}_patch_pair_{}_transl_mask_patch.nii.gz'.format(sub, ses, lesion_name, n + 1)
                            nib.save(patch_mask_obj_translated_scale_1, os.path.join(pos_patches_masks_path, patch_mask_name_transl_scale_1))

                            # Save image_patch
                            os.makedirs(pos_patch_path)  # create folder
                            patch_name_transl_scale_1 = '{}_{}_{}_patch_pair_{}_transl_pos_patch_angio.nii.gz'.format(sub, ses, lesion_name, n + 1)
                            nib.save(patch_obj_transl_scale_1, os.path.join(pos_patch_path, patch_name_transl_scale_1))

                            print("------------ patch_pair_{}".format(n + 1))
                            print("Original center coord in fsleyes: [x,y,z] = [{}, {}, {}]".format(x_center, y_center, z_central))
                            print("Patches created at translated center coord: [x,y,z] = [{}, {}, {}], seed = {}".format(x_transl, y_transl, z_transl, seed))

                        else:
                            print("Positive patch could not be created for {}".format(aneurysm_mask_path))
        # -------------------------------------------------------------------------------------
        # remove temporary folder for this subject
        if os.path.exists(tmp_folder) and os.path.isdir(tmp_folder):
            shutil.rmtree(tmp_folder)


def create_patch_ds(bids_dataset_path: str,
                    mni_landmark_points_path: str,
                    out_dataset_path: str,
                    id_out_dataset: str,
                    desired_spacing: tuple,
                    vessel_like_neg_patches: int,
                    random_neg_patches: int,
                    landmark_patches: bool,
                    pos_patches: int,
                    n_parallel_jobs: int,
                    overlapping: float,
                    subs_chuv_with_weak_labels_path: str,
                    subs_chuv_with_voxelwise_labels_path: str,
                    sub_ses_test: list,
                    patch_side: int = 64,
                    refine_weak_labels: bool = True,
                    convert_voxelwise_labels_into_weak: bool = False) -> None:
    """This function creates a dataset of patches starting from the 3D angio-TOF volume. For patients, it creates both
    positive (with aneurysm) and negative (without aneurysm) patches. For controls, it only creates negative patches.
    Args:
        bids_dataset_path: path to BIDS dataset
        mni_landmark_points_path: path to the csv file containing the landmark points coordinates
        out_dataset_path: path to folder where we create the output dataset
        id_out_dataset: unique identified for output folder where dataset will be created
        desired_spacing: list containing the desired voxel spacing for resampling the input volumes
        vessel_like_neg_patches: number of vessel-like negative patches to extract for each subject
        random_neg_patches: number of random negative patches to extract for each subject
        landmark_patches: if True, patches in correspondence of landmark points are extracted
        pos_patches: number of positive patches to extract for each patient (i.e. subject with aneurysm(s))
        n_parallel_jobs: number of jobs to run in parallel
        overlapping: amount of overlapping between patches in sliding-window approach
        subs_chuv_with_weak_labels_path: path to list containing patients with weak labels
        subs_chuv_with_voxelwise_labels_path: path to list containing patients with voxelwise labels
        sub_ses_test: sub_ses of the test set; we use it to take only the sub_ses of the training set
        patch_side: side of cubic patches that will be extracted (both negative and positive)
        refine_weak_labels: if set to True, the weak labels are refined with an intensity criterion
        convert_voxelwise_labels_into_weak: if set to True, it converts the voxel-wise labels into weak (i.e. it generates synthetic spheres around the aneurysm center)
    """
    # make sure all input paths exist
    assert os.path.exists(bids_dataset_path), f"Path {bids_dataset_path} does not exist"
    assert os.path.exists(mni_landmark_points_path), f"Path {mni_landmark_points_path} does not exist"
    assert os.path.exists(out_dataset_path), f"Path {out_dataset_path} does not exist"
    assert os.path.exists(subs_chuv_with_weak_labels_path), f"Path {subs_chuv_with_weak_labels_path} does not exist"
    assert os.path.exists(subs_chuv_with_voxelwise_labels_path), f"Path {subs_chuv_with_voxelwise_labels_path} does not exist"

    date = (datetime.today().strftime('%b_%d_%Y'))  # save today's date
    dataset_name = f"Data_Set_{date}_{id_out_dataset}"  # create dataset's name
    out_dataset_path = os.path.join(out_dataset_path, dataset_name)
    regexp_sub = re.compile(r'sub')  # create a substring template to match
    ext_gz = '.gz'  # type: str # set zipped files extension
    random.seed(123)  # fix random seed such that the extraction is identical every time the script is run

    # ----------------------------------------- CREATE NEGATIVE PATCHES -----------------------------------------
    intensity_thresholds = ()  # type: tuple # initialize as empty; if we want to extract vessel-like neg patches, this will contain numerical thresholds
    if vessel_like_neg_patches > 0:  # if we will create the vessel-like negative patches
        # we must extract some numerical thresholds to use for extracting vessel-like negative patches (i.e. neg patches similar to positive ones)
        print("\nComputing intensity thresholds...")
        intensity_thresholds = extract_thresholds_of_intensity_criteria(bids_dataset_path,
                                                                        sub_ses_test,
                                                                        patch_side,
                                                                        desired_spacing,
                                                                        out_dataset_path,
                                                                        n_parallel_jobs,
                                                                        overlapping)
        print("Done extracting intensity thresholds")

    # create input lists to create negative patches in parallel
    all_subdirs = []
    all_files = []
    for subdir, dirs, files in os.walk(bids_dataset_path):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()  # get the file extension
            if regexp_sub.search(file) and ext == ext_gz and "angio_N4bfc_brain_mask" in file and "N4_bias_field_corrected" in subdir:
                all_subdirs.append(subdir)
                all_files.append(file)
    assert all_subdirs and all_files, "Input lists must be non-empty"

    print("\nBegan extraction of negative patches...")
    Parallel(n_jobs=n_parallel_jobs, backend='loky')(delayed(extract_negative_patches)(all_subdirs[idx],
                                                                                       all_files[idx],
                                                                                       bids_dataset_path,
                                                                                       desired_spacing,
                                                                                       out_dataset_path,
                                                                                       mni_landmark_points_path,
                                                                                       intensity_thresholds,
                                                                                       extract_landmark_patches=landmark_patches,
                                                                                       nb_vessel_like_patches_per_sub=vessel_like_neg_patches,
                                                                                       nb_random_patches_per_sub=random_neg_patches,
                                                                                       patch_side=patch_side) for idx in range(len(all_subdirs)))

    # ----------------------------------------- CREATE POSITIVE PATCHES -----------------------------------------
    # create new input lists to create positive patches in parallel
    all_subdirs = []
    all_files = []
    for subdir, dirs, files in os.walk(bids_dataset_path):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()  # get the file extension
            if regexp_sub.search(file) and ext == ext_gz and "Lesion" in file and "registrations" not in subdir:
                all_subdirs.append(subdir)
                all_files.append(file)

    assert all_subdirs and all_files, "Input lists must be non-empty"

    print("\nBegan extraction of positive patches...")
    Parallel(n_jobs=n_parallel_jobs, backend='loky')(delayed(extract_positive_patches)(all_subdirs[idx],
                                                                                       all_files[idx],
                                                                                       bids_dataset_path,
                                                                                       desired_spacing,
                                                                                       out_dataset_path,
                                                                                       nb_pos_patches_per_sub=pos_patches,
                                                                                       patch_side=patch_side) for idx in range(len(all_subdirs)))

    # if we want to apply the intensity refinement to the weak labels
    if refine_weak_labels:
        # load list of subs with weak labels
        subs_with_weak_labels = load_pickle_list_from_disk(subs_chuv_with_weak_labels_path)

        pos_patches_path = os.path.join(out_dataset_path, "Positive_Patches")
        pos_masks_path = os.path.join(out_dataset_path, "Positive_Patches_Masks")

        # create new input lists to perform refinement in parallel
        all_pos_patches_path = []
        for sub_ses_lesion in os.listdir(pos_patches_path):
            sub = re.findall(r"sub-\d+", sub_ses_lesion)[0]  # extract sub
            ses = re.findall(r"ses-\w{6}\d+", sub_ses_lesion)[0]  # extract ses
            sub_ses = f"{sub}_{ses}"
            # only apply refinement to patients with weak labels (i.e. up to sub-449, because from sub-450 they already have voxel-wise labels)
            if sub_ses in subs_with_weak_labels:
                # loop over patch_pairs
                for patch_pair in os.listdir(os.path.join(pos_patches_path, sub_ses_lesion)):
                    # loop over files
                    for file in os.listdir(os.path.join(pos_patches_path, sub_ses_lesion, patch_pair)):
                        # get the file extension
                        ext = os.path.splitext(file)[-1].lower()  # type: str
                        if ext in ext_gz and "ADAM" not in file:  # ensure extension is correct and exclude ADAM subjects
                            all_pos_patches_path.append(os.path.join(pos_patches_path, sub_ses_lesion, patch_pair, file))

        if all_pos_patches_path:  # if list is non-empty
            print("\nBegan mask refinement...")
            Parallel(n_jobs=n_parallel_jobs, backend='loky')(delayed(refine_weak_label_one_sub)(all_pos_patches_path[idx],
                                                                                                pos_masks_path) for idx in range(len(all_pos_patches_path)))

    if convert_voxelwise_labels_into_weak:
        # load list of subs with weak labels
        subs_with_voxelwise_labels = load_pickle_list_from_disk(subs_chuv_with_voxelwise_labels_path)

        pos_patches_path = os.path.join(out_dataset_path, "Positive_Patches")
        pos_masks_path = os.path.join(out_dataset_path, "Positive_Patches_Masks")

        # create new input lists to perform refinement in parallel
        all_pos_patches_path = []
        for sub_ses_lesion in os.listdir(pos_patches_path):
            sub = re.findall(r"sub-\d+", sub_ses_lesion)[0]  # extract sub
            ses = re.findall(r"ses-\w{6}\d+", sub_ses_lesion)[0]  # extract ses
            sub_ses = "{}_{}".format(sub, ses)
            # only "weakify" (i.e. convert) labels of patients with voxelwise labels (i.e. from and above sub-450)
            if sub_ses in subs_with_voxelwise_labels:
                # loop over patch_pairs
                for patch_pair in os.listdir(os.path.join(pos_patches_path, sub_ses_lesion)):
                    # loop over files
                    for file in os.listdir(os.path.join(pos_patches_path, sub_ses_lesion, patch_pair)):
                        # get the file extension
                        ext = os.path.splitext(file)[-1].lower()  # type: str
                        if ext in ext_gz and "ADAM" not in file:  # ensure extension is correct and exclude ADAM subjects
                            all_pos_patches_path.append(os.path.join(pos_patches_path, sub_ses_lesion, patch_pair, file))

        if all_pos_patches_path:  # if list is non-empty
            print("\nBegan mask conversion (voxelwise_2_weak)...")
            Parallel(n_jobs=n_parallel_jobs, backend='loky')(delayed(weakify_voxelwise_label_one_sub)(all_pos_patches_path[idx],
                                                                                                      pos_masks_path) for idx in range(len(all_pos_patches_path)))


def main():
    start = time.time()  # start timer; used to compute the time needed to run this script
    # ---------------------------------------------------------------------------------------------
    # the code inside here is run only when THIS script is run, and not just imported
    config_dict = load_config_file()  # load input config file with argparser

    # extract input args
    bids_dataset_path = config_dict['bids_dataset_path']  # type: str # path to BIDS dataset (available on OpenNEURO)
    patch_side = config_dict['patch_side']  # type: int # size of training patches
    desired_spacing = tuple(config_dict['desired_spacing'])  # type: tuple # voxel spacing used for resampling (we resample all volumes to this spacing)
    overlapping = config_dict['overlapping']  # type: float # amount of overlapping between patches in sliding-window approach
    mni_landmark_points_path = config_dict['mni_landmark_points_path']  # type: str # path to file containining landmark points coordinates in physical space
    out_dataset_path = config_dict['out_dataset_path']  # type: str # path to output dataset
    id_out_dataset = config_dict['id_out_dataset']  # type: str # unique identifier used for naming the output folder where the dataset will be created
    subs_chuv_with_weak_labels_path = config_dict['subs_chuv_with_weak_labels_path']  # type: str # path to pickle file containin the subjects with weak labels
    subs_chuv_with_voxelwise_labels_path = config_dict['subs_chuv_with_voxelwise_labels_path']  # type: str  # path to pickle file containin the subjects with weak labels
    jobs_in_parallel = config_dict['jobs_in_parallel']  # type: int # nb. jobs to run in parallel (i.e. number of CPU (cores) to use); if set to -1, all available CPUs are used
    sub_ses_test = config_dict['sub_ses_test']  # type: list  # list containing sub_ses that will be used for test; empty by default; must be changed depending on the train-test split that is performed

    # ARGS for negative patches
    vessel_like_neg_patches = config_dict['vessel_like_neg_patches']  # type: int # number of vessel-like negative patches to extract
    random_neg_patches = config_dict['random_neg_patches']  # type: int # number of random negative patches to extract
    landmark_patches = str2bool(config_dict['landmark_patches'])  # type: bool # whether to extract patches in correspondence of landmark points or not

    # ARGS for positive patches
    pos_patches = config_dict['pos_patches']  # type: int # number of positive patches to extract for each aneurysm
    refine_weak_labels = str2bool(config_dict['refine_weak_labels'])  # type: bool # whether to refine weak labels or not
    convert_voxelwise_labels_into_weak = str2bool(config_dict['convert_voxelwise_labels_into_weak'])  # type: bool #  whether to weakify the voxelwise labels or not

    create_patch_ds(bids_dataset_path,
                    mni_landmark_points_path,
                    out_dataset_path,
                    id_out_dataset,
                    desired_spacing,
                    vessel_like_neg_patches,
                    random_neg_patches,
                    landmark_patches,
                    pos_patches,
                    jobs_in_parallel,
                    overlapping,
                    subs_chuv_with_weak_labels_path,
                    subs_chuv_with_voxelwise_labels_path,
                    sub_ses_test,
                    patch_side,
                    refine_weak_labels,
                    convert_voxelwise_labels_into_weak)
    # ---------------------------------------------------------------------------------------------
    end = time.time()  # stop timer
    print_running_time(start, end, "Dataset creation")


if __name__ == '__main__':
    main()

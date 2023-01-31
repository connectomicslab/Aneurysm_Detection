"""
Created on Apr 6, 2021

This script performs inference via the sliding-window approach.

"""

import os
import sys
PROJECT_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # extract directory of PyCharm project
sys.path.append(PROJECT_HOME)  # this line is needed to recognize the dir as a python package
from datetime import datetime
import re
from joblib import Parallel, delayed
import pandas as pd
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import tensorflow as tf
import shutil
from shutil import copyfile
import time
from inference.utils_inference import extract_reg_quality_metrics_one_sub, retrieve_registration_params, resample_volume, load_nifti_and_resample, round_half_up, \
    extracting_conditions_are_met, create_tf_dataset, create_output_folder, check_registration_quality, compute_patient_wise_metrics, create_input_lists, \
    save_and_print_results, convert_mni_to_angio, extract_thresholds_for_anatomically_informed, load_file_from_disk, save_sliding_window_mask_to_disk, \
    check_output_consistency_between_detection_and_segmentation, sanity_check_inputs, str2bool, print_running_time, load_config_file, create_dir_if_not_exist
from training.network_training import create_compiled_unet


__author__ = "Tommaso Di Noto"
__version__ = "0.0.1"
__email__ = "tommydino@hotmail.it"
__status__ = "Prototype"


def inference_one_subject(subdir,
                          file,
                          bids_dir_path,
                          sub_ses_test,
                          unet_checkpoint_path,
                          unet_patch_side,
                          unet_batch_size,
                          unet_threshold,
                          max_fp,
                          out_dir,
                          landmarks_physical_space_path,
                          new_spacing,
                          reg_quality_metrics_threshold,
                          intensity_thresholds,
                          distances_thresholds,
                          dark_fp_threshold,
                          ground_truth_dir,
                          min_aneurysm_volume,
                          unet,
                          anatomically_informed_sliding_window=True,
                          test_time_augmentation=True,
                          overlapping=0.,
                          reduce_fp=True,
                          reduce_fp_with_volume=True,
                          remove_dark_fp=True):
    """This function performs the sliding-window inference for one subject
    Args:
        subdir (str): folder where N4bfc_brain_mask is stored
        file (str): path to N4bfc_brain_mask volume
        bids_dir_path (str): path to BIDS dataset
        sub_ses_test (list): it contains the test sub-ses (i.e. those that were not used for training)
        unet_checkpoint_path (str): path to folder containing the saved parameters of the model
        unet_patch_side (int): patch side of cubic patches
        unet_batch_size (int): batch size to use (not really relevant since we are doing inference, but still needed)
        unet_threshold (float): threshold used to binarize the probabilistic U-Net's prediction
        max_fp (int): maximum number of allowed FPs per subject. If the U-Net predicts more than these, only the MAX_FP most probable are retained
        out_dir (str): path to folder where output files will be saved
        landmarks_physical_space_path (str): path to file containing the LPS landmark points in MNI physical space
        new_spacing (list): it contains the desired voxel spacing to which we will resample
        reg_quality_metrics_threshold (tuple): registration quality metrics; they will be used to decide whether the sliding-window is anatomically-informed or not
        intensity_thresholds (list): it contains [q5_local_vessel_mni, q5_global_vessel_mni, q5_local_tof_bet, q5_global_tof_bet, q5_nz_vessel_mni]
        distances_thresholds (list): it contains the thresholds to use for the distances from the patch centers to the landmark points
        dark_fp_threshold (float): threshold used to remove predictions which are on average too dark for being a true aneurysm
        ground_truth_dir (str): path to folder containing the ground truth volume and locations of the aneurysms
        min_aneurysm_volume (float): minimum aneurysm volume; if the model predicts an aneurysm with volume smaller than this, this prediction is removed
        unet (tf.keras.Model): trained network that we use for inference
        anatomically_informed_sliding_window (bool): whether to perform the sliding_window in an anatomically-informed fashion or not
        test_time_augmentation (bool): whether to perform test-time augmentation
        overlapping (float): rate of overlapping to use during the sliding-window; defaults to 0
        reduce_fp (bool): if set to True, only the "max_fp" most probable candidate aneurysm are retained; defaults to True
        reduce_fp_with_volume (bool): if set to True, only the candidate lesions that have a volume (mm^3) > than a specific threshold are retained; defaults to True
        remove_dark_fp (bool): if set to True, candidate aneurysms that are not brighter than a certain threshold (on average) are discarded; defaults to True
    Returns:
        None: if sub_ses not in sub_ses_test; otherwise it returns the output metrics as pd.Dataframe
    """
    # ----------------------- start timer ---------------------------
    start = time.time()  # stop timer; used at the end to compute the running time of this subject

    # ----------------------- extract sub_ses -----------------------
    sub = re.findall(r"sub-\d+", subdir)[0]  # extract sub
    ses = re.findall(r"ses-\w{6}\d+", subdir)[0]  # extract ses
    sub_ses = "{}_{}".format(sub, ses)
    assert len(sub) != 0, "Subject ID not found"
    assert len(ses) != 0, "Session not found"

    # since we are running patients in parallel, we must create separate tmp folders, otherwise we risk to overwrite/overload files of other subjects (racing conditions)
    tmp_path = os.path.join(out_dir, "tmp_{}_{}".format(sub, ses))

    # ------------------- only evaluate on test subjects for this cross-validation split -------------------
    if sub_ses in sub_ses_test:
        # check if output dir exists before starting inference; if it already exists, it means we already ran inference for this sub_ses
        if not os.path.exists(os.path.join(out_dir, sub, ses)):
            # uncomment line below for debugging
            # print("\nRunning inference on {}_{}".format(sub, ses))

            assert os.path.exists(unet_checkpoint_path), "Path {} does not exist".format(unet_checkpoint_path)
            assert os.path.exists(landmarks_physical_space_path), "Path {} does not exist".format(landmarks_physical_space_path)
            registration_params_dir = os.path.join(bids_dir_path, "derivatives", "registrations", "reg_params")
            assert os.path.exists(registration_params_dir), "Path {} does not exist".format(registration_params_dir)
            registration_metrics_dir = os.path.join(bids_dir_path, "derivatives", "registrations", "reg_metrics")
            assert os.path.exists(registration_metrics_dir), "Path {} does not exist".format(registration_metrics_dir)
            vessel_mni_registration_dir = os.path.join(bids_dir_path, "derivatives", "registrations", "vesselMNI_2_angioTOF")
            assert os.path.exists(vessel_mni_registration_dir), "path {0} does not exist".format(vessel_mni_registration_dir)  # make sure that path exists

            # retrieve useful registration parameters by invoking dedicated function
            mni_2_struct_mat_path, struct_2_tof_mat_path, mni_2_struct_warp_path, mni_2_struct_inverse_warp_path = retrieve_registration_params(os.path.join(registration_params_dir, sub, ses))

            # define half patch side
            shift_scale_1 = unet_patch_side // 2  # type: int # half patch side

            # save path of bias-field-corrected angio-TOF before BET
            if "ADAM" in subdir:  # if we are dealing with a subject from the ADAM dataset
                bfc_angio_path = os.path.join(subdir, "{}_{}_desc-angio_N4bfc_mask_ADAM.nii.gz".format(sub, ses))  # type: str # save path of bias-field-corrected angio TOF
            else:  # if instead we are dealing with a subject from the Lausanne dataset
                bfc_angio_path = os.path.join(subdir, "{}_{}_desc-angio_N4bfc_mask.nii.gz".format(sub, ses))  # type: str # save path of bias-field-corrected angio TOF
            assert os.path.exists(bfc_angio_path), "Path {} does not exist".format(bfc_angio_path)  # make sure path exists

            # load volume with sitk
            bfc_angio_volume_sitk = sitk.ReadImage(bfc_angio_path)  # type: sitk.Image

            # save path of corresponding vesselMNI co-registered volume
            if "ADAM" in vessel_mni_registration_dir:  # if we are dealing with a subject from the ADAM dataset
                vessel_mni_reg_volume_path = os.path.join(vessel_mni_registration_dir, sub, ses, "anat", "{}_{}_desc-vesselMNI2angio_deformed_ADAM.nii.gz".format(sub, ses))
            else:  # if instead we are dealing with a subject from the Lausanne dataset
                vessel_mni_reg_volume_path = os.path.join(vessel_mni_registration_dir, sub, ses, "anat", "{}_{}_desc-vesselMNI2angio_deformed.nii.gz".format(sub, ses))
            assert os.path.exists(vessel_mni_reg_volume_path), "Path {} does not exist".format(vessel_mni_reg_volume_path)  # make sure path exists

            # save path of bias-field-corrected angio brain after Brain Extraction Tool (BET)
            bet_bfc_angio_path = os.path.join(subdir, file)  # type: str # save path of bias-field-corrected angio brain after Brain Extraction Tool (BET)
            assert os.path.exists(bet_bfc_angio_path), "Path {} does not exist".format(bet_bfc_angio_path)  # make sure that path exists
            bet_bfc_angio_obj = nib.load(bet_bfc_angio_path)  # type: nib.Nifti1Image
            bet_bfc_angio = np.asanyarray(bet_bfc_angio_obj.dataobj)  # type: np.ndarray

            # load bias-field-corrected angio volume after BET and resample
            out_name = "{}_{}_bet_tof_bfc_resampled.nii.gz".format(sub, ses)
            nii_volume_after_bet_resampled_sitk, nii_volume_obj_after_bet_resampled, nii_volume_after_bet_resampled, aff_resampled = load_nifti_and_resample(bet_bfc_angio_path,
                                                                                                                                                             tmp_path,
                                                                                                                                                             out_name,
                                                                                                                                                             new_spacing)
            # save dimensions of resampled angio-BET volume
            rows_range, columns_range, slices_range = nii_volume_after_bet_resampled.shape
            # load corresponding vesselMNI volume and resample to new spacing
            out_path = os.path.join(tmp_path, "{}_{}_resampled_vessel_atlas.nii.gz".format(sub, ses))
            _, _, vessel_mni_volume_resampled = resample_volume(vessel_mni_reg_volume_path, new_spacing, out_path)

            # extract registration quality metrics for this subject and check (with the thresholds) if the registration was accurate enough
            sub_quality_metrics = extract_reg_quality_metrics_one_sub(os.path.join(registration_metrics_dir, sub, ses))  # type: tuple
            registration_accurate_enough = check_registration_quality(reg_quality_metrics_threshold, sub_quality_metrics)  # type: bool

            # load anatomical landmark coordinates with pandas
            df_landmarks_tof_space = pd.DataFrame()  # type: pd.DataFrame # initialize as empty dataframe; it will be modified if registration_accurate_enough == True
            if registration_accurate_enough:
                df_landmarks = pd.read_csv(landmarks_physical_space_path)  # type: pd.DataFrame # load csv file
                df_landmarks_tof_space = convert_mni_to_angio(df_landmarks, bfc_angio_volume_sitk, tmp_path, mni_2_struct_mat_path,
                                                              struct_2_tof_mat_path, mni_2_struct_inverse_warp_path)  # type: pd.DataFrame # register points from mni to subject space

            # ----------------------------- begin SLIDING-WINDOW -----------------------------
            nb_samples = 0  # type: int # it's a dummy variable to count how many patches are retained for each subject
            all_angio_patches_np_list = []  # type: list # empty list; it will contain the angio patches
            patch_center_coords = []  # type: list # will containt the center coords of the retained patches after the sliding-window approach
            retained_patches = np.zeros(nii_volume_after_bet_resampled.shape)  # volume to check which are the retained patches in the sliding window
            step = int(round_half_up((1 - overlapping) * unet_patch_side))  # type: int
            for i in range(shift_scale_1, rows_range, step):  # loop over rows
                for j in range(shift_scale_1, columns_range, step):  # loop over columns
                    for k in range(shift_scale_1, slices_range, step):  # loop over slices
                        patch_center_coordinates_resampled = [i, j, k]  # type: list # create list with coordinates of patch center

                        # uncomment line below for debugging
                        # print(patch_center_coordinates_resampled)

                        # ensure that the evaluated patch is not out of bound
                        if i - shift_scale_1 >= 0 and i + shift_scale_1 < rows_range and j - shift_scale_1 >= 0 and j + shift_scale_1 < columns_range and k - shift_scale_1 >= 0 and k + shift_scale_1 < slices_range:
                            # extract patch from resampled angio after BET
                            angio_patch_after_bet_scale_1 = nii_volume_after_bet_resampled[i - shift_scale_1:i + shift_scale_1,
                                                                                           j - shift_scale_1:j + shift_scale_1,
                                                                                           k - shift_scale_1:k + shift_scale_1]
                            # extract small-scale patch from resampled vesselMNI volume
                            vessel_mni_patch = vessel_mni_volume_resampled[i - shift_scale_1:i + shift_scale_1,
                                                                           j - shift_scale_1:j + shift_scale_1,
                                                                           k - shift_scale_1:k + shift_scale_1]
                            # since registrations were performed with original (non-resampled) volumes, we need to convert the coordinate back to original angio space
                            patch_center_coordinates_physical_space = nii_volume_after_bet_resampled_sitk.TransformIndexToPhysicalPoint(patch_center_coordinates_resampled)

                            # check that extracting conditions (e.g. intensity, distance-to-landmarks, not-out-of-bound) are met
                            if extracting_conditions_are_met(angio_patch_after_bet_scale_1, vessel_mni_patch, vessel_mni_volume_resampled, nii_volume_after_bet_resampled,
                                                             patch_center_coordinates_physical_space, unet_patch_side, df_landmarks_tof_space, registration_accurate_enough,
                                                             intensity_thresholds, distances_thresholds, anatomically_informed_sliding_window):
                                # if all conditions are met, we found a good candidate patch
                                patch_center_coords.append(patch_center_coordinates_resampled)  # append patch center to external list
                                nb_samples += 1  # increment counter variable to keep track of how many samples are evaluated for this subject
                                assert angio_patch_after_bet_scale_1.shape == (unet_patch_side, unet_patch_side, unet_patch_side), "Unexpected patch shape; expected ({},{},{})".format(unet_patch_side, unet_patch_side, unet_patch_side)
                                angio_patch_after_bet_scale_1 = tf.image.per_image_standardization(angio_patch_after_bet_scale_1)  # standardize patch to have mean 0 and variance 1
                                all_angio_patches_np_list.append(angio_patch_after_bet_scale_1)  # append standardized patch to external list

                                # fill mask volume with ones; this is just used to visualize which are the patches that were retained in the sliding-window
                                retained_patches[i - shift_scale_1:i + shift_scale_1,
                                                 j - shift_scale_1:j + shift_scale_1,
                                                 k - shift_scale_1:k + shift_scale_1] += 1

            # uncomment line below for debugging
            # print("Finished sliding window for {}".format(sub_ses))
            # time_sliding_window = time.time()  # stop timer
            # print_running_time(start, time_sliding_window, "Sliding-window {}_{}".format(sub, ses))

            assert len(all_angio_patches_np_list) == nb_samples == len(patch_center_coords), "{}: mismatch between all_angio_patches_np_list, nb_samples and patch_center_coords".format(sub_ses)
            batched_dataset = create_tf_dataset(all_angio_patches_np_list, unet_batch_size)

            # create output folder containing aneurysm center predictions and segmentation mask
            create_output_folder(batched_dataset,
                                 os.path.join(out_dir, sub, ses),
                                 unet_threshold,
                                 unet,
                                 nii_volume_after_bet_resampled,
                                 reduce_fp_with_volume,
                                 min_aneurysm_volume,
                                 nii_volume_obj_after_bet_resampled,
                                 patch_center_coords,
                                 shift_scale_1,
                                 bfc_angio_volume_sitk,
                                 aff_resampled,
                                 tmp_path,
                                 reduce_fp,
                                 max_fp,
                                 remove_dark_fp,
                                 dark_fp_threshold,
                                 bet_bfc_angio,
                                 sub_ses,
                                 test_time_augmentation,
                                 unet_batch_size)

            # ---------------------- SANITY CHECK ----------------------
            check_output_consistency_between_detection_and_segmentation(os.path.join(out_dir, sub, ses),
                                                                        sub,
                                                                        ses)
            # --------------------------------------------------- SAVE SLIDING-WINDOW MASK --------------------------------------------------
            save_sliding_window_mask_to_disk(retained_patches,
                                             aff_resampled,
                                             os.path.join(out_dir, sub, ses),
                                             bfc_angio_volume_sitk,
                                             tmp_path,
                                             out_filename="mask_sliding_window.nii.gz")
            # ------------------------------------------------------ REMOVE TMP FOLDER -------------------------------------------------------
            if os.path.exists(tmp_path) and os.path.isdir(tmp_path):
                shutil.rmtree(tmp_path)
            # ------------------------------------------ COMPUTE DETECTION and SEGMENTATION METRICS ------------------------------------------
            out_metrics = compute_patient_wise_metrics(os.path.join(out_dir, sub, ses),
                                                       os.path.join(ground_truth_dir, sub, ses),
                                                       sub,
                                                       ses)  # type: pd.DataFrame
            # uncomment line below for debugging
            # print("Output metrics for {}_{} = {}".format(sub, ses, out_metrics.values[0]))  # print detection and segmentation results for this subject

            # ---------------------- print inference time of this subject -------------
            end = time.time()  # stop timer
            print_running_time(start, end, "Inference {}_{}".format(sub, ses))

            return out_metrics

        # if instead output dir already exists
        else:
            if len(os.listdir(os.path.join(out_dir, sub, ses))) > 0:  # if the folder is not empty
                print("\n{}_{} already done".format(sub, ses))
            else:
                raise ValueError("Output dir exists but it's empty for {}_{}".format(sub, ses))


def main():
    # the code inside here is run only when THIS script is run, and not just imported
    config_dict = load_config_file()  # load input config file

    # args for patient-wise
    unet_patch_side = config_dict['unet_patch_side']
    unet_batch_size = config_dict['unet_batch_size']
    unet_threshold = config_dict['unet_threshold']
    overlapping = config_dict['overlapping']
    new_spacing = config_dict['new_spacing']
    conv_filters = tuple(config_dict['conv_filters'])
    lr = config_dict['lr']  # type: float # learning rate
    lambda_loss = config_dict['lambda_loss']  # type: float # value that weights the two terms of the hybrid loss
    cv_folds = config_dict['cv_folds']
    anatomically_informed_sliding_window = str2bool(config_dict['anatomically_informed_sliding_window'])
    test_time_augmentation = str2bool(config_dict['test_time_augmentation'])
    nb_parallel_jobs = config_dict['nb_parallel_jobs']

    # args for false positive reduction
    reduce_fp = str2bool(config_dict['reduce_fp'])
    max_fp = config_dict['max_fp']
    reduce_fp_with_volume = str2bool(config_dict['reduce_fp_with_volume'])
    min_aneurysm_volume = config_dict['min_aneurysm_volume']
    remove_dark_fp = str2bool(config_dict['remove_dark_fp'])

    # args for in-house CHUV paths
    bids_dir = config_dict['bids_dir_inhouse']
    all_test_sub_ses_inhouse = config_dict['all_test_sub_ses_inhouse']
    training_outputs_path = config_dict['training_outputs_path']

    # args for ADAM paths
    only_pretrain_on_adam = str2bool(config_dict['only_pretrain_on_adam'])
    bids_dir_adam = config_dict['bids_dir_adam']
    training_outputs_path_adam = config_dict['training_outputs_path_adam']

    # general args
    ground_truth_dir = config_dict['ground_truth_dir']
    id_output_dir = config_dict['id_output_dir']
    landmarks_physical_space_path = config_dict['landmarks_physical_space_path']
    inference_outputs_path = config_dict['inference_outputs_path']

    # make sure that inputs are fine
    sanity_check_inputs(unet_patch_side, unet_batch_size, unet_threshold, overlapping, new_spacing, conv_filters, cv_folds, anatomically_informed_sliding_window,
                        test_time_augmentation, reduce_fp, max_fp, reduce_fp_with_volume, min_aneurysm_volume, remove_dark_fp, bids_dir, training_outputs_path,
                        landmarks_physical_space_path, ground_truth_dir)

    #  ------------------ create input lists for running sliding-window in parallel across subjects
    all_subdirs, all_files = create_input_lists(bids_dir)

    # ------------------ COPY config file to output directory
    out_date_hours_minutes = (datetime.today().strftime('%b_%d_%Y_%Hh%Mm'))  # type: str # save today's date
    id_output_dir = "{}_{}".format(id_output_dir, out_date_hours_minutes)  # add date to dataset name
    path_config_file = sys.argv[2]  # type: str # save filename
    create_dir_if_not_exist(os.path.join(inference_outputs_path, id_output_dir))
    copyfile(path_config_file, os.path.join(inference_outputs_path, id_output_dir, "config_file.json"))  # copy config file to output dir to keep track of which were the input args

    # ------------------ loop over training folds of cross-validation
    metrics_cv_folds = []  # type: list # will contain the output metrics of all the subjects for each fold
    out_date = (datetime.today().strftime('%b_%d_%Y'))  # type: str # save today's date
    for cv_fold in range(1, cv_folds + 1):
        print("\nBegan fold {}".format(cv_fold))
        if only_pretrain_on_adam:  # if we load weights from a model only pre-trained on ADAM (i.e. which was not finetuned on the in-house dataset)
            unet_checkpoint_path = os.path.join(training_outputs_path_adam, "whole_dataset", "saved_models")  # type: str
            sub_ses_test = load_file_from_disk(all_test_sub_ses_inhouse)
            out_final_location_dir = os.path.join(inference_outputs_path, id_output_dir, "all_folds_inference_{}").format(cv_fold, out_date)  # type: str
        else:  # if instead we load weights from a model pre-trained on ADAM and then finetuned on in-house
            unet_checkpoint_path = os.path.join(training_outputs_path, "fold{}".format(cv_fold), "saved_models")  # type: str
            test_subs_path = os.path.join(training_outputs_path, "fold{}".format(cv_fold), "test_subs", "test_sub_ses.pkl")  # type: str # load test subjects of this CV fold
            sub_ses_test = load_file_from_disk(test_subs_path)  # type: list # load test subjects for this cross-validation split
            out_final_location_dir = os.path.join(inference_outputs_path, id_output_dir, "fold_{}_inference_{}").format(cv_fold, out_date)  # type: str

        assert os.path.exists(unet_checkpoint_path), "Path {} does not exist".format(unet_checkpoint_path)

        # --------------------------------------- create network and load weights
        # define input and create U-Net model
        inputs = tf.keras.Input(shape=(unet_patch_side, unet_patch_side, unet_patch_side, 1), name='TOF_patch')
        # create UNET and compile. There shouldn't be a need to compile since we're doing inference, but this solves a TF bug of the predict method, so we must compile the model
        unet = create_compiled_unet(inputs, lr, lambda_loss, conv_filters)
        # LOAD weights saved from a trained model somewhere else
        unet.load_weights(os.path.join(unet_checkpoint_path, "my_checkpoint")).expect_partial()

        # --------------- compute thresholds for anatomically-informed sliding-window
        reg_quality_metrics_threshold, intensity_thresholds, distances_thresholds, dark_fp_threshold = extract_thresholds_for_anatomically_informed(bids_dir,
                                                                                                                                                    sub_ses_test,
                                                                                                                                                    unet_patch_side,
                                                                                                                                                    new_spacing,
                                                                                                                                                    inference_outputs_path,
                                                                                                                                                    nb_parallel_jobs,
                                                                                                                                                    overlapping,
                                                                                                                                                    landmarks_physical_space_path,
                                                                                                                                                    out_final_location_dir,
                                                                                                                                                    only_pretrain_on_adam,
                                                                                                                                                    bids_dir_adam)

        print("\nreg_quality_metrics_threshold = {}".format(reg_quality_metrics_threshold))
        print("intensity_thresholds = {}".format(intensity_thresholds))
        print("distances_thresholds = {}".format(distances_thresholds))
        print("dark_fp_threshold = {}".format(dark_fp_threshold))

        # run patient-wise inference in parallel across subjects
        out_metrics_list = Parallel(n_jobs=nb_parallel_jobs, backend='threading')(delayed(inference_one_subject)(all_subdirs[idx],
                                                                                                                 all_files[idx],
                                                                                                                 bids_dir,
                                                                                                                 sub_ses_test,
                                                                                                                 unet_checkpoint_path,
                                                                                                                 unet_patch_side,
                                                                                                                 unet_batch_size,
                                                                                                                 unet_threshold,
                                                                                                                 max_fp,
                                                                                                                 out_final_location_dir,
                                                                                                                 landmarks_physical_space_path,
                                                                                                                 new_spacing,
                                                                                                                 reg_quality_metrics_threshold,
                                                                                                                 intensity_thresholds,
                                                                                                                 distances_thresholds,
                                                                                                                 dark_fp_threshold,
                                                                                                                 ground_truth_dir,
                                                                                                                 min_aneurysm_volume,
                                                                                                                 unet,
                                                                                                                 anatomically_informed_sliding_window=anatomically_informed_sliding_window,
                                                                                                                 test_time_augmentation=test_time_augmentation,
                                                                                                                 overlapping=overlapping,
                                                                                                                 reduce_fp=reduce_fp,
                                                                                                                 reduce_fp_with_volume=reduce_fp_with_volume,
                                                                                                                 remove_dark_fp=remove_dark_fp) for idx in range(len(all_subdirs)))
        # create unique dataframe with the metrics of all the subjects of this CV fold
        not_none_values = filter(None.__ne__, out_metrics_list)  # type: filter
        out_metrics_list = list(not_none_values)  # type: list  # remove None values
        if out_metrics_list:  # if list is non-empty
            out_metrics_df_cv_fold = pd.concat(out_metrics_list)  # type: pd.DataFrame
            metrics_cv_folds.append(out_metrics_df_cv_fold)  # append to external list

    # ------------------ SAVE and PRINT OUTPUT RESULTS
    save_and_print_results(metrics_cv_folds, os.path.join(inference_outputs_path, id_output_dir), out_date)


if __name__ == "__main__":
    main()

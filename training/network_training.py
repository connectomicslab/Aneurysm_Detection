"""
Created on Apr 6, 2021

This script performs the training of one fold of the cross-validation

"""

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
PROJECT_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # extract directory of PyCharm project
sys.path.append(PROJECT_HOME)  # this line is needed to recognize the dir as a python package
import tensorflow as tf
import time
from datetime import datetime
import re
import numpy as np
from pathlib import Path
import pickle
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from joblib import Parallel, delayed
import random
from typing import Tuple
from dataset_creation.utils_dataset_creation import print_running_time
from inference.utils_inference import load_config_file, str2bool, round_half_up, load_file_from_disk, create_dir_if_not_exist


__author__ = "Tommaso Di Noto"
__version__ = "0.0.1"
__email__ = "tommydino@hotmail.it"
__status__ = "Prototype"


def save_pickle_list_to_disk(list_to_save: list,
                             out_dir: str,
                             out_filename: str) -> None:
    """This function saves a list to disk
    Args:
        list_to_save (list): list that we want to save
        out_dir (str): path to output folder; will be created if not present
        out_filename (str): output filename
    Returns:
        None
    """
    if not os.path.exists(out_dir):  # if output folder does not exist
        os.makedirs(out_dir)  # create it
    open_file = open(os.path.join(out_dir, out_filename), "wb")
    pickle.dump(list_to_save, open_file)  # save list with pickle
    open_file.close()


def find_sub_ses_pairs(data_path: str):
    """This function extracts all the sub_ses of the dataset.
    Args:
        data_path (str): path to folder containing all training patches (both negative and positive)
    Returns:
        all_sub_ses (list): it contains the sub-ses of the whole cohort
    """
    neg_patch_path = os.path.join(data_path, "Negative_Patches")

    all_sub_ses = []
    for folders in os.listdir(neg_patch_path):  # every subject (both controls and patients) has negative patches, so we just loop over the negative patches directory
        sub = re.findall(r"sub-\d+", folders)[0]  # extract sub
        ses = re.findall(r"ses-\w{6}\d+", folders)[0]  # extract ses
        sub_ses = "{}_{}".format(sub, ses)  # combine into unique string
        all_sub_ses.append(sub_ses)

    return all_sub_ses


def define_output_folders(training_outputs_folder: str,
                          ds_path_: str,
                          cv_fold_numb_: int) -> Tuple[str, str, str, str]:
    """This function takes as input the path where the dataset is stored and the path where the script is stored. Then, it creates the output folders and checks
    that the dataset folder is within the script path. This avoids that we use one script, but we save into another dataset's output folder.
    Args:
        training_outputs_folder (str): path to output folder
        ds_path_ (str): path where the dataset is stored
        cv_fold_numb_ (int): number of cross-val fold
    Returns:
        model_path_ (str): path where we'll store (checkpoint) the best model
        plots_path_ (str): path where we'll store all the plots
        tensorboard_cbk_path_ (str): path were we'll store tensorboard logs
        test_subs_path_ (str): path where we'll store the test sub_ses for this split
    """
    model_path_ = os.path.join(ds_path_, training_outputs_folder, "fold{}".format(cv_fold_numb_), "saved_models", "my_checkpoint")
    create_dir_if_not_exist(model_path_)  # if path does not exist, create it

    plots_path_ = os.path.join(ds_path_, training_outputs_folder, "fold{}".format(cv_fold_numb_), "plots")
    create_dir_if_not_exist(plots_path_)  # if path does not exists, create it

    tensorboard_cbk_path_ = os.path.join(ds_path_, training_outputs_folder, "fold{}".format(cv_fold_numb_), "logs_tensorboard")
    create_dir_if_not_exist(tensorboard_cbk_path_)  # if path does not exist, create it

    test_subs_path_ = os.path.join(ds_path_, training_outputs_folder, "fold{}".format(cv_fold_numb_), "test_subs")
    create_dir_if_not_exist(test_subs_path_)  # if path does not exist, create it

    return model_path_, plots_path_, tensorboard_cbk_path_, test_subs_path_


def create_tf_dataset_from_patches_and_masks(all_angio_patches_list,
                                             all_angio_masks_list):
    """This function creates a tf.data.Dataset from two lists: the one containing the patches and the one containing the corresponding masks
    Args:
        all_angio_patches_list (list): it contains all the patches
        all_angio_masks_list (list): it contains all the corresponding masks
    Returns:
        dataset_ (tf.data.Dataset): the dataset containing patches and masks
        patch_side_ (int): the side of each cubic patch
        buffer_size_ (int): the number of patches in the created dataset
    Raises:
        AssertionError: if the two input lists do not have the same length
    """
    assert len(all_angio_patches_list) == len(all_angio_masks_list), "A different number of patches and masks was found"
    buffer_size_ = len(all_angio_patches_list)  # type: int # extract number of images (and masks); we'll use this to shuffle the dataset
    patch_side_ = all_angio_patches_list[0].shape[0]  # type: int # extract first dimension of first patch (since they are all cubic patches)

    all_angio_patches_tensors = tf.convert_to_tensor(all_angio_patches_list, dtype=tf.float32)  # convert from list to tf.Tensor
    all_angio_masks_tensors = tf.convert_to_tensor(all_angio_masks_list, dtype=tf.float32)  # convert from list to tf.Tensor

    dataset_ = tf.data.Dataset.from_tensor_slices((all_angio_patches_tensors, all_angio_masks_tensors))  # create tf.data.Dataset
    dataset_ = dataset_.shuffle(buffer_size_)  # shuffle dataset otherwise ADAM and CHUV samples may not be interleaved

    return dataset_, patch_side_, buffer_size_


def create_dataset_positives_one_sub(subdir: str,
                                     file: str):
    """This function creates a positive patch and the corresponding mask
    Args:
        subdir (str): folder where positive patch is stored
        file (str): filename of positive patch
    Returns:
        out_array (np.ndarray): it contains the positive patch and the corresponding mask stacked
    """
    pos_patch_obj = nib.load(os.path.join(subdir, file))  # type: nib.Nifti1Image
    pos_patch = np.asanyarray(pos_patch_obj.dataobj)  # type: np.ndarray

    subdir_masks = subdir.replace("Positive_Patches", "Positive_Patches_Masks")
    file_mask = file.replace("pos_patch_angio", "mask_patch")
    assert os.path.exists(os.path.join(subdir_masks, file_mask)), "Path {} does not exist".format(os.path.join(subdir_masks, file_mask))

    mask_patch_obj = nib.load(os.path.join(subdir_masks, file_mask))  # type: nib.Nifti1Image
    mask_patch_np = np.asanyarray(mask_patch_obj.dataobj)  # type: np.ndarray
    if not np.array_equal(mask_patch_np, mask_patch_np.astype(bool)) or np.sum(mask_patch_np) == 0:
        raise ValueError("Mask must be binary and non-empty for positive patches")

    assert pos_patch.shape == mask_patch_np.shape, "Patch and mask must have the same shape"

    out_array = np.asarray([pos_patch, mask_patch_np])

    return out_array


def create_dataset_positives_parallel(pos_patches_path: str,
                                      subs_to_use: list,
                                      n_parallel_jobs: int):
    """This function creates the dataset of all positive patches (and corresponding masks) in parallel
    Args:
        pos_patches_path (str): path to folder containing positive patches
        subs_to_use (list): it contains the sub_ses of all subjects to use to create the dataset of patches
        n_parallel_jobs (int): number of jobs to run in parallel
    Returns:
        dataset_ (tf.data.Dataset): dataset containing 3D patches and corresponding 3D mask volumes
        patch_side_ (int): side of cubic input patches
        buffer_size_ (int): number of patches (and masks) we'll work with
    """
    all_subdirs = []
    all_files = []
    regexp_sub = re.compile(r'sub')  # create a substring template to match
    ext_gz = ".gz"  # type: str # set extension to match

    for subdir, dirs, files in os.walk(pos_patches_path):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()  # get the file extension
            # save path of every positive patch
            if regexp_sub.search(file) and ext == ext_gz and "pos_patch" in file and "mask" not in file:
                sub = re.findall(r"sub-\d+", subdir)[0]
                # ses = re.findall(r"ses-\d+", subdir)[0]
                ses = re.findall(r"ses-\w{6}\d+", subdir)[0]  # extract ses
                sub_ses = "{}_{}".format(sub, ses)
                if sub_ses in subs_to_use:
                    all_subdirs.append(subdir)
                    all_files.append(file)

    assert all_subdirs and all_files, "Input lists must be non-empty"

    # UNCOMMENT LINES BELOW FOR QUICK debugging (only takes N patches)
    # all_subdirs = all_subdirs[0:100]
    # all_files = all_files[0:100]

    out_list = Parallel(n_jobs=n_parallel_jobs, backend='loky')(delayed(create_dataset_positives_one_sub)(all_subdirs[idx],
                                                                                                          all_files[idx]) for idx in range(len(all_subdirs)))

    out_list_np = np.asarray(out_list)  # type: np.ndarray # convert from list to numpy array
    all_angio_patches_list = list(out_list_np[:, 0, ...])  # extract patches
    all_angio_masks_list = list(out_list_np[:, 1, ...])  # extract masks

    dataset, patch_side, buffer_size = create_tf_dataset_from_patches_and_masks(all_angio_patches_list, all_angio_masks_list)

    return dataset, patch_side, buffer_size


def create_dataset_negatives_one_sub(subdir,
                                     file):
    """This function creates a negative patch and the corresponding empty mask
    Args:
        subdir (str): folder where negative patch is stored
        file (str): filename of negative patch
    Returns:
        out_array (np.ndarray): it contains the negative patch and the corresponding empty mask stacked
    """
    neg_patch_obj = nib.load(os.path.join(subdir, file))  # type: nib.Nifti1Image
    neg_patch = np.asanyarray(neg_patch_obj.dataobj)  # type: np.ndarray

    # for the mask, simply create an empty volume since these are negative patches (i.e. without aneurysm)
    neg_patch_mask = np.zeros(neg_patch.shape)  # type: np.ndarray

    assert neg_patch.shape == neg_patch_mask.shape, "Patch and mask must have the same shape"

    out_array = np.asarray([neg_patch, neg_patch_mask])

    return out_array


def create_dataset_negatives_parallel(neg_patches_path,
                                      subs_to_use,
                                      n_parallel_jobs):
    """This function creates the dataset of all negative patches (and corresponding empty masks) in parallel
    Args:
        neg_patches_path (str): path to folder containing negative patches
        subs_to_use (list): it contains the sub_ses of all subjects to use to create the dataset of patches
        n_parallel_jobs (int): number of jobs to run in parallel
    Returns:
        dataset_ (tf.data.Dataset): dataset containing 3D patches and corresponding 3D mask volumes
        patch_side_ (int): side of cubic input patches
        buffer_size_ (int): number of patches (and masks) we'll work with
    """
    all_subdirs = []
    all_files = []
    regexp_sub = re.compile(r'sub')  # create a substring template to match
    ext_gz = ".gz"  # type: str # set extension to match

    for subdir, dirs, files in os.walk(neg_patches_path):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()  # get the file extension
            # save path of every negative patch
            if regexp_sub.search(file) and ext == ext_gz and "neg_patch" in file and "mask" not in file:
                sub = re.findall(r"sub-\d+", subdir)[0]
                # ses = re.findall(r"ses-\d+", subdir)[0]
                ses = re.findall(r"ses-\w{6}\d+", subdir)[0]  # extract ses
                sub_ses = "{}_{}".format(sub, ses)

                if sub_ses in subs_to_use:
                    all_subdirs.append(subdir)
                    all_files.append(file)

    assert all_subdirs and all_files, "Input lists must be non-empty"

    # UNCOMMENT LINES BELOW FOR QUICK debugging (only takes N patches)
    # all_subdirs = all_subdirs[0:100]
    # all_files = all_files[0:100]

    out_list = Parallel(n_jobs=n_parallel_jobs, backend='loky')(delayed(create_dataset_negatives_one_sub)(all_subdirs[idx],
                                                                                                          all_files[idx]) for idx in range(len(all_subdirs)))

    out_list_np = np.asarray(out_list)  # type: np.ndarray # convert from list to numpy array
    all_angio_patches_list = list(out_list_np[:, 0, ...])  # extract patches
    all_angio_masks_list = list(out_list_np[:, 1, ...])  # extract masks

    dataset, patch_side, buffer_size = create_tf_dataset_from_patches_and_masks(all_angio_patches_list, all_angio_masks_list)

    return dataset, patch_side, buffer_size


def create_batched_augmented_tf_dataset(pos_patches_path: str,
                                        subs_to_use: list,
                                        neg_patches_path: str,
                                        batch_size: int,
                                        n_parallel_jobs: int,
                                        augment: bool = False):
    """This function creates a tf.data.Dataset from the folders containing the negative and positive patches.
    Args:
        pos_patches_path (str): path to folder where positive patches are stored
        subs_to_use (list): it contains the sub_ses of all subjects to use to create the dataset of patches
        neg_patches_path (str): path to folder where negative patches are stored
        batch_size (int): size of each training batch
        n_parallel_jobs (int): number of jobs to run in parallel (the higher, the faster)
        augment (bool): it indicates whether the dataset should be augmented or not; only the training dataset should be augmented; False by default
    Returns:
        batched_dataset (tf.data.Dataset): batched, standardized and augmented (only positive patches) version of the dataset to create
        pos_patch_side (int): patch side of positive (and negative) patches
    """
    pos_dataset, pos_patch_side, pos_buffer_size = create_dataset_positives_parallel(pos_patches_path, subs_to_use, n_parallel_jobs)
    neg_dataset, neg_patch_side, neg_buffer_size = create_dataset_negatives_parallel(neg_patches_path, subs_to_use, n_parallel_jobs)
    assert pos_patch_side == neg_patch_side, "Negative and positive samples have different dim"

    if augment:  # if we want to augment the training dataset
        augm_pos_dataset, orig_plus_augm_buffer_size = augment_dataset(pos_dataset, pos_dataset)  # we only augment the positive patches which are way less than the negatives
        buffer_size = neg_buffer_size + orig_plus_augm_buffer_size  # type: int # compute new buffer size
        print("\nThere are {} neg and {} aug pos sample ({} original and {} augmented)".format(neg_buffer_size,
                                                                                               orig_plus_augm_buffer_size,
                                                                                               pos_buffer_size,
                                                                                               orig_plus_augm_buffer_size - pos_buffer_size))
        dataset = neg_dataset.concatenate(augm_pos_dataset)  # merge negative and augmented-positive samples into one unique dataset

    # if instead augment is False (e.g. for the validation set), we don't perform data augmentation
    else:
        buffer_size = neg_buffer_size + pos_buffer_size  # type: int # compute new buffer size
        print("\nThere are {} neg and {} pos sample".format(neg_buffer_size, pos_buffer_size))
        dataset = neg_dataset.concatenate(pos_dataset)  # merge positive and negative samples into one unique dataset

    dataset = dataset.shuffle(buffer_size=buffer_size)  # shuffle dataset, otherwise pos and neg samples are not interleaved
    batched_dataset = create_standardized_batched_dataset_and_prefetch(dataset, batch_size)  # standardize samples and divide in batches
    return batched_dataset, pos_patch_side


def check_mask_is_binary_and_non_empty(mask):
    """This function ensures that the input mask is binary and non-empty
    Args:
        mask (tf.Tensor): mask volume to check
    Raises:
        ValueError: if mask is either not binary or empty
    """
    if not tf.reduce_all(tf.logical_or(tf.equal(mask, 0.0), tf.equal(mask, 1.0))).numpy() or tf.reduce_sum(mask).numpy() == 0:
        raise ValueError("Mask is either not binary or empty")


def augment_dataset(tf_dataset,
                    out_dataset):
    """This function performs the data augmentation of the positive patches (i.e. those that contain an aneurysm)
    Args:
        tf_dataset (tf.data.Dataset): input dataset to augment
        out_dataset (tf.data.Dataset): in the beginning, it corresponds to the input dataset; in the end, it is the concatenation of the original and the augmented datasets
    Returns:
        out_dataset (tf.data.Dataset): it is the concatenation of the original and the augmented datasets
        tot_numb_samples_orig_plus_augmented (int): number of samples (original + augmented)
    """
    def horizontal_flipping(sample, mask):
        hor_flip_patch_angio = tf.image.flip_left_right(sample)
        hor_flip_patch_mask = tf.image.flip_left_right(mask)
        check_mask_is_binary_and_non_empty(hor_flip_patch_mask)
        return hor_flip_patch_angio, hor_flip_patch_mask

    def vertical_flipping(sample, mask):
        ver_flip_patch_angio = tf.image.flip_up_down(sample)
        ver_flip_patch_mask = tf.image.flip_up_down(mask)
        check_mask_is_binary_and_non_empty(ver_flip_patch_mask)
        return ver_flip_patch_angio, ver_flip_patch_mask

    # Rotations (!counter-clockwise!); k indicates the number of times rotation occurs
    def rotate_270(sample, mask):
        rot_270_patch_angio = tf.image.rot90(sample, k=1, name="270_CC")
        rot_270_patch_mask = tf.image.rot90(mask, k=1, name="270_CC")
        check_mask_is_binary_and_non_empty(rot_270_patch_mask)
        return rot_270_patch_angio, rot_270_patch_mask

    def rotate_180(sample, mask):
        rot_180_patch_angio = tf.image.rot90(sample, k=2, name="180_CC")
        rot_180_patch_mask = tf.image.rot90(mask, k=2, name="180_CC")
        check_mask_is_binary_and_non_empty(rot_180_patch_mask)
        return rot_180_patch_angio, rot_180_patch_mask

    def rotate_90(sample, mask):
        rot_90_patch_angio = tf.image.rot90(sample, k=3, name="90_CC")
        rot_90_patch_mask = tf.image.rot90(mask, k=3, name="90_CC")
        check_mask_is_binary_and_non_empty(rot_90_patch_mask)
        return rot_90_patch_angio, rot_90_patch_mask

    def adjust_contrast(sample, mask):
        contr_adj_patch_angio = tf.image.adjust_contrast(sample, contrast_factor=2)  # adjust contrast
        contr_adj_patch_mask = mask  # no need to adjust the contrast of the mask cause it must remain binary
        check_mask_is_binary_and_non_empty(contr_adj_patch_mask)
        return contr_adj_patch_angio, contr_adj_patch_mask

    def gamma_correction(sample, mask):
        if (sample.numpy() > 0).all():  # if the patch has all non-negative values
            gamma_adj_patch_angio = tf.image.adjust_gamma(sample, gamma=0.2, gain=1)  # apply the correction Out = gain * In**gamma
            gamma_adj_patch_mask = mask  # no need to adjust the contrast of the mask cause it must remain binary
            check_mask_is_binary_and_non_empty(gamma_adj_patch_mask)
            return gamma_adj_patch_angio, gamma_adj_patch_mask
        else:
            return sample, mask

    def gaussian_noise(sample, mask):
        noise = tf.random.normal(shape=tf.shape(sample), mean=0.0, stddev=1, dtype=tf.float32)
        gauss_noise_patch_angio = tf.math.add(sample, noise)
        gauss_noise_patch_mask = mask  # no need to add noise to the mask cause it must remain binary
        check_mask_is_binary_and_non_empty(gauss_noise_patch_mask)
        return gauss_noise_patch_angio, gauss_noise_patch_mask

    # Add the augmentations to the dataset
    augmentations = [horizontal_flipping, vertical_flipping, rotate_270, rotate_180, rotate_90, adjust_contrast, gamma_correction, gaussian_noise]
    for augm in augmentations:
        # Apply the augmentation, running jobs in parallel
        augm_dataset = tf_dataset.map(lambda x, y: tf.py_function(func=augm, inp=[x, y], Tout=[tf.float32, tf.float32]),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
        out_dataset = out_dataset.concatenate(augm_dataset)  # concatenate augmented dataset to output dataset

    original_numb_samples = tf_dataset.cardinality().numpy()
    tot_numb_samples_orig_plus_augmented = out_dataset.cardinality().numpy()
    assert tot_numb_samples_orig_plus_augmented == original_numb_samples*(len(augmentations)+1), "Something wrong with out_dataset cardinality"

    return out_dataset, tot_numb_samples_orig_plus_augmented


def create_standardized_batched_dataset_and_prefetch(ds,
                                                     batch_size_):
    """This function takes as input a tf.data.Dataset, standardize all samples and returns the batched version of it.
    Args:
        ds (tf.data.Dataset): dataset that we want to split
        batch_size_ (int): size of each batch
    Returns:
        batched_train_dataset_ (tf.data.Dataset): batched version of training dataset
    """

    # standardize samples to have mean 0 and variance 1
    ds = ds.map(lambda x, y: (tf.image.per_image_standardization(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # divide dataset in batches
    batched_train_dataset_ = ds.batch(batch_size_)

    # ADD channel dimension to train dataset (both to the samples and to the labels) through the "map" method of tf.data.Dataset;
    # This ensures that the patches are readable by the first conv layer and that labels have comparable shape with the output of the net.
    batched_train_dataset_ = batched_train_dataset_.map(lambda x, y: (tf.expand_dims(x, axis=-1), tf.expand_dims(y, axis=-1)), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # prefetch as many batches of data as possible so that it is immediately ready for the next training loop
    batched_train_dataset_ = batched_train_dataset_.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return batched_train_dataset_


def dice_coeff(y_true: tf.Tensor,
               y_pred: tf.Tensor,
               smooth: float = 1.) -> float:
    """This function computes the soft dice coefficient between the predicted mask and the ground truth mask
    Args:
        y_true: ground truth mask
        y_pred: predicted mask
        smooth: value added for numerical stability (avoid division by 0)
    Returns:
        dice_coefficient: dice coefficient
    """
    # flatten vectors and cast to float32
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(tf.square(y_pred_f)) + tf.reduce_sum(tf.square(y_true_f))

    dice_coefficient = (2. * intersection + smooth) / (union + smooth)

    return dice_coefficient


def dice_loss(y_true: tf.Tensor,
              y_pred: tf.Tensor) -> float:
    """This function computes the dice loss as 1-dsc_coeff
    Args:
        y_true: ground truth mask
        y_pred: predicted mask
    Returns:
        dsc_loss: dice loss
    """
    dsc_loss = 1 - dice_coeff(y_true, y_pred)

    return dsc_loss


def bce_dice_loss(loss_lambda):
    """This function combines the binary cross-entropy loss with the Dice loss into one unique hybrid loss
    Args:
        loss_lambda (float): value to balance/weight the two terms of the loss
    Returns:
        loss (function)
    """
    def loss(y_true, y_pred):
        """This function computes the actual hybrid loss
        Args:
            y_true (tf.Tensor): label volume
            y_pred (tf.Tensor): prediction volume
        Returns:
            hybrid_loss (tf.Tensor): sum of the two losses
        """
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)  # compute binary cross entropy
        bce_loss = tf.reduce_mean(bce)  # reduce the result to get the final loss
        hybrid_loss = (1-loss_lambda) * bce_loss + loss_lambda * dice_loss(y_true, y_pred)  # sum the two losses
        return hybrid_loss

    return loss


def create_compiled_unet(inputs_: tf.keras.Input,
                         learning_rate: float,
                         lambda_loss: float,
                         conv_filters: tuple) -> tf.keras.Model:
    """This function creates a 3D U-Net starting from an input with modifiable dimensions and it returns the compiled model
    Args:
        inputs_: input to the model; used to set the input dimensions
        learning_rate: learning rate of the model
        lambda_loss: value that weights the two terms of the hybrid loss
        conv_filters: it contains the number of filters to use in the convolution layers
    Returns:
        model: the compiled U-Net
    """
    # DOWNWARD PATH (encoder)
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

    # UPWARD PATH (decoder)
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
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss=bce_dice_loss(lambda_loss), metrics=[dice_coeff, "binary_crossentropy"])

    return model


def save_ext_train_curves(train_loss,
                          train_dice,
                          image_path_):
    """ This function takes as input train_dice, train_loss, and plots them into a unique figure which is saved in the specified path
    Args:
        train_loss (list): training loss
        train_dice (list): training accuracy
        image_path_ (str): path where we want to save the image
    Returns:
        None
    """
    x_axis = np.arange(1, len(train_dice)+1, 1)  # since the two input vectors have same length, just use one of the two to extract epochs
    fig, ax1 = plt.subplots()  # create figure

    color_1 = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only keep integers in x axis
    ax1.set_ylabel('Dice')
    ax1.plot(x_axis, train_dice, color=color_1, label='Train dice')
    ax1.tick_params(axis='y', labelcolor=color_1)
    ax1.set_ylim([0, 1])  # set upper and lower bound for y-axis ticks

    ax2 = ax1.twinx()
    color_2 = 'tab:red'
    ax2.set_ylabel("Loss")
    ax2.plot(x_axis, train_loss, color=color_2, label='Train loss')
    ax2.tick_params(axis='y', labelcolor=color_2)
    ax2.set_ylim([0, 1])  # set upper and lower bound for y-axis ticks

    fig.suptitle('Training Curves', fontsize=16, fontweight='bold'), fig.legend(loc="upper right")
    fig.savefig(image_path_)  # save the full figure


def save_train_bce_curves(train_bce,
                          image_path_):
    """This function plots train binary cross entropy during learning
    Args:
        train_bce (list): train binary cross entropy
        image_path_ (str): path where to save the figure
        """
    x_axis = np.arange(1, len(train_bce) + 1, 1)  # since the two input vectors have same length, just use one of the two to extract epochs
    fig2, ax1 = plt.subplots()  # create figure
    color_1 = 'tab:green'
    ax1.plot(x_axis, train_bce, color=color_1, label='Train bce')
    ax1.set_xlabel('Epochs')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only keep integers in x axis
    fig2.suptitle('Training Curve', fontsize=16, fontweight='bold'), fig2.legend(loc="upper right")
    fig2.savefig(image_path_)  # save the full figure


def save_ext_train_val_curves(train_loss,
                              train_dice,
                              val_loss_,
                              val_dice_,
                              image_path_):
    """ This function takes as input train_dice, val_dice, train_loss, val_loss and plots them into a unique figure which is saved in the specified path
    Args:
        train_loss (list): training loss
        train_dice (list): training accuracy
        val_loss_ (list): validation loss
        val_dice_ (list): validation accuracy
        image_path_ (str): path where we want to save the image
    Returns:
        None
    """
    x_axis = np.arange(1, len(train_dice)+1, 1)  # since the two input vectors have same length, just use one of the two to extract epochs
    fig, ax1 = plt.subplots()  # create figure

    color_1 = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only keep integers in x axis
    ax1.set_ylabel('Dice')
    ax1.plot(x_axis, train_dice, color=color_1, label='Train dice')
    ax1.plot(x_axis, val_dice_, "--", color=color_1, label='Val dice')
    ax1.tick_params(axis='y', labelcolor=color_1)
    ax1.set_ylim([0, 1])  # set upper and lower bound for y-axis ticks

    ax2 = ax1.twinx()
    color_2 = 'tab:red'
    ax2.set_ylabel("Loss")
    ax2.plot(x_axis, train_loss, color=color_2, label='Train loss')
    ax2.plot(x_axis, val_loss_, "--", color=color_2, label='Val loss')
    ax2.tick_params(axis='y', labelcolor=color_2)
    ax2.set_ylim([0, 1])  # set upper and lower bound for y-axis ticks

    fig.suptitle('Train/Val Curves', fontsize=16, fontweight='bold'), fig.legend(loc="upper right")
    fig.savefig(image_path_)  # save the full figure


def save_train_val_bce_curves(train_bce,
                              val_bce,
                              image_path_):
    """This function plots train and validation binary cross entropy during learning
    Args:
        train_bce (list): train binary cross entropy
        val_bce (list): validation binary cross entropy
        image_path_ (str): path where to save the figure
        """
    x_axis = np.arange(1, len(train_bce) + 1, 1)  # since the two input vectors have same length, just use one of the two to extract epochs
    fig2, ax1 = plt.subplots()  # create figure
    color_1 = 'tab:green'
    ax1.plot(x_axis, train_bce, color=color_1, label='Train bce')
    ax1.plot(x_axis, val_bce, "--", color=color_1, label='Val bce')
    ax1.set_xlabel('Epochs')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only keep integers in x axis
    fig2.suptitle('Train/Val Curves', fontsize=16, fontweight='bold'), fig2.legend(loc="upper right")
    fig2.savefig(image_path_)  # save the full figure


def extract_unique_elements(lst: list,
                            ordered: bool = True) -> list:
    """This function extracts the unique elements of the input list (i.e. it removes duplicates)
    and returns them as an output list; if ordered=True (as by default), the returned list is ordered.
    Args:
        lst: input list from which we want to extract the unique elements
        ordered: whether the output list of unique values is sorted or not; True by default
    Returns:
        out_list: list containing unique values
    """
    out_list = list(set(lst))  # type: list

    if ordered:  # if we want to sort the list of unique values
        out_list.sort()  # type: list

    return out_list


def divide_sub_ses_into_train_and_val(percentage_validation_subs: float,
                                      all_sub_ses: list) -> Tuple[list, list]:
    """This function performs the train-validation split of the sub_ses but on a subject-wise level, cause some subs have multiple sessions and we want to avoid that
    one session goes to training and another session of the same subject goes to validation
    Args:
        percentage_validation_subs: percentage of sub_ses that will be used for validation
        all_sub_ses: list of all sub_ses
    """
    assert 0 < percentage_validation_subs <= 0.3, f"The percentage of validation subjects should be in the range (0., 0.3); found {percentage_validation_subs} instead"
    all_subs = [re.findall(r"sub-\d+", element)[0] for element in all_sub_ses]  # extract only subs
    all_subs_unique = extract_unique_elements(all_subs)  # extract unique elements (i.e. remove duplicates)
    nb_val_subs = int(round_half_up(percentage_validation_subs * len(all_subs_unique)))  # keep XX% of the subs for validation

    # extract validation and train subjects
    val_subs = random.sample(all_subs_unique, nb_val_subs)
    train_subs = [sub for sub in all_subs_unique if sub not in val_subs]

    # extract corresponding validation sub_ses
    idxs_val_subs = [idx for idx, value in enumerate(all_subs) if value in val_subs]
    val_sub_ses = [all_sub_ses[idx] for idx in idxs_val_subs]

    # extract corresponding training sub_ses
    idxs_train_subs = [idx for idx, value in enumerate(all_subs) if value in train_subs]
    train_sub_ses = [all_sub_ses[idx] for idx in idxs_train_subs]  # remove validation sub_ses from train_subs

    return train_sub_ses, val_sub_ses


def patch_wise_training(data_path: str,
                        all_sub_ses: list,
                        test_sub_ses: list,
                        lambda_loss: float,
                        epochs: int,
                        batch_size: int,
                        lr: float,
                        conv_filters: tuple,
                        cv_fold: int,
                        date: str,
                        percentage_validation_subs: float,
                        n_parallel_jobs: int,
                        training_outputs_folder: str,
                        path_previous_weights_for_pretraining: str,
                        use_validation_data: bool) -> None:
    """This function performs the patch-wise training
    Args:
        data_path: path to dataset of patches
        all_sub_ses: containing the training sub_ses
        test_sub_ses: containing the test sub_ses
        lambda_loss: value that weights the two terms of the hybrid loss
        epochs: number of training epochs
        batch_size: size of each batch
        lr: learning rate
        conv_filters: it contains the number of filters to use in the convolution layers
        cv_fold: cross validation fold
        date: today's date
        percentage_validation_subs: percentage of subjects to keep for validation
        n_parallel_jobs: number of jobs to run in parallel
        training_outputs_folder: path to output folder
        path_previous_weights_for_pretraining: path where previous weights are stored (used for pretraining)
        use_validation_data: if True, validation data is created to monitor the training curves
    Raises:
        AssertionError: if the path to the positive patches folder does not exist
        AssertionError: if the path to the positive masks folder does not exist
        AssertionError: if the path to the negative patches folder does not exist
        AssertionError: if the path to the positive and negative patches have different side
    """
    print("\nStarted CV fold {}".format(cv_fold))
    tf.random.set_seed(123)  # set fixed random seed for reproducibility (e.g. weight initialization will always be the same)
    random.seed(123)  # set fixed random seed for reproducibility
    start_global = time.time()  # start timer; used to compute the time needed to run this script
    # -----------------------------------------------------------------------------------------------------------------------
    pos_patches_path = os.path.join(data_path, "Positive_Patches")  # type: str
    assert os.path.exists(pos_patches_path), "Path {} does not exist".format(pos_patches_path)
    neg_patches_path = os.path.join(data_path, "Negative_Patches")  # type: str
    assert os.path.exists(neg_patches_path), "Path {} does not exist".format(neg_patches_path)

    # invoke external function to create output dirs
    parent_dir = str(Path(data_path).parent)  # extract parent dir of patches dataset
    model_path, plots_path, tensorboard_path, test_subs_path = define_output_folders(training_outputs_folder, parent_dir, cv_fold)

    # save sub_ses of test subjects as a list
    save_pickle_list_to_disk(test_sub_ses, test_subs_path, "test_sub_ses.pkl")

    # if we want a validation set to monitor training curves/metrics
    if use_validation_data:
        # ------------------------------ divide subjects into train and validation ---------------------------------
        train_sub_ses, val_sub_ses = divide_sub_ses_into_train_and_val(percentage_validation_subs,
                                                                       all_sub_ses)

        # ------------------------------- create TRAINING dataset --------------------------------------
        print("\nCreating training tf.dataset...")
        batched_train_dataset, train_patch_side = create_batched_augmented_tf_dataset(pos_patches_path,
                                                                                      train_sub_ses,
                                                                                      neg_patches_path,
                                                                                      batch_size,
                                                                                      n_parallel_jobs,
                                                                                      augment=True)

        # ------------------------------- create VALIDATION dataset -------------------------------------
        print("\nCreating validation tf.dataset...")
        batched_validation_dataset, val_patch_side = create_batched_augmented_tf_dataset(pos_patches_path,
                                                                                         val_sub_ses,
                                                                                         neg_patches_path,
                                                                                         batch_size,
                                                                                         n_parallel_jobs)

        assert train_patch_side == val_patch_side, "Train and validation patch side must be equal; train: {}, val: {}".format(train_patch_side, val_patch_side)

    # if instead we want to use all data for training (e.g. for last training before submission)
    else:
        # ------------------------------- create TRAINING dataset --------------------------------------
        print("\nCreating training tf.dataset...")
        batched_train_dataset, train_patch_side = create_batched_augmented_tf_dataset(pos_patches_path,
                                                                                      all_sub_ses,
                                                                                      neg_patches_path,
                                                                                      batch_size,
                                                                                      n_parallel_jobs,
                                                                                      augment=True)

    # ------------------------------- print "data loading" time -------------------------------------
    end_data_loading = time.time()  # stop timer
    print_running_time(start_global, end_data_loading, "Data loading")

    # ----------------------------------------- TRAIN UNet -------------------------------------------
    start_training = time.time()  # start timer

    inputs = tf.keras.Input(shape=(train_patch_side, train_patch_side, train_patch_side, 1), name='TOF_patch')
    unet = create_compiled_unet(inputs, lr, lambda_loss, conv_filters)

    if path_previous_weights_for_pretraining:  # if path to previous weights is not empty
        # LOAD weights saved from a trained model somewhere else (we do pretraining)
        print("\nLoading weights from a previous model...")
        unet.load_weights(os.path.join(path_previous_weights_for_pretraining, "my_checkpoint")).expect_partial()
    else:  # if instead path to previous weights is empty
        print("\nTraining from scratch...")

    # -------------------- ADD useful callback(s): they will be fed to the fit method ----------------
    print("\nDefining callback...")
    callbacks = []  # type: list # initialize empty list; it will store the callback(s)
    # reduce lr by 'factor' once dice stagnates. If no improvement is seen for a 'patience' number of epochs, lr is reduced.
    reduce_lr_on_plateau_cbk = tf.keras.callbacks.ReduceLROnPlateau(monitor='dice_coeff', factor=0.5, patience=50, min_lr=0.000001, verbose=1)
    callbacks.append(reduce_lr_on_plateau_cbk)
    # create tensorboard logs: compute activations and weight histograms every epoch for each layer of the model
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=1)
    callbacks.append(tensorboard_callback)
    # add checkpoint callback: it is used to save the weights of the model during training
    if use_validation_data:
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                                       save_weights_only=True,
                                                                       monitor='val_loss',
                                                                       mode='min',
                                                                       save_best_only=True)
    else:
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                                       save_weights_only=True,
                                                                       monitor='loss',
                                                                       mode='min',
                                                                       save_best_only=True)
    callbacks.append(model_checkpoint_callback)

    # ---------------------------------------- TRAIN model -------------------------------------
    print("\nBeginning training...")
    if use_validation_data:
        model_history = unet.fit(batched_train_dataset,
                                 epochs=epochs,
                                 validation_data=batched_validation_dataset,
                                 callbacks=callbacks)
    else:  # if we don't use any validation data
        model_history = unet.fit(batched_train_dataset,
                                 epochs=epochs,
                                 callbacks=callbacks)

    end_training = time.time()  # stop timer
    print_running_time(start_training, end_training, "Training")

    # ----------------------------------- PLOT training curves ----------------------------------------
    loss = model_history.history['loss']
    dice = model_history.history['dice_coeff']
    bce = model_history.history['binary_crossentropy']
    img_name = "train_curves_dice_and_loss_{}.png".format(date)
    save_ext_train_curves(loss, dice, os.path.join(plots_path, img_name))
    img_name_2 = "bce_curve_loss_{}.png".format(date)
    save_train_bce_curves(bce, os.path.join(plots_path, img_name_2))

    if use_validation_data:
        val_loss = model_history.history['val_loss']
        val_dice = model_history.history['val_dice_coeff']
        val_bce = model_history.history['val_binary_crossentropy']
        img_name_3 = "train_val_dice_and_loss_{}.png".format(date)
        save_ext_train_val_curves(loss, dice, val_loss, val_dice, os.path.join(plots_path, img_name_3))
        img_name_4 = "train_val_bce_curve_loss_{}.png".format(date)
        save_train_val_bce_curves(bce, val_bce, os.path.join(plots_path, img_name_4))

    # ---------------------------------- compute total running time ------------------------------------
    end_global = time.time()  # stop timer
    print_running_time(start_global, end_global, "Running")


def cross_validation(data_path: str,
                     lambda_loss: float,
                     epochs: int,
                     batch_size: int,
                     lr: float,
                     conv_filters: tuple,
                     percentage_validation_subs: float,
                     n_parallel_jobs: int,
                     date: str,
                     training_outputs_folder: str,
                     fold_to_do: int,
                     use_validation_data: bool,
                     path_previous_weights_for_pretraining: str,
                     train_test_split_to_replicate: str) -> None:
    """This function splits the data in train and test, ensuring that multiple sessions of the same subject are either all in train or all in test
    Args:
        data_path: path to dataset of patches
        lambda_loss: value that weights the two terms of the hybrid loss
        epochs: number of training epochs
        batch_size: batch size
        lr: learning rate
        conv_filters: it contains the number of filters to use in the convolution layers
        percentage_validation_subs: percentage of subjects to keep for validation
        date: today's date
        training_outputs_folder: path to output folder
        n_parallel_jobs: number of jobs to run in parallel
        fold_to_do: training fold that will be done
        use_validation_data: if True, validation data is created to monitor the training curves
        path_previous_weights_for_pretraining: path where previous weights are stored (used for pretraining); if empty, no pretraining is done
        train_test_split_to_replicate: path to directory containing the test subjects of each CV split
    """
    assert fold_to_do in range(1, 6), "Fold can only be 1, 2, 3, 4, 5; found {} instead".format(fold_to_do)
    print("\nTensorflow version used: {}".format(tf.version.VERSION))  # print tensorflow version
    # ------------------------------------------------------------------------
    all_sub_ses = find_sub_ses_pairs(data_path)  # extract all sub_ses pairs

    # retrieve train and test subs from previous experiment that we want to emulate
    test_subs_previous_split = load_file_from_disk(os.path.join(train_test_split_to_replicate, "fold{}".format(fold_to_do), "test_subs", "test_sub_ses.pkl"))
    train_subs_previous_split = [item for item in all_sub_ses if item not in test_subs_previous_split]  # exclude test subjects

    patch_wise_training(data_path,
                        train_subs_previous_split,
                        test_subs_previous_split,
                        lambda_loss,
                        epochs,
                        batch_size,
                        lr,
                        conv_filters,
                        fold_to_do,
                        date,
                        percentage_validation_subs,
                        n_parallel_jobs,
                        training_outputs_folder,
                        path_previous_weights_for_pretraining,
                        use_validation_data)


def main():
    # the code inside here is run only when THIS script is run, and not just imported
    config_dict = load_config_file()  # load input config file with argparser

    # extract input args from dictionary
    epochs = config_dict['epochs']  # type: int # number of training epochs
    lambda_loss = config_dict['lambda_loss']  # type: float # value that weights the two terms of the hybrid loss
    batch_size = config_dict['batch_size']  # type: int
    lr = config_dict['lr']  # type: float # learning rate
    conv_filters = tuple(config_dict['conv_filters'])  # type: tuple # number of filters in the convolutional layers
    fold_to_do = config_dict['fold_to_do']  # type: int # the five folds are 1, 2, 3, 4, 5
    use_validation_data = str2bool(config_dict['use_validation_data'])  # type: bool # whether to use validation data or not
    percentage_validation_subs = config_dict['percentage_validation_subs']   # type: float # percentage of samples to use for validation
    n_parallel_jobs = config_dict['n_parallel_jobs']  # type: int # nb. jobs to run in parallel (i.e. number of CPU (cores) to use); if set to -1, all available CPUs are used

    data_path = config_dict['data_path']  # type: str # path to dataset of patches
    input_ds_identifier = config_dict['input_ds_identifier']  # type: str # unique name given to rename output folders
    path_previous_weights_for_pretraining = config_dict['path_previous_weights_for_pretraining']  # type: str # path where weights of an already-trained model are stored; if empty, there is no pretraining
    train_test_split_to_replicate = config_dict['train_test_split_to_replicate']  # type: str # path to directory containing the test subjects of each CV split

    date = (datetime.today().strftime('%b_%d_%Y'))  # save today's date
    training_outputs_folder = f"Train_Outputs_{date}_{input_ds_identifier}"  # type: str # name of folder where all training outputs will be saved

    assert tf.test.is_built_with_cuda(), "TF was not built with CUDA"
    assert tf.config.experimental.list_physical_devices('GPU'), "A GPU is required to run this script"

    # begin cross validation
    cross_validation(data_path,
                     lambda_loss,
                     epochs,
                     batch_size,
                     lr,
                     conv_filters,
                     percentage_validation_subs,
                     n_parallel_jobs,
                     date,
                     training_outputs_folder,
                     fold_to_do,
                     use_validation_data,
                     path_previous_weights_for_pretraining,
                     train_test_split_to_replicate)


if __name__ == '__main__':
    main()

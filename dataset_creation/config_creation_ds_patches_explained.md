In this file, the input parameters of `config_creation_ds_patches.json` are explained:

- `bids_dataset_path (str)`: path to BIDS dataset. The dataset can be downloaded from this [OpenNEURO link](https://openneuro.org/datasets/ds003821/versions/1.0.0)
- `patch_side (int)`: side of the cubic training patches that we want to create (e.g. 64 will create 64x64x64-wide patches)
- `desired_spacing (list)`: pixel spacing in [mm] to which we want to resample our input patches
- `overlapping (float)`: amount of overlapping between patches in sliding-window approach
- `mni_landmark_points_path (str)`: path to the csv file containing the landmark points coordinates. This can be found [here](https://github.com/connectomicslab/Aneurysm_Detection/blob/main/extra_files/Landmarks_LPS_mm_Dec_05_2020.csv)
- `out_dataset_path (str)`: path to folder where we want to create the training dataset
- `id_out_dataset (str)`: unique identified for output folder where dataset will be created
- `subs_chuv_with_weak_labels_path (str)`: path to list containing patients with weak labels
- `subs_chuv_with_voxelwise_labels_path (str)`: path to list containing patients with voxelwise labels
- `jobs_in_parallel (int)`: number of CPUs to run subjects in parallel (the higher, the faster!); if set to `-1`, all available CPUs are used
- `vessel_like_neg_patches (int)`: number of vessel-like negative patches to extract for each subject; for more details, please check the dataset creation section of the paper
- `random_neg_patches (int)`: number of random negative patches to extract for each subject
- `landmark_patches (bool)`:  if True, patches in correspondence of landmark points are extracted
- `pos_patches (int)`: number of positive patches to extract for each patient (i.e. subject with aneurysm(s))
- `refine_weak_labels (bool)`: if set to True, the weak labels are refined with an intensity criterion
- `convert_voxelwise_labels_into_weak (bool)`: if set to True, it converts the voxel-wise labels into weak (i.e. it generates synthetic spheres around the aneurysm center); for more details, please check the creation of *weakened* labels in the paper
- `sub_ses_test (list)`: it contains the sub_ses of the subjects that will be used for testing. If you're running the cross-validation, then for every fold you should specify the test sub_ses. Instead, if
you are just performing one train-test split of the dataset, you can just specify the sub_ses of the test subjects
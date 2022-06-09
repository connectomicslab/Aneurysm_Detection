# "Towards automated brain aneurysm detection in TOF-MRA: open data, weak labels, and anatomical knowledge"
<p float="middle">
  <img src="https://github.com/connectomicslab/Aneurysm_Detection/blob/main/images/model_prediction_Nov_30_2021.png" width="335" />
  <img src="https://github.com/connectomicslab/Aneurysm_Detection/blob/main/images/anat_inf_sliding_window.png" width="340" /> 
</p>


This repository contains the code used for the [paper](https://arxiv.org/abs/2103.06168): "Towards automated brain aneurysm detection in TOF-MRA: open data, weak labels, and anatomical knowledge". 

Please cite the paper if you are using either our dataset or model.

## Installation
**Disclaimer**: the results of the paper were obtained with python 3.6 and a Linux (centOS) operating system. Reproducibility for different configurations is not guaranteed.

### Data
You can download the dataset used for this study from this [OpenNEURO link](https://openneuro.org/datasets/ds003949).

### Setup conda environment
1) Clone the repository
2) Create a conda environment using the `environment_clean.yml` file located inside the `install` directory. If you are not familiar with conda environments, please check out the [official documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Alternatively, feel free to use your favorite [IDE](https://en.wikipedia.org/wiki/Integrated_development_environment) such as [PyCharm](https://www.jetbrains.com/pycharm/download/#section=linux) or [Visual Studio](https://visualstudio.microsoft.com/downloads/) to set up the environment.
## Usage
**N.B.** if only interested in Inference, please skip to [Section 3 - Inference](#3-inference). Instead, to run the whole pipeline, begin from [Section 1 - Creation of training dataset](#1-creation-of-training-dataset)
### 1) Creation of training dataset
 The first step of the pipeline is the creation of the training dataset. This is composed of 3D TOF-MRA patches. The script used to create the dataset of patches is `create_dataset_patches_neg_and_pos.py` which is located inside the `dataset_creation` directory. Basically, the script loops over all subjects and for each of them creates training patches. For healthy controls, only negative patches (without aneurysms) are created. For patients, both negative and positive (with aneurysms) patches are created. To allow reproducibility with the results of the paper, the `dataset_creation` directory contains a config file named `config_creation_ds_patches.json`. A detailed explanation of each input argument can be found in [config_creation_ds_patches_explained.md](https://github.com/connectomicslab/Aneurysm_Detection/blob/main/dataset_creation/config_creation_ds_patches_explained.md). Before running the script, please read the .md file and make sure the paths in the config file are correct. This step does NOT need a GPU to be executed.
 
 **N.B.** also consider that running the script will generate the dataset of patches in the `out_dataset_path`. **Make sure you have at least 20 GB of free disk space** in `out_dataset_path`. When leaving default parameters, the dataset takes around 12 GB of disk space.
 
 Then, the script can be run with:
```python
create_dataset_patches_neg_and_pos.py --config config_creation_ds_patches.json
```
Since the dataset is created in parallel, consider increasing `jobs_in_parallel` in the config file to speed up the process (the higher, the better!). Feel free to modify the parameters inside the config file in case you would like to create a different dataset (e.g. different combinations of negative/positive patches).
### 2) Training
The second step of the pipeline is the training of the network. The script used to start training is `network_training.py` which is located inside the `training` directory. Similarly to step 1), the script should be run with the `config_training.json` file which is also located inside the `training` directory. A detailed explanation of each input argument can be found in [config_training_explained.md](https://github.com/connectomicslab/Aneurysm_Detection/blob/main/training/config_training_explained.md). Before running the script, please read the .md file and make sure the paths in the config file are correct.

**N.B.** In order to run `network_training.py`, you must have a GPU available. 

Also, consider that the user must manually specify which cross-validation (CV) fold to run. This can be modified in the `config_training.json` file. Ideally, if multiple GPUs are available, you should run the same script using one GPU per CV fold, modifying the config file accordingly. Before running the script, you should modify the paths in the config files according to the previous step.
Then, the script can be run with:
```python
network_training.py --config config_training.json
```
### 3) Inference
The third step of the pipeline is the patient-wise inference performed on the test subjects with the sliding-window approach. The script used to carry out inference is `patient_wise_sliding_window.py` which is located inside the `inference` directory, together with the config file `config_inference.json`.

If you ran the pipeline from the beginning (i.e. through steps 1) and 2)), then the `training_outputs_path` inside `config_inference.json` must correspond to the output folder that was created from [step 2)](#2-training). This output folder has been created in the same directory where the dataset of patches was created (`out_dataset_path` of [step 1)](#1-creation-of-training-dataset)) and should be named "Train_Outputs_%date_%input_ds_identifier".

If instead you only wish to run inference (with pretrained weights), then the `training_outputs_path` inside `config_inference.json`can simply be set to `Aneurysm_Detection/extra_files/Train_Outputs_Aug_16_2021_chuv_weak_plus_voxelwise_pretrain_adam_github/`

Before running the script, you should also modify the paths in the config files according to the previous steps.
Then, the script can be run with:
```python
patient_wise_sliding_window.py --config config_inference.json
```
### 4) Show/Plot results
In this last step of the pipeline, we want to print/plot results that we obtained from inference. All the scripts for showing test results are inside the directory `show_results`, together with the corresponding config files. For instance,
if we only want to print detection results, we can run:
```python
evaluation_detection_all_subs_with_CI.py --config config_evaluation_detection.json
```
If instead we want to compare two different models through a Wilcoxon signed-rank test as performed in the paper, we can run:
```python
wilcoxon_signed_rank_test_aufrocs_between_two_models.py --config config_wilcoxon_aufrocs.json
```
Finally, if we want to plot the FROC curves of several models, we can run:
```python
plot_multiple_FROC_curves_with_CI.py --config config_plot_FROC_curves.json
```
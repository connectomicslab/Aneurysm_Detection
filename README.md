# "Towards automated brain aneurysm detection in TOF-MRA: open data, weak labels, and anatomical knowledge"
<p float="middle">
  <img src="https://github.com/connectomicslab/Aneurysm_Detection/blob/main/images/model_prediction.png" width="250" />
  <img src="https://github.com/connectomicslab/Aneurysm_Detection/blob/main/images/anat_inf_sliding_window.png" width="340" /> 
</p>


This repository contains the code used for the [paper](https://arxiv.org/abs/2103.06168):
```
"Towards clinically applicable automated aneurysm detection in TOF-MRA: weak labels, anatomical knowledge, and open data"
```

## Installation
**Disclaimer**: the results of the paper were obtained with python 3.6 and a Linux (centOS) operating system. Reproducibility for different configurations is not guaranteed.

### Data
You can download the dataset used for this study from this [OpenNEURO link](https://openneuro.org/datasets/ds003821/versions/1.0.0).

### Setup conda environment
1) Clone the repository
2) Create a conda environment using the `environment.yml` file located inside the `install` directory. If you are not familiar with conda environments, please check out the [official documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Alternatively, feel free to use your favorite [IDE](https://en.wikipedia.org/wiki/Integrated_development_environment) such as [PyCharm](https://www.jetbrains.com/pycharm/download/#section=linux) or [Visual Studio](https://visualstudio.microsoft.com/downloads/) to set up the environment.
## Usage
### 1) Creation of training dataset
 The first step of the pipeline is the creation of the training dataset. This is composed of 3D TOF-MRA patches. The script used to create the dataset of patches is `create_dataset_patches_neg_and_pos.py` which is located inside the `dataset_creation` directory. Basically, the script loops over all subjects and for each of them creates training patches. For healthy controls, only negative patches (without aneurysms) are created. For patients, both negative and positive (with aneurysms) patches are created. To allow reproducibility with the results of the paper, the `dataset_creation` directory contains a config file named `config_creation_ds_patches.json`. 
 Before running the script, make sure the paths in the config file are correct. 
 
 **N.B.** also consider that running the script will generate the dataset of patches in the `out_dataset_path`. **Make sure you have at least 20 GB of free space** in `out_dataset_path`. When leaving default parameters, the dataset takes around 12 GB of disk space.
 
 Then, the script can be run with:
```python
random_scramble_sessions_bids_dataset.py --config config_creation_ds_patches.json
```
Since the dataset is created in parallel, consider increasing `jobs_in_parallel` in the config file to speed up the process (the higher, the better!). Feel free to modify the parameters inside the config file in case you would like to create a different dataset.
### 2) Training
The second step of the pipeline is the training of the network. The script used to start training is `network_training.py` and it is located inside the `training` directory. Similarly to step 1), the script should be run with the `config_training.json` file which is also located inside the `training` directory. In order to run this script, you must have a GPU available. Also, consider that the user must manually specify which cross-validation (CV) fold to run. This can be modified in the `config_training.json` file. Ideally, if multiple GPUs are available, you should run the same script using one GPU per CV fold, modifying the config file accordingly. Before running the script, you should modify the paths in the config files according to the previous step.
Then, the script can be run with:
```python
network_training.py --config config_training.json
```
### 3) Inference
The last step of the pipeline is the patient-wise inference performed on the test subjects with the sliding-window approach. The script used to carry out inference is `patient_wise_sliding_window.py` and it is located inside the `inference` directory, together with the config file `config_inference.json`. Before running the script, you should modify the paths in the config files according to the previous steps.
Then, the script can be run with:
```python
patient_wise_sliding_window.py --config config_inference.json
```

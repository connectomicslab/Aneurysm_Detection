import sys
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
sys.path.append('/home/to5743/aneurysm_project/Aneurysm_Detection/')  # this line is needed on the HPC cluster to recognize the dir as a python package
from inference.utils_inference import load_config_file
from show_results.utils_show_results import compute_areas_under_froc_curve


__author__ = "Tommaso Di Noto"
__version__ = "0.0.1"
__email__ = "tommydino@hotmail.it"
__status__ = "Prototype"


def compute_wilcoxon_signed_rank_test_aufrocs(prediction_dir_model1,
                                              prediction_dir_model2,
                                              ground_truth_dir,
                                              nb_parallel_jobs,
                                              max_nb_fp_per_subject,
                                              label_model1,
                                              label_model2):
    print("\nComputing aufrocs model1...")
    aufrocs_model1 = compute_areas_under_froc_curve(prediction_dir_model1, ground_truth_dir, nb_parallel_jobs, max_nb_fp_per_subject)
    df_aufrocs_model1 = pd.DataFrame(aufrocs_model1)  # type: pd.DataFrame # convert from list to dataframe
    distribution_model1 = np.ndarray.flatten(df_aufrocs_model1.values)  # type: np.ndarray # convert to numpy array

    print("\nComputing aufrocs model2...")
    aufrocs_model2 = compute_areas_under_froc_curve(prediction_dir_model2, ground_truth_dir, nb_parallel_jobs, max_nb_fp_per_subject)
    df_aufrocs_model2 = pd.DataFrame(aufrocs_model2)  # type: pd.DataFrame # convert from list to dataframe
    distribution_model2 = np.ndarray.flatten(df_aufrocs_model2.values)  # type: np.ndarray # convert to numpy array

    print("\n------ Performing Wilcoxon signed-rank test...")
    print("Comparing {} vs. {}".format(label_model1, label_model2))
    w1, p1 = wilcoxon(distribution_model1, distribution_model2, correction=True)
    print("W = {}; p-value = {}".format(w1, p1))


def main():
    # the code inside here is run only when THIS script is run, and not just imported
    config_dict = load_config_file()  # load input config file

    # extract paths needed to run this script
    prediction_dir_model1 = config_dict['prediction_dir_model1']  # path to dir containing segmentation predictions of model 1
    prediction_dir_model2 = config_dict['prediction_dir_model2']  # path to dir containing segmentation predictions of model 2
    ground_truth_dir = config_dict['ground_truth_dir']  # path to dir containing the ground truth masks
    nb_parallel_jobs = config_dict['nb_parallel_jobs']
    max_nb_fp_per_subject = config_dict['max_nb_fp_per_subject']
    label_model1 = config_dict['label_model1']
    label_model2 = config_dict['label_model2']

    compute_wilcoxon_signed_rank_test_aufrocs(prediction_dir_model1,
                                              prediction_dir_model2,
                                              ground_truth_dir,
                                              nb_parallel_jobs,
                                              max_nb_fp_per_subject,
                                              label_model1,
                                              label_model2)


if __name__ == '__main__':
    main()

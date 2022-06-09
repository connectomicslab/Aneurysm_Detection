import sys
sys.path.append('/home/to5743/aneurysm_project/Aneurysm_Detection/')  # this line is needed on the HPC cluster to recognize the dir as a python package
import numpy as np
import matplotlib.pyplot as plt
from inference.utils_inference import load_config_file
from show_results.utils_show_results import compute_sensitivities_froc_curve_with_conf_int


__author__ = "Tommaso Di Noto"
__version__ = "0.0.1"
__email__ = "tommydino@hotmail.it"
__status__ = "Prototype"


def plot_multiple_froc_curves(prediction_dir_model1,
                              prediction_dir_model2,
                              prediction_dir_model3,
                              max_nb_fp_per_subject,
                              nb_parallel_jobs,
                              ground_truth_dir,
                              label_model1,
                              label_model2,
                              label_model3):
    print("\nComputing sensitivities model1...")
    sensitivities_model1, lower_bound_model1, upper_bound_model1 = compute_sensitivities_froc_curve_with_conf_int(prediction_dir_model1, ground_truth_dir, nb_parallel_jobs, max_nb_fp_per_subject)

    print("\nComputing sensitivities model2...")
    sensitivities_model2, lower_bound_model2, upper_bound_model2 = compute_sensitivities_froc_curve_with_conf_int(prediction_dir_model2, ground_truth_dir, nb_parallel_jobs, max_nb_fp_per_subject)

    print("\nComputing sensitivities model3...")
    sensitivities_model3, lower_bound_model3, upper_bound_model3 = compute_sensitivities_froc_curve_with_conf_int(prediction_dir_model3, ground_truth_dir, nb_parallel_jobs, max_nb_fp_per_subject)

    fp_axis = np.arange(0, len(sensitivities_model1) + 1, dtype=int)  # create FP axis

    # add leading 0 because the starting point of the curves is (0,0)
    sensitivities_model1 = np.insert(sensitivities_model1, 0, 0)
    sensitivities_model2 = np.insert(sensitivities_model2, 0, 0)
    sensitivities_model3 = np.insert(sensitivities_model3, 0, 0)
    lower_bound_model1 = np.insert(lower_bound_model1, 0, 0)
    lower_bound_model2 = np.insert(lower_bound_model2, 0, 0)
    lower_bound_model3 = np.insert(lower_bound_model3, 0, 0)
    upper_bound_model1 = np.insert(upper_bound_model1, 0, 0)
    upper_bound_model2 = np.insert(upper_bound_model2, 0, 0)
    upper_bound_model3 = np.insert(upper_bound_model3, 0, 0)

    fig, ax = plt.subplots(1)
    ax.plot(fp_axis, sensitivities_model1, lw=2, color='red', label=label_model1)
    ax.fill_between(fp_axis, sensitivities_model1 + (upper_bound_model1 - sensitivities_model1), sensitivities_model1 - (sensitivities_model1 - lower_bound_model1), facecolor='red', alpha=0.5)
    ax.plot(fp_axis, sensitivities_model2, lw=2, color='blue', label=label_model2)
    ax.fill_between(fp_axis, sensitivities_model2 + (upper_bound_model2 - sensitivities_model2), sensitivities_model2 - (sensitivities_model2 - lower_bound_model2), facecolor='blue', alpha=0.5)
    ax.plot(fp_axis, sensitivities_model3, lw=2, color='green', label=label_model3)
    ax.fill_between(fp_axis, sensitivities_model3 + (upper_bound_model3 - sensitivities_model3), sensitivities_model3 - (sensitivities_model3 - lower_bound_model3), facecolor='green', alpha=0.5)
    ax.set_title("Mean FROC curves", weight="bold", fontsize=15)
    ax.set_xlabel('Nb. allowed FP per patient', fontsize=12)
    ax.set_ylabel('Sensitivity', fontsize=12)
    ax.set_xticks(fp_axis)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.grid()
    plt.show()


def main():
    # the code inside here is run only when THIS script is run, and not just imported
    config_dict = load_config_file()  # load input config file

    # extract paths needed to run this script
    prediction_dir_model1 = config_dict['prediction_dir_model1']  # path to dir containing segmentation predictions of model 1
    prediction_dir_model2 = config_dict['prediction_dir_model2']  # path to dir containing segmentation predictions of model 2
    prediction_dir_model3 = config_dict['prediction_dir_model3']  # path to dir containing segmentation predictions of model 3
    max_nb_fp_per_subject = config_dict['max_nb_fp_per_subject']
    nb_parallel_jobs = config_dict['nb_parallel_jobs']
    ground_truth_dir = config_dict['ground_truth_dir']  # path to dir containing the ground truth masks
    label_model1 = config_dict['label_model1']
    label_model2 = config_dict['label_model2']
    label_model3 = config_dict['label_model3']

    plot_multiple_froc_curves(prediction_dir_model1,
                              prediction_dir_model2,
                              prediction_dir_model3,
                              max_nb_fp_per_subject,
                              nb_parallel_jobs,
                              ground_truth_dir,
                              label_model1,
                              label_model2,
                              label_model3)


if __name__ == '__main__':
    main()

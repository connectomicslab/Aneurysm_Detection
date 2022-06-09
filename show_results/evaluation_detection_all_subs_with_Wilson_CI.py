import os
import sys
sys.path.append('/home/to5743/aneurysm_project/Aneurysm_Detection/')  # this line is needed on the HPC cluster to recognize the dir as a python package
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from show_results.utils_show_results import get_result_filename, detection_one_sub_for_conf_int, extract_unique_elements
from inference.utils_inference import load_config_file


__author__ = "Tommaso Di Noto"
__version__ = "0.0.1"
__email__ = "tommydino@hotmail.it"
__status__ = "Prototype"


def detection_all_sub_with_conf_int(prediction_dir, ground_truth_dir):
    only_dirs = [dirs for dirs in os.listdir(prediction_dir) if os.path.isdir(os.path.join(prediction_dir, dirs))]
    assert len(only_dirs) == 5, "There should be 5 test folds inside {}".format(prediction_dir)
    subj_ses_count = 0  # initialize counter
    sens_list = []
    tp_list = []
    cnt_list = []
    fp_list = []
    all_subs = []
    for fold in only_dirs:  # loop over test folds
        for sub in sorted(os.listdir(os.path.join(prediction_dir, fold))):  # loop over subjects
            all_subs.append(sub)
            for ses in sorted(os.listdir(os.path.join(prediction_dir, fold, sub))):  # loop over sessions
                subj_ses_count += 1  # increment counter
                print("\n{}) Subject {}_{}".format(subj_ses_count, sub, ses))

                participant_dir = os.path.join(prediction_dir, fold, sub, ses)
                test_dir = os.path.join(ground_truth_dir, sub, ses)

                result_filename = get_result_filename(participant_dir)

                froc_params = detection_one_sub_for_conf_int(result_filename, test_dir)

                fp_list.append(froc_params[0])  # append false positives

                # if there are no NaNs
                if not np.isnan(froc_params).any():
                    sens_list.append(froc_params[1])  # discard first element (not usable for the curve), extract sensitivity and append to external list
                    tp_list.append(froc_params[2])  # extract true positives and append to external list
                    cnt_list.append(froc_params[3])  # extract nb. aneurysms and append to external list

    all_subs_unique = extract_unique_elements(all_subs)  # only retain unique elements
    assert len(all_subs_unique) == 239, "There should be exactly 239 test subjects; found {} instead".format(len(all_subs_unique))

    sens_np = np.asarray(sens_list) * 100  # convert to %
    tps_df = pd.DataFrame(tp_list)
    cnts_df = pd.DataFrame(cnt_list)

    lower_bound = []
    upper_bound = []
    # loop over dataframe columns. Every column represents the nb. of allowed FP (e.g. first column 1 FP allowed, second column 2 FP allowed, ..)
    for column in tps_df:
        conf_int = proportion_confint(count=sum(tps_df[column]),
                                      nobs=sum(cnts_df[column]),
                                      alpha=0.05,
                                      method='wilson')  # compute 95% Wilson CI
        lower_bound.append(conf_int[0] * 100)
        upper_bound.append(conf_int[1] * 100)

    print("\n\n------------------------------------------------")
    print("\nMean sensitivity (95% Wilson CI): {}% ({}%, {}%)".format(np.mean(sens_np), lower_bound[0], upper_bound[0]))
    print("\nFP count = {}; average = {:.2f}".format(int(np.sum(fp_list)), np.mean(fp_list)))


def main():
    # the code inside here is run only when THIS script is run, and not just imported
    config_dict = load_config_file()  # load input config file

    # extract paths needed to run this script
    prediction_dir = config_dict['prediction_dir']  # path to dir containing segmentation predictions
    ground_truth_dir = config_dict['ground_truth_dir']  # path to dir containing the ground truth masks

    detection_all_sub_with_conf_int(prediction_dir, ground_truth_dir)


if __name__ == "__main__":
    main()

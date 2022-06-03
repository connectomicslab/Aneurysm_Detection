import os
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from show_results.utils_show_results import get_result_filename, detection_one_sub_for_conf_int


def detection_all_sub_with_conf_int(prediction_dir, ground_truth_dir):
    only_dirs = [dirs for dirs in os.listdir(prediction_dir) if os.path.isdir(os.path.join(prediction_dir, dirs))]
    assert len(only_dirs) == 5, "There should be 5 test folds inside {}".format(prediction_dir)
    subj_ses_count = 0  # initialize counter
    sens_list = []
    tp_list = []
    cnt_list = []
    fp_list = []
    for fold in only_dirs:  # loop over test folds
        for sub in sorted(os.listdir(os.path.join(prediction_dir, fold))):  # loop over subjects
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
    prediction_dir = "/media/newuser/HagmannHDD/Aneurysm_Project_Tommaso/Dataset_Patches/minor_review_neuroinformatics_May_2022/Inference/chuv_no_pretrain_on_adam_anatinf_slid_wind_May_29_2022/"
    ground_truth_dir = "/home/newuser/Desktop/MICCAI_Aneurysms/Ground_Truth_Dirs/Ground_Truth_Refined_Patient_Wise_CHUV_Aug_04_2021"
    detection_all_sub_with_conf_int(prediction_dir, ground_truth_dir)


if __name__ == "__main__":
    main()

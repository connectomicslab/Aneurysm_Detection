In this file, the input parameters of `config_training.json` are explained:

- `epochs (int)`: number of training epochs
- `lambda_loss (float)`: value that weights the two terms (dice and BCE) of the hybrid loss; must be between 0 and 1
- `batch_size (int)`: batch size to use during training
- `lr (float)`: learning rate to use during training
- `conv_filters (list)`: number of filters to use in the convolution layers of the UNET
- `fold_to_do (int)`: training fold that will be done; has to be one of [1,2,3,4,5]; we do one fold at a time so more folds can be done in parallel if more than one GPU is available
- `use_validation_data (str)`: if "True", validation data is created to monitor the training curves
- `percentage_validation_subs (float)`: percentage of subjects to keep for validation; must be between 0. and 0.3
- `n_parallel_jobs (int)`: number of CPUs to create the training dataset in parallel (the higher, the faster!); if set to `-1`, all available CPUs are used
 
 
- `data_path (str)`: path to directory containing dataset of patches (created during step 1 of the pipeline)
- `input_ds_identifier (str)`: unique name given to rename output folders
- `path_previous_weights_for_pretraining (str)`:  path where previous weights are stored (used for pretraining); if empty, no pretraining is done
- `train_test_split_to_replicate (str)`: path to directory containing the test subjects of each CV split; this can be found in the github directory [/Forced_CV_for_reproducibility](https://github.com/connectomicslab/Aneurysm_Detection/tree/70251d9c0d9d30385ec777c575e839a9dc12b2f7/extra_files/Forced_CV_for_reproducibility). There is no need to specify the fold since this will be added automatically using `fold_to_do`. This argument is needed in order to re-create the exact same cross-validation split used for the original paper. In this way, results are truly comparable.  
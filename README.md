# Updates

Full Data: [Google Drive](https://drive.google.com/drive/folders/1o2UgoPvMw6hunoewChw0ErumxXWJ4aTX?usp=sharing)

____________________________________________________________________________________
# Data

The data are available in the 'dataset' folder. Due to file size limitations, only the drug indexes for each fold and a small sample of data are provided to ensure reproducibility. The full dataset and model weights will be released on GitHub at a later time.
"model_name_or_path": "output/mult_task_training_base_drugbank_transductive_5e-4_fold2/runtime/checkpoint-24540",

This GitHub repository excludes large raw datasets, generated caches, local Conda environments, and model checkpoints. Use `data/dti_template/sample_dti.json` for the lightweight runtime smoke test, and download full data/model artifacts from the Google Drive link above when needed.

# Code

For all the training and inference scripts, first modify the configs file in the 'config' folder, then run 'python xxx.py' from the root folder. 


Prediction model:

'finetune_molt5_prediction.py' is used for training and inference.

ExDDI-S2S:

'finetune_molt5_joint_training.py' is used for training and inference.

ExDDI-MT:

'finetune_molt5_mult_task.py' is used for training and inference.

ExDDI-MTS:

'finetune_molt5_mult_task_inference.py' is used for inference. (Models are trained by ExDDI-MT).

 

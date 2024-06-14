# Multimodality with supernovae 

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
![Unittest](https://github.com/ThomasHelfer/multimodal-supernovae/actions/workflows/actions.yml/badge.svg)

<p align="center">
    <img src="https://github.com/ThomasHelfer/multimodal-supernovae/blob/main/imgs/logo_cropped.png" alt="no alignment" width="34%" height="auto"/>
</p>

## Overview
This codebase is dedicated to exploring different self-supervised pretraining methods. We integrate multimodal data from supernovae light curves with images of their host galaxies. Our goal is to leverage diverse data types to improve the prediction and understanding of astronomical phenomena. 

An overview over the the CLIP method and loss [link](https://lilianweng.github.io/posts/2021-05-31-contrastive/) 


## Installation

### Prerequisites
Before installing, ensure you have the following prerequisites:
- Python 3.8 or higher
- pip package manager

### Steps
1. #### Clone the Repository
   Clone the repository to your local machine and navigate into the directory:
   ```bash
   git clone git@github.com:ThomasHelfer/Multimodal-hackathon-2024.git
   cd Multimodal-hackathon-2024.git
   ```

2. #### Get data
   Unpack the dataset containing supernovae spectra, light curves and host galaxy images:
   ```bash
   unzip data/ZTFBTS.zip
   unzip data/ZTFBTS_spectra.zip   
   ```
   for larger simulation-based data download using wget 
   ```bash
   mkdir sim_data
   cd sim_data
   wget https://zenodo.org/records/6601211/files/scotch_z3.hdf5
   ```
  
4. #### Install Required Python Packages
   Install all dependencies listed in the requirements.txt file:
   ```bash
   pip install -r requirements.txt 
   ```

5. #### Run the Script
   Execute the main script to start your analysis:
   ```bash
   python script.py --config_path configs/config.yaml
   ```
6. #### Restart script
   To start off from a checkpoint use 
   ```bash
   python script.py --ckpt_path analysis/batch_sweep/worldly-sweep-4/model.ckpt
   ```
   where there should be a config.yaml corresponing to this run in the same folder
   
### Setting Up a Hyperparameter Scan with Weights & Biases

1. #### Create a Weights & Biases Account
   Sign up for an account at [Weights & Biases]((https://wandb.ai)) if you haven't already.
2. #### Configure Your Project
   Edit the configuration file to specify your project name. Ensure the name matches the project you create on [wand.ai](https://wandb.a). You can define sweep parameters within the [config file](https://github.com/ThomasHelfer/Multimodal-hackathon-2024/blob/main/configs/config_grid.yaml) .
3. #### Choose important parameters
   In the config file you can choose
   ```yaml
   extra_args
     regression: True
   ```
   if true, script_wandb.py performs a regression for redshift.
   Similarly for
   ```yaml
   extra_args
     classification: True
   ```
   if true, script_wandb.py performs a classification.
   if neither are true, it will perform a normal clip pretraining.
   Lastly, for
   ```yaml
   extra_args
     pretrain_lc_path: 'path_to_checkpoint/checkpoint.ckpt'
     freeze_backbone_lc: True
   ```
   preloads a pretrained model in script_wandb.py or allows to restart a run from a checkpoint for retraining_wandb.py
5. #### Run the Sweep Script
   Start the hyperparameter sweep with the following command:
   ```bash
   python script_wandb.py configs/config_grid.yaml 
   ```
   Resume a sweep with the following command:
   ```bash
   python script_wandb.py [sweep_id]
   ```
   or for pretraining the lightcurve encoder please use:
   ```bash
   python pretraining_wandb.py configs/config_grid.yaml
   ```
6. #### API Key Configuration
   The first execution will prompt you for your Weights & Biases API key, which can be found [here]([https://wandb.ai](https://wandb.ai/authorize)https://wandb.ai/authorize). 
 Alternatively, you can set your API key as an environment variable, especially if running on a compute node:
      ```bash
   export WANDB_API_KEY=...
   ```
7. #### View Results
   Monitor and analyze your experiment results on your Weights & Biases project page. [wand.ai](https://wandb.ai)

### Running a k-fold cross-validation
   We can run a k-fold cross validation by defining the variable 
   ```yaml
    extra_args:
      kfolds: 5 # for strat Crossvaildation
   ```
   as this can take serially very long, one can choose to split your runs for different submission by just choosing certain folds for each submission    
   ```yaml
      foldnumber:
        values: [1,2,3]
   ```


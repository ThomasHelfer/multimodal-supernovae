# Multimodality with supernovae 

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)


## Overview
This codebase is dedicated to exploring multimodal learning approaches by integrating data from supernovae light curves with images of their host galaxies. Our goal is to leverage diverse data types to improve the prediction and understanding of astronomical phenomena.

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

2. #### Unzip the Data
   Unpack the dataset containing supernovae light curves and host galaxy images:
   ```bash
   unzip data/ZTFBTS.zip  
   ```

3. #### Install Required Python Packages
   Install all dependencies listed in the requirements.txt file:
   ```bash
   pip install -r requirements.txt 
   ```

4. #### Run the Script
   Execute the main script to start your analysis:
   ```bash
   python script.py
   ```
   
### Setting Up a Hyperparameter Scan with Weights & Biases

1. #### Create a Weights & Biases Account
   Sign up for an account at [Weights & Biases]((https://wandb.ai)) if you haven't already.
2. #### Configure Your Project
   Edit the configuration file to specify your project name. Ensure the name matches the project you create on [wand.ai](https://wandb.a). You can define sweep parameters within the [config file](https://github.com/ThomasHelfer/Multimodal-hackathon-2024/blob/main/sweep_configs/config_grid.yaml) .
3. #### Run the Sweep Script
   Start the hyperparameter sweep with the following command:
   ```bash
   python script_wandb.py sweep_configs/config_grid.yaml 
   ```
4. #### API Key Configuration
   The first execution will prompt you for your Weights & Biases API key, which can be found [here]([https://wandb.ai](https://wandb.ai/authorize)https://wandb.ai/authorize). 
 Alternatively, you can set your API key as an environment variable, especially if running on a compute node:
      ```bash
   export WANDB_API_KEY=...
   ```
5. #### View Results
   Monitor and analyze your experiment results on your Weights & Biases project page. [wand.ai](https://wandb.ai)

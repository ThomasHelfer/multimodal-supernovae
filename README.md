# Multimodal study 

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)


## Overview
This is a small codebase to explore multimodal learning

## Installation

### Prerequisites
Before installing, ensure you have the following prerequisites:
- Python 3.8 or higher
- pip package manager

### Steps
1. Clone the repository:

   ```bash
   git clone git@github.com:ThomasHelfer/Multimodal-hackathon-2024.git
   cd Multimodal-hackathon-2024.git
   ```

2. Unzip the data:

   ```bash
   unzip data/ZTFBTS.zip  
   ```

3. Install requirements
   ```bash
   pip install -r requirements.txt 
   ```

4. Run Script

   ```bash
   python script.py
   ```
### Set up hyperparameter scan 

1. Set up account with [wand.ai](https://wandb.ai)
2. Define name of project within [config file](https://github.com/ThomasHelfer/Multimodal-hackathon-2024/blob/bc8b55767276fd6c08af08fd668e5dbf1ad646de/sweep_configs/config_grid.yaml#L5) and create project with same name in [wand.ai](https://wandb.a) and choose parameter for sweep.
3. Run
   ```bash
   python script_wandb.py sweep_configs/config_grid.yaml 
   ```
4. Running for the first time will require you to give API key available [here]([https://wandb.ai](https://wandb.ai/authorize)https://wandb.ai/authorize) or define your API key in your enviorment beforehand (best option if submitted to compute node)
      ```bash
   export WANDB_API_KEY=...
   ```
5. Get results on [wand.ai](https://wandb.ai)

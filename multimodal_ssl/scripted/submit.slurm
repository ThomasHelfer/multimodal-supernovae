#!/bin/bash

#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH -p GPU-shared
#SBATCH -t 0:20:00
#SBATCH --gres=gpu:1 # Request 1 GPU. --gpus=v100-32:1  is also fine.

set -x  # Echo commands to stdout

#cd /jet/home/thelfer/Multimodal-hackathon-2024/multimodal_ssl/notebooks

echo $SLURM_JOB_ID
source module_load.sh
module list
# Run code
python script.py 
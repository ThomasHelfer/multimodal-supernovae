#!/bin/bash
#SBATCH -Job-name=Multimodal
#SBATCH -t 07:00:00
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH -N 1
##SBATCH -n 14
#SBATCH -A tedwar41_gpu
#SBATCH -o Multimodal-%J.log
source dev/bin/activate
python script.py

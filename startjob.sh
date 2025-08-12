#!/bin/bash

#SBATCH --mem 250gb
#SBATCH --cpus-per-task 40
#SBATCH --time 24:00:00
#SBATCH --gpus a100:2
#SBATCH --partition viprgs


cd /scratch/asalvi/Diffusion/velDiff
module add anaconda3
source activate Husky_CS_SB3
python3 train_v5.py > train_diff.log 2>&1

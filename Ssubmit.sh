#!/bin/bash
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p queue 
#SBATCH -c 32
#SBATCH --time 240
#SBATCH --job-name NAMD3_deepdriveMD 
#SBATCH -q normal
#SBATCH --export=ALL

pwd; hostname; date
eval "$(conda shell.bash hook)"
source ~/.bashrc
conda activate eddmd

nvidia-cuda-mps-control -d

bash main.sh
wait

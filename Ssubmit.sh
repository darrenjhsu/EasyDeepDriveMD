#!/bin/bash
#BSUB -nnodes 1
#BSUB -q debug
#BSUB -W 2:00
#BSUB -J DDMD_NAMD3 
#BSUB -P CHM188


# Point to your EasyDeepDriveMD conda environment
export EDDMD_env=/path/to/your/env
# For example mine is
export EDDMD_env=/gpfs/alpine/chm188/scratch/djh992/CDDMD/CDDMD_env

module load cuda/11.0.3 gcc/9.1.0 fftw
conda activate $EDDMD_env
export OMP_NUM_THREADS=1
export PYTHON_UNBUFFERED=1
export CUDA_HOME=/sw/summit/cuda/11.0.3
export NAMDHOME_A13=/gpfs/alpine/chm188/proj-shared/software/NAMD/Linux-POWER-g++.summit30a13 # Use yours


sh main.sh > test0.log
wait

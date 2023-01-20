#!/bin/bash
#BSUB -nnodes 1
#BSUB -q debug
#BSUB -W 2:00
#BSUB -J DDMD_NAMD3 
#BSUB -P CHM188

source /gpfs/alpine/chm188/scratch/djh992/MD_setup.sh
source ~/.cddmdrc

sh main.sh > test0.log
wait

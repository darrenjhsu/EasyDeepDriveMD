#!/bin/bash
#BSUB -nnodes 1
#BSUB -q batch-hm
#BSUB -W 6:00
#BSUB -J DDMD_NAMD3 
#BSUB -P STF006

source /gpfs/alpine/chm188/scratch/djh992/MD_setup.sh
source ~/.cddmdrc

sh main.sh > test6.log
wait

#!/bin/bash

# The main workflow for adaptive sampling through CVAE

# Source config
n_rounds=150
n_sims=6
psf=../Structures/1fme_wb_ion.psf
template=../Template
init_coord=BBA_init
sim_config0=BBA0.conf
sim_config=BBA.conf
# This has to match the output of your simulations
sample_dcd=BBA_sample
target=BBA_target.pdb
fftw=FFTW_NAMD_3.0alpha13_Linux-x86_64-multicore-CUDA.txt
#latent=10

mkdir -p ../Simulations/data

# TODO: determine the progress

restart_round=0
restart_stage=0
if test -f "../Simulations/progress"; then 
  source ../Simulations/progress
fi
echo round $restart_round stage $restart_stage


for round in `seq $restart_round $n_rounds`
do
  echo -e "\n`date` We are in round ${round}!\n"

  if [ $round -ge $restart_round ] && [ $restart_stage -le 0 ] ; then
    # Stage 0
    mkdir -p ../Simulations/$round
    # Copy simulation config
    echo "`date` Copy sim config for round $round"
    for idx in `seq 0 $((n_sims -1))`
    do
      mkdir -p ../Simulations/$round/$idx
      if [ $round -eq 0 ]; then
        cp $template/$sim_config0 ../Simulations/$round/$idx/$sim_config
      else
        cp $template/$sim_config ../Simulations/$round/$idx/
      fi
      cp $template/$fftw ../Simulations/$round/$idx/
    done
     
    # Copy initial coordinates for round 0 and 1
    if [ $round -eq 0 ]; then
      echo "`date` Copy initial coord for round 0"
      for idx in `seq 0 $((n_sims -1))`
      do
        cp $template/${init_coord}.pdb ../Simulations/$round/$idx/
      done
    fi

    restart_stage=1
    echo restart_round=$round > ../Simulations/progress 
    echo restart_stage=1 >> ../Simulations/progress
  else
    echo Skipping initial prep for round $round 
  fi

  if [ $round -ge $restart_round ] && [ $restart_stage -le 1 ]; then
    # Stage 1: Run simulation and post processing (sim.sh)
    echo "`date` Run simulations "
    for idx in `seq 0 $((n_sims - 1))`
    do
      sh sim.sh $psf $sample_dcd $round $idx $template/$target &
    done
    # In addition, use the remaining CPUs to calculate and store
    # pairwise distances as well as suggest new inits
    wait
    
    restart_stage=2
    echo restart_round=$round > ../Simulations/progress 
    echo restart_stage=2 >> ../Simulations/progress
  else
    echo Skipping simulation for round $round
  fi

  if [ $round -ge $restart_round ] && [ $restart_stage -le 2 ]; then
    # Stage 2: post process the simulation 
    echo "`date` Run simulations and post processing"
    for idx in `seq 0 $((n_sims - 1))`
    do
      python postprocess.py $psf $sample_dcd $round $idx $template/$target &
    done
    wait
    
    restart_stage=3
  else
    echo Skipping post-processing for round $round
  fi

  if [ $round -ge $restart_round ] && [ $restart_stage -le 3 ]; then
    echo "`date` Run pairwise using data from round 0 to $round"
    python pairwise.py $round $n_sims $psf $sample_dcd $template/$init_coord 
    restart_stage=0
    echo restart_round=$((round+1)) > ../Simulations/progress
    echo restart_stage=0 >> ../Simulations/progress
  fi

  #  # Run training (CVAE), also suggests new initial coordinates
  #  echo "`date` Train CVAE and suggest new inits"
  #  jsrun -n1 -g1 -c42 python train.py $round $n_sims $psf $sample_dcd $init_coord $latent $target
  #

  # Tidy up simulation data 

done
wait

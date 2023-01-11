#!/bin/bash

# The main workflow for adaptive sampling through CVAE

# Source config
n_rounds=300
n_sims=6
psf=../Structures/8b7i_wb_Cl.psf
init_coord=../Template/HSP90_init.pdb
sim_config=../Template/HSP90.conf
config=HSP90.conf
# This has to match the output of your simulations
sample_dcd=HSP90_sample.dcd
target=../Template/HSP90_target.pdb
#latent=10

mkdir -p ../Simulations
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
      cp $sim_config ../Simulations/$round/$idx/
    done
     
    # Copy initial coordinates for round 0 and 1
    if [ $round -eq 0 ] || [ $round -eq 1 ]; then
      echo "`date` Copy initial coord for round 0"
      for idx in `seq 0 $((n_sims -1))`
      do
        cp $init_coord ../Simulations/$round/$idx/
      done
    fi

    restart_stage=1
    echo -e restart_round=$round > ../Simulations/progress 
    echo -e restart_stage=1 >> ../Simulations/progress
  else
    echo Skipping initial prep for round $round 
  fi

  if [ $round -ge $restart_round ] && [ $restart_stage -le 1 ]; then
    # Stage 1: Run simulation and post processing (sim.sh)
    echo "`date` Run simulations "
    if [ $round -ge 1 ]; then
      echo "`date` also run post-processing using data from round 0 to $((round-1))"
      jsrun -n1 -c36 -brs python pairwise.py $round $n_sims $psf $sample_dcd $init_coord &
    else
      echo "`date` First round, skip pairwise distance"
    fi
    for idx in `seq 0 $((n_sims - 1))`
    do
      sh sim.sh $psf $sample_dcd $round $idx $config &
    done
    # In addition, use the remaining CPUs to calculate and store
    # pairwise distances as well as suggest new inits
    wait
    
    restart_stage=2
    echo -e restart_round=$round > ../Simulations/progress 
    echo -e restart_stage=2 >> ../Simulations/progress
  else
    echo Skipping simulation for round $round
  fi

  if [ $round -ge $restart_round ] && [ $restart_stage -le 2 ]; then
    # Stage 2: post process the simulation 
    echo "`date` Run simulations and post processing"
    for idx in `seq 0 $((n_sims - 1))`
    do
      jsrun -n1 -c7 python postprocess.py $psf $sample_dcd $round $idx $target &
    done
    wait
    
    restart_stage=0
    echo -e restart_round=$((round+1)) > ../Simulations/progress 
    echo -e restart_stage=0 >> ../Simulations/progress
  else
    echo Skipping post-processing for round $round
  fi

  #if [ $round -ge $restart_round ] && [ $restart_stage -le 3 ]; then
  #  # Run training (CVAE), also suggests new initial coordinates
  #  echo "`date` Train CVAE and suggest new inits"
  #  jsrun -n1 -g1 -c42 python train.py $round $n_sims $psf $sample_dcd $init_coord $latent $target
  #
  #  restart_stage=0
  #  echo -e restart_round=$((round+1)) > ../Simulations/progress
  #  echo -e restart_stage=0 >> ../Simulations/progress
  #fi

  # Tidy up simulation data 

done
wait

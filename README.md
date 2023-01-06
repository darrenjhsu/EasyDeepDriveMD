# FAST Adaptive MD

Poor person's DeepDriveMD (DDMD) implementation (in fact it is a FAST implementation now). 
FAST stands for "fluctuation amplification of specific traits".

## Caveat

This implementation is very primitive and only works on adaptive/guided MD to 
approach a target structure or explore conformational space.


## DDMD vs. FAST

DeepDriveMD uses a convolutional variational autoencoder (CVAE) trained on the 
contact map (CA atoms, 8 A cutoff) and DBSCAN on the latent space to predict 
conformational outliers. The [implementation](!https://github.com/DeepDriveMD/DeepDriveMD-pipeline)
is quite sophisticated.

FAST is a multi-armed bandit approach to explore/exploit conformational (reward space).
The reward space can be defined as e.g. low/high RMSD to a target, energy, SASA, etc.
In the [implementation](!https://pubs.acs.org/doi/10.1021/acs.jctc.5b00737), they used
an ambiguous k-centers method to cluster their trajectories, based on pairwise RMSD.
In this implementation we use DBSCAN to determine the clusters.

## Reason

Because I was waiting for Slate access from OLCF to run actual DeepDriveMD,
which requires MongoDB and RabbitMQ.
So I decided to implement a slimmer DeepDriveMD and repurpose it for FAST adaptive MD,
which are very similar in spirit.

## Requirement

1. scikit-learn (for DBSCAN)
1. MDAnalysis (for calculating RMSD to target)
1. Your favorite MD package (I use NAMD)

If you are on Summit like I do, you can use the open-ce as your base environment (which is an overkill for FAST):

```bash
module load cuda gcc fftw
# with your conda manager
conda create -p /your/conda/env/path --clone /sw/summit/open-ce/anaconda-base/envs/open-ce-1.5.2-py39-0
conda activate -p /your/conda/env/path
pip install MDAnalysis
```

## Structure

I suggest something like this:

```
Experiment/
  |- Force_fields/
  |    |- all your force field files
  |- Structures/
  |    |- protein.psf
  |    |- protein.pdb
  |- Templates/
  |    |- protein_init.pdb
  |    |- sampling.conf
  |- EasyDeepDriveMD/ (this repo)
  |
```

Over the course of DDMD, the `Simulations` folder will be generated with simulation result and associated data in it.

```
Experiment/
  |- ...
  |- Simulations/
       |- progress
       |- data/
       |- saved_models/
       |- 0/ (rounds of simulations)
          |- 0/ (individual simulations)
          |- 1/
          |- ...
       |- 1/
       |- ...
```

## How it works

The `main.sh` has three stages for each round: prepare / sample / pairwise RMSD & DBSCAN.

In the prepare stage, the script (EDDMD) copies `sim_config` to each simulation folder for that round.
If it is the first round, EDDMD copies `init_coord` to each simulation folder as well.

In the simulation stage, EDDMD calls `sim.sh` for every individual simulations with different arguments.
The `sim.sh` does both simulation and postprocessing by calling the MD engine and then `postprocess.py`.
The `postprocess.py` for now calculates and stores atom positions of interest (e.g. CA or backbone + DB) 
and calculate RMSD of each snapshot to the target (if needed).
`postprocess.py` saves results in `Simulations/data/{round}_{sim_idx}.npy`.

In the pairwise RMSD & DBSCAN stage, EDDMD calls `pairwise.py` to load all `.npy` data, 
calculate pairwise RMSD of all snapshots (this is a compute-heavy step),
and use DBSCAN to cluster based on that. 
It then outputs specific frames from the pool of snapshots as new initial coordinates for the next round.

This workflow is basically identical to the file-based DeepDriveMD (DeepDriveMD-F) implementation.


## How to use (on Summit)

You basically have to edit every `.sh` and `.py` files in this folder, tailored to your simulation goals,
which is why I recommend having individual copies of this repo for every `Experiment/`.

1. Place all necessary files
1. Edit `main.sh` so that the initial variables point to correct files
1. Edit `sim.sh` so that your computer calls the correct MD engine
1. Edit `postprocess.py` so that it generates the data you want 
(output is `N_frames * (3 + N_features)` for every simulation. 
The first three are round, simulation index, and frame. 
E.g. In the case of contact map matrices, output is `N_frames * (3 + N_residue * N_residue)`)
E.g. In the case of atom position matrices, output is `N_frames * (3 + 3 * N_atoms)`)
1. Edit `train.py / pairwise.py` so that it takes your data and processes it correctly. 
You don't necessarily need to use a CVAE - you can do any model, ML or not, 
so long as it selects some frames for next rounds
1. Edit `model.py` if you choose to use CVAE - make sure the input dimensions match number of residues of your protein.
In case you have an odd number of residues, use ZeroPadding2D in encoder and Cropping2D in decoder. Use `model_dim.py` to check your model dimensions.

With all things edited, launch by `sh main.sh`. You should see progress.


## How to use on other clusters

Modify the `jsrun` commands in `main.sh` and `sim.sh` to whatever your scheduler wants.

## Restarting

The `Simulations/progress` files contains keywords of restarting round and stages: 
0 = new round, 1 = do simulation and propose, 2 = postprocessing. 
In `main.sh` these are updated once a stage is completed. 
If the workflow is terminated (e.g. due to wall time limit or completed required rounds),
it will restart from that stage.

## Other features

Calculating the pairwise RMSD is very expensive, and the squared matrix is huge.
so it is cached as a scipy.sparse matrix. 
Since we use DBSCAN we can use sparse distance input as precomputed metric.
Every round, the matrix is extended to include more pairwise RMSD values.
Any pairwise values less than a threshold (e.g. 1.3 A) are not stored.
This saves about 99.99 % of space (275 GB -> 40 MB) when using 1.3 A cutoff.

Currently, the calculation of pairwise RMSD and suggesting new structures are done 
***the next round***, because each NAMD3 simulation uses 1 core and 1 GPU, and 
caluclating pairwise RMSD works well with multiprocessing, we can hide the latency 
by having this calculation done during simulations.

# Folding BBA

Following the use case of DDMD paper folding the BBA protein, 
I also tested it with the FAST approach. It works!



# EasyDeepDriveMD
Poor person's DeepDriveMD implementation


## Requirement

1. scipy (for DBSCAN)
1. MDAnalysis (for contact maps)
1. tensorflow(-gpu) (for training)
1. h5py and pyyaml (for saving model weights)
1. Your favorite MD package (I use NAMD)

If you are on Summit like I do, you can use the open-ce as your base environment:

```bash
module load cuda gcc fftw
# with your conda env manager - if you don't have one, see below
conda create -p /your/conda/env/path --clone /sw/summit/open-ce/anaconda-base/envs/open-ce-1.5.2-py39-0
conda activate -p /your/conda/env/path
pip install MDAnalysis
```

Afterwards, make sure MDAnalysis runs by
```bash
python
import MDAnalysis as mda
```

### If you don't have a conda env manager (on Summit)

Installing one is easy; highly recommended

``` bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-ppc64le.sh
bash Miniconda3-latest-Linux-ppc64le.sh -b -p miniconda
# Initialize your ~/.bash_profile
miniconda/bin/conda init bash
source ~/.bashrc
```

## Structure

I suggest something like this (for NAMD runs):

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

If you run with AMBER, perhaps something like this would work:

```
Experiment/
  |- Structures/
  |    |- protein.parm7
  |    |- protein.rst7
  |- Templates/
  |    |- protein_init.rst7
  |    |- sampling.in
  |- EasyDeepDriveMD/ (this repo)
  |
```

Over the course of DDMD, the `Simulations` folder will be generated with simulation result and associated data in it.

```
Experiment/
  |- ...
  |- Simulations/
       |- progress
       |- eps
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

The `main.sh` is the main script. It has three stages for each round: prepare / sample / train & propose.

In the prepare stage, the script (EDDMD) copies `sim_config` to each simulation folder for that round.
If it is the first round, EDDMD copies `init_coord` to each simulation folder as well.

In the simulation stage, EDDMD calls `sim.sh` for every individual simulations with different arguments.
The `sim.sh` does both simulation and postprocessing by calling the MD engine and then `postprocess.py`.
The `postprocess.py` for now calculates contact map of CA atoms using 8 Angstrom cutoff.
You can modify it so it generates something that fits your scientific goal 
(e.g. torsional angle of all backbone + side chains).
`postprocess.py` saves results in `Simulations/data/{round}_{sim_idx}.npy`.

In the train & propose stage, EDDMD calls `train.py` to load all `.npy` data, train a convolutional 
variational autoencoder (CVAE), use DBSCAN to pick outliers in the latent space of CVAE, and 
outputs specific frames from the pool of snapshots as new initial coordinates for the next round.

This workflow is basically identical to the file-based DeepDriveMD (DeepDriveMD-F) implementation.


## How to use (on Summit)

You basically have to edit every `.sh` and `.py` files in this folder, tailored to your simulation engines and goals,
which is why I recommend having individual copies of this repo for every `Experiment/`.

1. Place all necessary files
1. Edit `main.sh` so that the initial variables point to correct files
1. Edit `sim.sh` so that your computer calls the correct MD engine with correct arguments
1. Edit `postprocess.py` so that it generates the data you want 
(output is `N_frames * (3 + N_features)` for every simulation. 
The first three are round, simulation index, and frame. 
E.g. In the case of contact map matrices, output is `N_frames * (3 + N_residue * N_residue)`)
1. Edit `train.py` so that it takes your data and processes it correctly. 
You don't necessarily need to use a CVAE - you can do any model, ML or not, 
so long as it selects some frames for the next round.
1. Edit `model.py` if you choose to use CVAE (or create another `model.py`).
For CVAE, make sure the input dimensions match number of residues of your protein.
In case you have an odd number of residues, use ZeroPadding2D in encoder and Cropping2D in decoder. 
Use `model_dim.py` to check your model dimensions.
For other models, just make sure the VAE requirement (that the model tries to reconstruct original data) is met.

With all things edited, launch by `sh main.sh`. You should see progress.


## How to use on other clusters

Modify the `jsrun` commands in `main.sh` and `sim.sh` to whatever your scheduler wants.

## Restarting

The `Simulations/progress` files contains keywords of restarting round and stages: 
0 = new round, 1 = do simulation, 2 = do training & propose.
In `main.sh` these are updated once a stage is completed. 
If the workflow is terminated (e.g. due to wall time limit or completed required rounds),
it will restart from that stage.

## Other features

Since the `eps` of DBSCAN dictates how many outliers there are, I allowed it to be varied within and across rounds.
The goal is to select an `eps` that generates slightly more outliers than number of individual simulations.
E.g. if I try to select 6 simulations, then 6 outliers is better than 8 than 20 than 4 than 0.

Also, in smaller systems the CA maps are frequently duplicated,
so one can actually group them in DBSCAN by putting `sample_weights`
on every unique CA map vector (flattened matrix). 
This saves tremendous amount of DBSCAN time (for me 50s -> 0.4 s).


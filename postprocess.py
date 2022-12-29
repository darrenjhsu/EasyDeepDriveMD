
import sys
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import contacts, distances

if len(sys.argv) < 5:
    print("Usage: postprocess.py <relative psf file path> <just the dcd> <round index> <simulation index>")
    exit()

psf = sys.argv[1]
dcd = sys.argv[2]
round_idx = int(sys.argv[3])
sim_idx = int(sys.argv[4])

U = mda.Universe(psf, f'../Simulations/{round_idx}/{sim_idx}/{dcd}')

CA = U.select_atoms('name CA')
CM = []

for ts in U.trajectory:
    CM.append(np.hstack([round_idx, sim_idx, ts.frame, distances.contact_matrix(CA.positions, cutoff=8).flatten()]))

CM = np.array(CM)

np.save(f'../Simulations/data/{round_idx}_{sim_idx}.npy', CM)

#print(f'Running post processing python script for round {round_idx}, simulation {sim_idx}')

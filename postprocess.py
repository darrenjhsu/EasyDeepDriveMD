
import sys

if len(sys.argv) < 5:
    print("Usage: postprocess.py <relative psf file path> <just the dcd> <round index> <simulation index> <target pdb (optional)>")
    exit()

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import contacts, distances, rms


psf = sys.argv[1]
dcd = sys.argv[2]
round_idx = int(sys.argv[3])
sim_idx = int(sys.argv[4])
try:
    target = sys.argv[5]
except:
    target = None

U = mda.Universe(psf, f'../Simulations/{round_idx}/{sim_idx}/{dcd}')


CA = U.select_atoms('name CA')
CA_list = []
for ts in U.trajectory:
    CA_list.append(np.hstack([round_idx, sim_idx, ts.frame, CA.positions.flatten()]))

CA_list = np.array(CA_list)

np.save(f'../Simulations/data/CA_{round_idx}_{sim_idx}.npy', CA_list)


if target is not None:
    T = mda.Universe(target)

    # https://www.nature.com/articles/s41467-022-35399-8
    # lid = 98-136
    # rigid parts = 11-97 137-223
    Umob = U.select_atoms('protein and name CA and resid 98-136') 
    Tmob = T.select_atoms('protein and name CA and resid 98-136')

    R = rms.RMSD(U, T, select='protein and name CA and resid 11-97 137-223', tol_mass=np.inf)

    # This also aligns the structures in place
    R.run()

    RMSD = []
    for ts in U.trajectory:
        RMSD.append(rms.rmsd(Umob.positions, Tmob.positions)) # Do not align

    RMSD = np.hstack([CA_list[:,:3], np.array(RMSD)[:, None]])#R.rmsd[:,-1][:, None]])
    
    np.save(f'../Simulations/data/RMSD_{round_idx}_{sim_idx}.npy', RMSD)

#print(f'Running post processing python script for round {round_idx}, simulation {sim_idx}')

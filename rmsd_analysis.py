
import numpy as np

RMSD_list = np.load('../Simulations/data/RMSD_rounds_0_to_299.npy')
print(len(RMSD_list))

print(RMSD_list[:, -1].min(), RMSD_list[np.argmin(RMSD_list[:, -1])])

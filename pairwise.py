
import sys, os

if len(sys.argv) < 6:
    print('Usage: python train.py <round index> <num of sim per round> <relative psf file path> <just the dcd> <initial coord file name>')
    exit()

import numpy as np
import MDAnalysis as mda
import multiprocessing as mp
import time
from itertools import repeat
from selection import *
from scipy.sparse import csr_matrix
import scipy.sparse as sp

def pairwise_RMSD_row(P, Q):
    # P is the one to be aligned in batch(B * N * 3)
    # Q is the ref (N * 3)
    PU = P
    QU = Q
    PC = PU - PU.mean(axis=1, keepdims=True) # Center points
    QC = QU - QU.mean(axis=0) # Center points
    # Kabsch method
    #print(PC.shape)
    C = np.dot(np.transpose(PC, (0, 2, 1)), QC)
    #print(C.shape)
    V, S, W = np.linalg.svd(C,full_matrices=False)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    #print(V.shape, S.shape, W.shape, d.shape)
    S[d, -1] = -S[d, -1]
    V[d, :, -1] = -V[d, :, -1]
    # Create Rotation matrix U
    #print(V.shape, S.shape, W.shape)
    U = np.matmul(V, W)
    #print(U.shape)
    #print(P.shape)
    P = P - PU.mean(axis=1, keepdims=True) # Move all points
    #print(P.shape)
    Q = Q - QU.mean(axis=0) # Move all points
    P = np.matmul(P, U) # Rotate P
    diff = P - Q
    #print(diff.shape)
    N = len(Q)
    return np.sqrt((diff * diff).sum((1, 2)) / N)

round_idx = int(sys.argv[1])
n_sim = int(sys.argv[2])
psf = sys.argv[3]
dcd_fname = sys.argv[4]
init_fname = sys.argv[5].split('/')[-1]


t0 = time.time()
t00 = time.time()
if round_idx > 1:
    print(f"Loading npy of previous rounds: round 0 to {round_idx-2}")
    CA_prev = np.load(f'../Simulations/data/CA_rounds_0_to_{round_idx-2}.npy')
    PD_prev_sp = sp.load_npz(f'../Simulations/data/PD_rounds_0_to_{round_idx-2}.npz')
    #PD_prev = np.load(f'../Simulations/data/PD_rounds_0_to_{round_idx-2}.npy')
    RMSD_prev = np.load(f'../Simulations/data/RMSD_rounds_0_to_{round_idx-2}.npy')
    print(CA_prev.shape, PD_prev_sp.shape)
t1 = time.time()
print(f"Load old: {t1-t0:.3f} s")
t0 = time.time()
CA_this = []
RMSD_this = []

for i in range(n_sim):
    print(f"Loading npy of last round: idx {i}")
    try:
        CA_this.append(np.load(f'../Simulations/data/CA_{round_idx-1}_{i}.npy'))
        RMSD_this.append(np.load(f'../Simulations/data/RMSD_{round_idx-1}_{i}.npy'))
    except Exception as e:
        print(e)

t1 = time.time()
print(f"Load new: {t1-t0:.3f} s")
t0 = time.time()
CA_this = np.concatenate(CA_this, axis=0)
RMSD_this = np.concatenate(RMSD_this, axis=0)

print("Concatenating npy of this round with all previous rounds")
if round_idx > 1:
    CA_all = np.concatenate([CA_prev, CA_this], axis=0)
    RMSD_all = np.concatenate([RMSD_prev, RMSD_this], axis=0)
else:
    CA_all = CA_this
    CA_prev = CA_this
    RMSD_all = RMSD_this

n_frames = len(CA_all)
RMSD_dict = {(x[0], x[1], x[2]): x[3] for x in RMSD_all}


t1 = time.time()
print(f"Concat: {t1-t0:.3f} s")
t0 = time.time()

print(f"Saving npys of round 0 to {round_idx-1}")
np.save(f'../Simulations/data/CA_rounds_0_to_{round_idx-1}.npy', CA_all)
np.save(f'../Simulations/data/RMSD_rounds_0_to_{round_idx-1}.npy', RMSD_all)
t1 = time.time()
print(f"Save file: {t1-t0:.3f} s")
t0 = time.time()

frames = CA_all[:, :3]
CA_all = CA_all[:, 3:].reshape(len(CA_all), -1, 3)
CA_prev = CA_prev[:, 3:].reshape(len(CA_prev), -1, 3)
CA_this = CA_this[:, 3:].reshape(len(CA_this), -1, 3)

t1 = time.time()
print(f"Reshape: {t1-t0:.3f} s")
t0 = time.time()

with mp.Pool(36) as p:
    # this vs this
    res_tt = p.starmap(pairwise_RMSD_row, zip(repeat(CA_this), [CA_this[i] for i in range(len(CA_this))])) 
    if round_idx > 1:
        # prev vs this
        res_pt = p.starmap(pairwise_RMSD_row, zip(repeat(CA_prev), [CA_this[i] for i in range(len(CA_this))]))
    #res_at = p.starmap(pairwise_RMSD_row, zip(repeat(CA_all), [CA_this[i] for i in range(len(CA_this))])) 
PD_tt = np.vstack(res_tt)
#PD_at = np.vstack(res_at)
PD_tt *= (PD_tt < 1.3)
if round_idx > 1:
    PD_pt = np.vstack(res_pt)
    PD_pt *= (PD_pt < 1.3)
    PD_tp = PD_pt.T
t1 = time.time()
print(f'MP row method: {(t1-t0)*1000:.1f} ms')
t0 = time.time()

if round_idx > 1:
    PD_len = len(CA_prev) + len(CA_this)
    # Calculate ground truth
    #PD_all = np.zeros((PD_len, PD_len))
    #PD_all[:len(PD_prev), :len(PD_prev)] = PD_prev
    #PD_all[len(PD_prev):] = PD_at
    #PD_all[:, len(PD_prev):] = PD_at.T

    # Calculate sparse
    PD_prev_sp.resize((PD_len, PD_len))
    PD_pt_idx1, PD_pt_idx2 = np.nonzero(PD_pt)
    PD_tp_idx1, PD_tp_idx2 = np.nonzero(PD_tp)
    PD_tt_idx1, PD_tt_idx2 = np.nonzero(PD_tt)
    PD_pt_data = PD_pt[np.nonzero(PD_pt)]
    PD_tp_data = PD_tp[np.nonzero(PD_tp)]
    PD_tt_data = PD_tt[np.nonzero(PD_tt)]
    PD_pt_sp = csr_matrix((PD_pt_data, (PD_pt_idx1 + len(CA_prev), PD_pt_idx2)), shape=(PD_len, PD_len))
    PD_tp_sp = csr_matrix((PD_tp_data, (PD_tp_idx1, PD_tp_idx2 + len(CA_prev))), shape=(PD_len, PD_len))
    PD_tt_sp = csr_matrix((PD_tt_data, (PD_tt_idx1 + len(CA_prev), PD_tt_idx2 + len(CA_prev))), shape=(PD_len, PD_len))
    PD_all_sp = PD_prev_sp + PD_pt_sp + PD_tp_sp + PD_tt_sp
else:
    PD_len = len(PD_tt)
    #PD_all = PD_at
    PD_all_sp = csr_matrix(PD_tt * (PD_tt < 1.3))


#print('Max diff between dense and sparse:', np.sqrt(np.max((PD_all * (PD_all < 1.3) - PD_all_sp.toarray())**2)))


t1 = time.time()
print(f"Assign: {t1-t0:.3f} s")
t0 = time.time()

sp.save_npz(f'../Simulations/data/PD_rounds_0_to_{round_idx-1}.npz', PD_all_sp)
#np.save(f'../Simulations/data/PD_rounds_0_to_{round_idx-1}.npy', PD_all)

t1 = time.time()
print(f"Save PD: {t1-t0:.3f} s")
t0 = time.time()

from sklearn.cluster import DBSCAN

cls = DBSCAN(eps=1.0, min_samples=5, metric='precomputed')
#cls.fit(PD_all)
#print('Number of clusters (dense):', cls.labels_.max()+1)
#print('Number of outliers (dense):', np.sum(cls.labels_ == -1))
cls.fit(PD_all_sp)
print('Number of clusters (sparse):', cls.labels_.max()+1)
print('Number of outliers (sparse):', np.sum(cls.labels_ == -1))

t1 = time.time()
print(f"DBSCAN: {t1-t0:.3f} s")
t0 = time.time()

select = select_FAST(frames, cls.labels_, n_sim, RMSD_dict, metric='min', cls=cls)
print(select)

t1 = time.time()
print(f"FAST: {t1-t0:.3f} s")
t0 = time.time()

print("Finding and outputting specific frames ...")

for idx, sel in enumerate(select):
    t01 = time.time()
    os.makedirs(f'../Simulations/{round_idx+1}/{idx}', exist_ok=True)
    t11 = time.time()
    #print(f"  makedir: {t11-t01:.3f} s")
    t01 = time.time()
    U = mda.Universe(psf, f'../Simulations/{sel[0]}/{sel[1]}/{dcd_fname}')
    t11 = time.time()
    #print(f"  loadtraj: {t11-t01:.3f} s")
    t01 = time.time()
    U.trajectory[sel[2]]
    t11 = time.time()
    #print(f"  goto: {t11-t01:.3f} s")
    t01 = time.time()
    Uall = U.select_atoms('all')
    t11 = time.time()
    #print(f"  select: {t11-t01:.3f} s")
    t01 = time.time()
    Uall.write(f'../Simulations/{round_idx+1}/{idx}/{init_fname}', bonds=None) # bonds=None makes write out much faster
    t11 = time.time()
    #print(f"  write: {t11-t01:.3f} s")
    t01 = time.time()
    t1 = time.time()
    print(f"Output one file: {t1-t0:.3f} s")
    t0 = time.time()
t10 = time.time()
print(f"Overall: {t10-t00:.3f} s")

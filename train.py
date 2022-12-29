

import sys, os

if len(sys.argv) < 4:
    print('Usage: python train.py <round index> <num of sim per round> <relative psf file path> <just the dcd> <initial coord file name>')
    exit()

import numpy as np

import glob
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time
from model import *
from sklearn.cluster import DBSCAN
import MDAnalysis as mda

round_idx = int(sys.argv[1])
n_sim = int(sys.argv[2])
psf = sys.argv[3]
dcd_fname = sys.argv[4]
init_fname = sys.argv[5].split('/')[-1]

CM_this = []

if round_idx > 0:
    print(f"Loading npy of previous rounds: round 0 to {round_idx-1}")
    CM_prev = np.load(f'../Simulations/data/rounds_0_to_{round_idx-1}.npy')

for i in range(n_sim):
    print(f"Loading npy of this round: idx {i}")
    try:
        CM_this.append(np.load(f'../Simulations/data/{round_idx}_{i}.npy'))
    except Exception as e:
        print(e)

CM_this = np.concatenate(CM_this, axis=0)

print("Concatenating npy of this round with all previous rounds")
if round_idx > 0:
    CM_all = np.concatenate([CM_prev, CM_this], axis=0)
else:
    CM_all = CM_this

print(f"Saving npys of round 0 to {round_idx}")
np.save(f'../Simulations/data/rounds_0_to_{round_idx}.npy', CM_all)

np.random.shuffle(CM_all)

n_res = np.sqrt(CM_all.shape[-1]).astype(int)
n_frames = len(CM_all)
print(f'There are {n_res} residues and {n_frames} frames')

CM_tuple = [tuple(x) for x in list(CM_all[:,3:])]
CM_dict = {}
CM_category_map = {}
for idx, x in enumerate(set(CM_tuple)):
    CM_dict[x] = 0
    CM_category_map[x] = idx
for x in CM_tuple:
    CM_dict[x] += 1
print(f'There are {len(set(CM_tuple))} unique combinations')
CM_list = np.array([list(x) for x in CM_dict.keys()]).reshape(-1, n_res, n_res, 1)
CM_weight = np.array([CM_dict[x] for x in CM_dict.keys()])

CM_category = [CM_category_map[x] for x in CM_tuple] # This is a map of frame -> combination index

print("Preprocess data")
frames, data = CM_all[:, :3], CM_all[:, 3:].reshape(-1, n_res, n_res, 1).astype('float32')
train_val_split = int(0.8 * len(CM_all))  # type: ignore[arg-type]
train_data, valid_data = (
    data[:train_val_split],  # type: ignore[index]
    data[train_val_split:],  # type: ignore[index]
)

batch_size = min(2048, max(int(len(train_data) / 8) // 16 * 16, 1))
print(f'Batch size is {batch_size}')
all_dataset = (tf.data.Dataset.from_tensor_slices(data).batch(batch_size))
train_dataset = (tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size))
valid_dataset = (tf.data.Dataset.from_tensor_slices(valid_data).batch(batch_size))

print("Training CVAE ...")
model = CVAE(10)
if round_idx > 0:
    try:
        latest = tf.train.latest_checkpoint(f'../Simulations/saved_models/round_{round_idx-1}/')
        model.load_weights(latest)
        print(f'Loaded pretrained model from rounds 0 to {round_idx-1}')
    except Exception as e:
        print(e)
else:
    print("Round 0: Start fresh model")

optimizer = tf.keras.optimizers.Adam(1e-4)
epochs = 400

def earlystop(current, history, patience):
    if len(history) == 0:
        return False
    if len(history) - np.argmax(np.array(history)) > patience:
        return True
    else:
        return False

elbo_history = []

start_time = time.time()
for epoch in range(1, epochs + 1):
    for train_x in train_dataset:
        train_step(model, train_x, optimizer)
    end_time = time.time()
  
    loss = tf.keras.metrics.Mean()
    for valid_x in valid_dataset:
        loss(compute_loss(model, valid_x))
    elbo = -loss.result()
    if earlystop(elbo, elbo_history, patience=20):
        print(f'Epoch: {epoch:3d}, Test set ELBO: {elbo:8.3f}, time elapsed: {end_time - start_time:8.3f} s')
        print('Terminating due to patience of 20 epochs')
        break
    else:
        elbo_history.append(elbo)
        if len(elbo_history) == 1:
            print(f'Epoch: {epoch:3d}, Test set ELBO: {elbo:8.3f}, time elapsed: {end_time - start_time:8.3f} s | reset patience')
        elif elbo_history[-1] > np.max(elbo_history[:-1]):
            print(f'Epoch: {epoch:3d}, Test set ELBO: {elbo:8.3f}, time elapsed: {end_time - start_time:8.3f} s | reset patience')
        else:
            print(f'Epoch: {epoch:3d}, Test set ELBO: {elbo:8.3f}, time elapsed: {end_time - start_time:8.3f} s')

os.makedirs(f'../Simulations/saved_models/round_{round_idx}', exist_ok=True)
model.save_weights(f'../Simulations/saved_models/round_{round_idx}/checkpoint')

print("Encode all data ...")
#CM_embed = model.encode(data)
CM_embed = model.encode(CM_list)
print(CM_embed[0].shape)


print("Running DBSCAN in latent space ...")
if round_idx > 0:
    try:
        with open('../Simulations/eps','r') as f:
            eps_init = float(f.read().strip())
    except Exception as e:
        eps_init = 0.3
        print(e)
if eps_init < 0.25:
    eps_choices = np.linspace(0.45, 0.05, num = 9)
else:
    eps_choices = np.linspace(eps_init + 0.2, eps_init - 0.2, num = 9)

#eps_choices = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5][::-1]
num_outliers = []
cls_collection = []
t0 = time.time()
for eps in eps_choices:
    cls = DBSCAN(eps=eps, n_jobs=1, algorithm='ball_tree')
    cls.fit(CM_embed[0], sample_weight=CM_weight)
    #print("Suggesting outliers in latent space ...")
    sum_outliers = np.sum(CM_weight[cls.labels_ == -1])
    num_outliers.append(sum_outliers)
    cls_collection.append(cls)
    #print(outliers)
    t1 = time.time()
    print(f'There are {sum_outliers} outliers for eps = {eps} ({(t1-t0)*1000:.1f} ms)')
    #if sum_outliers >= n_sim:
    #    break

# Select best DBSCAN model ... best is exactly n_sim outliers, 
# then slightly more than n_sim outliers,
# then a lot more, then a little less, then none
outlier_rank = np.array(num_outliers)
outlier_rank[outlier_rank < n_sim] = outlier_rank.max() + n_sim + 1 - outlier_rank[outlier_rank < n_sim]
cls_sel = np.argmin(outlier_rank)
print(f'Using eps = {eps_choices[cls_sel]}, and there are {num_outliers[cls_sel]} outliers')
cls = cls_collection[cls_sel]
t1 = time.time()
print(f'Determining best DBSCAN model took {(t1-t0)*1000:.1f} ms')

category_to_cls_label = {}
for idx, x in enumerate(cls.labels_):
    category_to_cls_label[idx] = x

# Put back assignment
CM_label = np.array([category_to_cls_label[x] for x in CM_category])
outliers = frames[CM_label == -1]

print(f'There are {len(outliers)} outliers')

if len(outliers) < n_sim:
    extra_select = frames[np.random.choice(np.array(np.arange(len(data)))[np.nonzero(CM_label > -1)], n_sim - len(outliers), replace=False)]
    if len(outliers) == 0:
        select = extra_select
    else:
        select = np.vstack((outliers, extra_select))
elif len(outliers) > n_sim:
    select = outliers[np.random.choice(len(outliers), n_sim, replace=False)]
else: # len outliers match n_sim
    select = outliers

print(select)

print("Finding and outputting specific frames ...")

for idx, sel in enumerate(select):
    os.makedirs(f'../Simulations/{round_idx+1}/{idx}', exist_ok=True)
    U = mda.Universe(psf, f'../Simulations/{sel[0]}/{sel[1]}/{dcd_fname}')
    U.trajectory[sel[2]]
    Uall = U.select_atoms('all')
    Uall.write(f'../Simulations/{round_idx+1}/{idx}/{init_fname}')

with open('../Simulations/eps', 'w') as f:
    f.write(f'{eps_choices[cls_sel]:.2f}')

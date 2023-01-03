
import numpy as np

def select_explore(frames, CM_label, n_sim):
    # This is for exploring conformational space
    outliers = frames[CM_label == -1]
    if len(outliers) < n_sim:
        extra_select = frames[np.random.choice(np.array(np.arange(len(frames)))[np.nonzero(CM_label > -1)], n_sim - len(outliers), replace=False)]
        if len(outliers) == 0:
            select = extra_select
        else:
            select = np.vstack((outliers, extra_select))
    elif len(outliers) > n_sim:
        select = outliers[np.random.choice(len(outliers), n_sim, replace=False)]
    else: # len outliers match n_sim
        select = outliers

    return select

def select_guided_DDMD(frames, CM_label, n_sim, guide, metric='min'):
    # This is for guided search (select outlier with best guide metric)
    outliers = frames[CM_label == -1]
    if len(outliers) < n_sim:
        extra_select = frames[np.random.choice(np.array(np.arange(len(frames)))[np.nonzero(CM_label > -1)], n_sim - len(outliers), replace=False)]
        if len(outliers) == 0:
            select = extra_select
        else:
            select = np.vstack((outliers, extra_select))
    elif len(outliers) > n_sim:
        # Calculate rmsd
        outliers_rmsd = []
        for idx, sel in enumerate(outliers): 
            #print(f'Outlier {idx} has rmsd of {r:.3f} A')
            outliers_rmsd.append(guide[(sel[0], sel[1], sel[2])])
        if metric == 'min':
            select = outliers[np.argsort(outliers_rmsd)[:n_sim]]
            print(f'Selected outliers with guiding metric: {np.sort(outliers_rmsd)[:n_sim]}')
        elif metric == 'max': 
            select = outliers[np.argsort(outliers_rmsd)[::-1][:n_sim]]
            print(f'Selected outliers with guiding metric: {np.sort(outliers_rmsd)[::-1][:n_sim]}')
    else: # len outliers match n_sim
        select = outliers
    return select


def select_FAST(frames, CM_label, n_sim, guide, metric='min', cls=None, select_outliers=True):
    # Using the FAST adaptive sampling method
    # Every outlier is treated as its own cluster, but the undirected reward is 1
    # The actual bottom for that reward is determined by the min_sample parameter of the cls (if provided)
    # Otherwise it is the size of the smallest cluster that is not an outlier
    outliers = frames[CM_label == -1]
    cluster_count = {x: np.sum(CM_label == x) for x in range(0, CM_label.max()+1)}
    print(cluster_count)
    # Determine min samples
    if cls is not None:
        min_samples = cls.min_samples
    else:
        min_samples = np.array(list(cluster_count.values())).min()
    # Determine undirected reward
    max_samples = np.array(list(cluster_count.values())).max()
    cluster_ur = {x: (max_samples - cluster_count[x]) / (max_samples - min_samples) for x in cluster_count.keys()}
    cluster_ur[-1] = 1.0
    sample_ur = np.array([cluster_ur[x] for x in CM_label])
    sample_ur[sample_ur > 1] = 1.0

    # Determine directed reward
    guide_max = np.array(list(guide.values())).max()
    guide_min = np.array(list(guide.values())).min()
    if metric == 'min':
        sample_dr = np.array([(guide_max - guide[(x[0], x[1], x[2])]) / (guide_max - guide_min) for x in frames])
    elif metric == 'max':
        sample_dr = np.array([(guide[(x[0], x[1], x[2])] - guide_min) / (guide_max - guide_min) for x in frames])

    sample_r = sample_ur + sample_dr # Currently they are evenly divided

    candidates_outlier = np.where(CM_label == -1)[0]

    CM_idx = np.arange(len(CM_label))
    candidates_cluster = np.array([CM_idx[CM_label == x][np.argmax(sample_r[CM_label == x])] for x in range(0, CM_label.max()+1)])

    if select_outliers:
        candidates = np.concatenate([candidates_outlier, candidates_cluster])
    else:
        candidates = candidates_cluster
    candidates_r = sample_r[candidates]
    print(candidates[np.argsort(candidates_r)[::-1][:n_sim]])
    print(np.sort(candidates_r)[::-1][:n_sim])
    print(CM_label[candidates][np.argsort(candidates_r)[::-1][:n_sim]])
    select = frames[candidates[np.argsort(candidates_r)[::-1][:n_sim]]].astype(int)
    print([guide[(x[0], x[1], x[2])] for x in select])

    if len(select) < n_sim:
        extra_select = frames[np.random.choice(np.arange(len(frames)), n_sim - len(select), replace=False)].astype(int)
    if len(select) == 0:
        select = extra_select
    elif len(select) < n_sim:
        select = np.vstack((select, extra_select))

    return select
#    pass

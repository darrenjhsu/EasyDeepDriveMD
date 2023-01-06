
import numpy as np

# Build tree ... this is going to be tricky
def backtrace(frame, RMSD_frame, start_frame, print_trace=True):
    traj_length = 0
    trace = []
    trace.append(RMSD_frame[frame])
    r, s, f, rmsd = RMSD_frame[frame]
    fr, fs, ff, frmsd = RMSD_frame[frame]
    #print(f'Frame {frame:6d} is round {int(r):3d}, simulation {int(s):3d} and frame {int(f):3d}, rmsd {rmsd:6.3f} A')
    while r > 0:
        last_frame = start_frame[int(r)][int(s)]
        trace.append(RMSD_frame[last_frame])
        r, s, f, rmsd = RMSD_frame[last_frame]
    #    print(f'  which is from round {int(r):3d}, simulation {int(s):3d} and frame {int(f):3d}, rmsd {rmsd:6.3f} A')
    if print_trace:
        print(f'To arrive at frame {frame:6d} (round {int(fr):3d}, simulation {int(fs):3d} and frame {int(ff):3d}, rmsd {frmsd:6.3f} A):')
    for t in trace[::-1]: 
        r, s, f, rmsd = t
        traj_length += int(f)
        if print_trace:
            print(f'  From round {int(r):3d}, simulation {int(s):3d} and frame 0 to {int(f):3d} (rmsd {rmsd:6.3f} A)')
    if print_trace:
        print(f'Total traj length = {traj_length / 100} ns')
    return trace

def output_traj(frame, RMSD_frame, start_frame, psf=None, fname='traj.dcd'):
    try:
        import mdtraj
    except:
        return None
    if psf is None:
        print("Supply psf and try again")
        return None
    trace = backtrace(frame, RMSD_frame, start_frame, print_trace=False)[::-1]
    for idx, t in enumerate(trace):
        r, s, f, rmsd = t
        print(f'  From round {int(r):3d}, simulation {int(s):3d} and frame 0 to {int(f):3d} (rmsd {rmsd:6.3f} A)')
        if idx == 0:
            #traj = mdtraj.load(f'../Simulations/{int(r)}/{int(s)}/BBA_init.pdb', top=psf)
            traj = mdtraj.load(f'../Simulations/{int(r)}/{int(s)}/BBA_sample.dcd', top=psf)[:int(f)]
        else:
            #traj += mdtraj.load(f'../Simulations/{int(r)}/{int(s)}/BBA_init.pdb', top=psf)
            traj += mdtraj.load(f'../Simulations/{int(r)}/{int(s)}/BBA_sample.dcd', top=psf)[:int(f)]
    traj.save(fname)



# Get starting frames every round
start_frame = {}
for log in ['test6.log','test7.log','test8.log']:
    print(log)
    with open(log,'r') as f:
        cont = f.readlines()
    for idx, line in enumerate(cont):
        if "We are in round" in line:
            current_round = int(line.split()[-1].strip('!'))
        if "DBSCAN" in line:
            for idy in range(idx, idx+10):
                if cont[idy].startswith('['):
                    #print(cont[idy])
                    start_frame[current_round] = [int(x.strip(']')) for x in cont[idy].strip('[').split()]
                    break

RMSD_frame = np.load('../Simulations/data/RMSD_rounds_0_to_299.npy')
#print(RMSD_frame.shape)
#print(start_frame)

for x in np.argsort(RMSD_frame[:, -1])[:1]:
    #backtrace(x, RMSD_frame, start_frame)
    output_traj(x, RMSD_frame, start_frame, psf='../Structures/1FME_wb_Cl.psf')
    print()


print(f'There are {np.sum(RMSD_frame[:, -1] < 2.5)} frames < 2.5 A RMSD to target')

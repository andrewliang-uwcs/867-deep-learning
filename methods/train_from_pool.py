# -*- coding: utf-8 -*-
"""
Semi-supervised baseline on a STORED scene pool: random initial labels -> supervised steps
on the labelled scenes -> unsupervised steps on the whole pool. This is the starting point
(round 0) of the active-calibration loop and checks that lib/scene_pool.py trains.

Run from methods/ after generate_scenes.py:
    $env:PYTHONPATH = ".."
    python train_from_pool.py
"""
import os
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../lib/')
from simulation_parameters import *
from lib.scene_pool import SimConfig, load_scenes, num_scenes, train_steps
from lib.identifiability import ObsModel, calibration_error, translation_direction

file_name = 'pool_SSL'
cfg = {'smoke': dict(n_labels=8,   sl_steps=5,    ul_steps=5),
       'cpu':   dict(n_labels=50,  sl_steps=1000, ul_steps=1000),
       'paper': dict(n_labels=500, sl_steps=int(1e4), ul_steps=int(7.5e4))}[run_mode]
seed = 0

sim = SimConfig.from_globals(globals())
m = ObsModel(sim)
gauge = translation_direction(m).reshape(-1, 1)      # unfixable direction, ignored in the error

path = os.path.join('data', f'scenes_{run_mode}_train.pt')
if not os.path.exists(path):
    sys.exit(f'{path} not found: run generate_scenes.py first')
pool, meta = load_scenes(path, device)
N = num_scenes(pool)

g = torch.Generator().manual_seed(seed)
perm = torch.randperm(N, generator=g)
labelled, unlabelled = perm[:cfg['n_labels']], perm[cfg['n_labels']:]
print(f'Pool {path}: {N} scenes, {len(labelled)} labelled at random (seed {seed})')

err = [calibration_error(network.d, ant_d, m, gauge)]
print(f'Calibration error before training: {err[-1]:.4f}')

print('*** Supervised steps on labelled scenes ***')
loss_sl = train_steps(network, pool, labelled, cfg['sl_steps'], batch_size, optimizerSuper, sim,
                      supervised=True, generator=g, print_every=max(1, cfg['sl_steps'] // 5))
err.append(calibration_error(network.d, ant_d, m, gauge))
print(f'Calibration error after SL: {err[-1]:.4f}')

print('*** Unsupervised steps on the whole pool (labels not used) ***')
loss_ul = train_steps(network, pool, torch.arange(N), cfg['ul_steps'], batch_size, optimizerUnsuper,
                      sim, supervised=False, generator=g, print_every=max(1, cfg['ul_steps'] // 5))
err.append(calibration_error(network.d, ant_d, m, gauge))
print(f'Calibration error after UL: {err[-1]:.4f}')

print('*** Sensing test (original test function, fresh test scenes) ***')
network.eval()
pd, pfa, rmse_angle, rmse_range, rmse_pos = \
    testNetworkSensing(sigma_sens, theta_min_sens_test, theta_max_sens_test, range_min_sens_test,
                       range_max_sens_test, Ngrid_angle, range_grid, pixels_angle, pixels_range,
                       K, S, N0, Delta_f, lamb, ant_d, network.d, refConst,
                       target_pfa, delta_pfa, thresholds_pfa, batch_size, nTestSamples, device=device)
print(f'Pd {np.mean(pd):.4f}  Pfa {np.mean(pfa):.4f}  position RMSE {np.mean(rmse_pos):.3f} m')

out_file = result_path(file_name)
print(f'Saving results to {out_file}')
np.savez(out_file, labelled=labelled.numpy(), loss_sl=np.array(loss_sl), loss_ul=np.array(loss_ul),
         calibration_error=np.array(err), pd=pd, pfa=pfa, rmse_angle=rmse_angle,
         rmse_range=rmse_range, rmse_pos=rmse_pos,
         learned_d=network.d.detach().cpu().numpy(), pool_file=path)

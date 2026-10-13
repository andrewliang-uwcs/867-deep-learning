# -*- coding: utf-8 -*-
"""
Generate and store fixed pools of simulated ISAC sensing scenes.

Run from methods/ (same as the other scripts):
    $env:PYTHONPATH = ".."
    python generate_scenes.py

Writes data/scenes_<run_mode>_<split>.pt. Pool sizes follow run_mode in
lib/simulation_parameters.py. Scenes use the same training distribution as trainNetwork
(random angular sector of 10-20 deg centred in [-60, 60] deg, range in [0, 200] m).
All splits contain target-present scenes only (proposal Appendix A, Step 1); the
sensing test sets are still drawn by the original test functions.
"""
import os
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../lib/')
from simulation_parameters import *
from lib.scene_pool import SimConfig, draw_scenes, save_scenes, num_scenes

data_dir = 'data'
pool_sizes = {'smoke': dict(train=128, val=64),
              'cpu':   dict(train=2000, val=500),
              'paper': dict(train=20000, val=5000)}[run_mode]
seeds = dict(train=1000, val=2000)          # disjoint seeds -> disjoint pools

sim = SimConfig.from_globals(globals())
os.makedirs(data_dir, exist_ok=True)

for split, n in pool_sizes.items():
    scenes = draw_scenes(n, sim, theta_mean_min, theta_mean_max, span_min_theta, span_max_theta,
                         range_mean_min, range_mean_max, span_min_range, span_max_range,
                         p_present=1.0, seed=seeds[split])
    path = os.path.join(data_dir, f'scenes_{run_mode}_{split}.pt')
    save_scenes(scenes, path, meta=dict(run_mode=run_mode, split=split, n=n,
                                        seed=seeds[split], p_present=1.0))
    th = scenes['theta'] * 180 / np.pi
    print(f'{split:5s}: {num_scenes(scenes)} scenes -> {path} '
          f'(theta {float(th.min()):.1f}..{float(th.max()):.1f} deg, '
          f'range {float(scenes["range"].min()):.1f}..{float(scenes["range"].max()):.1f} m)')

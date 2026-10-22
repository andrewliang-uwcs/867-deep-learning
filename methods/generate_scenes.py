# -*- coding: utf-8 -*-
"""
Generate and store fixed pools of simulated ISAC sensing scenes.

Run from the repository root:
    python methods/generate_scenes.py

Writes methods/data/scenes_<run_mode>_th<theta_mean_max_deg>_<split>.pt. Pool sizes follow run_mode in
lib/simulation_parameters.py. Scenes use the same training distribution as trainNetwork
(random angular sector of 10-20 deg centred in [-60, 60] deg, range in [0, 200] m).
All splits contain target-present scenes only (proposal Appendix A, Step 1); the
sensing test sets are still drawn by the original test functions.
"""
import os
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
# Put the repository root and lib/ on the import path relative to this file, so the
# script runs from any working directory without setting PYTHONPATH.
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[1:1] = [REPO_ROOT, os.path.join(REPO_ROOT, 'lib')]
from simulation_parameters import *
from lib.scene_pool import SimConfig, draw_scenes, save_scenes, num_scenes

data_dir = DATA_DIR                          # methods/data, independent of the working directory
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
    path = os.path.join(data_dir, f'scenes_{pool_tag}_{split}.pt')
    save_scenes(scenes, path, meta=dict(run_mode=run_mode, theta_mean_max_deg=float(theta_mean_max_deg), split=split, n=n,
                                        seed=seeds[split], p_present=1.0))
    th = scenes['theta'] * 180 / np.pi
    print(f'{split:5s}: {num_scenes(scenes)} scenes -> {path} '
          f'(theta {float(th.min()):.1f}..{float(th.max()):.1f} deg, '
          f'range {float(scenes["range"].min()):.1f}..{float(scenes["range"].max()):.1f} m)')

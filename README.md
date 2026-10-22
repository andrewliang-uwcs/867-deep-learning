# 867-deep-learning
This repository contains the codebase for the project/paper component of the CISC 867 Deep Learning course.

The primary intent is to reproduce the results from the following paper:

J. M. Mateos-Ramos, B. Chatelier, C. Häger, M. F. Keskin, L. L. Magoarou, and H. Wymeersch, "Semi-Supervised End-to-End Learning for Integrated Sensing and Communications," in IEEE International Conference on Machine Learning for Communication and Networking (ICMLCN), Stockholm, Sweden, 2024.

The initial source code is pulled from the github repository for that paper, which can be found here: https://github.com/josemateosramos/SSLISAC

On top of the reproduction, the repository implements our extension, **identifiability-guided active calibration**: choosing which target-position labels to acquire so that they constrain poorly identified antenna-spacing directions.

The team members of this project are Andrew Liang, Mohammad Al-Shammeri, and Muneef Mohammed.

## Repository layout

| Path | Contents |
|---|---|
| `lib/simulation_parameters.py` | All configuration: system model, run size (`run_mode`), angular coverage (`theta_mean_max_deg`), seeds, optimisers, data/results folders |
| `lib/functions.py` | Original SSLISAC signal model, `trainNetwork()`, sensing and ISAC test functions |
| `lib/learnable_parameters.py` | `PertNet`: the learnable antenna-spacing vector |
| `lib/scene_pool.py` | Fixed pools of scenes, measurement re-synthesis, training on a pool |
| `lib/identifiability.py` | Fisher information, nuisance elimination, weak directions, calibration error |
| `lib/acquisition.py` | The six label-acquisition rules |
| `methods/*.py` | Runnable experiment scripts (see below) |
| `methods/data/` | Stored scene pools (`scenes_<run_mode>_th<deg>_<split>.pt`) |
| `methods/results/` | Saved outputs (`<name>_<run_mode>_<timestamp>.npz`) |

## Setup (Windows, PowerShell)

Everything runs on CPU; no GPU is needed. Use Python 3.10.

```powershell
cd 867-deep-learning
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

All commands below are run **from the repository root**. The scripts locate `lib/`, `methods/data/` and `methods/results/` relative to their own files, so `PYTHONPATH` does not need to be set.

## Configuration and seeds

Settings live in `lib/simulation_parameters.py`:

* `run_mode`: `'smoke'` (about a minute, checks the code runs), `'cpu'` (hours on a CPU), `'paper'` (full published settings).
* `theta_mean_max_deg`: sector centres are drawn in `[-theta_mean_max_deg, theta_mean_max_deg]` degrees (60 in the paper, 10 in our active-calibration pilot).
* Seeds are fixed in code, so every run is deterministic for a given seed:
  * `SEED` seeds the global torch and NumPy generators when the configuration is imported. It defaults to `SEED_REPORTED = 10` (the seed of the reported baseline runs). `SEED_COURSE = 42` is the course-checklist seed. Select a seed without editing the file:
    ```powershell
    $env:ISAC_SEED = "42"     # remove with: Remove-Item Env:ISAC_SEED
    ```
  * Active calibration takes its seeds on the command line (`--seeds`); the reported pilot uses `0 1 2 3 4`, and `--seeds 42` runs the course seed.
  * Scene pools use fixed seeds 1000 (train) and 2000 (validation); held-out evaluation uses seed 12345.

## Reproducing the report

Each row gives the command that produces the table or figure. Outputs are written to `methods/results/`. The published reference values in Table III and Figs. 2-3 were read off the paper's plotted curves and are embedded in the report source, not produced by this code.

To start a new test run, use

```python methods\generate_scenes.py
python methods\identifiability_check.py
python methods\active_calibration.py --rules weak random --seeds 0
python methods\summarize_active.py --mode smoke
python methods\supervised_learning.py
python methods\baseline.py
```

### Baseline reproduction (Table III, Figs. 2 and 3)

These runs use the reproduction column of Table II. Before running, set in `lib/simulation_parameters.py`:

```python
run_mode           = 'paper'      # batch 3000, 1.2M test samples, 8x8 ISAC beam sweep
theta_mean_max_deg = 60           # sector centres in [-60, 60] deg
```
and set `train_it = int(1e4)` in the `'paper'` entry of `train_it` (10,000 updates). Note that the `StepLR` step size is computed from `train_it`.

| Report item | Command | Output |
|---|---|---|
| Table III and Fig. 2, SL curve | `python methods\supervised_learning.py` | `SL_paper_<time>.npz` (`pd_inter`, `rmse_pos_inter`, `num_iterations`) |
| Table III and Fig. 2, UL curve | `python methods\unsupervised_learning.py` | `UL_paper_<time>.npz` |
| Uncalibrated reference (Fig. 2 dashed line, Table III baseline row) | `python methods\baseline.py` | `baseline_unknown_impairments_paper_<time>.npz` |
| Fig. 3, ISAC Pareto frontiers | same three runs (fields `ser_isac`, `pd_isac`, `rmse_pos_isac`) | as above |
| Sequential SSL (pending) | `python methods\semisupervised_sl_ul.py` | `SSL_SL_UL_<n>_paper_<time>.npz` |

The misdetection probability in Table III is `1 - pd_inter` at the listed iterations.

A full-settings run processes 30 million training observations and 1.2 million test observations per evaluation, so it takes many hours on a CPU. For a quick check, use `run_mode = 'smoke'`.

### Active-calibration pilot (Tables IV-VII, Fig. 4)

These use the committed defaults (`run_mode = 'cpu'`, `theta_mean_max_deg = 10`).

| Report item | Command |
|---|---|
| Scene pools used by all pilot results | `python methods\generate_scenes.py` |
| Model checks (Sec. V-B) and Table IV, narrow coverage | `python methods\identifiability_check.py` |
| Table IV, broad coverage | set `theta_mean_max_deg = 60`, then `python methods\generate_scenes.py` and `python methods\identifiability_check.py` |
| All 30 pilot runs (6 rules x 5 seeds) | `python methods\active_calibration.py --seeds 0 1 2 3 4` |
| Fig. 4, Table VI (paired differences vs random, `paired_vs_random.csv`), SL-step diagnostic, acquisition times | `python methods\summarize_active.py --since <timestamp of your runs>` |
| Table V (final calibration error, position RMSE, median error) and Table VII (scenes queried by each rule) | see snippets below |

One pilot run takes about 12 minutes on a laptop CPU (training 711-736 s plus acquisition), so the 30 runs take about 6 hours. `identifiability_check.py` saves `identifiability_th<deg>_cpu_<time>.npz`; the eigenvalue ratios in Table IV are `eig_*[1] / eig_*[-1]`.

For the reported runs, pass `--since 20261013-165331`; runs saved before that time predate a fix to the shared starting spacing and are not valid comparisons.

Table V and the query statistics in Table VII are computed from the saved runs:

```powershell
python -c "import numpy as np, glob; [print(r, *['%.4f +/- %.4f' % (a.mean(), a.std(ddof=1)) for a in [np.array([np.load(f)[k][-1] for f in sorted(glob.glob('methods/results/active_%s_s[0-4]_th10_cpu_20261013-165331.npz' % r))]) for k in ('calib_err', 'rmse_pos', 'median_pos_err')]]) for r in ('weak', 'full', 'maxangle', 'diversity', 'uncertainty', 'random')]"
```

```powershell
python -c "import torch, numpy as np; p = torch.load('methods/data/scenes_cpu_th10_train.pt', weights_only=True)['scenes']; a = p['alpha'].abs().flatten().numpy(); t = p['theta'].abs().flatten().numpy(); [print(r, np.median(a[i]), np.degrees(np.median(t[i]))) for r in ('weak', 'full', 'maxangle', 'diversity', 'uncertainty', 'random') for i in [np.concatenate([np.load('methods/results/active_%s_s%d_th10_cpu_20261013-165331.npz' % (r, s))['chosen'] for s in range(5)])]]"
```

The second command prints, per rule, the median reflection-gain magnitude and median absolute angle of the 250 queried scenes (5 seeds x 50 queries), as in Table VII.

### Known issues (documented in the report)

* `train_it_super` equals `train_it` in `lib/simulation_parameters.py`, so `semisupervised_sl_ul.py` performs no unsupervised updates until `train_it_super` is reduced (e.g. 1,000 of 10,000).
* `active_calibration.py` stores `learned_d` as NumPy views of the parameter tensor, so every saved round repeats the final spacing. Use the scalar metrics (`calib_err`, `rmse_pos`, ...), which are recorded correctly.
* Acquisition scoring draws from the global torch generator, so rules with the same seed do not share identical training-noise streams.

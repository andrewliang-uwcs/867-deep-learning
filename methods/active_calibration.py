# -*- coding: utf-8 -*-
"""
Identifiability-guided active calibration loop (proposal Appendix A, Steps 4-5, Figure 1).

For each (rule, seed):
  1. fresh model at the assumed spacing; random initial labels L0 (same L0 for every rule
     with the same seed)
  2. initialise: SL steps on L0, then UL steps on the whole pool
  3. repeat until the label budget is used:
       estimate (receiver estimates; information matrices if the rule needs them)
       -> choose a batch of unlabelled scenes -> reveal their labels
       -> warm-start SL on all labels, then UL on the whole pool
       -> evaluate on the held-out pool and log
Every rule gets identical budgets, batch sizes and update counts.

Run from methods/ after generate_scenes.py (same run_mode and theta_mean_max_deg):
    $env:PYTHONPATH = ".."
    python active_calibration.py                          # all rules, seed 0
    python active_calibration.py --rules weak random maxangle --seeds 0 1 2

Outputs results/active_<rule>_s<seed>_th<deg>_<run_mode>_<time>.npz per run, and prints a
summary table. Evaluation uses the val pool as held-out data (position RMSE, angle RMSE)
plus calibration error against the true spacing; the learned spacing is saved at every
round so the original Pd/SER test functions can be run on any checkpoint later.
"""
import argparse
import os
import sys
import time
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../lib/')
from simulation_parameters import *
from lib.scene_pool import (SimConfig, load_scenes, num_scenes, subset, train_steps,
                            synthesize, run_receiver)
from lib.identifiability import (ObsModel, pool_information, current_estimates,
                                 calibration_error, translation_direction, complement_basis)
from lib.acquisition import RULES, NEEDS_INFO, choose

parser = argparse.ArgumentParser(description='Active calibration loop')
parser.add_argument('--rules', nargs='+', default=list(RULES), choices=RULES)
parser.add_argument('--seeds', nargs='+', type=int, default=[0])
parser.add_argument('--no-redundancy-control', action='store_true',
                    help='ablation: score the whole batch at once for weak/full')
args = parser.parse_args()

# Matched across all rules. n_cand: unlabelled candidates scored per round (random subset).
AL = {'smoke': dict(n_init=4,  per_round=2,  budget=8,   sl_steps=3,    ul_steps=3,
                    batch=16, n_cand=32,   n_eval=64,   tau=1e-3),
      'cpu':   dict(n_init=10, per_round=5,  budget=60,  sl_steps=100,  ul_steps=100,
                    batch=64, n_cand=300,  n_eval=500,  tau=1e-3),
      'paper': dict(n_init=20, per_round=10, budget=200, sl_steps=1000, ul_steps=1000,
                    batch=batch_size, n_cand=1000, n_eval=5000, tau=1e-3)}[run_mode]
lr_scale = 10     # multiplies lr_super / lr_unsuper; raise it if calibration error barely moves

sim = SimConfig.from_globals(globals())
# Private copy of the starting spacing. On CPU, PertNet(assumed_d.cpu()) shares memory with
# assumed_d, so training would overwrite it and the next run would start where the last ended.
d_init = assumed_d.detach().cpu().clone()
m = ObsModel(sim)
gauge = translation_direction(m).reshape(-1, 1)          # unfixable direction (see identifiability_check)
Q = complement_basis(gauge, K)

pool_path = os.path.join('data', f'scenes_{pool_tag}_train.pt')
eval_path = os.path.join('data', f'scenes_{pool_tag}_val.pt')
for p in (pool_path, eval_path):
    if not os.path.exists(p):
        sys.exit(f'{p} not found: run generate_scenes.py first (same run_mode / theta_mean_max_deg)')
pool, _ = load_scenes(pool_path, device)
eval_pool, _ = load_scenes(eval_path, device)
eval_pool = subset(eval_pool, list(range(min(AL['n_eval'], num_scenes(eval_pool)))))
N = num_scenes(pool)
print(f'Pool {pool_path}: {N} scenes; eval {eval_path}: {num_scenes(eval_pool)} scenes; settings {AL}')


@torch.no_grad()
def evaluate(net, scenes, batch=64, eval_seed=12345):
    """Position RMSE [m], angle RMSE [deg] and median position error [m] of the receiver on
    held-out target-present scenes. Uses the same noise/symbols every call (fixed seed), so
    differences between rounds come from the model, not from re-drawn noise. The global RNG
    state is restored afterwards so training randomness is unaffected."""
    rng_state = torch.get_rng_state()
    torch.manual_seed(eval_seed)
    se_pos, se_ang, n, errs = 0.0, 0.0, 0, []
    for start in range(0, num_scenes(scenes), batch):
        sc = subset(scenes, list(range(start, min(start + batch, num_scenes(scenes)))))
        Y, sym, _, A, b_angle = synthesize(sc, net.d, sim)
        _, th, r = run_receiver(net.d, sc, sim, Y, sym, A, b_angle)
        x_e, y_e = r * torch.cos(th), r * torch.sin(th)
        x_t, y_t = sc['range'] * torch.cos(sc['theta']), sc['range'] * torch.sin(sc['theta'])
        e2 = (x_e - x_t) ** 2 + (y_e - y_t) ** 2
        errs.append(e2.reshape(-1).sqrt().cpu())
        se_pos += float(e2.sum())
        se_ang += float(((th - sc['theta']) ** 2).sum())
        n += th.shape[0]
    torch.set_rng_state(rng_state)
    return (se_pos / n) ** 0.5, (se_ang / n) ** 0.5 * 180 / np.pi, float(torch.cat(errs).median())


def run(rule, seed):
    torch.manual_seed(seed)                               # noise / symbols during training
    g_train = torch.Generator().manual_seed(seed)         # mini-batch picks
    g_init = torch.Generator().manual_seed(10_000 + seed) # L0: identical for every rule
    net = PertNet(d_init.clone()).to(device)
    assert torch.equal(net.d.detach().cpu(), d_init), 'every run must start from the same spacing'
    opt_sl = torch.optim.Adam(net.parameters(), lr=lr_super * lr_scale)
    opt_ul = torch.optim.Adam(net.parameters(), lr=lr_unsuper * lr_scale, maximize=True)

    labelled = torch.randperm(N, generator=g_init)[:AL['n_init']].tolist()
    log = dict(n_labels=[], calib_err=[], calib_err_after_sl=[], median_pos_err=[], rmse_pos=[], rmse_angle_deg=[], acq_time=[],
               train_time=[], learned_d=[], chosen=[], chosen_round=[], chosen_true_theta=[])

    def train_and_log(acq_time):
        t0 = time.perf_counter()
        train_steps(net, pool, labelled, AL['sl_steps'], AL['batch'], opt_sl, sim,
                    supervised=True, generator=g_train)
        log['calib_err_after_sl'].append(calibration_error(net.d, ant_d, m, gauge))
        train_steps(net, pool, torch.arange(N), AL['ul_steps'], AL['batch'], opt_ul, sim,
                    supervised=False, generator=g_train)
        t_train = time.perf_counter() - t0
        rmse_pos, rmse_ang, med_pos = evaluate(net, eval_pool)
        log['median_pos_err'].append(med_pos)
        log['n_labels'].append(len(labelled)); log['acq_time'].append(acq_time)
        log['train_time'].append(t_train)
        log['calib_err'].append(calibration_error(net.d, ant_d, m, gauge))
        log['rmse_pos'].append(rmse_pos); log['rmse_angle_deg'].append(rmse_ang)
        log['learned_d'].append(net.d.detach().cpu().numpy().reshape(-1))
        print(f'  [{rule} s{seed}] labels {len(labelled):4d}  calib err {log["calib_err"][-1]:.4f} '
              f'(after SL {log["calib_err_after_sl"][-1]:.4f})  median pos err {med_pos:6.3f} m  '
              f'pos RMSE {rmse_pos:7.3f} m  angle RMSE {rmse_ang:6.3f} deg  '
              f'(acq {acq_time:5.1f}s, train {t_train:5.1f}s)')

    train_and_log(0.0)                                    # round 0: initialise on L0
    rnd = 0
    while len(labelled) < AL['budget']:
        rnd += 1
        t0 = time.perf_counter()
        labelled_set = set(labelled)
        unlabelled = [i for i in range(N) if i not in labelled_set]
        if not unlabelled:
            break
        g_cand = torch.Generator().manual_seed(seed * 1000 + rnd)
        sel = torch.randperm(len(unlabelled), generator=g_cand)[:AL['n_cand']].tolist()
        cand = [unlabelled[j] for j in sel]
        n_pick = min(AL['per_round'], AL['budget'] - len(labelled))
        labelled_theta = pool['theta'][labelled].reshape(-1).cpu()   # revealed labels only

        if rule in NEEDS_INFO:
            # information at the current d for labelled + candidates (labels not used:
            # everything is evaluated at receiver estimates)
            idx = labelled + cand
            F_U, dF, est = pool_information(net, subset(pool, idx), sim, m,
                                            batch_size=min(AL['batch'], 64))
            nL = len(labelled)
            F_current = (F_U[:nL] + dF[:nL]).sum(0) + F_U[nL:].sum(0)
            est_c = {k: v[nL:] for k, v in est.items()}
            picks = choose(rule, n_pick, cand, est_c, dF=dF[nL:], F_current=F_current, Q=Q,
                           tau=AL['tau'], redundancy_control=not args.no_redundancy_control)
        else:
            est_c = current_estimates(net, subset(pool, cand), sim, batch_size=min(AL['batch'], 64))
            picks = choose(rule, n_pick, cand, est_c, labelled_theta=labelled_theta,
                           generator=g_cand)
        acq_time = time.perf_counter() - t0

        labelled += picks                                 # oracle reveals these labels
        log['chosen'] += picks; log['chosen_round'] += [rnd] * len(picks)
        log['chosen_true_theta'] += pool['theta'][picks].reshape(-1).cpu().tolist()  # analysis only
        train_and_log(acq_time)

    suffix = '_noRC' if args.no_redundancy_control and rule in ('weak', 'full') else ''
    out_file = result_path(f'active_{rule}{suffix}_s{seed}_th{theta_mean_max_deg:g}')
    np.savez(out_file, rule=rule, seed=seed, settings=str(AL), lr_scale=lr_scale,
             pool_file=pool_path, eval_file=eval_path, labelled=np.array(labelled),
             true_d=ant_d.detach().cpu().numpy().reshape(-1),
             **{k: np.array(v) for k, v in log.items()})
    print(f'  saved {out_file}')
    return log


summary = []
for seed in args.seeds:
    for rule in args.rules:
        print(f'*** rule {rule}, seed {seed} ***')
        log = run(rule, seed)
        summary.append((rule, seed, log['n_labels'][-1], log['calib_err'][0], log['calib_err'][-1],
                        log['rmse_pos'][-1], sum(log['acq_time'])))

print('\nrule         seed  labels  calib err (start -> end)  pos RMSE [m]  acq time [s]')
for rule, seed, nl, e0, e1, rp, ta in summary:
    print(f'{rule:12s} {seed:4d}  {nl:6d}  {e0:.4f} -> {e1:.4f}          {rp:8.3f}     {ta:8.1f}')

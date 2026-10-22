# -*- coding: utf-8 -*-
"""
Sanity checks for lib/scene_pool.py and lib/identifiability.py, plus a first look at the
identifiability structure (proposal Appendix A, Step 2).

Run from the repository root after generate_scenes.py (it draws a pool itself if none is stored):
    python methods/identifiability_check.py

Checks 1-4 should all print PASS; they confirm the refactored simulator and the Fisher
model agree with the original code. Section 5 prints the spectrum and saves it to methods/results/.
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
from lib.scene_pool import (SimConfig, draw_scenes, load_scenes, subset, num_scenes,
                            precoder_from_sector, draw_symbols, synthesize)
from lib.identifiability import (ObsModel, mean_obs_complex, mean_obs_real, pack_params,
                                 jacobian, fisher_obs, info_blocks, current_estimates,
                                 scale_direction, translation_direction, rayleigh,
                                 gauge_basis, complement_basis, weak_directions,
                                 calibration_error, select_batch)

sim = SimConfig.from_globals(globals())
m = ObsModel(sim)
c128 = torch.complex128
results = {}


def report(name, value, tol):
    """Print PASS/FAIL for a relative-error check, store the value, return value < tol."""
    ok = value < tol
    print(f'[{"PASS" if ok else "FAIL"}] {name}: {value:.2e} (tol {tol:.0e})')
    results[name] = value
    return ok


# ---------------------------------------------------------------- 1. simulator equivalence
B = 4
th = torch.tensor([[-0.6], [-0.1], [0.3], [0.9]], device=device)
r = torch.tensor([[20.0], [75.0], [130.0], [190.0]], device=device)
sec_min, sec_max = th - 0.1, th + 0.1
with torch.no_grad():
    prec, _, _ = precoder_from_sector(sec_min, sec_max, network.d, sim)
    prec_b = prec.transpose(-2, -1).reshape(B, K, 1)
    symbols = draw_symbols(B, sim)
    torch.manual_seed(7)   # sensingChannel draws the complex gain first
    Y_ref, _, _ = sensingChannel(sigma_sens, th, th, prec_b, symbols, r, r,
                                 torch.tensor(0.0, device=device), torch.ones(B, 1, 1, device=device),
                                 Delta_f, lamb, ant_d, device=device)
    torch.manual_seed(7)
    alpha = noise(sigma_sens ** 2, (B, 1, 1), device).reshape(B, 1)
    scenes_chk = dict(theta_min=sec_min, theta_max=sec_max, theta=th, range_min=r, range_max=r,
                      range=r, alpha=alpha, presence=torch.ones(B, 1, device=device))
    Y_new, _, _, _, _ = synthesize(scenes_chk, network.d, sim, symbols=symbols, add_noise=False)
report('1 synthesize() == sensingChannel()',
       float((Y_new - Y_ref).abs().max() / Y_ref.abs().max()), 1e-4)

# ---------------------------------------------------------------- 2. Fisher model == simulator
d_true = ant_d.detach().cpu().double().reshape(-1)
f0 = prec_b[0, :, 0].detach().cpu().to(c128)
x0 = symbols[0, :, 0].detach().cpu().to(c128)
mu0 = mean_obs_complex(d_true, torch.tensor(float(th[0]), dtype=torch.float64),
                       torch.tensor(float(r[0]), dtype=torch.float64),
                       torch.tensor(complex(alpha[0, 0]), dtype=c128), f0, x0, m)
report('2 mean_obs_complex() == synthesize()',
       float((mu0 - Y_new[0].detach().cpu().to(c128)).abs().max() / mu0.abs().max()), 1e-4)

# ---------------------------------------------------------------- 3. Jacobian vs finite differences
p0 = pack_params(d_true, float(th[0]), float(r[0]), complex(alpha[0, 0]), m)
J = jacobian(p0, f0, x0, m)
h, worst = 1e-6, 0.0
for j in [0, K // 2, K - 1, m.TH, m.R, m.AR, m.AI]:
    e = torch.zeros_like(p0); e[j] = h
    fd = (mean_obs_real(p0 + e, f0, x0, m) - mean_obs_real(p0 - e, f0, x0, m)) / (2 * h)
    worst = max(worst, float((fd - J[:, j]).norm() / J[:, j].norm()))
report('3 jacfwd Jacobian vs finite differences', worst, 1e-5)

# ---------------------------------------------------------------- 4. exact symmetries
c = 1.003
th0 = torch.tensor(float(th[3]), dtype=torch.float64)                  # a wide angle
r0 = torch.tensor(float(r[3]), dtype=torch.float64)
a0 = torch.tensor(complex(alpha[3, 0]), dtype=c128)
f3 = prec_b[3, :, 0].detach().cpu().to(c128); x3 = symbols[3, :, 0].detach().cpu().to(c128)
mu_a = mean_obs_complex(d_true, th0, r0, a0, f3, x3, m)
mu_s = mean_obs_complex(c * d_true, torch.asin(torch.sin(th0) / c), r0, a0, f3, x3, m)
report('4a scale: (c d, asin(sin th / c)) gives the same measurement',
       float((mu_s - mu_a).abs().max() / mu_a.abs().max()), 1e-9)
delta = 1e-4
kk = torch.arange(-(K - 1) / 2.0, (K - 1) / 2.0 + 1, dtype=torch.float64)
mu_t = mean_obs_complex(d_true + delta / kk, th0, r0, a0, f3, x3, m)
phase = torch.exp(torch.tensor(-2j * 2 * np.pi / m.lamb * delta, dtype=c128) * torch.sin(th0))
report('4b translation: shifted array = same measurement times a phase (absorbed by alpha)',
       float((mu_t - phase * mu_a).abs().max() / mu_a.abs().max()), 1e-9)

# ---------------------------------------------------------------- 5. pool information spectrum
n_analysis = {'smoke': 32, 'cpu': 300, 'paper': 1000}[run_mode]
path = os.path.join(DATA_DIR, f'scenes_{pool_tag}_train.pt')
if os.path.exists(path):
    pool, _ = load_scenes(path, device)
    print(f'\nUsing stored pool {path}')
else:
    pool = draw_scenes(n_analysis, sim, theta_mean_min, theta_mean_max, span_min_theta,
                       span_max_theta, range_mean_min, range_mean_max, span_min_range,
                       span_max_range, p_present=1.0, seed=1000)
    print('\nNo stored pool found (run generate_scenes.py); drew a temporary one')
pool = subset(pool, list(range(min(n_analysis, num_scenes(pool)))))
N = num_scenes(pool)
print(f'Computing per-observation information for {N} scenes at the current network.d ...')

est = current_estimates(network, pool, sim, batch_size=min(batch_size, 64))
d_hat = network.d.detach().cpu().double().reshape(-1)
span = (pool['theta_max'] - pool['theta_min']).detach().cpu().double().reshape(-1)
F_U = torch.empty(N, K, K, dtype=torch.float64)
F_U_np = torch.empty_like(F_U)          # without the angular-sector prior
dF = torch.empty_like(F_U)
for i in range(N):
    p = pack_params(d_hat, est['theta'][i], est['range'][i], est['alpha'][i], m)
    I = fisher_obs(p, est['precoder'][i].to(c128), est['symbols'][i].to(c128), m)
    F_U[i], _, dF[i] = info_blocks(I, m, 12.0 / float(span[i]) ** 2)
    F_U_np[i], _, _ = info_blocks(I, m, 0.0)
    if (i + 1) % max(1, N // 5) == 0:
        print(f'  {i + 1}/{N}')

FU_tot, FUnp_tot = F_U.sum(0), F_U_np.sum(0)
FL_tot = F_U.sum(0) + dF.sum(0)
null = gauge_basis(FL_tot)
Q = complement_basis(null, K)
v_scale, v_trans = scale_direction(d_hat, m), translation_direction(m)


def smallest(F, n=4):
    """The n smallest eigenvalues of F divided by its largest eigenvalue."""
    w = torch.linalg.eigvalsh(F)
    return (w[:n] / w[-1]).numpy()


print('\nSmallest eigenvalues / largest (unlabelled = position unknown, labelled = known):')
print('  sum F_U, no sector prior :', np.array2string(smallest(FUnp_tot), precision=2))
print('  sum F_U, sector prior    :', np.array2string(smallest(FU_tot), precision=2))
print('  sum F_L (all labelled)   :', np.array2string(smallest(FL_tot), precision=2))
print(f'Gauge (unfixable) dimension found: {null.shape[1]}  '
      f'(overlap with translation direction: '
      f'{float((null.T @ v_trans).norm()) if null.shape[1] else 0.0:.3f})')
lam_max = float(torch.linalg.eigvalsh(FL_tot)[-1])
print(f'Absolute scale (compare across coverage settings): largest eig of sum F_L {lam_max:.3e}; '
      f'scale info F_U(prior) {rayleigh(FU_tot, v_scale):.3e}, F_L {rayleigh(FL_tot, v_scale):.3e}; '
      f'mean sin^2(theta_hat) {float(est["theta"].double().sin().pow(2).mean()):.3f}')
print('Rayleigh quotients / largest eigenvalue of sum F_L:')
for name, v in [('scale', v_scale), ('translation', v_trans)]:
    print(f'  {name:11s}: F_U(no prior) {rayleigh(FUnp_tot, v) / lam_max:.2e}   '
          f'F_U(prior) {rayleigh(FU_tot, v) / lam_max:.2e}   F_L {rayleigh(FL_tot, v) / lam_max:.2e}')

V_weak, _ = weak_directions(FU_tot, Q, tau=1e-3)
print(f'Weak directions of sum F_U (tau=1e-3): {V_weak.shape[1]}; '
      f'overlap with scale direction: {float((V_weak.T @ v_scale).norm()):.3f}')

gain_scale = torch.einsum('k,ikl,l->i', v_scale, dF, v_scale).numpy()
abs_sin = est['theta'].abs().sin().numpy()
corr = float(np.corrcoef(gain_scale, abs_sin)[0, 1]) if N > 2 else float('nan')
print(f'Corr(label gain in scale direction, |sin theta_hat|): {corr:.3f}  (expected clearly > 0)')

# Demo of the acquisition step: first few scenes labelled, pick 5 more
n0 = min(4, N - 1)
F_now = dF[:n0].sum(0) + F_U.sum(0)            # sum F_L over L + sum F_U over U
cand = list(range(n0, N))
for rule in ('weak', 'full'):
    pick = select_batch(F_now, dF, cand, 5, Q, rule=rule)
    print(f'select_batch({rule!r}) -> {pick}; |theta_hat| deg = '
          f'{np.round(est["theta"][pick].abs().numpy() * 180 / np.pi, 1)}')

print(f'Calibration error of current network.d (gauge removed): '
      f'{calibration_error(network.d, ant_d, m, null):.4f} (units of lambda/2)')

out_file = result_path(f'identifiability_th{theta_mean_max_deg:g}')
print(f'Saving results to {out_file}')
np.savez(out_file, check_names=np.array(list(results.keys())), check_values=np.array(list(results.values())),
         eig_FU_noprior=torch.linalg.eigvalsh(FUnp_tot).numpy(),
         eig_FU=torch.linalg.eigvalsh(FU_tot).numpy(), eig_FL=torch.linalg.eigvalsh(FL_tot).numpy(),
         gain_scale=gain_scale, abs_sin_theta_hat=abs_sin, gauge_basis=null.numpy(),
         theta_hat=est['theta'].numpy(), range_hat=est['range'].numpy())

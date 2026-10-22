# -*- coding: utf-8 -*-
"""
Supervised learning (SL) baseline of Mateos-Ramos et al. (ICMLCN 2024).

Trains the learnable antenna spacing (PertNet, lib/learnable_parameters.py) for train_it
iterations by minimising the target-position MSE on freshly simulated, labelled batches,
evaluates the misdetection probability and position RMSE at the iterations in
epoch_test_list, then sweeps the ISAC trade-off beam and measures the symbol error rate.
All settings come from lib/simulation_parameters.py (run_mode, seed, sizes).

Run from the repository root:
    python methods/supervised_learning.py
Writes methods/results/SL_<run_mode>_<timestamp>.npz (pd_inter, rmse_pos_inter,
num_iterations, loss_super, *_isac, learned_d).
"""
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
# Put the repository root and lib/ on the import path relative to this file, so the
# script runs from any working directory without setting PYTHONPATH.
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[1:1] = [REPO_ROOT, os.path.join(REPO_ROOT, 'lib')]
from simulation_parameters import *

file_name = 'SL'

"""# Training and intermediate testing"""

loss_super, _, num_iterations, pd_inter, pfa_inter, \
    rmse_angle_inter, rmse_range_inter, \
    rmse_pos_inter = trainNetwork(network, 0, train_it, sigma_sens, theta_mean_min, theta_mean_max, span_min_theta,
                                  span_max_theta, range_mean_min, range_mean_max, span_min_range,
                                  span_max_range, K, S, Delta_f, lamb, Ngrid_angle,
                                  range_grid, pixels_angle, pixels_range, msg_card, refConst, N0,
                                  ant_d, batch_size, optimizerSuper, scheduler_super, criterionLoss,
                                  epoch_test_list, theta_min_sens_test, theta_max_sens_test,
                                  range_min_sens_test, range_max_sens_test, target_pfa, delta_pfa,
                                  thresholds_pfa, nTestSamples, SL_flag=True, sch_flag=True, device=device)

"""## Plot performance as a function of the number of iterations"""

plt.figure()
plt.semilogy(num_iterations, 1-np.array(pd_inter), 'o-', label='Supervised learning (SL)')
plt.grid()
plt.xlabel('Number of training iterations')
plt.ylabel('Misdetection probability')
plt.legend()

"""# Test ISAC"""

pd_isac, pfa_isac, rmse_angle_isac, rmse_range_isac, rmse_pos_isac, ser_isac = \
    testNetworkISAC(sigma_sens, sigma_vector_comm, theta_min_sens_test, theta_max_sens_test, theta_min_comm_test,
                    theta_max_comm_test, range_min_sens_test,
                    range_max_sens_test, Ngrid_angle, range_grid, pixels_angle, pixels_range,
                    K, S, N0, Delta_f, lamb, ant_d, network.d, refConst,
                    rho, phi, target_pfa, delta_pfa, thresholds_pfa, batch_size, nTestSamples, device)

"""### Plot ISAC results"""

plt.figure()
plt.semilogx(ser_isac, 1-np.array(pd_isac), 'o-', label=f'Supervised learning (SL)')
plt.grid()
plt.xlabel('Symbol error rate')
plt.ylabel('Misdetection probability')
plt.legend()

plt.figure()
plt.semilogx(ser_isac, rmse_pos_isac, 'o-', label=f'Supervised learning (SL)')
plt.grid()
plt.xlabel('Symbol error rate')
plt.ylabel('Position RMSE [m]')
plt.legend()

"""# Save results"""

out_file = result_path(file_name)
print(f'Saving results to {out_file}')
np.savez(out_file, \
        pd_inter = pd_inter, pfa_inter = pfa_inter, \
        rmse_angle_inter = rmse_angle_inter, \
        rmse_range_inter = rmse_range_inter, \
        rmse_pos_inter = rmse_pos_inter, \
        num_iterations = num_iterations, loss_super = loss_super, \
        pd_isac = pd_isac, pfa_isac = pfa_isac, \
        rmse_angle_isac = rmse_angle_isac, rmse_range_isac = rmse_range_isac, \
        rmse_pos_isac = rmse_pos_isac, ser_isac = ser_isac, \
        learned_d = network.d.cpu().detach().numpy()
        )
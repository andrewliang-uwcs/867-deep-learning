# -*- coding: utf-8 -*-
"""
Label-acquisition rules for active calibration (proposal Appendix A, Steps 3-4, plus the
comparison methods from the evaluation plan). Every rule sees only receiver estimates
for unlabelled candidates and the revealed labels of already-labelled scenes, never
the hidden labels.

Rules
  weak        : proposal eq. (1), tr(V^T dF_i V) over weak, label-fixable directions,
                greedy with redundancy control
  full        : D-optimal log det gain on the gauge-free subspace ('full-information' ablation)
  random      : uniform at random
  maxangle    : largest |theta_hat| (the simple heuristic the identifiability check
                suggests, since label information about d grows with sin^2 theta)
  diversity   : farthest-point on theta_hat, measured against angles already labelled
  uncertainty : largest position CRB (target-prediction uncertainty)
"""
import torch

from lib.identifiability import select_batch

RULES = ('weak', 'full', 'random', 'maxangle', 'diversity', 'uncertainty')
NEEDS_INFO = ('weak', 'full', 'uncertainty')     # need pool_information() each round


def choose(rule, n, cand, est, dF=None, F_current=None, Q=None, labelled_theta=None,
           tau=1e-3, redundancy_control=True, generator=None):
    """Pick n scenes from cand (list of pool indices).
    est, dF are aligned with cand: row j describes pool scene cand[j].
    F_current: sum of F_L over labelled + sum of F_U over candidates (weak/full only).
    labelled_theta: revealed angles of labelled scenes (diversity only).
    Returns a list of pool indices."""
    n = min(n, len(cand))
    if n <= 0:
        return []
    if rule == 'random':
        picks = torch.randperm(len(cand), generator=generator)[:n].tolist()
    elif rule == 'maxangle':
        picks = torch.argsort(est['theta'].abs(), descending=True)[:n].tolist()
    elif rule == 'uncertainty':
        picks = torch.argsort(est['pos_crb'], descending=True)[:n].tolist()
    elif rule == 'diversity':
        theta_hat = est['theta'].double()
        ref = [float(t) for t in (labelled_theta if labelled_theta is not None else [])]
        remaining, picks = list(range(len(cand))), []
        for _ in range(n):
            if ref:
                ref_t = torch.tensor(ref, dtype=torch.float64)
                dist = (theta_hat[remaining].view(-1, 1) - ref_t.view(1, -1)).abs().min(dim=1).values
                j = remaining[int(torch.argmax(dist))]
            else:
                j = remaining[int(torch.argmax(theta_hat[remaining].abs()))]
            picks.append(j); remaining.remove(j); ref.append(float(theta_hat[j]))
    elif rule in ('weak', 'full'):
        picks = select_batch(F_current, dF, list(range(len(cand))), n, Q, rule=rule, tau=tau,
                             redundancy_control=redundancy_control)
    else:
        raise ValueError(f'unknown rule {rule!r}; choose from {RULES}')
    return [cand[j] for j in picks]

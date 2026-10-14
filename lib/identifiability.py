# -*- coding: utf-8 -*-
"""
Identifiability analysis and information-based label scoring (proposal Appendix A,
Steps 2-3).

Measurement model for one target-present observation (same as sensingChannel):
    mu(d, theta, r, alpha) = alpha/sqrt(S) * (a^T f) * a (x * rho)^T,
    a_k = exp(-j 2pi/lamb * d_k k sin(theta)),   rho_s = exp(-j 2pi s Delta_f 2r/c),
with f the precoder that was actually transmitted (known, held fixed) and x the symbols.
Noise is CN(0, N0), so the Fisher information is I = (2/N0) J^T J for the Jacobian J of
the real-stacked mean.

Parameter vector (normalised so the blocks have comparable scales):
    p = [ d / d_scale (K) , theta , r / r_scale , Re(alpha) , Im(alpha) ]

Per observation:
    F_U  = Schur complement of I removing (theta, r, alpha)  -> position unknown
    F_L  = Schur complement of I removing (alpha) only       -> position labelled
    dF   = F_L - F_U  (>= 0): extra impairment information the label would add.

Two directions in d-space are worth knowing about (both checked in
methods/identifiability_check.py):
  * scale:       d -> c d together with sin(theta) -> sin(theta)/c leaves a unchanged, so it
                 is (near) unidentifiable WITHOUT labels; labels at large |sin theta| fix it.
  * translation: element positions p_k = d_k k shifted by delta multiply mu by a phase that
                 alpha absorbs (Delta d_k ∝ 1/k). Unidentifiable even WITH labels, and harmless
                 for positioning: a gauge direction that is projected out.

Everything here runs on CPU in float64. Nothing reads the hidden labels: estimates come
from the receiver at the current network.d.
"""
import math

import numpy as np
import torch
from torch.func import jacfwd

from lib.scene_pool import (num_scenes, subset, synthesize, run_receiver, steering,
                            range_phases, unit_gain_reflection)


class ObsModel:
    """Scalar constants of the measurement model as Python floats (float64 maths)."""
    def __init__(self, sim, d_scale=None, r_scale=None):
        self.K, self.S = int(sim.K), int(sim.S)
        self.lamb = float(sim.lamb)
        self.Delta_f = float(sim.Delta_f)
        self.N0 = float(sim.N0)
        self.d_scale = float(d_scale) if d_scale is not None else self.lamb / 2.0
        # range resolution c/(2 S Delta_f) as the natural range unit
        self.r_scale = float(r_scale) if r_scale is not None else 3e8 / (2 * self.S * self.Delta_f)
        # indices into p
        self.D = list(range(self.K))
        self.TH, self.R, self.AR, self.AI = self.K, self.K + 1, self.K + 2, self.K + 3


def mean_obs_complex(d, theta, rng, alpha, f, x, m):
    """Noise-free measurement (K,S) for one observation. d (K,), theta/rng real scalars,
    alpha complex scalar, f (K,) and x (S,) complex."""
    K, S = m.K, m.S
    k = torch.arange(-(K - 1) / 2.0, (K - 1) / 2.0 + 1, dtype=d.dtype)
    a = torch.exp(-1j * (2 * math.pi / m.lamb) * d * k * torch.sin(theta))
    s = torch.arange(S, dtype=d.dtype)
    rho = torch.exp(-1j * (2 * math.pi * m.Delta_f * 2 / 3e8) * s * rng)
    return alpha / math.sqrt(S) * (a @ f) * torch.outer(a, x * rho)


def mean_obs_real(p, f, x, m):
    """Real-stacked mean as a function of the normalised parameter vector p."""
    K = m.K
    mu = mean_obs_complex(p[:K] * m.d_scale, p[m.TH], p[m.R] * m.r_scale,
                          p[m.AR] + 1j * p[m.AI], f, x, m)
    return torch.cat([mu.real.reshape(-1), mu.imag.reshape(-1)])


def pack_params(d, theta, rng, alpha, m):
    d = torch.as_tensor(d, dtype=torch.float64).reshape(-1)
    alpha = complex(alpha)
    return torch.cat([d / m.d_scale,
                      torch.tensor([float(theta), float(rng) / m.r_scale,
                                    alpha.real, alpha.imag], dtype=torch.float64)])


def jacobian(p, f, x, m):
    """d mean_obs_real / d p, shape (2KS, K+4). Forward mode: few inputs, many outputs."""
    return jacfwd(lambda q: mean_obs_real(q, f, x, m))(p)


def fisher_obs(p, f, x, m):
    """Fisher information of one observation w.r.t. p, shape (K+4, K+4)."""
    J = jacobian(p, f, x, m)
    return (2.0 / m.N0) * (J.T @ J)


def schur(I, keep, nuis, ridge=1e-12):
    """Effective information for `keep` after eliminating nuisance parameters `nuis`."""
    A = I[keep][:, keep]
    B = I[keep][:, nuis]
    C = I[nuis][:, nuis]
    C = C + ridge * torch.trace(C) / len(nuis) * torch.eye(len(nuis), dtype=I.dtype)
    F = A - B @ torch.linalg.solve(C, B.T)
    return 0.5 * (F + F.T)


def info_blocks(I, m, theta_prior_prec=0.0):
    """(F_U, F_L, dF) for one observation. theta_prior_prec adds the angular-sector prior
    (uniform on [theta_min, theta_max] -> precision 12/span^2) to the unlabelled case."""
    I_u = I.clone()
    I_u[m.TH, m.TH] += theta_prior_prec
    F_U = schur(I_u, m.D, [m.TH, m.R, m.AR, m.AI])
    F_L = schur(I, m.D, [m.AR, m.AI])
    return F_U, F_L, F_L - F_U


# ----------------------------------------------------------------------------------------
# Plug-in estimates for a pool (no hidden labels used)
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def current_estimates(network, scenes, sim, batch_size=64):
    """Synthesise each scene with the current precoder, run the receiver, and fit alpha by
    least squares under the model's belief d = network.d.
    Returns dict of CPU tensors: theta (N,), range (N,), alpha (N,) complex,
    precoder (N,K) complex, symbols (N,S) complex, metric (N,)."""
    out = {k: [] for k in ('theta', 'range', 'alpha', 'precoder', 'symbols', 'metric')}
    N = num_scenes(scenes)
    for start in range(0, N, batch_size):
        sc = subset(scenes, list(range(start, min(start + batch_size, N))))
        Y, sym, prec, A, b_angle = synthesize(sc, network.d, sim)
        metric, th, r = run_receiver(network.d, sc, sim, Y.clone(), sym, A, b_angle)
        g = unit_gain_reflection(steering(th, network.d, sim), prec, sym, range_phases(r, sim), sim.S)
        alpha = (g.conj() * Y).sum(dim=(1, 2)) / (g.abs() ** 2).sum(dim=(1, 2))
        out['theta'].append(th.reshape(-1)); out['range'].append(r.reshape(-1))
        out['alpha'].append(alpha); out['precoder'].append(prec.reshape(prec.shape[0], -1))
        out['symbols'].append(sym.reshape(sym.shape[0], -1)); out['metric'].append(metric.reshape(-1))
    return {k: torch.cat(v).detach().cpu() for k, v in out.items()}


def pool_information(network, scenes, sim, m=None, batch_size=64, use_theta_prior=True,
                     estimates=None, print_every=0):
    """Per-observation F_U and dF for every scene in `scenes`, evaluated at the current
    network.d and the receiver's estimates. Returns (F_U (N,K,K), dF (N,K,K), estimates).
    estimates also gets 'pos_crb' (N,): Cramer-Rao bound on the target position [m^2] given
    the current d (alpha unknown) -- the target-prediction uncertainty used by the
    'uncertainty' acquisition baseline."""
    m = m or ObsModel(sim)
    est = estimates if estimates is not None else current_estimates(network, scenes, sim, batch_size)
    d_hat = network.d.detach().cpu().double().reshape(-1)
    span = (scenes['theta_max'] - scenes['theta_min']).detach().cpu().double().reshape(-1)
    N = num_scenes(scenes)
    F_U = torch.empty(N, m.K, m.K, dtype=torch.float64)
    dF = torch.empty_like(F_U)
    pos_crb = torch.empty(N, dtype=torch.float64)
    for i in range(N):
        p = pack_params(d_hat, est['theta'][i], est['range'][i], est['alpha'][i], m)
        I = fisher_obs(p, est['precoder'][i].to(torch.complex128),
                       est['symbols'][i].to(torch.complex128), m)
        prior = 12.0 / float(span[i]) ** 2 if use_theta_prior else 0.0
        F_U[i], _, dF[i] = info_blocks(I, m, prior)
        pos_crb[i] = position_crb(I, float(est['theta'][i]), float(est['range'][i]), m)
        if print_every and (i + 1) % print_every == 0:
            print(f'  information for {i + 1}/{N} observations')
    est['pos_crb'] = pos_crb
    return F_U, dF, est


def position_crb(I, theta, rng, m, ridge=1e-12):
    """CRB on the (x, y) target position [m^2] from one observation's Fisher matrix I,
    treating d as known and alpha as nuisance."""
    F = schur(I, [m.TH, m.R], [m.AR, m.AI])
    F = F + ridge * torch.trace(F) * torch.eye(2, dtype=F.dtype)
    C = torch.linalg.inv(F)                       # covariance of (theta, r / r_scale)
    Jp = torch.tensor([[-rng * math.sin(theta), m.r_scale * math.cos(theta)],
                       [rng * math.cos(theta),  m.r_scale * math.sin(theta)]], dtype=F.dtype)
    return float(torch.trace(Jp @ C @ Jp.T))


# ----------------------------------------------------------------------------------------
# Special directions, weak directions, calibration error
# ----------------------------------------------------------------------------------------
def scale_direction(d, m):
    """Unit vector (normalised coords) for d -> (1+eps) d."""
    v = torch.as_tensor(d, dtype=torch.float64).reshape(-1) / m.d_scale
    return v / v.norm()


def translation_direction(m):
    """Unit vector for shifting all element positions d_k k by the same delta (Delta d_k ∝ 1/k)."""
    k = torch.arange(-(m.K - 1) / 2.0, (m.K - 1) / 2.0 + 1, dtype=torch.float64)
    v = 1.0 / k
    return v / v.norm()


def rayleigh(F, v):
    return float(v @ F @ v)


def gauge_basis(F_L_total, rel_tol=1e-9):
    """Orthonormal basis of the numerical null space of the FULLY LABELLED information,
    i.e. directions no label can fix (expected: the translation direction)."""
    w, V = torch.linalg.eigh(0.5 * (F_L_total + F_L_total.T))
    return V[:, w < rel_tol * w.max()]


def complement_basis(null_basis, K):
    """Orthonormal basis Q of the orthogonal complement of null_basis."""
    if null_basis.shape[1] == 0:
        return torch.eye(K, dtype=torch.float64)
    P = torch.eye(K, dtype=torch.float64) - null_basis @ null_basis.T
    w, V = torch.linalg.eigh(P)
    return V[:, w > 0.5]


def weak_directions(F, Q, tau=1e-3):
    """Eigenvectors of F restricted to span(Q) (gauge removed) whose eigenvalue is below
    tau * the largest. Returns (V (K, n_weak), eigenvalues of the restricted F)."""
    Fc = Q.T @ F @ Q
    w, Vc = torch.linalg.eigh(0.5 * (Fc + Fc.T))
    sel = w < tau * w.max()
    if not bool(sel.any()):
        sel[0] = True                      # always keep at least the weakest direction
    return Q @ Vc[:, sel], w


def calibration_error(d_hat, d_true, m, null_basis=None):
    """|| d_hat - d_true || in normalised units, ignoring gauge (unfixable) directions."""
    e = (torch.as_tensor(d_hat).detach().cpu().double().reshape(-1)
         - torch.as_tensor(d_true).detach().cpu().double().reshape(-1)) / m.d_scale
    if null_basis is not None and null_basis.shape[1] > 0:
        e = e - null_basis @ (null_basis.T @ e)
    return float(e.norm())


# ----------------------------------------------------------------------------------------
# Acquisition scores (proposal eq. (1)) and greedy batch selection
# ----------------------------------------------------------------------------------------
def score_weak(dF, V):
    """s_i = tr(V^T dF_i V) for every candidate. dF (N,K,K), V (K,n)."""
    return torch.einsum('kn,ikl,ln->i', V, dF, V)


def score_full_information(dF, F, Q, ridge=1e-9):
    """D-optimal gain log det(F + dF_i) - log det(F) on the gauge-free subspace
    (the 'full-information' ablation / conventional information-based baseline)."""
    Fc = Q.T @ F @ Q
    Fc = Fc + ridge * torch.trace(Fc) / Fc.shape[0] * torch.eye(Fc.shape[0], dtype=F.dtype)
    base = torch.logdet(Fc)
    return torch.stack([torch.logdet(Fc + Q.T @ dF[i] @ Q) - base for i in range(dF.shape[0])])


def select_batch(F_current, dF, candidates, n_select, Q, rule='weak', tau=1e-3,
                 redundancy_control=True):
    """Greedy batch selection.
    F_current : current pool information sum(F_L over labelled) + sum(F_U over unlabelled).
    dF        : (N,K,K) per-observation label gains (indexed by pool position).
    candidates: list of pool indices still unlabelled.
    rule      : 'weak' (eq. (1)) or 'full' (D-optimal).
    With redundancy_control, F is updated after each pick so near-duplicates lose score."""
    F = F_current.clone()
    cand = list(candidates)
    chosen = []
    for _ in range(min(n_select, len(cand))):
        idx = torch.as_tensor(cand, dtype=torch.long)
        if rule == 'weak':
            V, _ = weak_directions(F, Q, tau)
            s = score_weak(dF[idx], V)
        elif rule == 'full':
            s = score_full_information(dF[idx], F, Q)
        else:
            raise ValueError(f'unknown rule {rule}')
        best = cand[int(torch.argmax(s))]
        chosen.append(best)
        cand.remove(best)
        if redundancy_control:
            F = F + dF[best]
    return chosen

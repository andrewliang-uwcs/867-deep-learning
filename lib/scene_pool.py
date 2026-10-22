# -*- coding: utf-8 -*-
"""
Fixed pools of simulated ISAC sensing scenes.

The original code (lib/functions.py -> trainNetwork) draws a brand-new random batch at
every training iteration. Active label acquisition needs a FIXED pool of observations
whose position labels stay hidden until queried, so this module separates:

  * the random *scene* (angular/range sector, true angle, true range, complex gain,
    target presence)  -> drawn once by draw_scenes() and stored with save_scenes(), and
  * the *measurement* Y -> re-synthesised from a scene with synthesize() every time the
    scene is used.

Why re-synthesise instead of storing Y: the transmit precoder is built from the learnable
spacing network.d, so a measurement recorded under an old estimate of d goes stale. The
position label belongs to the scene, not to a particular measurement, so we keep the scene
fixed and re-transmit with the current precoder. Communication symbols and receiver noise
are drawn fresh on every synthesis (PSK symbols cancel in the receiver; the noise is not
part of the scene). Use the same convention for every method you compare.

lib/functions.py is NOT modified, so the original reproduction scripts keep working.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch

from lib.functions import noise, steeringMatrix, sensingReceiver, createMessages

SCENE_KEYS = ('theta_min', 'theta_max', 'theta', 'range_min', 'range_max', 'range',
              'alpha', 'presence')


@dataclass
class SimConfig:
    """Physical/receiver constants needed to synthesise and process scenes."""
    K: int
    S: int
    Delta_f: torch.Tensor
    lamb: torch.Tensor
    N0: torch.Tensor
    sigma_sens: torch.Tensor
    ant_d: torch.Tensor          # TRUE (impaired) inter-antenna spacings, shape (K,1)
    refConst: torch.Tensor
    Ngrid_angle: int
    range_grid: torch.Tensor
    pixels_angle: int
    pixels_range: int
    device: str = 'cpu'

    @classmethod
    def from_globals(cls, g):
        """Build from the namespace created by `from simulation_parameters import *`,
        e.g. SimConfig.from_globals(globals())."""
        return cls(K=g['K'], S=g['S'], Delta_f=g['Delta_f'], lamb=g['lamb'], N0=g['N0'],
                   sigma_sens=g['sigma_sens'], ant_d=g['ant_d'], refConst=g['refConst'],
                   Ngrid_angle=g['Ngrid_angle'], range_grid=g['range_grid'],
                   pixels_angle=g['pixels_angle'], pixels_range=g['pixels_range'],
                   device=g['device'])


# ----------------------------------------------------------------------------------------
# Drawing, storing and indexing scenes
# ----------------------------------------------------------------------------------------
def draw_scenes(n, sim, theta_mean_min, theta_mean_max, span_min_theta, span_max_theta,
                range_mean_min, range_mean_max, span_min_range, span_max_range,
                p_present=1.0, seed=0):
    """Draw n scenes with the same distributions as trainNetwork/sensingChannel:
    sector = mean +/- span/2 with mean, span uniform; theta, range uniform in the sector;
    alpha ~ CN(0, sigma_sens^2); presence ~ Bernoulli(p_present).
    Uses its own generator, so a pool is reproducible from its seed alone.
    Returns a dict of tensors, each with first dimension n."""
    g = torch.Generator(device='cpu').manual_seed(int(seed))

    def U(lo, hi):
        """n draws uniform on [lo, hi), shape (n,1), from the pool's own generator."""
        return torch.rand(n, 1, generator=g) * (float(hi) - float(lo)) + float(lo)

    th_mean, th_span = U(theta_mean_min, theta_mean_max), U(span_min_theta, span_max_theta)
    theta_min, theta_max = th_mean - th_span / 2.0, th_mean + th_span / 2.0
    r_mean, r_span = U(range_mean_min, range_mean_max), U(span_min_range, span_max_range)
    range_min, range_max = r_mean - r_span / 2.0, r_mean + r_span / 2.0

    theta = theta_min + torch.rand(n, 1, generator=g) * (theta_max - theta_min)
    rng = range_min + torch.rand(n, 1, generator=g) * (range_max - range_min)
    std = math.sqrt(float(sim.sigma_sens) ** 2 / 2.0)
    alpha = std * torch.complex(torch.randn(n, 1, generator=g), torch.randn(n, 1, generator=g))
    presence = (torch.rand(n, 1, generator=g) < p_present).float()

    scenes = dict(theta_min=theta_min, theta_max=theta_max, theta=theta,
                  range_min=range_min, range_max=range_max, range=rng,
                  alpha=alpha, presence=presence)
    return {k: v.to(sim.device) for k, v in scenes.items()}


def save_scenes(scenes, path, meta=None):
    """Save a scene pool (plus a dict of simple metadata) to a .pt file."""
    torch.save({'scenes': {k: v.detach().cpu() for k, v in scenes.items()},
                'meta': dict(meta or {})}, path)


def load_scenes(path, device='cpu'):
    """Load a pool saved by save_scenes. Returns (scenes, meta)."""
    obj = torch.load(path, map_location=device, weights_only=True)
    return {k: v.to(device) for k, v in obj['scenes'].items()}, obj.get('meta', {})


def num_scenes(scenes):
    """Number of scenes in a pool dict."""
    return scenes['theta'].shape[0]


def subset(scenes, idx):
    """Scenes at the given indices (list, numpy array or LongTensor)."""
    idx = torch.as_tensor(idx, dtype=torch.long, device=scenes['theta'].device)
    return {k: v[idx] for k, v in scenes.items()}


# ----------------------------------------------------------------------------------------
# Deterministic signal model (same equations as lib/functions.py, without random draws)
# ----------------------------------------------------------------------------------------
def steering(theta, d, sim):
    """Steering vectors e^{-j 2pi/lamb d_k k sin(theta)}. theta (B,1) -> (B,K,1)."""
    K = sim.K
    k = torch.arange(-(K - 1) / 2.0, (K - 1) / 2.0 + 1, device=sim.device).view(1, K, 1)
    pos = d.reshape(1, K, 1) * k                                   # element positions d_k * k
    return torch.exp(-1j * 2 * np.pi / sim.lamb * pos * torch.sin(theta).reshape(-1, 1, 1))


def range_phases(rng, sim):
    """Subcarrier phase shifts e^{-j 2pi s Delta_f 2R/c}. rng (B,1) -> (B,S,1)."""
    s = torch.arange(sim.S, device=sim.device, dtype=torch.float32).view(1, sim.S, 1)
    return torch.exp(-1j * 2 * np.pi * sim.Delta_f * 2 / 3e8 * s * rng.reshape(-1, 1, 1))


def precoder_from_sector(theta_min, theta_max, d, sim):
    """createPrecoder() for GIVEN sectors (no random draw).
    Returns precoder (K,B) with unit-norm columns, A (K,Ngrid), b_matrix (Ngrid,B) in {0,1}."""
    K = sim.K
    angle_vector = torch.linspace(-np.pi / 2, np.pi / 2, sim.Ngrid_angle, device=sim.device)
    b_matrix = K * ((angle_vector >= theta_min) & (angle_vector <= theta_max)) \
        .type(torch.cfloat).transpose(1, 0)
    # steeringMatrix with theta_min == theta_max returns exactly angle_vector (its rand() is
    # multiplied by zero), matching createPrecoder.
    A, _ = steeringMatrix(angle_vector, angle_vector, K, d, sim.lamb, device=sim.device)
    precoder = torch.linalg.lstsq(A.transpose(1, 0), b_matrix).solution
    precoder = precoder / torch.sqrt(torch.sum(torch.abs(precoder) ** 2, dim=0, keepdim=True))
    return precoder, A, b_matrix / K


def draw_symbols(batch_size, sim):
    """Random PSK symbols, shape (B,S,1)."""
    msg = createMessages(len(sim.refConst), num=batch_size * sim.S, device=sim.device)
    return sim.refConst[msg].reshape(batch_size, sim.S, 1)


def unit_gain_reflection(a, precoder, symbols, rho, S):
    """1/sqrt(S) * a a^T f (x * rho)^T for alpha = 1. Shapes (B,K,1),(B,K,1),(B,S,1),(B,S,1)."""
    return 1 / np.sqrt(S) * a @ a.permute(0, 2, 1) @ precoder @ (symbols * rho).permute(0, 2, 1)


def synthesize(scenes, d_tx, sim, symbols=None, add_noise=True):
    """Measurement for each scene, transmitting with a precoder designed from d_tx
    (normally network.d) over the TRUE array sim.ant_d. Same equation as sensingChannel.
    Returns Y (B,K,S), symbols (B,S,1), precoder (B,K,1), A (K,Ngrid), b_angle (Ngrid,B)."""
    B = num_scenes(scenes)
    precoder, A, b_angle = precoder_from_sector(scenes['theta_min'], scenes['theta_max'], d_tx, sim)
    precoder = precoder.transpose(-2, -1).reshape(B, sim.K, 1)
    if symbols is None:
        symbols = draw_symbols(B, sim)
    a = steering(scenes['theta'], sim.ant_d, sim)
    rho = range_phases(scenes['range'], sim)
    Y = scenes['alpha'].reshape(B, 1, 1) * unit_gain_reflection(a, precoder, symbols, rho, sim.S)
    Y = Y * scenes['presence'].reshape(B, 1, 1)
    if add_noise:
        Y = Y + noise(sim.N0, (B, sim.K, sim.S), device=sim.device)
    return Y, symbols, precoder, A, b_angle


def run_receiver(d_rx, scenes, sim, Y, symbols, A, b_angle):
    """Apply the original sensingReceiver. NOTE: it divides Y by the symbols in place,
    so pass Y.clone() if you still need Y afterwards. Returns metric, est_angle, est_range (B,1)."""
    B = num_scenes(scenes)
    b_range = ((sim.range_grid >= scenes['range_min'].reshape(B, 1, 1)) &
               (sim.range_grid <= scenes['range_max'].reshape(B, 1, 1))).type(torch.cfloat)
    return sensingReceiver(Y, symbols, A, b_angle, b_range, sim.range_grid, sim.pixels_angle,
                           sim.pixels_range, sim.S, sim.Delta_f, d=d_rx, lamb=sim.lamb,
                           device=sim.device)


def position_mse(est_angle, est_range, scenes, criterion=None):
    """MSE between estimated and true (x,y) positions over target-present scenes."""
    criterion = criterion or torch.nn.MSELoss()
    mask = scenes['presence'].reshape(-1).bool()
    def xy(r, th):
        """Polar (range, angle) to Cartesian (x, y), shape (B,2)."""
        return torch.cat((r * torch.cos(th), r * torch.sin(th)), dim=-1)
    return criterion(xy(est_range[mask], est_angle[mask]),
                     xy(scenes['range'][mask], scenes['theta'][mask]))


# ----------------------------------------------------------------------------------------
# Training on a fixed pool
# ----------------------------------------------------------------------------------------
def train_steps(network, pool, idx, n_steps, batch_size, optimizer, sim, supervised,
                scheduler=None, criterion=None, generator=None, print_every=0):
    """n_steps optimisation steps on mini-batches drawn (with replacement) from pool[idx].
    supervised=True : position MSE using the scenes' labels (idx must be labelled,
                      target-present scenes) -> pair with optimizerSuper.
    supervised=False: mean receiver peak metric, no labels used -> pair with
                      optimizerUnsuper (maximize=True), as in the original UL.
    Fresh noise/symbols are drawn every step, so repeated scenes are not identical samples.
    Returns the list of loss values."""
    idx = torch.as_tensor(idx, dtype=torch.long)
    if len(idx) == 0 or n_steps <= 0:
        return []
    if supervised:
        assert bool(pool['presence'][idx.to(pool['presence'].device)].all()), \
            'supervised steps need target-present scenes'
    network.train()
    losses = []
    for it in range(n_steps):
        pick = idx[torch.randint(len(idx), (batch_size,), generator=generator)]
        sc = subset(pool, pick)
        Y, symbols, _, A, b_angle = synthesize(sc, network.d, sim)
        metric, est_angle, est_range = run_receiver(network.d, sc, sim, Y, symbols, A, b_angle)
        loss = position_mse(est_angle, est_range, sc, criterion) if supervised \
            else torch.mean(metric[:, 0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
        if print_every and (it + 1) % print_every == 0:
            print(f"  {'SL' if supervised else 'UL'} step {it + 1}/{n_steps}  loss {loss.item():.4g}")
    return losses

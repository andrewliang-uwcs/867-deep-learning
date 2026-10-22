# -*- coding: utf-8 -*-
"""
Summarise active-calibration runs (methods/results/active_*.npz from active_calibration.py).

Run from the repository root:
    python methods/summarize_active.py --since 20261013-165331
    python methods/summarize_active.py --th 10 --mode cpu --since 20261013-165331 --rules weak full random

--since drops runs saved before that timestamp. USE IT: runs saved before the fix for
the shared-starting-spacing bug (before 20261013-165331 in your results folder) are
not valid comparisons. If a (rule, seed) pair was run more than once, the latest is used.

Prints, and writes to methods/results/summary_th<deg>_<mode>/:
  * summary.csv          per-rule mean +/- 95% CI of: mean calibration error over the
                         acquisition rounds (area under the curve / budget range), final
                         calibration error, final median position error, acquisition time
  * paired_vs_random.csv per-rule paired difference to 'random' (same seeds), CI and wins
  * calib_err.png        calibration error vs labels used (mean, 95% CI band)
  * median_pos_err.png   median position error vs labels used
  * sl_step_effect.png   how much each supervised step raises calibration error
"""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Summarise active-calibration results')
parser.add_argument('--results', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'),
                    help='folder with active_*.npz (default: methods/results next to this script)')
parser.add_argument('--mode', default='cpu', help='run_mode in the file names')
parser.add_argument('--th', default='10', help='theta_mean_max_deg in the file names')
parser.add_argument('--since', default=None, help='ignore files saved before YYYYMMDD-HHMMSS')
parser.add_argument('--rules', nargs='+', default=None, help='subset/order of rules to show')
args = parser.parse_args()

NAME = re.compile(r'active_(?P<rule>[a-z]+(?:_noRC)?)_s(?P<seed>\d+)_th(?P<th>[\d.]+)_'
                  r'(?P<mode>[a-z]+)_(?P<ts>\d{8}-\d{6})\.npz$')
ORDER = ['weak', 'weak_noRC', 'full', 'full_noRC', 'maxangle', 'diversity', 'uncertainty', 'random']
# Reference categorical palette (fixed order, colour follows the rule); random = neutral reference
COLORS = {'weak': '#2a78d6', 'full': '#eb6834', 'maxangle': '#1baf7a', 'diversity': '#eda100',
          'uncertainty': '#e87ba4', 'weak_noRC': '#6250d6', 'full_noRC': '#e34948',
          'random': '#52514e'}
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
       9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}


def t95(df):
    """Two-sided 95% Student-t critical value for df degrees of freedom (normal above 30)."""
    if df <= 0:
        return float('nan')
    keys = [k for k in sorted(T95) if k <= df]
    return T95[keys[-1]] if df <= 30 else 1.96


def mean_ci(x):
    """(mean, 95% CI half-width, n) of a 1-D sample; half-width is NaN for n < 2."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return float('nan'), float('nan'), 0
    if n == 1:
        return float(x[0]), float('nan'), 1
    return float(x.mean()), t95(n - 1) * float(x.std(ddof=1)) / np.sqrt(n), n


# ------------------------------------------------------------------ load
runs = {}
skipped_old = 0
for path in glob.glob(os.path.join(args.results, 'active_*.npz')):
    m = NAME.search(os.path.basename(path))
    if not m or m['mode'] != args.mode or float(m['th']) != float(args.th):
        continue
    if args.since and m['ts'] < args.since:
        skipped_old += 1
        continue
    key = (m['rule'], int(m['seed']))
    if key not in runs or m['ts'] > runs[key]['ts']:
        z = np.load(path, allow_pickle=False)
        runs[key] = dict(ts=m['ts'], path=path, **{k: z[k] for k in z.files})

if not runs:
    raise SystemExit('No matching runs found (check --results/--mode/--th/--since).')
if not args.since:
    print('WARNING: no --since given; runs from before the starting-spacing fix may be included.')
print(f'Using {len(runs)} runs ({skipped_old} older files ignored):')

by_rule = defaultdict(dict)
for (rule, seed), r in sorted(runs.items()):
    by_rule[rule][seed] = r
rules = [r for r in (args.rules or ORDER) if r in by_rule] + \
        [r for r in sorted(by_rule) if r not in ORDER and not args.rules]
for rule in rules:
    print(f'  {rule:12s} seeds {sorted(by_rule[rule])}')

# common label grid
grids = {tuple(r['n_labels'].tolist()) for r in runs.values()}
if len(grids) != 1:
    raise SystemExit(f'Runs use different label schedules: {grids}. Filter with --since/--rules.')
labels = np.array(next(iter(grids)))


def stack(rule, field):
    """(sorted seeds, array seeds x rounds) of a logged field for one rule."""
    seeds = sorted(by_rule[rule])
    return seeds, np.array([by_rule[rule][s][field] for s in seeds], dtype=float)


def has(rule, field):
    """True if every run of the rule logged the field (older runs may lack it)."""
    return all(field in r for r in by_rule[rule].values())


# per-run scalar summaries (acquisition rounds = all rounds after round 0)
def per_run(rule):
    """Per-seed scalar summaries for one rule (mean/final calibration error, final median
    position error, SL-step increase, total acquisition time)."""
    seeds, ce = stack(rule, 'calib_err')
    out = dict(seeds=seeds, auc=ce[:, 1:].mean(axis=1), final_ce=ce[:, -1],
               acq=np.array([by_rule[rule][s]['acq_time'].sum() for s in seeds]))
    if has(rule, 'median_pos_err'):
        out['final_med'] = stack(rule, 'median_pos_err')[1][:, -1]
    if has(rule, 'calib_err_after_sl'):
        _, sl = stack(rule, 'calib_err_after_sl')
        out['sl_jump'] = (sl[:, 1:] - ce[:, :-1]).mean(axis=1)   # after-SL minus previous end
    return out


stats = {rule: per_run(rule) for rule in rules}

# ------------------------------------------------------------------ tables
out_dir = os.path.join(args.results, f'summary_th{args.th}_{args.mode}')
os.makedirs(out_dir, exist_ok=True)


def fmt(m, c):
    """Format mean +/- CI (mean only if the CI is undefined)."""
    return f'{m:.4f} ± {c:.4f}' if np.isfinite(c) else f'{m:.4f}'


cols = [('auc', 'mean calib err (rounds 1+)'), ('final_ce', 'final calib err'),
        ('final_med', 'final median pos err [m]'), ('sl_jump', 'SL-step increase'),
        ('acq', 'acq time [s]')]
print('\nPer rule, mean ± 95% CI over seeds (lower is better):')
print(f'{"rule":12s} {"n":>2s}  ' + '  '.join(f'{h:>26s}' for _, h in cols))
with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['rule', 'n_seeds'] + sum([[h + ' mean', h + ' ci95'] for _, h in cols], []))
    for rule in rules:
        s, cells, row = stats[rule], [], [rule, len(stats[rule]['seeds'])]
        for key, _ in cols:
            m, c, _ = mean_ci(s[key]) if key in s else (float('nan'), float('nan'), 0)
            cells.append(fmt(m, c)); row += [m, c]
        print(f'{rule:12s} {len(s["seeds"]):2d}  ' + '  '.join(f'{x:>26s}' for x in cells))
        w.writerow(row)

if 'random' in stats:
    print('\nPaired difference to random on the same seeds (negative = better than random):')
    print(f'{"rule":12s} {"n":>2s}  {"mean calib err diff":>24s}  {"final calib err diff":>24s}  wins')
    with open(os.path.join(out_dir, 'paired_vs_random.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rule', 'n_pairs', 'auc_diff_mean', 'auc_diff_ci95', 'final_diff_mean',
                    'final_diff_ci95', 'wins_auc'])
        ref = {s: i for i, s in enumerate(stats['random']['seeds'])}
        for rule in rules:
            if rule == 'random':
                continue
            s = stats[rule]
            common = [i for i, seed in enumerate(s['seeds']) if seed in ref]
            if not common:
                continue
            j = [ref[s['seeds'][i]] for i in common]
            d_auc = s['auc'][common] - stats['random']['auc'][j]
            d_fin = s['final_ce'][common] - stats['random']['final_ce'][j]
            ma, ca, n = mean_ci(d_auc); mf, cf, _ = mean_ci(d_fin)
            wins = int((d_auc < 0).sum())
            print(f'{rule:12s} {n:2d}  {fmt(ma, ca):>24s}  {fmt(mf, cf):>24s}  {wins}/{n}')
            w.writerow([rule, n, ma, ca, mf, cf, wins])
    print('A difference is clearly non-zero only if its CI excludes 0.')

# ------------------------------------------------------------------ figures
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#8a8984', 'axes.labelcolor': '#0b0b0b',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'grid.color': '#e4e3df'})


def curve_plot(field, ylabel, fname, title):
    """Plot a logged field against labels used (mean and 95% CI band per rule) and save it."""
    usable = [r for r in rules if has(r, field)]
    if not usable:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for rule in usable:
        _, y = stack(rule, field)
        m = y.mean(axis=0)
        c = (np.array([t95(len(y) - 1)]) * y.std(axis=0, ddof=1) / np.sqrt(len(y))
             if len(y) > 1 else np.zeros_like(m))
        col = COLORS.get(rule, '#8a8984')
        ls = '--' if rule == 'random' else '-'
        ax.plot(labels, m, ls, color=col, lw=2, marker='o', ms=4, label=f'{rule} (n={len(y)})')
        ax.fill_between(labels, m - c, m + c, color=col, alpha=0.12, lw=0)
    ax.set_xlabel('Labels used')
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left', fontsize=11)
    ax.grid(True, axis='y', lw=0.6)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname))
    plt.close(fig)
    print(f'  wrote {os.path.join(out_dir, fname)}')


print('\nFigures:')
curve_plot('calib_err', 'Calibration error (units of λ/2, gauge removed)', 'calib_err.png',
           f'Calibration error vs labels (mean, 95% CI), ±{args.th}° coverage')
curve_plot('median_pos_err', 'Median position error [m]', 'median_pos_err.png',
           'Held-out median position error vs labels (mean, 95% CI)')

sl_rules = [r for r in rules if 'sl_jump' in stats[r]]
if sl_rules:
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    ms = [mean_ci(stats[r]['sl_jump']) for r in sl_rules]
    x = np.arange(len(sl_rules))
    ax.bar(x, [m for m, _, _ in ms], width=0.55, color=[COLORS.get(r, '#8a8984') for r in sl_rules],
           yerr=[c if np.isfinite(c) else 0 for _, c, _ in ms], capsize=3, ecolor='#52514e')
    ax.axhline(0, color='#8a8984', lw=1)
    ax.set_xticks(x, sl_rules)
    ax.set_ylabel('Mean rise in calibration error\nfrom one supervised step')
    ax.set_title('Effect of each supervised step (positive = SL makes calibration worse)',
                 loc='left', fontsize=11)
    ax.grid(True, axis='y', lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'sl_step_effect.png'))
    plt.close(fig)
    print(f'  wrote {os.path.join(out_dir, "sl_step_effect.png")}')

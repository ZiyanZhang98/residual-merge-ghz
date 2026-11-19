"""Paper figure: why the PTA correction cannot repair merging (n=4). Read-only, CPU, compute node.

(a), (b) Whole simulations, n=4 independent test set, hard families (AB, AABB, ABB): per-round relative error of P
    for Fixed-K merging [M0] and Fixed-K + PTA [F2] from the same rollouts (same K, same atoms; the only difference is
    the additive PTA correction), K = 16 and K = 64. Median over tasks with interquartile band; the Monte Carlo
    reference's own standard error (median) and the 1 % tolerance are drawn for scale.

A third panel (one merge, from the training labels) was dropped on 2026-11-07 (user decision); its last rendering is
results/causefig_28973173/fig_cause_pta_repair_full.*.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

if not os.environ.get('SLURM_JOB_ID'):
    sys.exit('Run on a Slurm compute node (sbatch ml_guided_pta/residual_merge/cmpfig.sbatch)')

HERE = Path(__file__).resolve().parent
RES = HERE/'results'
HARD = ('AB', 'AABB', 'ABB')
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e9e8e4'
BEFORE = dict(color='#cf8fc4', ls=':', label='$F_0$')   # paper name of Fixed-K (internal M0)
AFTER = dict(color='#b0479b', ls='-', label='$F_1$')    # paper name of Fixed-K + PTA (internal F2)


def q(x, p):
    return np.nanpercentile(x, p, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-job', default='28851717')
    ap.add_argument('--K', type=int, nargs=2, default=[16, 64])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)

    # ---- whole simulations (test set)
    cells, mc = [], {}
    for d in sorted(glob.glob(str(RES/('test_%s_*' % a.test_job)))):
        cells += json.loads(Path(d, 'cells.json').read_text())
        for f in glob.glob(str(Path(d, 'mc_*.npz'))):
            z = np.load(f)
            mc[Path(f).stem[3:]] = z['se_P_relative']
    hard_tasks = sorted({c['task_id'] for c in cells if c['family'] in HARD})
    sim = {}
    for K in a.K:
        e = {}
        for m in ('M0', 'F2'):
            rows = {c['task_id']: np.array(c['P_relative'])*100 for c in cells
                    if c['method'] == m and c.get('K') == K and c['family'] in HARD and c.get('status') == 'OK'}
            if sorted(rows) != hard_tasks:
                raise ValueError('Missing %s K=%d cells' % (m, K))
            e[m] = np.array([rows[t] for t in hard_tasks])
        sim[K] = e
    noise = np.array([mc[t]*100 for t in hard_tasks])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                         'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    rounds = np.arange(1, 41)
    data = dict(test_job=a.test_job, hard_tasks=len(hard_tasks), panels={})
    floor = 1e-4

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.2), gridspec_kw=dict(wspace=0.32))
    for ax, K, tag in zip(axes, a.K, ('a', 'b')):
        for m, st in (('M0', BEFORE), ('F2', AFTER)):
            x = sim[K][m]
            med, lo, hi = q(x, 50), q(x, 25), q(x, 75)
            ax.fill_between(rounds, np.maximum(lo, floor), np.maximum(hi, floor), color=st['color'], alpha=0.18, lw=0)
            ax.plot(rounds, np.maximum(med, floor), color=st['color'], ls=st['ls'], lw=1.7, label=st['label'])
        ax.plot(rounds, np.maximum(q(noise, 50), floor), color=INK2, lw=0.8, ls=(0, (1, 2)),
                label='Monte Carlo std. error')
        ax.axhline(1.0, color=INK, lw=0.8, ls='--')
        ax.text(1.5, 1.12, 'tolerance 1%', fontsize=6, color=INK2, va='bottom')
        ax.set_yscale('log')
        ax.set_xlim(1, 40)
        ax.set_xlabel('Round')
        ax.set_title(f'({tag}) K = {K}', fontsize=7, pad=4)
        ax.grid(True, color=GRID, lw=0.4)
        ax.spines[['top', 'right']].set_visible(False)
        mb, ma = q(sim[K]['M0'], 50), q(sim[K]['F2'], 50)
        data['panels'][tag] = dict(K=K, rounds={str(r): dict(before=float(mb[r-1]), after=float(ma[r-1]))
                                                for r in (2, 5, 8, 12, 16, 20, 25, 30, 35, 40)})
    axes[0].set_ylabel('Relative error $P$ (%)')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.07), fontsize=6.3)
    fig.savefig(out/'fig_cause_pta_repair.pdf', bbox_inches='tight')
    fig.savefig(out/'fig_cause_pta_repair.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ---- subfigure version: one PDF per panel (K = 16, K = 64) with the legend inside the first panel. Fixed page size (no tight bbox)
    # so the panels line up; 1.72 in = 0.49 of the IEEE column, so text prints at 8 pt (legend and notes 7 pt).
    # No in-figure titles: the subcaptions give "(a) K = 16" etc.
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
                         'font.size': 8})
    for K, tag in zip(a.K, ('a', 'b')):
        fig = plt.figure(figsize=(1.72, 1.6))
        ax = fig.add_axes([0.31, 0.25, 0.65, 0.71])
        for m, st in (('M0', BEFORE), ('F2', AFTER)):
            x = sim[K][m]
            med, lo, hi = q(x, 50), q(x, 25), q(x, 75)
            ax.fill_between(rounds, np.maximum(lo, floor), np.maximum(hi, floor), color=st['color'], alpha=0.18, lw=0)
            ax.plot(rounds, np.maximum(med, floor), color=st['color'], ls=st['ls'], lw=1.4, label=st['label'])
        ax.plot(rounds, np.maximum(q(noise, 50), floor), color=INK2, lw=0.7, ls=(0, (1, 2)),
                label='Monte Carlo\nstd. error')
        ax.axhline(1.0, color=INK, lw=0.7, ls='--')
        if tag == 'a':   # one legend for the pair, inside the first panel (empty lower-right corner)
            leg = ax.legend(loc='lower right', fontsize=6, handlelength=1.6, handletextpad=0.4, labelspacing=0.25,
                            borderpad=0.3, borderaxespad=0.2, frameon=True, framealpha=0.9, edgecolor='none')
            leg.get_frame().set_facecolor('white')
        ax.text(1.5, 1.12, 'tolerance 1%', fontsize=7, color=INK2, va='bottom')
        ax.set_yscale('log')
        ax.set_xlim(1, 40)
        ax.set_xticks([10, 20, 30, 40])
        ax.set_xlabel('Round', labelpad=1)
        ax.set_ylabel('Relative error $P$ (%)')
        ax.grid(True, color=GRID, lw=0.4)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(top=max(ax.get_ylim()[1], 3.0))   # room above the 1 % line for its label
        fig.savefig(out/('fig_cause_pta_repair_%s.pdf' % tag))
        fig.savefig(out/('fig_cause_pta_repair_%s.png' % tag), dpi=300)
        plt.close(fig)
    (out/'cause.json').write_text(json.dumps(data, indent=1))
    print(json.dumps(data, indent=1))


if __name__ == '__main__':
    main()

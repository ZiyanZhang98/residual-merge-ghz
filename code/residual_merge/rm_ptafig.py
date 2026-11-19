"""Paper figures: PTA approximate simulator vs. Monte Carlo ground truth, per round, for P and for F (n=4 test set).

Read-only: uses the stored PTA-simulator cells (job 28921046) and the stored Monte Carlo references of the test set
(job 28851717). No simulation, training or new truth. Must run on a Slurm compute node.

One panel per word family (all resolvable test tasks of the family). In each panel:
  - PTA simulator: thick line = mean over the family's tasks; error bars = min and max over the tasks, every round;
  - Monte Carlo truth: black line = mean over the same tasks; grey band = min to max over the tasks.
The mean is geometric for P (log axis; P spans decades across tasks) and arithmetic for F.
Outputs fig_pta_truth_P and fig_pta_truth_F (all six families; appendix candidates), fig_pta_truth_ABB (paper
version: the hardest family, P and F side by side; only this one is copied to paper/figures) and stats.json.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

if not os.environ.get('SLURM_JOB_ID'):
    sys.exit('Run on a Slurm compute node (sbatch ml_guided_pta/residual_merge/ptafig.sbatch)')

HERE = Path(__file__).resolve().parent
RES = HERE/'results'
FAMILIES = (('AB', 'AABB', 'ABB'), ('A', 'AAB', 'B'))
TOL = {'P': 0.01, 'F': 0.01}
INK, INK2, PTA_C, BAND = '#0b0b0b', '#52514e', '#eb6834', '#d9d8d3'


def load(pta_job, test_job):
    cells = json.loads((RES/('pta_%s' % pta_job)/'cells.json').read_text())['test']
    mc = {}
    for f in glob.glob(str(RES/('test_%s_*' % test_job)/'mc_*.npz')):
        z = np.load(f)
        mc[Path(f).stem[3:]] = {k: z[k] for k in z.files}
    return cells, mc


def first_fail(err, tol):
    bad = np.nonzero(np.asarray(err) > tol)[0]
    return int(bad[0])+1 if len(bad) else None


def family_arrays(rows, mc, qty):
    sim = np.array([r[qty] for r in rows], dtype=float)
    truth = np.array([r['P_reference' if qty == 'P' else 'F_reference'] for r in rows], dtype=float)
    for r, t in zip(rows, truth):
        if r['task_id'] not in mc:
            raise KeyError('Missing Monte Carlo reference for %s' % r['task_id'])
        if not np.allclose(t, mc[r['task_id']][qty], rtol=1e-12, atol=1e-15):
            raise ValueError('Stored reference does not match the Monte Carlo file for %s' % r['task_id'])
    if qty == 'P':
        if (sim <= 0).any() or (truth <= 0).any():
            raise ValueError('Non-positive P cannot be averaged on a log scale')
        mean = lambda x: np.exp(np.log(x).mean(0))  # noqa: E731
        err = np.abs(sim-truth)/truth
    else:
        mean = lambda x: x.mean(0)  # noqa: E731
        err = np.abs(sim-truth)
    return sim, truth, mean(sim), mean(truth), err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pta-job', default='28921046')
    ap.add_argument('--test-job', default='28851717')
    ap.add_argument('--main-family', default='ABB', help='family of the one-figure paper version (the hardest)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import LogLocator, NullFormatter, MaxNLocator
    plt.rcParams.update({'font.size': 7, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                         'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})

    cells, mc = load(a.pta_job, a.test_job)
    rounds = np.arange(1, 41)
    stats = {}

    for qty in ('P', 'F'):
        fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.3), sharex=True, gridspec_kw=dict(hspace=0.42, wspace=0.32))
        for i, row in enumerate(FAMILIES):
            for j, fam in enumerate(row):
                ax = axes[i, j]
                rows = [c for c in cells if c['family'] == fam and c['status'] == 'OK']
                sim, truth, sim_m, truth_m, err = family_arrays(rows, mc, qty)
                fails = [first_fail(e, TOL[qty]) for e in err]
                passed = sum(k is None for k in fails)
                failed_at = [k for k in fails if k is not None]

                ax.fill_between(rounds, truth.min(0), truth.max(0), color=BAND, lw=0, zorder=1)
                ax.errorbar(rounds, sim_m, yerr=np.clip([sim_m-sim.min(0), sim.max(0)-sim_m], 0, None), fmt='none', ecolor=PTA_C,
                            elinewidth=0.7, capsize=1.3, capthick=0.7, alpha=0.75, zorder=2)
                ax.plot(rounds, sim_m, color=PTA_C, lw=1.8, zorder=3)
                ax.plot(rounds, truth_m, color=INK, lw=1.1, zorder=4)
                if qty == 'P':
                    ax.set_yscale('log')
                    ax.yaxis.set_major_locator(LogLocator(numticks=5))
                    ax.yaxis.set_minor_formatter(NullFormatter())
                else:
                    ax.yaxis.set_major_locator(MaxNLocator(5))
                if not failed_at:
                    sub = '%d/%d tasks pass' % (passed, len(rows))
                elif passed == 0:
                    sub = '0/%d pass; first failure at round %g (median)' % (len(rows), float(np.median(failed_at)))
                else:
                    sub = '%d/%d pass; %d fail%s from round %g' % (
                        passed, len(rows), len(failed_at), '' if len(failed_at) > 1 else 's', float(np.median(failed_at)))
                ax.set_title('%s\n%s' % (fam, sub), fontsize=6.5, color=INK, pad=3)
                ax.set_xlim(0.5, 40.5)
                ax.grid(True, color='#e9e8e4', lw=0.4)
                ax.spines[['top', 'right']].set_visible(False)
                if i == 1:
                    ax.set_xlabel('Round')
                if j == 0:
                    ax.set_ylabel(('Success probability $P$' if qty == 'P' else 'Fidelity $F$')
                                  + ('\n(hard families)' if i == 0 else '\n(easy families)'))
                stats.setdefault(qty, {})[fam] = dict(
                    tasks=len(rows), within_tolerance=passed, first_failure_median=(
                        float(np.median(failed_at)) if failed_at else None),
                    first_failure_range=[min(failed_at), max(failed_at)] if failed_at else None,
                    round40=dict(sim_mean=float(sim_m[-1]), sim_min=float(sim.min(0)[-1]), sim_max=float(sim.max(0)[-1]),
                                 truth_mean=float(truth_m[-1]), truth_min=float(truth.min(0)[-1]),
                                 truth_max=float(truth.max(0)[-1])))
        mean_word = 'geometric mean' if qty == 'P' else 'mean'
        handles = [Line2D([], [], color=INK, lw=1.1, label='Monte Carlo truth (%s)' % mean_word),
                   Patch(color=BAND, label='Monte Carlo truth (min to max over tasks)'),
                   Line2D([], [], color=PTA_C, lw=1.8, label='PTA simulator (%s)' % mean_word),
                   Line2D([], [], color=PTA_C, lw=0.7, alpha=0.75, marker='_', ms=4,
                          label='PTA simulator (min to max over tasks)')]
        fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02),
                   fontsize=6.5, handlelength=2.2)
        name = 'fig_pta_truth_%s' % qty
        fig.savefig(out/(name+'.pdf'), bbox_inches='tight')
        fig.savefig(out/(name+'.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # ---- paper version: one family (the hardest, ABB), P and F side by side, single column
    fam = a.main_family
    rows = [c for c in cells if c['family'] == fam and c['status'] == 'OK']
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.05), gridspec_kw=dict(wspace=0.45))
    for ax, qty in zip(axes, ('P', 'F')):
        sim, truth, sim_m, truth_m, err = family_arrays(rows, mc, qty)
        fails = [first_fail(e, TOL[qty]) for e in err]
        passed = sum(k is None for k in fails)
        failed_at = [k for k in fails if k is not None]
        ax.fill_between(rounds, truth.min(0), truth.max(0), color=BAND, lw=0, zorder=1)
        ax.errorbar(rounds, sim_m, yerr=np.clip([sim_m-sim.min(0), sim.max(0)-sim_m], 0, None), fmt='none',
                    ecolor=PTA_C, elinewidth=0.6, capsize=1.0, capthick=0.6, alpha=0.75, zorder=2)
        ax.plot(rounds, sim_m, color=PTA_C, lw=1.5, zorder=3)
        ax.plot(rounds, truth_m, color=INK, lw=1.0, zorder=4)
        if qty == 'P':
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(LogLocator(numticks=4))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_ylabel('Success probability $P$')
        else:
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.set_ylabel('Fidelity $F$')
        ax.set_xlim(0.5, 40.5)
        if failed_at:   # vertical line at the median round in which the PTA simulator first leaves the tolerance
            kf = float(np.median(failed_at))
            ax.axvline(kf, color=INK2, lw=0.8, ls='--', zorder=5)
            ticks = [kf, 20, 40]   # the failure round is labelled on the x axis
            ax.set_xticks(ticks)
            ax.set_xticklabels(['%g' % t for t in ticks])
            ax.get_xticklabels()[0].set_color(INK)
            ax.get_xticklabels()[0].set_fontweight('bold')
        ax.set_xlabel('Round')
        ax.grid(True, color='#e9e8e4', lw=0.4)
        ax.spines[['top', 'right']].set_visible(False)
        stats.setdefault('paper_%s' % fam, {})[qty] = dict(tasks=len(rows), within_tolerance=passed,
                                                            first_failure_median=(float(np.median(failed_at))
                                                                                  if failed_at else None))
    # The spread (truth band, PTA error bars) and the failure line are described in the caption, not the legend.
    handles = [Line2D([], [], color=INK, lw=1.0, label='Monte Carlo truth'),
               Line2D([], [], color=PTA_C, lw=1.5, label='PTA simulator')]
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06),
               fontsize=6.3, handlelength=2.0, columnspacing=1.5)
    name = 'fig_pta_truth_%s' % fam
    fig.savefig(out/(name+'.pdf'), bbox_inches='tight')
    fig.savefig(out/(name+'.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ---- subfigure version: one PDF per panel (P, F) with the legend inside the first panel. Fixed page size (no tight bbox) so the
    # panels line up; 1.72 in = 0.49 of the IEEE column (3.5 in), so text prints at its nominal 8 pt (legend 7 pt).
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
                         'font.size': 8})
    for qty in ('P', 'F'):
        sim, truth, sim_m, truth_m, err = family_arrays(rows, mc, qty)
        failed_at = [k for k in (first_fail(e, TOL[qty]) for e in err) if k is not None]
        fig = plt.figure(figsize=(1.72, 1.6))
        ax = fig.add_axes([0.33, 0.25, 0.63, 0.71])
        ax.fill_between(rounds, truth.min(0), truth.max(0), color=BAND, lw=0, zorder=1)
        ax.errorbar(rounds, sim_m, yerr=np.clip([sim_m-sim.min(0), sim.max(0)-sim_m], 0, None), fmt='none',
                    ecolor=PTA_C, elinewidth=0.5, capsize=0.8, capthick=0.5, alpha=0.75, zorder=2)
        ax.plot(rounds, sim_m, color=PTA_C, lw=1.3, zorder=3, label='PTA simulator')
        ax.plot(rounds, truth_m, color=INK, lw=0.9, zorder=4, label='Monte Carlo truth')
        if qty == 'P':
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(LogLocator(numticks=4))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_ylabel('Success probability $P$')
            # one legend for the pair, inside the P panel (empty lower-left corner)
            h, l = ax.get_legend_handles_labels()
            leg = ax.legend(h[::-1], l[::-1], loc='lower left', fontsize=6, handlelength=1.6, handletextpad=0.4,
                            labelspacing=0.25, borderpad=0.3, borderaxespad=0.2, frameon=True, framealpha=0.9,
                            edgecolor='none')
            leg.get_frame().set_facecolor('white')
            leg.set_zorder(6)
        else:
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.set_ylabel('Fidelity $F$')
        ax.set_xlim(0.5, 40.5)
        if failed_at:
            kf = float(np.median(failed_at))
            ax.axvline(kf, color=INK2, lw=0.8, ls='--', zorder=5)
            ax.set_xticks([kf, 20, 40])
            ax.set_xticklabels(['%g' % kf, '20', '40'])
            ax.get_xticklabels()[0].set_fontweight('bold')
            ax.get_xticklabels()[0].set_color(INK)
        ax.set_xlabel('Round', labelpad=1)
        ax.grid(True, color='#e9e8e4', lw=0.4)
        ax.spines[['top', 'right']].set_visible(False)
        fig.savefig(out/('%s_%s.pdf' % (name, qty)))
        fig.savefig(out/('%s_%s.png' % (name, qty)), dpi=300)
        plt.close(fig)

    (out/'stats.json').write_text(json.dumps(dict(
        pta_job=a.pta_job, test_job=a.test_job, set='n=4 test, resolvable tasks',
        mean='geometric for P, arithmetic for F', tolerance=TOL,
        note='within_tolerance counts tasks whose error stays within the tolerance of that quantity in all 40 rounds',
        stats=stats), indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()

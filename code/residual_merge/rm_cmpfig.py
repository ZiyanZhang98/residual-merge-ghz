"""Paper figures for the comparison experiments on the n=4 independent test set. Read-only, CPU, compute node.

Only stored quantities of the original runs are plotted: needed atoms (physical_calls at the suffix-rule point, the
registered metric), total atoms per run, per-round P and F, and the stored per-round P relative error and F absolute
error. Method names follow the paper (internal codes in brackets):
  Learned (ours) [L], Geometric M_0 [C1], Fixed-K F_0 [M0], Fixed-K + PTA F_1 [F2], Fixed-K x PTA F_2 [F2mul].
  Note: the paper's M_0 is the internal C1 (Geometric), NOT the internal M0 (Fixed-K without correction, paper F_0).

  figA_cost_ladder:     atoms needed to reach the 1 % tolerance, one dot per task, per method (hard / easy panels).
  figB_L_vs_baselines:  needed atoms of Learned against Fixed-K + PTA and against Geometric, one dot per task.
  figA_*_{a,b}, figB_*_{a,b}: the same panels as separate PDFs for LaTeX subfigures (minimal text; details in captions).
  figR_ratio:           required-atom ratio baseline / Learned per family, 95% bootstrap interval, preparations not
                        worse (the registered numbers of the analysis; replaces the per-family ratio table).
  figE_alpha_sweep:    Learned over its alpha grid: total atoms and largest P error per run (forced merges marked).
  figD_error_vs_atoms:  largest P relative error over 40 rounds against total atoms, per family; each point is one
                        grid setting of a method (median over the family's tasks of the atoms and of the error).

Checks (job fails otherwise): needed atoms recomputed here for L, C1, F2 equal the registered analysis (28851720)
task by task; M0 / F2mul prep-equal ratios and the plain Monte Carlo hard median reproduce the same analysis.
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if not os.environ.get('SLURM_JOB_ID'):
    sys.exit('Run on a Slurm compute node (sbatch ml_guided_pta/residual_merge/cmpfig.sbatch)')

HERE = Path(__file__).resolve().parent
RES = HERE/'results'
HARD, EASY = ('AB', 'AABB', 'ABB'), ('A', 'AAB', 'B')
TOL_P = 0.01
INK, INK2, GRID, GRAY = '#0b0b0b', '#52514e', '#e9e8e4', '#8a8984'
# Paper names (2026-11-16): internal code -> symbol. M_0 = Geometric (internal C1); F_0 = Fixed-K without correction
# (internal M0); F_1 = Fixed-K + additive PTA (internal F2); F_2 = Fixed-K + multiplicative PTA (internal F2mul).
NAME = {'L': 'Learned (ours)', 'C1': '$M_0$', 'M0': '$F_0$', 'F2': '$F_1$', 'F2mul': '$F_2$'}
# Learned blue, Geometric teal, Fixed-K + PTA plum (Fixed-K: light plum, same runs without correction); line style
# and marker as a second cue. None is close to the PTA-simulator orange of the PTA figures (palette checked).
STYLE = {'L': dict(color='#2a78d6', ls='-', marker='o'),
         'C1': dict(color='#2e9e8f', ls='--', marker='s'),
         'F2': dict(color='#b0479b', ls='-.', marker='^'),
         'M0': dict(color='#cf8fc4', ls=':', marker='v')}
# Figure B colours the task groups (not the methods): slots 1-3 of the reference categorical palette, the only
# three that pass the all-pairs CVD and normal-vision checks for a scatter; the marker shape is the second cue.
FAM_STYLE = {HARD: dict(color='#2a78d6', marker='o'), ('A', 'AAB'): dict(color='#eb6834', marker='s'),
             ('B',): dict(color='#1baf7a', marker='^')}
REGION = '#f1f0ec'   # neutral tint for "Learned needs fewer branches"; text stays in ink, never in a series colour


def needed(points):
    """Same rule as rm_deploy.needed: fewest atoms from which every costlier grid point passes Q_QoI <= 1."""
    pts = sorted(points)
    ok = [q is not None and q <= 1 for _, q in pts]
    for i in range(len(pts)):
        if all(ok[i:]):
            return pts[i][0], False
    return 2*max(c for c, _ in pts), True


def load(test_job):
    cells, mc = [], []
    for d in sorted(glob.glob(str(RES/('test_%s_*' % test_job)))):
        cells += json.loads(Path(d, 'cells.json').read_text())
        mc += json.loads(Path(d, 'mc.json').read_text())
    return cells, mc


def prep_ratio(vals, a, b, tasks):
    preps = sorted({tasks[t]['prep'] for t in tasks})
    r = []
    for p in preps:
        ids = [t for t in tasks if tasks[t]['prep'] == p]
        r.append(np.median([vals[a][i][0] for i in ids])/np.median([vals[b][i][0] for i in ids]))
    return float(math.exp(np.mean(np.log(r))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-job', default='28851717')
    ap.add_argument('--analysis-job', default='28851720')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    ana = json.loads((RES/('testa_%s' % a.analysis_job)/'analysis.json').read_text())

    cells, mcs = load(a.test_job)
    by = defaultdict(lambda: defaultdict(list))
    tasks = {}
    for c in cells:
        by[c['method']][c['task_id']].append(c)
        tasks[c['task_id']] = dict(family=c['family'], prep=c['prep_id'])
    methods = ('L', 'C1', 'F2', 'M0', 'F2mul')
    vals = {m: {t: needed([(c['physical_calls'], c.get('Q_QoI') if c.get('status') == 'OK' else None)
                           for c in by[m][t]]) for t in tasks} for m in methods}

    # ---- checks against the registered analysis
    reg = ana['tables']['Q_QoI']['needed']
    for m in ('L', 'C1', 'F2'):
        for t in tasks:
            r = reg[m][t]
            if (vals[m][t][0], vals[m][t][1]) != (r['atoms'], r['censored']):
                raise ValueError('Needed atoms differ from the analysis: %s %s %s vs %s' % (m, t, vals[m][t], r))
    for g, fams in (('hard', HARD), ('easy', EASY)):
        sub = {t: v for t, v in tasks.items() if v['family'] in fams}
        for m in ('M0', 'F2mul'):
            mine, theirs = prep_ratio(vals, m, 'L', sub), ana['reference_methods'][g]['L_vs_'+m]['ratio']
            if abs(mine-theirs) > 1e-9*theirs:
                raise ValueError('L vs %s (%s) differs from the analysis: %r vs %r' % (m, g, mine, theirs))
    mc_cost = {x['task_id']: x['paths']*(x['max_se_P_relative']/(TOL_P/2))**2*x['H'] for x in mcs if x['resolved']}
    hard_med = float(np.median([mc_cost[t] for t in tasks if tasks[t]['family'] in HARD]))
    if abs(hard_med-ana['plain_mc']['hard']['median_rounds']) > 1e-6*hard_med:
        raise ValueError('Plain Monte Carlo median differs from the analysis')
    pta = {c['task_id']: c for c in json.loads((RES/'pta_28921046'/'cells.json').read_text())['test']}
    pta_pass = {g: sum(pta[t]['Q_QoI'] <= 1 for t in tasks if tasks[t]['family'] in fams)
                for g, fams in (('hard', HARD), ('easy', EASY))}

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 7, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                         'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    data = {}

    def save(fig, name):
        fig.savefig(out/(name+'.pdf'), bbox_inches='tight')
        fig.savefig(out/(name+'.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def style(ax):
        ax.grid(True, color=GRID, lw=0.4)
        ax.spines[['top', 'right']].set_visible(False)

    # ---------------------------------------------------------------- A: cost ladder
    rows = [('PTA simulator', 'PTA'), ('Monte Carlo', None), (NAME['M0'], 'M0'), (NAME['F2'], 'F2'),
            (NAME['F2mul'], 'F2mul'), (NAME['C1'], 'C1'), (NAME['L'], 'L')]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0), sharey=True, gridspec_kw=dict(wspace=0.06))
    data['A'] = {}
    for ax, (g, fams) in zip(axes, (('hard', HARD), ('easy', EASY))):
        ts = sorted(t for t in tasks if tasks[t]['family'] in fams)
        data['A'][g] = {}
        for i, (label, m) in enumerate(rows):
            y = len(rows)-i
            if m == 'PTA':
                ax.text(0.5, y, 'reaches 1%% on %d of %d tasks at any cost' % (pta_pass[g], len(ts)),
                        transform=ax.get_yaxis_transform(), ha='center', va='center', fontsize=6.5, color=INK2)
                data['A'][g][label] = dict(passed=pta_pass[g], tasks=len(ts))
                continue
            if m is None:
                x, cen = np.array([mc_cost[t] for t in ts]), np.zeros(len(ts), bool)
            else:
                x = np.array([vals[m][t][0] for t in ts], float)
                cen = np.array([vals[m][t][1] for t in ts])
            col = STYLE['L']['color'] if m == 'L' else GRAY
            jy = y+rng.uniform(-0.22, 0.22, len(ts))
            ax.scatter(x[~cen], jy[~cen], s=9, color=col, alpha=0.75, lw=0, zorder=3)
            if cen.any():
                ax.scatter(x[cen], jy[cen], s=9, facecolors='none', edgecolors=col, lw=0.7, zorder=3)
            med = float(np.median(x))
            ax.plot([med, med], [y-0.35, y+0.35], color=INK, lw=1.4, zorder=4)
            ax.annotate('{:,.0f}'.format(med), (med, y+0.35), xytext=(0, 1), textcoords='offset points',
                        ha='center', va='bottom', fontsize=6, color=INK)
            data['A'][g][label] = dict(median=med, tasks=len(ts), censored=int(cen.sum()))
        ax.set_xscale('log')
        ax.set_title('%s families (%s), %d tasks' % ('Hard' if g == 'hard' else 'Easy', ', '.join(fams), len(ts)),
                     fontsize=7, color=INK, pad=4)
        ax.set_xlabel('Branches needed to reach 1% accuracy')
        ax.set_ylim(0.4, len(rows)+0.75)
        style(ax)
        ax.grid(False, axis='y')
    axes[0].set_yticks(range(len(rows), 0, -1))
    axes[0].set_yticklabels([r[0] for r in rows])
    axes[0].get_yticklabels()[-1].set_color(STYLE['L']['color'])
    fig.legend(handles=[Line2D([], [], ls='', marker='o', ms=3.5, color=GRAY, label='one task'),
                        Line2D([], [], ls='', marker='o', ms=3.5, mfc='none', mec=GRAY,
                               label='not reached on the grid (plotted at 2x the largest grid point)'),
                        Line2D([], [], color=INK, lw=1.4, label='median')],
               loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.55, 1.06), fontsize=6.5)
    save(fig, 'figA_cost_ladder')

    # ---------------------------------------------------------------- B: Learned against the two baselines, per task
    groups = [('Hard families (AB, AABB, ABB)', HARD, 'o'), ('Families A, AAB', ('A', 'AAB'), 's'),
              ('Family B', ('B',), '^')]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.3), gridspec_kw=dict(wspace=0.28))
    data['B'] = {}
    for ax, base in zip(axes, ('F2', 'C1')):
        allx = [vals[base][t][0] for t in tasks]+[vals['L'][t][0] for t in tasks]
        lo, hi = min(allx)/1.6, max(allx)*1.6
        ax.fill_between([lo, hi], [lo, lo], [lo, hi], color=REGION, lw=0, zorder=0)
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, zorder=1)
        ax.text(hi/1.25, lo*1.5, 'Learned needs fewer branches', ha='right', va='bottom', fontsize=6.5, color=INK2)
        ax.text(lo*1.4, hi/1.3, 'Learned needs more branches', ha='left', va='top', fontsize=6.5, color=INK2)
        for label, fams, _ in groups:
            fs = FAM_STYLE[fams]
            ts = [t for t in tasks if tasks[t]['family'] in fams]
            x = np.array([vals[base][t][0] for t in ts], float)
            y = np.array([vals['L'][t][0] for t in ts], float)
            cen = np.array([vals[base][t][1] or vals['L'][t][1] for t in ts])
            ax.scatter(x[~cen], y[~cen], s=18, color=fs['color'], marker=fs['marker'], alpha=0.85,
                       edgecolors='white', lw=0.4, zorder=3, label=label)
            if cen.any():
                ax.scatter(x[cen], y[cen], s=18, facecolors='white', edgecolors=fs['color'], marker=fs['marker'],
                           lw=0.9, zorder=3)
        fewer = sum(vals['L'][t][0] < vals[base][t][0] for t in tasks)
        same = sum(vals['L'][t][0] == vals[base][t][0] for t in tasks)
        more = len(tasks)-fewer-same
        hard_t = [t for t in tasks if tasks[t]['family'] in HARD]
        data['B'][NAME[base]] = dict(tasks=len(tasks), learned_fewer=fewer, same=same, learned_more=more,
                                     hard_tasks=len(hard_t),
                                     hard_learned_fewer=sum(vals['L'][t][0] < vals[base][t][0] for t in hard_t))
        ax.set_title('Learned vs. %s\nfewer branches on %d, equal on %d, more on %d of %d tasks'
                     % (NAME[base], fewer, same, more, len(tasks)), fontsize=7, color=INK, pad=4)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.set_xlabel('%s: branches needed to reach 1%%' % NAME[base])
        ax.set_ylabel('Learned (ours): branches needed to reach 1%')
        style(ax)
    h, l = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], ls='', marker='o', ms=4, mfc='white', mec=INK2))
    l.append('open: not reached on the grid by one method (2x largest grid point)')
    fig.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04), fontsize=6.5)
    save(fig, 'figB_L_vs_baselines')

    # ---------------------------------------------------------------- A, B as LaTeX subfigures
    # One PDF per panel, fixed page size (no tight bbox) so the two panels of a pair line up; serif 8 pt as in the
    # cause figure. No titles, no explanatory text: groups, medians, open markers and counts go to the captions.
    # A: 3.45 in = 0.49 of the IEEE text width (figure*); B: 1.72 in = 0.49 of the column width (figure).
    SUB = {'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix', 'font.size': 8}

    def med_label(v):
        return '{:,.0f}'.format(v) if v < 1e5 else r'%.1f$\times10^{%d}$' % (v/10**int(math.log10(v)), int(math.log10(v)))

    with plt.rc_context(SUB):
        rng = np.random.default_rng(0)   # same draws, same order as the combined figure A
        for (g, fams), tag in zip((('hard', HARD), ('easy', EASY)), 'ab'):
            ts = sorted(t for t in tasks if tasks[t]['family'] in fams)
            fig = plt.figure(figsize=(3.45, 2.15))
            ax = fig.add_axes([0.25, 0.17, 0.72, 0.81])
            for i, (label, m) in enumerate(rows):
                y = len(rows)-i
                if m == 'PTA':
                    ax.text(0.5, y, 'never reaches 1%' if pta_pass[g] == 0 else '%d/%d reach 1%%' % (pta_pass[g], len(ts)),
                            transform=ax.get_yaxis_transform(), ha='center', va='center', fontsize=7, color=INK2)
                    continue
                if m is None:
                    x, cen = np.array([mc_cost[t] for t in ts]), np.zeros(len(ts), bool)
                else:
                    x = np.array([vals[m][t][0] for t in ts], float)
                    cen = np.array([vals[m][t][1] for t in ts])
                c = STYLE['L']['color'] if m == 'L' else GRAY
                jy = y+rng.uniform(-0.22, 0.22, len(ts))*0.8
                ax.scatter(x[~cen], jy[~cen], s=6, color=c, alpha=0.75, lw=0, zorder=3)
                if cen.any():
                    ax.scatter(x[cen], jy[cen], s=6, facecolors='none', edgecolors=c, lw=0.6, zorder=3)
                med = float(np.median(x))
                ax.plot([med, med], [y-0.28, y+0.28], color=INK, lw=1.3, zorder=4)
                ax.annotate(med_label(med), (med, y+0.28), xytext=(0, 0.5), textcoords='offset points',
                            ha='center', va='bottom', fontsize=6.5, color=INK, zorder=5,
                            bbox=dict(boxstyle='square,pad=0.08', fc='white', ec='none', alpha=0.85))
            ax.set_xscale('log')
            ax.set_xlabel('Branches to reach 1%', labelpad=1)
            ax.set_ylim(0.45, len(rows)+0.6)
            ax.set_yticks(range(len(rows), 0, -1))
            ax.set_yticklabels([r[0] for r in rows])
            ax.get_yticklabels()[-1].set_color(STYLE['L']['color'])
            style(ax)
            ax.grid(False, axis='y')
            fig.savefig(out/('figA_cost_ladder_%s.pdf' % tag))
            fig.savefig(out/('figA_cost_ladder_%s.png' % tag), dpi=300)
            plt.close(fig)

        short = {HARD: 'AB, AABB, ABB', ('A', 'AAB'): 'A, AAB', ('B',): 'B'}
        for base, tag in zip(('F2', 'C1'), 'ab'):
            allx = [vals[base][t][0] for t in tasks]+[vals['L'][t][0] for t in tasks]
            lo, hi = min(allx)/1.6, max(allx)*1.6
            fig = plt.figure(figsize=(1.72, 1.72))
            ax = fig.add_axes([0.27, 0.24, 0.69, 0.69])
            ax.fill_between([lo, hi], [lo, lo], [lo, hi], color=REGION, lw=0, zorder=0)
            ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.7, zorder=1)
            for _, fams, _ in groups:
                fs = FAM_STYLE[fams]
                ts = [t for t in tasks if tasks[t]['family'] in fams]
                x = np.array([vals[base][t][0] for t in ts], float)
                y = np.array([vals['L'][t][0] for t in ts], float)
                cen = np.array([vals[base][t][1] or vals['L'][t][1] for t in ts])
                ax.scatter(x[~cen], y[~cen], s=11, color=fs['color'], marker=fs['marker'], alpha=0.85,
                           edgecolors='white', lw=0.3, zorder=3, label=short[fams])
                if cen.any():
                    ax.scatter(x[cen], y[cen], s=11, facecolors='white', edgecolors=fs['color'], marker=fs['marker'],
                               lw=0.7, zorder=3)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect('equal')
            ax.set_xlabel('%s (branches)' % NAME[base], labelpad=1)
            ax.set_ylabel('Learned (branches)', labelpad=1)
            style(ax)
            if tag == 'a':   # one marker key for the pair, inside the first panel (upper left), frameless, under the points
                leg = ax.legend(loc='upper left', fontsize=6, handlelength=0.8, handletextpad=0.3, labelspacing=0.2,
                                borderpad=0.2, borderaxespad=0.1, frameon=False)
                leg.set_zorder(2)
            fig.savefig(out/('figB_L_vs_baselines_%s.pdf' % tag))
            fig.savefig(out/('figB_L_vs_baselines_%s.png' % tag), dpi=300)
            plt.close(fig)

    # ---------------------------------------------------------------- R: required-atom ratios per family (replaces Table 1)
    # Registered numbers only (analysis 28851720): prep-equal ratio baseline / Learned, 95% bootstrap interval over
    # preparations, preparations not worse. Colour and marker = baseline (as figure D); open = Learned right-censored on
    # every task of the row (ratio between censored values). Text stays in ink.
    boot, grp = ana['bootstrap'], ana['tables']['Q_QoI']['groups']
    ROWS = [('hard', 'Hard', True), ('AB', 'AB', False), ('AABB', 'AABB', False), ('ABB', 'ABB', False), None,
            ('easy', 'Easy', True), ('A', 'A', False), ('AAB', 'AAB', False), ('B', 'B', False)]
    BASES = (('F2', +0.2), ('C1', -0.2))
    data['R'] = {}
    with plt.rc_context(SUB):
        fig = plt.figure(figsize=(3.45, 2.95))
        ax = fig.add_axes([0.20, 0.13, 0.60, 0.75])
        y, ticks, labels = len(ROWS), [], []
        for row in ROWS:
            y -= 1
            if row is None:
                continue
            key, name, bold = row
            g = grp[key]
            allcen = g['right_censored']['L'] == g['tasks']
            data['R'][key] = dict(tasks=g['tasks'], right_censored=g['right_censored'])
            if bold:
                ax.axhspan(y-0.45, y+0.45, color=REGION, lw=0, zorder=0)
            for base, dy in BASES:
                s, b, c = STYLE[base], boot[key]['L_vs_'+base], g['L_vs_'+base]
                ax.plot([b['low'], b['high']], [y+dy, y+dy], color=s['color'], lw=1.0, zorder=2, solid_capstyle='butt')
                kw = dict(mfc='white', mec=s['color'], mew=0.9) if allcen else dict(mfc=s['color'], mec='white', mew=0.4)
                ax.plot([b['ratio']], [y+dy], ls='', marker=s['marker'], ms=4.2, zorder=3, **kw)
                ax.text(1.04, y+dy, '%d/%d' % (c['not_worse'], c['preparations']), transform=ax.get_yaxis_transform(),
                        ha='left', va='center', fontsize=6.5, color=INK)
                data['R'][key][NAME[base]] = dict(ratio=b['ratio'], low=b['low'], high=b['high'],
                                                  not_worse=c['not_worse'], preparations=c['preparations'])
            ticks.append(y)
            labels.append(('%s (%d)' % (name, g['tasks'])) + ('$^{\\dagger}$' if allcen else ''))
        ax.axvline(1.0, color=INK2, lw=0.7, zorder=1)
        ax.set_xscale('log')
        ax.set_xlim(0.3, 6)
        ax.set_xticks([0.5, 1, 2, 4])
        ax.set_xticklabels(['0.5', '1', '2', '4'])
        ax.minorticks_off()
        ax.set_ylim(-0.6, len(ROWS)-0.4)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        for lab, row in zip(ax.get_yticklabels(), [r for r in ROWS if r]):
            if row[2]:
                lab.set_fontweight('bold')
        ax.set_xlabel('Required branches: baseline / Learned', labelpad=1)
        ax.text(1.04, len(ROWS)-0.45, 'not\nworse', transform=ax.get_yaxis_transform(), ha='left', va='bottom',
                fontsize=6.5, color=INK2, linespacing=0.9)
        ax.text(0.98, 0.015, 'Learned cheaper $\\rightarrow$', transform=ax.transAxes, ha='right', va='bottom',
                fontsize=6.5, color=INK2)
        style(ax)
        ax.grid(False, axis='y')
        fig.legend(handles=[Line2D([], [], color=STYLE[b_]['color'], marker=STYLE[b_]['marker'], ms=4, lw=1.0,
                                   mec='white', mew=0.4, label='vs. '+NAME[b_]) for b_, _ in BASES],
                   loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.48, 1.0), fontsize=7,
                   handlelength=1.6, columnspacing=1.2)
        fig.savefig(out/'figR_ratio.pdf')
        fig.savefig(out/'figR_ratio.png', dpi=300)
        plt.close(fig)

    # ---------------------------------------------------------------- D: largest P error against atoms, per family
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.4), sharex=True, sharey=True,
                             gridspec_kw=dict(hspace=0.3, wspace=0.08))
    data['D'] = {}
    for i, row in enumerate((HARD, EASY)):
        for j, fam in enumerate(row):
            ax = axes[i, j]
            ts = [t for t in tasks if tasks[t]['family'] == fam]
            data['D'][fam] = {}
            for m in ('F2', 'C1', 'L'):
                setting = 'K' if m == 'F2' else 'budget'
                grid = sorted({c[setting] for t in ts for c in by[m][t]})
                xs, ys = [], []
                for gv in grid:
                    cs = [c for t in ts for c in by[m][t] if c[setting] == gv and c.get('status') == 'OK']
                    if len(cs) < len(ts):
                        continue
                    xs.append(float(np.median([c['physical_calls'] for c in cs])))
                    ys.append(float(np.median([max(c['P_relative'])*100 for c in cs])))
                o = np.argsort(xs)
                xs, ys = np.array(xs)[o], np.array(ys)[o]
                s = STYLE[m]
                ax.plot(xs, ys, color=s['color'], ls=s['ls'], marker=s['marker'], ms=3.2, lw=1.3 if m == 'L' else 1.0,
                        zorder=3 if m == 'L' else 2, label=NAME[m])
                data['D'][fam][NAME[m]] = dict(atoms=xs.tolist(), max_P_rel_error_pct=ys.tolist())
            ax.axhline(1.0, color=INK, lw=0.8, ls=':', zorder=1)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title('%s (%d tasks)' % (fam, len(ts)), fontsize=7, color=INK, pad=3)
            if i == 1:
                ax.set_xlabel('Total branches (median over tasks)')
            if j == 0:
                ax.set_ylabel('Largest relative error of $P$\nover 40 rounds (%%, median)\n(%s families)'
                              % ('hard' if i == 0 else 'easy'))
            style(ax)
    axes[0, 0].text(axes[0, 0].get_xlim()[0]*1.3, 1.12, 'tolerance', fontsize=6, color=INK2, va='bottom')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=6.5)
    save(fig, 'figD_error_vs_atoms')

    # ---------------------------------------------------------------- E: alpha sweep of Learned
    alphas = sorted({c['budget'] for t in tasks for c in by['L'][t]})
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.6), sharex=True, gridspec_kw=dict(hspace=0.32, wspace=0.3))
    blue = STYLE['L']['color']
    rngE = np.random.default_rng(1)
    data['E'] = {}
    for i, (g, fams) in enumerate((('hard', HARD), ('easy', EASY))):
        ts = sorted(t for t in tasks if tasks[t]['family'] in fams)
        cs = [[next(c for c in by['L'][t] if c['budget'] == al) for t in ts] for al in alphas]
        if any(c.get('status') != 'OK' for row in cs for c in row):
            raise ValueError('Learned run not OK in the alpha sweep')
        atoms = np.array([[c['physical_calls'] for c in row] for row in cs], float)
        err = np.array([[max(c['P_relative'])*100 for c in row] for row in cs])
        forced = np.array([[c['forced'] > 0 for c in row] for row in cs])
        passed = np.array([[c['Q_QoI'] <= 1 for c in row] for row in cs])
        al = np.array(alphas)
        # left: atoms used
        ax = axes[i, 0]
        ax.fill_between(al, np.percentile(atoms, 25, 1), np.percentile(atoms, 75, 1), color=blue, alpha=0.15, lw=0)
        ax.plot(al, np.median(atoms, 1), color=blue, marker='o', ms=3.2, lw=1.4)
        ax.axhline(256*40, color=INK2, lw=0.8, ls=':')
        ax.text(al[-1], 256*40*1.12, 'cap: 256 branches per round', ha='right', va='bottom', fontsize=6, color=INK2)
        ax.set_yscale('log')
        ax.set_ylabel('%s families (%d tasks)\nTotal branches used' % ('Hard' if g == 'hard' else 'Easy', len(ts)))
        # right: largest P error, every run
        ax = axes[i, 1]
        jit = np.exp(rngE.uniform(-0.13, 0.13, err.shape))
        x = al[:, None]*jit
        e = np.maximum(err, 1e-3)
        ax.scatter(x[~forced], e[~forced], s=7, color=blue, alpha=0.45, lw=0, zorder=2)
        ax.scatter(x[forced], e[forced], s=7, facecolors='none', edgecolors=GRAY, lw=0.6, zorder=2)
        ax.plot(al, np.median(e, 1), color=blue, lw=1.6, zorder=3)
        ax.plot(al, al, color=INK2, lw=0.9, ls='--', zorder=1)
        ax.axhline(1.0, color=INK, lw=0.8, ls=':', zorder=1)
        ax.set_yscale('log')
        ax.set_ylabel('Largest relative error\nof $P$ over 40 rounds (%)')
        k = alphas.index(0.5)
        ax.set_title('at the default alpha = 0.5: %d of %d tasks within tolerance'
                     % (int(passed[k].sum()), len(ts)), fontsize=6.5, color=INK, pad=3)
        data['E'][g] = {str(a_): dict(median_atoms=float(np.median(atoms[j])), median_err_pct=float(np.median(err[j])),
                                       within_tolerance=int(passed[j].sum()), runs_with_forced_merges=int(forced[j].sum()),
                                       tasks=len(ts)) for j, a_ in enumerate(alphas)}
    for ax in axes.flat:
        ax.set_xscale('log', base=2)
        ax.set_xticks(alphas)
        ax.set_xticklabels(['1/8', '1/4', '1/2', '1', '2', '4', '8', '16', '32', '64', '128'][:len(alphas)])
        ax.axvline(0.5, color=INK2, lw=0.6, ls=(0, (1, 2)), zorder=0)
        style(ax)
    for ax in axes[1]:
        ax.set_xlabel('Error budget scale alpha (log scale)')
    for ax in axes[:, 0]:
        ax.text(0.52, 0.04, ' default 0.5', fontsize=6, color=INK2, va='bottom', transform=ax.get_xaxis_transform())
    fig.legend(handles=[Line2D([], [], color=blue, lw=1.6, marker='o', ms=3, label='Learned (ours): median'),
                        Line2D([], [], ls='', marker='o', ms=3, color=blue, alpha=0.6, label='one run'),
                        Line2D([], [], ls='', marker='o', ms=3, mfc='none', mec=GRAY,
                               label='one run that hit the branch cap (forced merges)'),
                        Line2D([], [], color=INK2, lw=0.9, ls='--', label='error allowed by the budget (alpha x 1%)'),
                        Line2D([], [], color=INK, lw=0.8, ls=':', label='tolerance 1%')],
               loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04), fontsize=6.3)
    save(fig, 'figE_alpha_sweep')

    (out/'data.json').write_text(json.dumps(dict(test_job=a.test_job, analysis_job=a.analysis_job, checks='passed',
                                                 names=NAME, figures=data), indent=1))
    print(json.dumps({k: v for k, v in data.items() if k != 'D'}, indent=1))


if __name__ == '__main__':
    main()

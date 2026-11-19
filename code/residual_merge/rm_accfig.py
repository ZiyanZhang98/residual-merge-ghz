"""Running maximum of the P and F error over rounds, per family, at a matched branch budget (post hoc, descriptive).
Learned at the frozen default alpha = 0.5. On each task, F_1 runs at its first K grid point with at least as many total
branches as Learned; F_0 and F_2 use the same K (same merges, different correction). M_0 at its first budget grid
point with at least as many branches. Per task the running max over rounds; then the median over the family's tasks."""
import json, glob, sys, numpy as np
from collections import defaultdict
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
out = Path(sys.argv[1]); ALPHA = 0.5
cells, se = [], {}
for d in sorted(glob.glob('results/test_28851717_*')):
    cells += json.load(open(d+'/cells.json'))
    for f in glob.glob(d+'/mc_*.npz'):
        z = np.load(f); se[Path(f).stem[3:]] = dict(P=z['se_P_relative']*100, F=z['se_F'])
by = defaultdict(lambda: defaultdict(list)); fam = {}
for c in cells:
    if c.get('status') == 'OK': by[c['method']][c['task_id']].append(c)
    fam[c['task_id']] = c['family']
pick = defaultdict(dict)
for t, cs in by['L'].items():
    l = [c for c in cs if c['budget'] == ALPHA][0]; pick['L'][t] = l
    for m in ('F2', 'C1'):
        cand = sorted(by[m][t], key=lambda c: c['physical_calls'])
        pick[m][t] = [c for c in cand if c['physical_calls'] >= l['physical_calls']][0]
    for m in ('M0', 'F2mul'):
        same = [c for c in by[m][t] if c['K'] == pick['F2'][t]['K']]
        assert len(same) == 1 and same[0]['physical_calls'] == pick['F2'][t]['physical_calls'], (m, t)
        pick[m][t] = same[0]
def runmax(c, key, scale):
    return np.maximum.accumulate(np.abs(np.array(c[key]))*scale) if key in c else np.full(40, np.inf)
INK, INK2, BAND = '#0b0b0b', '#52514e', '#e4e2dc'
ORDER = ('M0', 'F2', 'F2mul', 'C1', 'L')
NAME = {'M0': '$F_0$', 'F2': '$F_1$', 'F2mul': '$F_2$', 'C1': '$M_0$', 'L': 'Learned (ours)'}
STY = {'M0': dict(color='#cf8fc4', ls=':', lw=1.1), 'F2': dict(color='#b0479b', ls='-.', lw=1.0),
       'F2mul': dict(color='#6e2a62', ls=(0, (5, 1.6)), lw=1.0), 'C1': dict(color='#2e9e8f', ls=(0, (2.2, 1.4)), lw=1.0),
       'L': dict(color='#2a78d6', ls='-', lw=1.5)}
SUB = {'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix', 'font.size': 8,
       'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK, 'axes.linewidth': 0.6,
       'xtick.labelsize': 8, 'ytick.labelsize': 8}
FAMS = [('AB', 'Hard'), ('AABB', 'Hard'), ('ABB', 'Hard'), ('A', 'Easy'), ('AAB', 'Easy'), ('B', 'Easy')]
R = np.arange(1, 41); table = {}
with plt.rc_context(SUB):
    fig, axes = plt.subplots(2, 6, figsize=(7.16, 2.85), sharex=True, sharey='row',
                             gridspec_kw=dict(hspace=0.10, wspace=0.17, left=0.08, right=0.99, top=0.74, bottom=0.12))
    for j, (f, _) in enumerate(FAMS):
        ts = sorted(t for t in pick['L'] if fam[t] == f); table[f] = {}
        for i, (key, scale, tol, sk, ylab, lim) in enumerate((
                ('P_relative', 100, 1.0, 'P', 'Error of $P$(%)', (0.02, 20)),
                ('F_absolute', 1, 0.01, 'F', 'Error of $F$', (1e-5, 0.1)))):
            ax = axes[i, j]
            band = np.maximum.accumulate(2*np.median([se[t][sk] for t in ts], axis=0))
            ax.fill_between(R, 1e-12, band, color=BAND, lw=0, zorder=0)
            ax.axhline(tol, color=INK, lw=0.6, zorder=1)
            for m in ORDER:
                y = np.median([runmax(pick[m][t], key, scale) for t in ts], axis=0)
                ax.plot(R, y, zorder=4 if m == 'L' else 2, label=NAME[m], **STY[m])
                table[f].setdefault(m, {})[sk] = float(y[-1])
            ax.set_yscale('log'); ax.set_xlim(1, 40); ax.set_ylim(*lim)
            ax.spines[['top', 'right']].set_visible(False); ax.tick_params(length=2.5, pad=1.5)
            if i == 0: ax.set_title(f, fontsize=8, pad=2)
            if j == 0: ax.set_ylabel(ylab, labelpad=1, linespacing=1.0)
        axes[1, j].set_xticks([1, 20, 40])
    fig.supxlabel('Round', fontsize=8, y=0.005)
    for a0, a1, t in ((0, 2, 'Hard group'), (3, 5, 'Easy group')):
        l, r = axes[0, a0].get_position().x0, axes[0, a1].get_position().x1
        fig.text((l+r)/2, 0.835, t, ha='center', va='bottom', fontsize=8, color=INK)
        fig.add_artist(plt.Line2D([l, r], [0.83, 0.83], color=INK2, lw=0.6))
    h, lb = axes[0, 0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color=INK, lw=0.6), Patch(color=BAND)]; lb += ['tolerance', 'reference uncertainty']
    fig.legend(h, lb, loc='upper center', ncol=7, frameon=False, bbox_to_anchor=(0.535, 1.0), fontsize=7.5,
               handlelength=2.2, columnspacing=1.0, handletextpad=0.5)
    fig.savefig(out/'fig_accuracy_rounds.png', dpi=300); fig.savefig(out/'fig_accuracy_rounds.pdf')
for f, d in table.items():
    print(f, '  '.join('%s P %.2f F %.4f' % (m, d[m]['P'], d[m]['F']) for m in ORDER))

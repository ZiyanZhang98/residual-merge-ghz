"""Paper figures and summaries from existing cells (HANDOFF 5j item 22.40, 2026-11-04, user-approved). CPU, read-only:
no simulation, no new truth, no training; the frozen 45-task final is never accessed.

Inputs: the n=4 test shards (22.37), the n=3 evaluation shards (22.38) and the n=4 development grids (22.35, via
rm_test_report.development_cells). Each source is verified shard by shard as in its own analysis, and the needed atoms
recomputed here must equal the stored analysis tables exactly (otherwise the job fails), so the figures share the
registered definition (rm_deploy.needed: fewest total atoms from which every costlier grid point passes Q_QoI <= 1).

Outputs:
  figs_data.json  accuracy-atoms curves (median atoms and Q per grid point), pass fraction against an atom budget,
                  alpha calibration of L (Q against alpha, split by whether the 256-atom cap forced merges), forced-merge
                  counts for L and C1, wall-clock break-even per atom against F2 and C1, the needed-atom cross-check.
  points.json     every cell used (task, family, method, grid value, atoms, Q, Q_high, forced, seconds) for replotting.
  fig2_pass_fraction, fig2b_accuracy_atoms, fig3_alpha_calibration, fig_n3_L_vs_C1 (.png and .pdf) if matplotlib is
  available; the JSON is written either way.
"""
import rm_runtime as rt
from collections import defaultdict
import numpy as np
from rm_deploy import needed, HARD
from rm_ext import EASY
from rm_eval2 import verified_shard

METHODS = ('L', 'C1', 'F2')
FAMILIES = HARD+EASY
GROUPS = dict(hard=HARD, easy=EASY, **{f: (f,) for f in FAMILIES})
# validated categorical slots (dataviz reference palette, light surface), fixed order
METHOD_COLOR = dict(L='#2a78d6', C1='#eb6834', F2='#1baf7a')
METHOD_STYLE = dict(L=('-', 'o'), C1=('--', 's'), F2=(':', '^'))
FAMILY_COLOR = dict(zip(FAMILIES, ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300')))
FAMILY_MARKER = dict(zip(FAMILIES, ('o', 's', '^', 'D', 'v', 'P')))
INK, MUTED = '#0b0b0b', '#52514e'


def grid_value(c):
    return c['K'] if c['method'] == 'F2' else c['budget']


def q_of(c, key='Q_QoI'):
    return c.get(key) if c.get('status') == 'OK' else None


# ------------------------------------------------------------------ sources

def test_source(cfg, fc, inputs):
    from rm_test import DEPLOY_KEYS, DEPLOY_FILES
    roots = sorted((rt.HERE/'results').glob('test_%s_*' % fc['test_run_job']))
    if len(roots) != cfg['test']['shards']:
        raise ValueError('Expected %d test shards, found %d' % (cfg['test']['shards'], len(roots)))
    cells = []
    for r in roots:
        verified_shard(r, 'TEST_SHARD_COMPLETE', DEPLOY_KEYS, DEPLOY_FILES)
        inputs[r.name] = rt.sha(r/'manifest.json')
        cells += rt.read(r/'cells.json')
    a = rt.HERE/'results'/('testa_%s' % fc['test_analysis_job'])
    inputs[a.name] = rt.sha(a/'manifest.json')
    return cells, rt.read(a/'analysis.json')['tables']['Q_QoI']['needed']


def n3_source(cfg, fc, inputs):
    from rm_n3 import N3_KEYS, N3_FILES
    roots = sorted((rt.HERE/'results').glob('n3eval_%s_*' % fc['n3_eval_job']))
    if len(roots) != cfg['n3']['eval_shards']:
        raise ValueError('Expected %d n=3 shards, found %d' % (cfg['n3']['eval_shards'], len(roots)))
    cells = []
    for r in roots:
        verified_shard(r, 'N3_EVAL_SHARD_COMPLETE', N3_KEYS, N3_FILES)
        inputs[r.name] = rt.sha(r/'manifest.json')
        cells += rt.read(r/'cells.json')
    a = rt.HERE/'results'/('n3a_%s' % fc['n3_analysis_job'])
    inputs[a.name] = rt.sha(a/'manifest.json')
    return cells, rt.read(a/'analysis.json')['tables']['Q_QoI']['needed']


def dev_source(cfg, fc, inputs):
    from rm_test_report import development_cells
    cells, tasks = development_cells(cfg, inputs)
    fam = {t['id']: t['family'] for t in tasks}
    ids = set(fam)
    cells = [dict(c, family=c.get('family') or fam[c['task_id']]) for c in cells if c['task_id'] in ids]
    a = rt.HERE/'results'/('exta_%s' % cfg['test']['ext_analysis_job'])
    inputs[a.name] = rt.sha(a/'manifest.json')
    return cells, rt.read(a/'analysis.json')['extended']['Q_QoI']['needed']


# ------------------------------------------------------------------ summaries

def buckets(cells):
    b = {m: defaultdict(list) for m in METHODS}
    for c in cells:
        if c.get('method') in b:
            b[c['method']][c['task_id']].append(c)
    return b


def needed_atoms(b):
    return {m: {t: needed([(c['physical_calls'], q_of(c)) for c in v]) for t, v in b[m].items()} for m in METHODS}


def crosscheck(need, stored):
    bad = []
    for m in METHODS:
        for t, (atoms, cens) in need[m].items():
            s = stored[m].get(t)
            if s is None or s['atoms'] != atoms or bool(s['censored']) != bool(cens):
                bad.append([m, t, atoms, cens, s])
        missing = sorted(set(stored[m])-set(need[m]))
        bad += [[m, t, 'missing'] for t in missing]
    return dict(tasks=len(need['L']), mismatches=len(bad), examples=bad[:20], passed=not bad)


def family_of(b):
    return {t: v[0]['family'] for m in METHODS for t, v in b[m].items()}


def curves(b, fam):
    """Per group and method: for each grid value, median atoms and the 10/50/90 % quantiles of Q over tasks."""
    out = {}
    for g, fams in GROUPS.items():
        res = {}
        for m in METHODS:
            per = defaultdict(list)
            for t, v in b[m].items():
                if fam[t] in fams:
                    for c in v:
                        per[grid_value(c)].append(c)
            rows = []
            for gv in sorted(per):
                cs = per[gv]
                qs = [q_of(c) for c in cs if q_of(c) is not None]
                rows.append(dict(grid=gv, tasks=len(cs), median_atoms=float(np.median([c['physical_calls'] for c in cs])),
                                 Q10=float(np.quantile(qs, .1)) if qs else None, Q50=float(np.median(qs)) if qs else None,
                                 Q90=float(np.quantile(qs, .9)) if qs else None,
                                 pass_fraction=sum(q is not None and q <= 1 for q in [q_of(c) for c in cs])/len(cs)))
            res[m] = rows
        out[g] = res
    return out


def pass_curves(need, fam):
    """Share of tasks whose needed atoms (registered suffix rule) are within an atom budget; censored never pass."""
    allv = [a for m in METHODS for a, _ in need[m].values()]
    budgets = np.unique(np.round(np.geomspace(max(1, min(allv)), max(allv), 60))).tolist()
    out = {}
    for g, fams in GROUPS.items():
        res = dict(budgets=budgets)
        for m in METHODS:
            vals = [(a, cz) for t, (a, cz) in need[m].items() if fam[t] in fams]
            res[m] = dict(tasks=len(vals), censored=sum(cz for _, cz in vals),
                          fraction=[sum((not cz) and a <= B for a, cz in vals)/len(vals) if vals else None for B in budgets],
                          needed_sorted=sorted([a, bool(cz)] for a, cz in vals))
        out[g] = res
    return out


def alpha_calibration(b, fam):
    """L: Q against alpha. Item 22.37 T4 predicts Q <= alpha if every predicted cost bounded its true residual; merges
    forced by the 256-atom cap are outside the budget, so runs with forced merges are reported separately."""
    out = {}
    for g, fams in GROUPS.items():
        rows = []
        per = defaultdict(list)
        for t, v in b['L'].items():
            if fam[t] in fams:
                for c in v:
                    per[c['budget']].append(c)
        for a in sorted(per):
            cs = [c for c in per[a] if q_of(c) is not None]
            row = dict(alpha=a, runs=len(per[a]))
            for name, sel in (('all', cs), ('no_forced', [c for c in cs if not c.get('forced')]),
                              ('forced', [c for c in cs if c.get('forced')])):
                qs = np.array([q_of(c) for c in sel])
                row[name] = dict(runs=len(sel), median_Q=float(np.median(qs)) if len(qs) else None,
                                 median_Q_over_alpha=float(np.median(qs/a)) if len(qs) else None,
                                 share_Q_le_alpha=float(np.mean(qs <= a)) if len(qs) else None,
                                 share_Q_le_1=float(np.mean(qs <= 1)) if len(qs) else None)
            rows.append(row)
        out[g] = rows
    return out


def forced_counts(b, fam):
    out = {}
    for m in ('L', 'C1'):
        per = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for t, v in b[m].items():
            for c in v:
                for g in ('all', fam[t]):
                    x = per[c['budget']][g]
                    x[0] += 1
                    x[1] += bool(c.get('forced'))
        out[m] = {str(k): {g: dict(runs=x[0], with_forced=x[1]) for g, x in v.items()} for k, v in sorted(per.items())}
    return out


def breakeven(b, need, fam):
    """Wall-clock break-even against a baseline: at each method's needed point, the per-atom propagation cost at which
    L's extra seconds equal the seconds the baseline spends on its extra atoms, c* = (T_L - T_base)/(A_base - A_L), for
    tasks where L needs fewer atoms and neither is censored. Compared with the observed non-correction seconds per atom
    of the F2 run at its needed point (propagation and Lloyd merging)."""
    def at(m, t):
        a, cz = need[m][t]
        if cz:
            return None
        cs = [c for c in b[m][t] if c['physical_calls'] == a and q_of(c) is not None]
        return min(cs, key=lambda c: c.get('seconds') or 0.) if cs else None
    out = {}
    for base in ('F2', 'C1'):
        rows = []
        for t in need['L']:
            cl, cb = at('L', t), at(base, t)
            if cl is None or cb is None or cl.get('seconds') is None or cb.get('seconds') is None:
                continue
            al, ab = cl['physical_calls'], cb['physical_calls']
            if ab <= al:
                continue
            f2 = at('F2', t)
            per_atom = None
            if f2 is not None and f2.get('seconds') is not None:
                per_atom = (f2['seconds']-(f2.get('correction_seconds') or 0.))/f2['physical_calls']
            rows.append(dict(task=t, family=fam[t], L_atoms=al, base_atoms=ab, L_seconds=cl['seconds'],
                             L_merge_seconds=cl.get('merge_seconds'), base_seconds=cb['seconds'],
                             breakeven_seconds_per_atom=(cl['seconds']-cb['seconds'])/(ab-al),
                             F2_observed_seconds_per_atom=per_atom))
        res = {}
        for g, fams in GROUPS.items():
            rs = [r for r in rows if r['family'] in fams]
            if not rs:
                continue
            be = np.array([r['breakeven_seconds_per_atom'] for r in rs])
            ob = np.array([r['F2_observed_seconds_per_atom'] for r in rs if r['F2_observed_seconds_per_atom'] is not None])
            ratio = [r['breakeven_seconds_per_atom']/r['F2_observed_seconds_per_atom'] for r in rs
                     if r['F2_observed_seconds_per_atom']]
            res[g] = dict(tasks=len(rs), median_breakeven_seconds_per_atom=float(np.median(be)),
                          q25=float(np.quantile(be, .25)), q75=float(np.quantile(be, .75)),
                          median_F2_observed_seconds_per_atom=float(np.median(ob)) if len(ob) else None,
                          median_ratio_breakeven_to_observed=float(np.median(ratio)) if ratio else None)
        out['L_vs_'+base] = dict(groups=res, rows=rows)
    return out


def points(cells):
    keep = ('task_id', 'family', 'method', 'physical_calls', 'Q_QoI', 'Q_QoI_high', 'forced', 'seconds',
            'merge_seconds', 'correction_seconds', 'status')
    return [dict({k: c.get(k) for k in keep}, grid=grid_value(c)) for c in cells if c.get('method') in METHODS]


def summarize(cells, stored):
    b = buckets(cells)
    fam = family_of(b)
    need = needed_atoms(b)
    return b, fam, need, dict(crosscheck=crosscheck(need, stored), curves=curves(b, fam), pass_curves=pass_curves(need, fam),
                              alpha_calibration=alpha_calibration(b, fam), forced=forced_counts(b, fam),
                              breakeven=breakeven(b, need, fam))


# ------------------------------------------------------------------ figures

def figures(root, data, raw):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:   # the JSON is the deliverable; figures are a convenience
        return dict(made=False, error=repr(exc))
    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.5,
                         'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                         'savefig.facecolor': '#fcfcfb', 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
    made = []
    title = dict(test='n=4 test set', n3='n=3 held-out set', dev='n=4 development set')

    def save(fig, name):
        for ext in ('png', 'pdf'):
            fig.savefig(root/(name+'.'+ext), dpi=200, bbox_inches='tight')
        plt.close(fig)
        made.append(name)

    settings = [s for s in ('test', 'n3') if s in data]
    # fig 2: pass fraction against total atoms
    fig, ax = plt.subplots(len(settings), 2, figsize=(7, 2.6*len(settings)), squeeze=False, constrained_layout=True)
    for i, s in enumerate(settings):
        for j, g in enumerate(('hard', 'easy')):
            pc, a = data[s]['pass_curves'][g], ax[i][j]
            for m in METHODS:
                ls, _ = METHOD_STYLE[m]
                a.step(pc['budgets'], pc[m]['fraction'], where='post', color=METHOD_COLOR[m], lw=2, ls=ls,
                       label='%s (%d/%d censored)' % (m, pc[m]['censored'], pc[m]['tasks']))
            a.set_xscale('log')
            a.set_ylim(0, 1.02)
            a.set_title('%s, %s families' % (title[s], g), color=INK)
            a.set_xlabel('total atoms (40 rounds)')
            a.set_ylabel('share of tasks at Q <= 1')
            a.legend(loc='upper left')
    save(fig, 'fig2_pass_fraction')
    # fig 2b: accuracy against atoms per family (median over tasks at each grid point)
    fig, ax = plt.subplots(len(settings), len(FAMILIES), figsize=(13, 2.5*len(settings)), squeeze=False, sharey=True, constrained_layout=True)
    for i, s in enumerate(settings):
        for j, f in enumerate(FAMILIES):
            a = ax[i][j]
            for m in METHODS:
                rows = [r for r in data[s]['curves'][f][m] if r['Q50'] is not None]
                if not rows:
                    continue
                ls, mk = METHOD_STYLE[m]
                a.plot([r['median_atoms'] for r in rows], [max(r['Q50'], 1e-3) for r in rows], color=METHOD_COLOR[m],
                       lw=2, ls=ls, marker=mk, ms=4, label=m)
            a.axhline(1, color=MUTED, lw=1, ls='--')
            a.set_xscale('log')
            a.set_yscale('log')
            a.set_title('%s: %s' % ('n=4 test' if s == 'test' else 'n=3', f), color=INK)
            if i == len(settings)-1:
                a.set_xlabel('median total atoms')
            if j == 0:
                a.set_ylabel('median Q (error / tolerance)')
    ax[0][0].legend(loc='lower left')
    save(fig, 'fig2b_accuracy_atoms')
    # fig 3: alpha calibration of L (every run; hollow = the atom cap forced merges)
    cal = [s for s in ('test', 'n3', 'dev') if s in raw]
    fig, ax = plt.subplots(1, len(cal), figsize=(4*len(cal), 3.4), squeeze=False, sharey=True, constrained_layout=True)
    rng = np.random.default_rng(0)
    for j, s in enumerate(cal):
        a = ax[0][j]
        for f in FAMILIES:
            cs = [c for c in raw[s] if c.get('method') == 'L' and c['family'] == f and q_of(c) is not None]
            if not cs:
                continue
            x = np.array([c['budget'] for c in cs])*np.exp(rng.uniform(-0.12, 0.12, len(cs)))
            y = np.clip([q_of(c) for c in cs], 1e-3, 1e3)
            forced = np.array([bool(c.get('forced')) for c in cs])
            a.scatter(x[~forced], y[~forced], s=12, color=FAMILY_COLOR[f], marker=FAMILY_MARKER[f], label=f, lw=0)
            a.scatter(x[forced], y[forced], s=12, facecolors='none', edgecolors=FAMILY_COLOR[f], marker=FAMILY_MARKER[f], lw=0.8)
        al = np.geomspace(0.1, 160, 50)
        a.plot(al, al, color=INK, lw=1, label='Q = alpha')
        a.axhline(1, color=MUTED, lw=1, ls='--')
        a.set_xscale('log')
        a.set_yscale('log')
        a.set_xlabel('alpha (total error budget / tolerance)')
        a.set_title(title[s]+' (hollow: cap forced merges)', color=INK)
    ax[0][0].set_ylabel('Q (error / tolerance)')
    ax[0][0].legend(loc='upper left', ncol=2)
    save(fig, 'fig3_alpha_calibration')
    # n=3 AB and AABB: L and C1 per task
    if 'n3' in raw:
        fig, ax = plt.subplots(1, 2, figsize=(8, 3.2), squeeze=False, sharey=True, constrained_layout=True)
        for j, f in enumerate(('AB', 'AABB')):
            a = ax[0][j]
            for m in ('L', 'C1'):
                per = defaultdict(list)
                for c in raw['n3']:
                    if c.get('method') == m and c['family'] == f and q_of(c) is not None:
                        per[c['task_id']].append(c)
                ls, mk = METHOD_STYLE[m]
                for k, (t, cs) in enumerate(sorted(per.items())):
                    cs = sorted(cs, key=lambda c: c['physical_calls'])
                    a.plot([c['physical_calls'] for c in cs], [max(q_of(c), 1e-3) for c in cs], color=METHOD_COLOR[m],
                           lw=1, ls=ls, alpha=0.45, label=m if k == 0 else None)
            a.axhline(1, color=MUTED, lw=1, ls='--')
            a.set_xscale('log')
            a.set_yscale('log')
            a.set_xlabel('total atoms')
            a.set_title('n=3 %s: one line per task' % f, color=INK)
        ax[0][0].set_ylabel('Q (error / tolerance)')
        ax[0][0].legend(loc='lower left')
        save(fig, 'fig_n3_L_vs_C1')
    return dict(made=True, files=made)


def run_figs(cfg, root, inputs, save):
    fc = cfg['figs']
    data, raw = {}, {}
    for name, src in (('test', test_source), ('n3', n3_source), ('dev', dev_source)):
        cells, stored = src(cfg, fc, inputs)
        b, fam, need, out = summarize(cells, stored)
        if not out['crosscheck']['passed']:
            raise ValueError('Needed atoms differ from the stored %s analysis: %s' % (name, out['crosscheck']['examples'][:3]))
        data[name] = out
        raw[name] = [c for c in cells if c.get('method') in METHODS]
    save('points.json', {k: points(v) for k, v in raw.items()})
    data['figures'] = figures(root, data, raw)
    return data

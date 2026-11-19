"""Registered analysis of the independent n=4 test set (HANDOFF 5j item 22.37), written and smoke-checked on the
development grids before any test result exists; it runs once on the test set.

Needed atoms, prep-equal comparisons and censoring counts are the 22.35 table (rm_ext.table), so test and development
numbers share one definition. Added: the main reading (L against the stronger of F2 and C1 on the hard families:
REPRODUCED if >= 1.3 with >= 2/3 preparations not worse, SAME_DIRECTION if > 1, else NOT_REPRODUCED), the easy-family
non-regression against F2 (>= 1.0), preparation bootstrap intervals, pass rates at the default alpha (0.5) with Wilson
intervals, the reference methods M0 (merge without correction) and F2mul, plain Monte Carlo cost at 1 % accuracy,
truth resolvability and run costs. Final data are never accessed.
"""
import rm_runtime as rt
import math
from collections import defaultdict
import numpy as np
from rm_deploy import needed, compare, HARD
from rm_ext import table, EASY
from rm_eval2 import m3_cells, verified_shard, M3_KEYS, M3_MODEL_FILES

MAIN = ('L', 'C1', 'F2')
EXTRA = ('M0', 'F2mul')


def groups():
    g = {'hard': HARD, 'easy': EASY, 'all': HARD+EASY}
    g.update({f: (f,) for f in HARD+EASY})
    return g


def buckets_of(cells, ids, methods):
    b = {m: defaultdict(list) for m in methods}
    for c in cells:
        if c.get('method') in b and c['task_id'] in ids:
            b[c['method']][c['task_id']].append(c)
    return b


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return None
    p, d = k/n, 1+z*z/n
    c, h = (p+z*z/(2*n))/d, z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-h, c+h]


def bootstrap(values, a, b, ts, reps, seed):
    """Percentile interval of compare()'s prep-equal geometric-mean ratio a/b, resampling preparations."""
    preps = sorted({t['prep_signature'] for t in ts})
    if not preps:
        return None
    x = []
    for p in preps:
        ids = [t['id'] for t in ts if t['prep_signature'] == p]
        x.append(math.log(float(np.median([values[a][i][0] for i in ids]))/float(np.median([values[b][i][0] for i in ids]))))
    x = np.asarray(x)
    m = np.exp(x[np.random.default_rng(seed).integers(0, len(x), size=(reps, len(x)))].mean(1))
    return dict(ratio=float(np.exp(x.mean())), low=float(np.quantile(m, 0.025)), high=float(np.quantile(m, 0.975)),
                preparations=len(preps), reps=reps)


def reading(cfg, hard):
    d = cfg['test']['decision']
    vs = {b: hard['L_vs_'+b] for b in ('F2', 'C1')}
    best = min(vs, key=lambda b: vs[b]['ratio'])
    m = vs[best]
    if m['ratio'] >= d['ratio'] and m['not_worse'] >= d['fraction']*m['preparations']-1e-9:
        r = 'REPRODUCED'
    elif m['ratio'] > 1.0:
        r = 'SAME_DIRECTION'
    else:
        r = 'NOT_REPRODUCED'
    return dict(stronger_baseline=best, ratio=m['ratio'], not_worse=m['not_worse'], preparations=m['preparations'],
                reading=r, rule='REPRODUCED if >= %.2f with >= 2/3 preparations not worse; SAME_DIRECTION if > 1; '
                                'else NOT_REPRODUCED' % d['ratio'])


def summarize(cfg, tasks, cells):
    tc, G = cfg['test'], groups()
    ids = {t['id'] for t in tasks}
    main = table(buckets_of(cells, ids, MAIN), tasks, G)
    out = dict(tables=main, tasks=len(tasks))
    vals = {key: {m: {t: (v['atoms'], v['censored']) for t, v in vm.items()} for m, vm in main[key]['needed'].items()}
            for key in main}
    out['reading'] = reading(cfg, main['Q_QoI']['groups']['hard'])
    out['reading_Q_high'] = reading(cfg, main['Q_QoI_high']['groups']['hard'])
    e = main['Q_QoI']['groups']['easy']['L_vs_F2']
    out['easy_not_worse'] = dict(ratio=e['ratio'], not_worse=e['not_worse'], preparations=e['preparations'],
                                 passed=e['ratio'] is not None and e['ratio'] >= tc['decision']['easy_ratio'])
    bs = tc['bootstrap']
    out['bootstrap'] = {g: {'L_vs_'+b: bootstrap(vals['Q_QoI'], b, 'L', [t for t in tasks if t['family'] in fams],
                                                 bs['reps'], bs['seed']) for b in ('F2', 'C1')} for g, fams in G.items()}
    # default alpha: share of tasks whose L run at alpha=default meets the tolerance without looking at the reference
    da, out['default_alpha'] = tc['default_alpha'], {}
    for g, fams in G.items():
        ts = [t['id'] for t in tasks if t['family'] in fams]
        rows = [c for c in cells if c.get('method') == 'L' and c.get('budget') == da and c['task_id'] in ts]
        ok = lambda c, key: c.get('status') == 'OK' and c.get(key) is not None and c[key] <= 1  # noqa: E731
        k = sum(ok(c, 'Q_QoI') for c in rows)
        qs = [c['Q_QoI'] for c in rows if c.get('Q_QoI') is not None]
        out['default_alpha'][g] = dict(alpha=da, tasks=len(ts), cells=len(rows), passed=k,
                                       passed_by_low=sum(ok(c, 'Q_QoI_low') for c in rows),
                                       passed_by_high=sum(ok(c, 'Q_QoI_high') for c in rows),
                                       fraction=k/len(ts) if ts else None, wilson95=wilson(k, len(ts)),
                                       max_Q=max(qs) if qs else None)
    # reference methods from the F2 rollouts (no correction; multiplicative PTA), where every task has points
    xb = buckets_of(cells, ids, EXTRA)
    point = lambda c: (c['physical_calls'], c.get('Q_QoI') if c.get('status') == 'OK' else None)  # noqa: E731
    xv = {m: {t: needed([point(c) for c in v]) for t, v in xb[m].items()} for m in EXTRA}
    vv = dict(vals['Q_QoI'], **xv)
    out['reference_methods'] = {}
    for g, fams in G.items():
        ts = [t for t in tasks if t['family'] in fams]
        out['reference_methods'][g] = {'L_vs_'+m: compare(vv, m, 'L', ts) for m in EXTRA
                                       if ts and all(t['id'] in xv[m] for t in ts)}
    return out


# ------------------------------------------------------------------ development grids (smoke check)

def development_cells(cfg, inputs):
    """The 22.35 inputs: M3 (L, C1), the extension shards (L, C1, F2 K=1,2) and step-1 F2 cells (K 4..256)."""
    from rm_deploy import task_list
    tasks, npz, step1 = task_list(cfg, inputs)
    cells = [c for c in m3_cells(cfg, inputs) if c.get('method') in ('L', 'C1')]
    roots = sorted((rt.HERE/'results').glob('ext_%s_*' % cfg['test']['ext_deploy_job']))
    if len(roots) != cfg['ext']['shards']:
        raise ValueError('Extension shards missing')
    for r in roots:
        verified_shard(r, 'EXT_SHARD_COMPLETE', M3_KEYS+('ext',), M3_MODEL_FILES+('rm_ext.py', 'rm_deploy.py', 'rm_eval2.py'))
        inputs[r.name] = rt.sha(r/'manifest.json')
        cells += rt.read(r/'cells.json')
    cells += [dict(c, method='F2') for c in step1 if c['arm'] == cfg['f2']['arm'] and c['K'] in cfg['f2']['K']]
    return cells, tasks


def matches_ext(cfg, s):
    """Every group ratio, not-worse count and censoring count equals the stored 22.35 analysis (extended grids)."""
    ext = rt.read(rt.HERE/'results'/('exta_%s' % cfg['test']['ext_analysis_job'])/'analysis.json')['extended']
    bad = []
    for key in ('Q_QoI', 'Q_QoI_high'):
        for g, want in ext[key]['groups'].items():
            got = s['tables'][key]['groups'][g]
            for k in ('L_vs_F2', 'L_vs_C1', 'C1_vs_F2'):
                a, b = got[k], want[k]
                if a['not_worse'] != b['not_worse'] or (a['ratio'] is None) != (b['ratio'] is None) or (
                        a['ratio'] is not None and abs(a['ratio']-b['ratio']) > 1e-12):
                    bad.append([key, g, k])
            if got['left_censored'] != want['left_censored'] or got['right_censored'] != want['right_censored']:
                bad.append([key, g, 'censoring'])
    return dict(groups=len(ext['Q_QoI']['groups']), mismatches=bad, passed=not bad)


# ------------------------------------------------------------------ test analysis (runs once)

def analyze(cfg, pool_job, run_job, inputs):
    from rm_test import DEPLOY_KEYS, DEPLOY_FILES, grid
    tc = cfg['test']
    proot = rt.HERE/'results'/('testpool_%s' % pool_job)
    verified_shard(proot, 'TEST_POOL_COMPLETE', ('test',), ())
    inputs[proot.name] = rt.sha(proot/'manifest.json')
    pool = rt.read(proot/'pool.json')
    if not pool['isolation']['passed']:
        raise ValueError('Test pool failed isolation')
    tasks = pool['tasks']
    roots = sorted((rt.HERE/'results').glob('test_%s_*' % run_job))
    if len(roots) != tc['shards']:
        raise ValueError('Expected %d test shards, found %d' % (tc['shards'], len(roots)))
    refs, cells, runs = [], [], []
    for r in roots:
        verified_shard(r, 'TEST_SHARD_COMPLETE', DEPLOY_KEYS, DEPLOY_FILES)
        inputs[r.name] = rt.sha(r/'manifest.json')
        refs += rt.read(r/'mc.json')
        cells += rt.read(r/'cells.json')
        runs += rt.read(r/'runs.json')
    if sorted(x['task_id'] for x in refs) != sorted(t['id'] for t in tasks):
        raise ValueError('Truth rows do not cover the test tasks exactly once')
    resolved = {x['task_id'] for x in refs if x['resolved']}
    rtasks = [t for t in tasks if t['id'] in resolved]
    # completeness of the frozen grids on resolved tasks (numeric failures are reported, not filled in)
    gr = grid(cfg)
    expect = dict(L=len(gr['L']), C1=len(gr['C1']), F2=len(gr['F2']), M0=len(gr['F2']), F2mul=len(gr['F2']))
    have = defaultdict(int)
    for c in cells:
        have[(c['method'], c['task_id'])] += 1
    incomplete = [[m, t['id'], have[(m, t['id'])]] for t in rtasks for m in expect if have[(m, t['id'])] != expect[m]]
    out = summarize(cfg, rtasks, cells)
    out['incomplete_grids'] = incomplete
    out['failures'] = [x for x in runs if x['status'] != 'OK']
    # truth
    fam = defaultdict(list)
    for x in refs:
        fam[x['family']].append(x)
    out['truth'] = {f: dict(tasks=len(v), resolved=sum(x['resolved'] for x in v),
                            unresolved=[x['task_id'] for x in v if not x['resolved']],
                            median_paths=float(np.median([x['paths'] for x in v])),
                            seconds=sum(x['seconds'] for x in v)) for f, v in sorted(fam.items())}
    # plain Monte Carlo at 1 % P accuracy (2 sigma), single-state rounds, as hard_regime step 1
    tol = cfg['tolerance']['P_relative']
    mcc = {x['task_id']: x['paths']*(x['max_se_P_relative']/(tol/2))**2*x['H'] for x in refs if x['resolved']}
    Lneed = out['tables']['Q_QoI']['needed']['L']
    out['plain_mc'] = {g: dict(median_rounds=float(np.median([mcc[t['id']] for t in rtasks if t['family'] in fams])),
                               median_L_atoms=float(np.median([Lneed[t['id']]['atoms'] for t in rtasks if t['family'] in fams])))
                       for g, fams in groups().items() if any(t['family'] in fams for t in rtasks)}
    # cost per run
    cost = defaultdict(list)
    for x in runs:
        if x['status'] == 'OK':
            cost[x['method']].append(x)
    out['run_cost'] = {m: dict(runs=len(v), median_seconds=float(np.median([x['seconds'] for x in v])),
                               median_merge_seconds=float(np.median([x.get('merge_seconds', 0.) for x in v])),
                               total_seconds=sum(x['seconds'] for x in v)) for m, v in cost.items()}
    out['resolved_tasks'] = len(rtasks)
    out['pool_isolation'] = pool['isolation']
    return out

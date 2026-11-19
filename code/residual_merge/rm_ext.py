"""Extended evaluation grids for the frozen configuration (HANDOFF 5j item 22.35, 2026-11-02, user request).

M3 compared methods at the ends of their own grids: L stopped at alpha=16 while F2 went down to K=4 (160 atoms), so on
easy tasks every method passed at its cheapest point and the comparison read grid ends (22.29). Here, on all resolved
n=4 M3 tasks, the frozen L (train 28736643, Pauli-coordinate kNN) is extended to alpha 32/64/128 where its M3 alpha=16
point passes (central Q or the reference band's lower bound, as 22.29), C1 to budgets 0.3/1 and F2 to K=1/2 (40 and 80
atoms). Everything else is M3. The analysis merges these points with M3 (L, C1) and hard_regime step 1 (F2 K 4..256)
and reports needed atoms by family on the extended grids next to the M3-only grids; the 22.29 extended runs on the
B family and the hard check set serve as a reproduction check. Final data are never accessed.
"""
import rm_runtime as rt
from collections import defaultdict
import numpy as np
from hr_eval import context as task_context, shard as take_shard
from rm_deploy import run_one, task_list, needed, compare, HARD
from rm_eval2 import m3_cells, passes16, f2_cell, verified_shard, M3_KEYS, M3_MODEL_FILES

EASY = ('A', 'AAB', 'B')


def deploy_ext(cx, cfg, shard, shards, inputs, save):
    from rm_curv import old_model
    ec = cfg['ext']
    model = old_model(cfg, cx.device, inputs)
    tasks, npz, _ = task_list(cfg, inputs)
    m3 = m3_cells(cfg, inputs)
    mine = take_shard(tasks, shard, shards)
    save('unit.json', dict(index=shard, shards=shards, mine=[t['id'] for t in mine]))
    cells, runs = [], []
    for task in mine:
        states, weights, ctx = task_context(cx, task)
        ref = dict(np.load(npz[task['id']]))
        grid = [('C1', b) for b in ec['geo_ext']]
        if passes16(m3, task['id'], 'L'):
            grid = [('L', a) for a in ec['alpha_ext']]+grid
        for method, b in grid:
            c, r = run_one(cx, cfg, task, ctx, states, weights, ref, method, b, None if method == 'C1' else model)
            runs.append(r)
            if c is not None:
                c['method'] = method
                cells.append(c)
        for K in ec['f2_ext']:
            c = f2_cell(cx, cfg, task, ctx, states, weights, ref, K)
            cells.append(c)
            runs.append(dict(task_id=task['id'], family=task['family'], method='F2', K=K, status=c['status'],
                             physical_calls=c['physical_calls'], Q_QoI=c.get('Q_QoI')))
        save('cells.json', cells)
        save('runs.json', runs)
    return dict(tasks=len(mine), runs=len(runs), failures=sum(r['status'] != 'OK' for r in runs))


def point(c, key):
    return c['physical_calls'], (c.get(key) if c.get('status') == 'OK' else None)


def table(buckets, tasks, groups):
    """Needed atoms per method/task (central Q and Q_high) and the prep-equal comparisons per group."""
    out = {}
    for key in ('Q_QoI', 'Q_QoI_high'):
        vals = {m: {t['id']: needed([point(c, key) for c in buckets[m][t['id']]]) for t in tasks} for m in buckets}
        res = {}
        for g, fams in groups.items():
            ts = [t for t in tasks if t['family'] in fams]
            res[g] = dict(tasks=len(ts), L_vs_F2=compare(vals, 'F2', 'L', ts), L_vs_C1=compare(vals, 'C1', 'L', ts),
                          C1_vs_F2=compare(vals, 'F2', 'C1', ts),
                          left_censored={m: sum(bool(buckets[m][t['id']]) and
                                                min(buckets[m][t['id']], key=lambda c: c['physical_calls']).get(key) is not None and
                                                min(buckets[m][t['id']], key=lambda c: c['physical_calls'])[key] <= 1
                                                and not vals[m][t['id']][1] for t in ts) for m in buckets},
                          right_censored={m: sum(vals[m][t['id']][1] for t in ts) for m in buckets})
        out[key] = dict(groups=res, needed={m: {t: dict(atoms=v[0], censored=v[1]) for t, v in vm.items()} for m, vm in vals.items()})
    return out


def m3_reading(cfg, cmp_hard):
    vs = {b: cmp_hard['L_vs_'+b] for b in ('F2', 'C1')}
    best = min(vs, key=lambda b: vs[b]['ratio'])
    d, m = cfg['m3']['decision'], vs[best]
    reading = ('EXPAND' if m['ratio'] >= d['ratio'] and m['not_worse'] >= d['fraction']*m['preparations']-1e-9
               else 'ITERATE' if m['ratio'] > 1.0 else 'DIAGNOSE')
    return dict(stronger_baseline=best, ratio=m['ratio'], not_worse=m['not_worse'], preparations=m['preparations'], reading=reading)


def analyze_ext(cfg, job, inputs):
    ec, f2 = cfg['ext'], cfg['f2']
    tasks, npz, step1 = task_list(cfg, inputs)
    roots = sorted((rt.HERE/'results').glob('ext_%s_*' % job))
    if len(roots) != ec['shards']:
        raise ValueError('Expected %d extension shards, found %d' % (ec['shards'], len(roots)))
    ext, failures = [], []
    for r in roots:
        verified_shard(r, 'EXT_SHARD_COMPLETE', M3_KEYS+('ext',), M3_MODEL_FILES+('rm_ext.py', 'rm_deploy.py', 'rm_eval2.py'))
        inputs[r.name] = rt.sha(r/'manifest.json')
        ext += rt.read(r/'cells.json')
        failures += [x for x in rt.read(r/'runs.json') if x['status'] != 'OK']
    m3 = m3_cells(cfg, inputs)
    base = {m: defaultdict(list) for m in ('L', 'C1', 'F2')}
    for c in m3:
        if c.get('method') in ('L', 'C1'):
            base[c['method']][c['task_id']].append(c)
    for c in step1:
        if c['arm'] == f2['arm'] and c['K'] in f2['K']:
            base['F2'][c['task_id']].append(dict(c, method='F2'))
    full = {m: defaultdict(list, {t: list(v) for t, v in base[m].items()}) for m in base}
    for c in ext:
        full[c['method']][c['task_id']].append(c)
    groups = {'hard': HARD, 'easy': EASY, 'all': HARD+EASY}
    groups.update({f: (f,) for f in HARD+EASY})
    out = dict(failures=failures, extended=table(full, tasks, groups), m3_grid=table(base, tasks, groups))
    out['m3_rule'] = dict(extended=m3_reading(cfg, out['extended']['Q_QoI']['groups']['hard']),
                          m3_grid=m3_reading(cfg, out['m3_grid']['Q_QoI']['groups']['hard']))
    out['L_extended_tasks'] = sorted({c['task_id'] for c in ext if c['method'] == 'L'})
    # reproduction check against the 22.29 extended runs (old L on the B family; C1 and F2 on B and the hard check set)
    prior = []
    for job2, batch in ((ec['check_jobs'][0], 'B'), (ec['check_jobs'][1], 'hard')):
        for r in sorted((rt.HERE/'results').glob('eval2_%s_*' % job2)):
            verified_shard(r, 'EVAL2_SHARD_COMPLETE', M3_KEYS+('s1', 'eval2'), M3_MODEL_FILES+('rm_lik.py', 'rm_signed.py', 'rm_deploy.py'))
            inputs[r.name] = rt.sha(r/'manifest.json')
            prior += rt.read(r/'cells.json')
    def ident(c):
        m = 'L' if c['method'] in ('L', 'Lold') else c['method']
        return (c['task_id'], m, c['K'] if m == 'F2' else c.get('budget'))
    mine = {ident(c): c for c in ext}
    rows = []
    for c in prior:
        if c['method'] not in ('Lold', 'C1', 'F2'):
            continue
        k = ident(c)
        if k in mine:
            a, b = mine[k], c
            rows.append(dict(key=list(map(str, k)), calls_equal=a['physical_calls'] == b['physical_calls'],
                             dQ=None if a.get('Q_QoI') is None or b.get('Q_QoI') is None else abs(a['Q_QoI']-b['Q_QoI'])))
    out['reproduction'] = dict(cells=len(rows), all_calls_equal=all(x['calls_equal'] for x in rows),
                               max_dQ=max([x['dQ'] for x in rows if x['dQ'] is not None] or [0.]))
    return out

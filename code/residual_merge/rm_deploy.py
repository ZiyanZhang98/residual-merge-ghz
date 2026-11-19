"""M3 of HANDOFF 22.28: deployment of L (learned residual cost) and C1 (geometric Ward cost), both with the same
budgeted agglomeration, PTA additive correction and atom cap, on the resolved n=4 development tasks, scored against
the Monte Carlo truth of hard_regime step 1; analysis against F2 (fixed-K merge+PTA cells of step 1).
No training here; final data are never accessed.
"""
import rm_runtime as rt
import math
import time
from collections import defaultdict
import numpy as np
import torch
from g1_core import check_deadline, Deadline
from qa_sources import PTA
from hr_eval import load_tasks, context as task_context, scored, shard as take_shard
from rm_core import GeoCost, Agglomerative, adaptive_rollout
from rm_model import LearnedCost, load_model

HARD = ('AB', 'AABB', 'ABB')


class LearnedPolicy(Agglomerative):
    """Agglomerative with the learned cost; the cost is prepared (ensemble PTA level, caches) per compression."""
    def __call__(self, sim, states, w):
        self.cost.prepare(sim, states, w)
        labels, info = super().__call__(sim, states, w)
        info['pairs_priced'] = self.cost.evaluated
        return labels, info


def task_list(cfg, inputs):
    refs, npz, step1 = rt.references(inputs, cfg['hard_regime']['step1_job'])
    resolved = {r['task_id'] for r in refs if r['resolved']}
    tasks = [t for t in load_tasks(rt.hr.config()) if t['id'] in resolved and t['n'] == cfg['m3']['n']]
    return tasks, npz, step1


def policy(method, budget, model, cfg):
    mg = cfg['merge']
    if method == 'L':
        return LearnedPolicy(LearnedCost(model, cfg['tolerance']), budget/cfg['m3']['horizon'], mg['cap'], mg['knn'], mg['max_rounds'])
    return Agglomerative(GeoCost(), budget, mg['cap'], mg['knn'], mg['max_rounds'])


def run_one(cx, cfg, task, ctx, states, weights, ref, method, budget, model):
    check_deadline(cx.hard)
    t0 = time.perf_counter()
    try:
        sim, rho, ell, est = adaptive_rollout(policy(method, budget, model, cfg), task['word'], ctx, cx.c, states, weights,
                                              {'M_PTA': PTA()}, cx.hard)
    except Deadline:
        raise
    except (FloatingPointError, RuntimeError) as exc:
        return None, dict(task_id=task['id'], family=task['family'], method=method, budget=budget,
                          status='NUMERIC_FAILURE', error=repr(exc))
    log = sim.merge_log
    extra = dict(method=method, budget=budget, physical_calls=sim.cost['physical_calls'], seconds=time.perf_counter()-t0,
                 max_atoms=max([e['after'] for e in log] or [0]), forced=sum(e.get('forced', 0) for e in log),
                 merge_seconds=sum(e['seconds'] for e in log), correction_seconds=sum(sim.correction_seconds.values()))
    c = scored(task, method+'+M_PTA', None, est['M_PTA'], ref, cfg['tolerance'], extra)
    run = dict(task_id=task['id'], family=task['family'], prep_id=task['prep_signature'], status=c['status'],
               Q_QoI=c.get('Q_QoI'), **extra)
    print('RM_RUN %s %s budget=%g calls=%d max_atoms=%d forced=%d Q=%s %.1fs (merge %.1fs)' % (
        task['id'], method, budget, extra['physical_calls'], extra['max_atoms'], extra['forced'], run['Q_QoI'],
        extra['seconds'], extra['merge_seconds']), flush=True)
    return c, run


def profile(cx, cfg, model, inputs):
    """Seconds per rollout of L at the profile budgets on one hard task (sizes the M3 array)."""
    tasks, npz, _ = task_list(cfg, inputs)
    task = next(t for t in tasks if t['family'] == cfg['m3']['profile_family'])
    states, weights, ctx = task_context(cx, task)
    ref = dict(np.load(npz[task['id']]))
    runs = []
    for b in cfg['m3']['profile_alpha']:
        runs.append(run_one(cx, cfg, task, ctx, states, weights, ref, 'L', b, model)[1])
    return dict(task_id=task['id'], runs=runs)


def deploy(cx, cfg, shard, shards, model, inputs, save):
    m3 = cfg['m3']
    tasks, npz, _ = task_list(cfg, inputs)
    mine = take_shard(tasks, shard, shards)
    save('unit.json', dict(index=shard, shards=shards, mine=[t['id'] for t in mine]))
    cells, runs = [], []
    for task in mine:
        states, weights, ctx = task_context(cx, task)
        ref = dict(np.load(npz[task['id']]))
        grid = (('L', m3['alpha']), ('C1', m3['geo_budgets']))
        for method, budgets in grid:
            for b in budgets:
                c, r = run_one(cx, cfg, task, ctx, states, weights, ref, method, b, model)
                runs.append(r)
                if c is not None:
                    cells.append(c)
        save('cells.json', cells)
        save('runs.json', runs)
    return dict(tasks=len(mine), runs=len(runs), failures=sum(r['status'] != 'OK' for r in runs))


# ------------------------------------------------------------------ analysis (CPU)

def needed(points):
    """Fewest total atoms (physical_calls) at which this and every costlier grid point pass Q_QoI <= 1."""
    pts = sorted(points)
    ok = [q is not None and q <= 1 for _, q in pts]
    for i in range(len(pts)):
        if all(ok[i:]):
            return pts[i][0], False
    return 2*max(c for c, _ in pts), True   # right-censored: one grid step beyond the largest point


def gmean(x):
    return float(math.exp(np.mean(np.log(x)))) if len(x) else None


def compare(values, a, b, tasks):
    """Prep-equal comparison of b against a: geometric mean of a/b over preps (prep value = median over its tasks)."""
    preps = sorted({t['prep_signature'] for t in tasks})
    ratios, not_worse = [], 0
    for p in preps:
        ids = [t['id'] for t in tasks if t['prep_signature'] == p]
        va = float(np.median([values[a][i][0] for i in ids]))
        vb = float(np.median([values[b][i][0] for i in ids]))
        ratios.append(va/vb)
        not_worse += vb <= va
    uncensored = [t for t in tasks if not values[a][t['id']][1] and not values[b][t['id']][1]]
    task_ratio = gmean([values[a][t['id']][0]/values[b][t['id']][0] for t in uncensored])
    return dict(preparations=len(preps), ratio=gmean(ratios), not_worse=not_worse,
                tasks_uncensored_ratio=task_ratio, tasks_uncensored=len(uncensored))


def analyze(cfg, deploy_job, inputs):
    m3, f2 = cfg['m3'], cfg['f2']
    tasks, npz, step1 = task_list(cfg, inputs)
    cells = []
    for r in sorted((rt.HERE/'results').glob('deploy_%s_*' % deploy_job)):
        rt.artifact(r, ['DEPLOY_SHARD_COMPLETE'])
        inputs[r.name] = rt.sha(r/'manifest.json')
        cells += rt.read(r/'cells.json')
    points = defaultdict(lambda: defaultdict(list))
    for c in cells:
        points[c['method']][c['task_id']].append((c['physical_calls'], c.get('Q_QoI') if c['status'] == 'OK' else None))
    for c in step1:
        if c['arm'] == f2['arm'] and c['K'] in f2['K']:
            points['F2'][c['task_id']].append((c['physical_calls'], c.get('Q_QoI') if c['status'] == 'OK' else None))
    expect = {'L': len(m3['alpha']), 'C1': len(m3['geo_budgets']), 'F2': len(f2['K'])}
    missing = [(m, t['id']) for m in expect for t in tasks if len(points[m][t['id']]) != expect[m]]
    if missing:
        raise ValueError('Incomplete deployment/F2 grid: %s' % missing[:10])
    values = {m: {t['id']: needed(points[m][t['id']]) for t in tasks} for m in expect}
    hard = [t for t in tasks if t['family'] in HARD]
    easy = [t for t in tasks if t['family'] not in HARD]
    out = dict(tasks=len(tasks), hard_tasks=len(hard), easy_tasks=len(easy), by_family={}, hard={}, easy={})
    for fam in sorted({t['family'] for t in tasks}):
        ts = [t for t in tasks if t['family'] == fam]
        out['by_family'][fam] = {m: dict(median_atoms=float(np.median([values[m][t['id']][0] for t in ts])),
                                         censored=sum(values[m][t['id']][1] for t in ts)) for m in expect}
    for name, ts in (('hard', hard), ('easy', easy)):
        out[name] = {'L_vs_C1': compare(values, 'C1', 'L', ts), 'L_vs_F2': compare(values, 'F2', 'L', ts),
                     'C1_vs_F2': compare(values, 'F2', 'C1', ts)}
    best = 'C1' if out['hard']['L_vs_C1']['ratio'] <= out['hard']['L_vs_F2']['ratio'] else 'F2'
    main = out['hard']['L_vs_'+best]
    d = m3['decision']
    if main['ratio'] >= d['ratio'] and main['not_worse'] >= d['fraction']*main['preparations']-1e-9:
        reading = 'EXPAND'
    elif main['ratio'] > 1.0:
        reading = 'ITERATE'
    else:
        reading = 'DIAGNOSE'
    out['decision'] = dict(stronger_baseline=best, ratio=main['ratio'], not_worse=main['not_worse'],
                           preparations=main['preparations'], reading=reading,
                           rule='L vs the stronger of F2 and C1 on the hard tasks (prep-equal geometric mean of needed '
                                'total atoms): EXPAND if >= %.2f with >= 2/3 preparations not worse; ITERATE if > 1; '
                                'else DIAGNOSE (resource decision only)' % d['ratio'])
    out['points'] = {m: {t: sorted(v) for t, v in pm.items()} for m, pm in points.items()}
    out['needed'] = {m: {t: dict(atoms=v[0], censored=v[1]) for t, v in vm.items()} for m, vm in values.items()}
    return out


class Windowed:
    """Development diagnostic policy: `inner` (L) outside the rounds lo..hi; inside them either no merging
    (alt None) or the alternative policy (C1). Replacing L's decisions in one window at a time locates the
    rounds whose merges cause an error."""
    def __init__(self, inner, window, alt):
        self.inner, self.window, self.alt = inner, window, alt

    def __call__(self, sim, states, w):
        lo, hi = self.window
        if lo <= sim.t <= hi:
            if self.alt is None:
                return torch.arange(len(w), device=w.device), dict(policy='none', before=len(w), after=len(w),
                                                                   merges=0, forced=0, spent=0.)
            return self.alt(sim, states, w)
        return self.inner(sim, states, w)


def first_violation(c):
    P, F = c.get('P_relative') or [], c.get('F_absolute') or []
    for k, (p, f) in enumerate(zip(P, F)):
        if p > 0.01 or f > 0.01:
            return dict(round=k+1, P=p, F=f)
    return None


def diagnose(cx, cfg, model, inputs):
    """Window-swap diagnostics of L on chosen tasks (B-family failure, ABB plateau, AB control)."""
    dg = cfg['diag']
    tasks, npz, _ = task_list(cfg, inputs)
    out = []
    for spec in dg['tasks']:
        task = next(t for t in tasks if t['id'] == spec['task_id'])
        states, weights, ctx = task_context(cx, task)
        ref = dict(np.load(npz[task['id']]))
        a = spec['alpha']
        variants = [('L', None, None), ('L_no_merge_0-2', (0, 2), 'none')]
        variants += [('L_C1_%d-%d' % tuple(win), tuple(win), 'C1') for win in dg['windows']]
        variants += [('C1', None, None)]
        rec = dict(task_id=task['id'], family=task['family'], alpha=a, runs=[])
        for name, win, alt in variants:
            check_deadline(cx.hard)
            if name == 'C1':
                pol = policy('C1', dg['c1_budget'], model, cfg)
            else:
                pol = policy('L', a, model, cfg)
                if win is not None:
                    pol = Windowed(pol, win, None if alt == 'none' else policy('C1', dg['c1_budget'], model, cfg))
            t0 = time.perf_counter()
            sim, rho, ell, est = adaptive_rollout(pol, task['word'], ctx, cx.c, states, weights, {'M_PTA': PTA()}, cx.hard)
            c = scored(task, 'x', None, est['M_PTA'], ref, cfg['tolerance'], {})
            raw = scored(task, 'x', None, est['M0'], ref, cfg['tolerance'], {})
            log = [{k: e.get(k) for k in ('t', 'before', 'after', 'merges', 'forced', 'spent', 'policy')} for e in sim.merge_log]
            run = dict(variant=name, physical_calls=sim.cost['physical_calls'], seconds=time.perf_counter()-t0,
                       Q_QoI=c.get('Q_QoI'), Q_P=c.get('Q_P'), Q_F=c.get('Q_F'), raw_Q_P=raw.get('Q_P'), raw_Q_F=raw.get('Q_F'),
                       first_violation=first_violation(c), P_relative=c.get('P_relative'), F_absolute=c.get('F_absolute'),
                       merge_log=log)
            rec['runs'].append(run)
            fv = run['first_violation']
            print('RM_DIAG %-34s %-16s calls=%-6d Q=%-7.3g QP=%-7.3g QF=%-7.3g raw QP=%-7.3g QF=%-7.3g first=%s' % (
                task['id'], name, run['physical_calls'], run['Q_QoI'], run['Q_P'], run['Q_F'], run['raw_Q_P'], run['raw_Q_F'],
                None if fv is None else 'r%d(P %.3g F %.3g)' % (fv['round'], fv['P'], fv['F'])), flush=True)
            if name == 'L':
                for e in sim.merge_log[:6]:
                    print('      t=%d before=%d after=%d merges=%s forced=%s spent=%.3g' % (
                        e['t'], e['before'], e['after'], e.get('merges'), e.get('forced'), e.get('spent') or 0.), flush=True)
        out.append(rec)
    return out

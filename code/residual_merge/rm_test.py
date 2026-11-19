"""Independent n=4 test set of the frozen configuration (HANDOFF 5j item 22.37; user decisions 2026-11-02: default
alpha 0.5, plan B, Monte Carlo path cap 2^21).

Preparations: the distribution, class rotation and isolation rules of pc_guard.generate_pool with a new seed, also
excluding the development step-B pool (48 fit + 12 monitor). Tasks: the hard families (AB, AABB, ABB) on all
preparations, the easy families (A, AAB, B) on the first ones (balanced over the 4 classes). Truth: hard_regime's
unbiased path sampling with the development settings; unresolved tasks get no method runs. Methods on the frozen
extended grids: L (train 28736643), C1, F2 (fixed-K merge + additive PTA; the same rollout also yields merge without
correction, M0, and multiplicative PTA, F2mul). The analysis is in rm_test_report.py.
The smoke touches development tasks only. Final data are never accessed.
"""
import rm_runtime as rt
import random
import time
import numpy as np
from g1_core import check_deadline, Deadline
from qa_sources import PTA
from hr_eval import context as task_context, scored, shard as take_shard
from hr_goal import rollout as goal_rollout
from hr_mc import reference
import pc_guard as pg
from rm_deploy import run_one, task_list

DEPLOY_KEYS = ('test', 'm3', 'merge', 'tolerance', 'hard_regime', 'f2', 's1')
DEPLOY_FILES = ('rm_core.py', 'rm_labels.py', 'rm_runtime.py', 'rm_report.py', 'rm_model.py', 'rm_deploy.py', 'rm_test.py')
F2_ARMS = (('M_PTA', 'F2'), ('M0', 'M0'), ('M_PTA__mul', 'F2mul'))


# ------------------------------------------------------------------ preparations and tasks

def pc_config():
    return rt.hr.qb.qa.prt.config()


def forbidden_sets(pc_cfg, physical, extra=()):
    """What pc_guard.generate_pool excludes (evaluation/final specifications, G1a evaluation metadata, every physical
    history), plus the `extra` preparations."""
    preps = [p for p in pg.physical_plan(physical) if p['split'] in ('evaluation', 'final')]
    preps += [i['prep'] for i in pg.metadata(pc_cfg)]+list(extra)
    sigs = {pg.signature(p, physical) for p in preps}
    hists = [pg.history(p) for p in preps]+[tuple(h) for hs in physical['histories'].values() for h in hs]
    return sigs, hists


def make_pool(pc_cfg, physical, n, seed, splits, extra=()):
    """pc_guard.generate_pool with a chosen seed, splits and extra excluded preparations: same draws, class rotation
    (noise family by index parity, history depth by (index//2) parity, restarting per split) and isolation rules.
    With seed=pc_cfg['seed'] and splits fit/monitor it is generate_pool itself (smoke check)."""
    forbidden_s, forbidden_h = forbidden_sets(pc_cfg, physical, extra)
    pool = pc_cfg['pool']
    rng = random.Random(seed+n)
    seen, made, out = set(), [], []
    for split, count in splits:
        for j in range(count):
            for _ in range(10000):
                p = dict(n=n, family=('depolarizing', 'phase_biased')[j % 2], depth=(2, 4)[(j//2) % 2], split='calibration',
                         F0=rng.uniform(max(pool['F0'][0], 1/2**n+1e-6), pool['F0'][1]),
                         phi=rng.uniform(*pool['phi']), gamma=rng.uniform(*pool['gamma']))
                s, h = pg.signature(p, physical), pg.history(p)
                if s not in forbidden_s | seen and not any(pg.near(h, x) for x in forbidden_h+made):
                    break
            else:
                raise ValueError('Unable to generate isolated history')
            seen.add(s)
            made.append(h)
            p['id'] = 'pc_'+s[:20]
            out.append(dict(prep=p, split=split, signature=s))
    return out


def test_pool(cfg, physical):
    """(test pool, development step-B pool)."""
    tc, pc_cfg = cfg['test'], pc_config()
    dev = pg.generate_pool(pc_cfg, physical, tc['n'])
    pool = make_pool(pc_cfg, physical, tc['n'], tc['seed'], (('test', tc['preparations']),), [e['prep'] for e in dev])
    return pool, dev


def prep_class(prep):
    return '%s/depth%d' % (prep['family'], prep['depth'])


def test_tasks(cfg, pool):
    tc = cfg['test']
    tasks = []
    for j, e in enumerate(pool):
        for fam in tc['hard_families']+(tc['easy_families'] if j < tc['easy_preparations'] else []):
            tasks.append(dict(id='ts_n%d_%s_%s' % (tc['n'], e['signature'][:16], fam), n=tc['n'], H=tc['H'],
                              gamma=tc['gamma'], family=fam, word=(fam*tc['H'])[:tc['H']], prep=e['prep'],
                              prep_signature=e['signature'], prep_index=j, prep_class=prep_class(e['prep'])))
    if len({t['id'] for t in tasks}) != len(tasks):
        raise ValueError('Duplicate test task ids')
    return tasks


def isolation(cfg, physical, pool, dev):
    """Independent re-check of the test pool: no shared signature with, and no history within 1e-10 of, anything
    excluded (development pool included); distinct preparations; balanced classes overall and among the first
    `easy_preparations`."""
    tc = cfg['test']
    sigs, hists = forbidden_sets(pc_config(), physical, [e['prep'] for e in dev])
    test_s = [e['signature'] for e in pool]
    test_h = [pg.history(e['prep']) for e in pool]
    gap = min(max(abs(x-y) for x, y in zip(h, f)) for h in test_h for f in hists)
    inner = min([max(abs(x-y) for x, y in zip(a, b)) for i, a in enumerate(test_h) for b in test_h[i+1:]] or [1.])
    count = lambda es: {k: sum(prep_class(e['prep']) == k for e in es) for k in sorted({prep_class(e['prep']) for e in pool})}  # noqa: E731
    classes, first = count(pool), count(pool[:tc['easy_preparations']])
    out = dict(preparations=len(pool), shared_signatures=sum(s in sigs for s in test_s), min_history_gap=gap,
               min_history_gap_within_test=inner, distinct=len(set(test_s)) == len(test_s),
               signatures_recomputed=all(pg.signature(e['prep'], physical) == e['signature'] for e in pool),
               development_pool_signatures=len(dev), classes=classes, classes_easy=first,
               parameter_ranges={k: [min(e['prep'][k] for e in pool), max(e['prep'][k] for e in pool)] for k in ('F0', 'phi', 'gamma')})
    out['passed'] = (out['shared_signatures'] == 0 and gap > 1e-10 and inner > 1e-10 and out['distinct']
                     and out['signatures_recomputed'] and len(pool) == tc['preparations']
                     and len(set(classes.values())) == 1 and len(set(first.values())) == 1)
    return out


def write_pool(cx, cfg, save):
    pool, dev = test_pool(cfg, cx.c)
    tasks = test_tasks(cfg, pool)
    iso = isolation(cfg, cx.c, pool, dev)
    save('pool.json', dict(pool=pool, tasks=tasks, isolation=iso))
    return iso, tasks


def load_pool(cx, cfg, job, inputs):
    """The test pool artifact, checked against its hashes, the unchanged 'test' settings and a deterministic
    regeneration."""
    root = rt.HERE/'results'/('testpool_%s' % job)
    m = rt.read(root/'manifest.json')
    if m['status'] != 'TEST_POOL_COMPLETE' or m['config']['test'] != cfg['test']:
        raise ValueError('Incomplete/incompatible test pool: '+str(root))
    for filename, value in m['files'].items():
        if rt.sha(root/filename) != value:
            raise ValueError('Test pool content changed: '+str(root/filename))
    inputs[root.name] = rt.sha(root/'manifest.json')
    stored = rt.read(root/'pool.json')
    pool, _ = test_pool(cfg, cx.c)
    tasks = test_tasks(cfg, pool)
    if [t['id'] for t in tasks] != [t['id'] for t in stored['tasks']] or not stored['isolation']['passed']:
        raise ValueError('Test pool does not regenerate or failed isolation')
    return tasks


# ------------------------------------------------------------------ truth and methods

def grid(cfg):
    tc = cfg['test']
    return dict(L=tc['alpha'], C1=tc['geo_budgets'], F2=tc['f2_K'])


def f2_cells(cx, cfg, task, ctx, states, weights, ref, K):
    """Fixed-K merge through hard_regime step 1's code path (hr_goal.rollout, no look-ahead): additive PTA (F2),
    no correction (M0) and multiplicative PTA (F2mul) from one rollout."""
    check_deadline(cx.hard)
    t0 = time.perf_counter()
    try:
        sim, rho, ell, est = goal_rollout(K, task['word'], ctx, cx.c, states, weights, {'M_PTA': PTA()}, cx.hard)
    except Deadline:
        raise
    except (FloatingPointError, RuntimeError) as exc:
        return [], dict(task_id=task['id'], family=task['family'], method='F2', K=K, status='NUMERIC_FAILURE', error=repr(exc))
    extra = dict(physical_calls=sim.cost['physical_calls'], seconds=time.perf_counter()-t0,
                 correction_seconds=sum(sim.correction_seconds.values()))
    cells = []
    for arm, method in F2_ARMS:
        try:
            c = scored(task, arm, K, est[arm], ref, cfg['tolerance'], extra)
        except FloatingPointError as exc:
            c = dict(task_id=task['id'], n=task['n'], family=task['family'], prep_id=task['prep_signature'], arm=arm, K=K,
                     status='NUMERIC_FAILURE', error=repr(exc), **extra)
        c.update(method=method, budget=None)
        cells.append(c)
    run = dict(task_id=task['id'], family=task['family'], prep_id=task['prep_signature'], method='F2', K=K,
               status=cells[0]['status'], Q_QoI=cells[0].get('Q_QoI'), **extra)
    print('RM_TEST_F2 %s K=%d calls=%d Q=%s %.1fs' % (task['id'], K, extra['physical_calls'], run['Q_QoI'], extra['seconds']), flush=True)
    return cells, run


def run_task(cx, cfg, task, ctx, states, weights, ref, model, points):
    cells, runs = [], []
    for method in ('L', 'C1'):
        for b in points[method]:
            c, r = run_one(cx, cfg, task, ctx, states, weights, ref, method, b, model)
            runs.append(r)
            if c is not None:
                cells.append(c)
    for K in points['F2']:
        cs, r = f2_cells(cx, cfg, task, ctx, states, weights, ref, K)
        runs.append(r)
        cells += cs
    return cells, runs


def mc_settings(cfg):
    mc = rt.hr.config()['mc']
    if mc['max_paths'] != cfg['test']['mc_max_paths']:
        raise ValueError('Monte Carlo path cap differs from the registered one')
    return mc


def run_shard(cx, cfg, pool_job, shard, shards, root, inputs, save):
    """Per task: Monte Carlo truth; if resolved, every method on its full frozen grid (the grid never depends on
    results)."""
    from rm_curv import old_model
    mc = mc_settings(cfg)
    tasks = load_pool(cx, cfg, pool_job, inputs)
    model = old_model(cfg, cx.device, inputs)
    mine = take_shard(tasks, shard, shards)
    save('unit.json', dict(index=shard, shards=shards, mine=[t['id'] for t in mine], mc=mc, grid=grid(cfg)))
    refs, cells, runs = [], [], []
    for task in mine:
        states, weights, ctx = task_context(cx, task)
        seed = mc['seed']+int(rt.hr.digest(task['id'])[:8], 16) % 1_000_000
        t0 = time.perf_counter()
        ref, info = reference(task['prep'], task['word'], ctx['letters'], ctx['aux'], cx.c, mc, cx.device, cx.hard, seed)
        np.savez_compressed(root/('mc_%s.npz' % task['id']), **ref)
        refs.append(dict(task_id=task['id'], n=task['n'], family=task['family'], prep_id=task['prep_signature'],
                         prep_class=task['prep_class'], H=task['H'], seed=seed, seconds=time.perf_counter()-t0, **info))
        save('mc.json', refs)
        print('RM_TEST_MC', task['id'], info, '%.0fs' % refs[-1]['seconds'], flush=True)
        if info['resolved']:
            c, r = run_task(cx, cfg, task, ctx, states, weights, ref, model, grid(cfg))
            cells += c
            runs += r
        save('cells.json', cells)
        save('runs.json', runs)
    return dict(tasks=len(mine), resolved=sum(r['resolved'] for r in refs), runs=len(runs),
                failures=sum(r['status'] != 'OK' for r in runs))


# ------------------------------------------------------------------ smoke (development tasks only)

def smoke(cx, cfg, inputs):
    from rm_curv import old_model
    from rm_eval2 import m3_cells
    from rm_test_report import development_cells, summarize, matches_ext
    tc, sm = cfg['test'], cfg['test']['smoke']
    pc_cfg, physical = pc_config(), cx.c
    out = {}
    # 1-2: generator and isolation
    pool, dev = test_pool(cfg, physical)
    again = make_pool(pc_cfg, physical, tc['n'], pc_cfg['seed'],
                      (('fit', pc_cfg['pool']['fit']), ('monitor', pc_cfg['pool']['monitor'])))
    out['generator_reproduces_development_pool'] = dict(
        preparations=len(dev), passed=[(e['signature'], e['split']) for e in again] == [(e['signature'], e['split']) for e in dev])
    out['isolation'] = isolation(cfg, physical, pool, dev)
    tasks = test_tasks(cfg, pool)
    dev_tasks, npz, step1 = task_list(cfg, inputs)
    monitor = {e['signature'] for e in dev if e['split'] == 'monitor'}
    fams = sorted({t['family'] for t in tasks})
    out['tasks'] = dict(count=len(tasks), by_family={f: sum(t['family'] == f for t in tasks) for f in fams},
                        development_preparations_in_monitor_pool=all(t['prep_signature'] in monitor for t in dev_tasks),
                        same_protocol_as_development=all(t['H'] == tc['H'] and t['gamma'] == tc['gamma']
                                                         and t['word'] == (t['family']*tc['H'])[:tc['H']] for t in dev_tasks))
    expect = len(tc['hard_families'])*tc['preparations']+len(tc['easy_families'])*tc['easy_preparations']
    out['tasks']['passed'] = (len(tasks) == expect and out['tasks']['development_preparations_in_monitor_pool']
                              and out['tasks']['same_protocol_as_development'])
    # 3: Monte Carlo truth reproduces a development reference (same seed)
    mc = mc_settings(cfg)
    refs, npz_mc, _ = rt.references(inputs, cfg['hard_regime']['step1_job'])
    row = next(r for r in refs if r['task_id'] == sm['mc_task'])
    task = next(t for t in dev_tasks if t['id'] == sm['mc_task'])
    states, weights, ctx = task_context(cx, task)
    t0 = time.perf_counter()
    ref, info = reference(task['prep'], task['word'], ctx['letters'], ctx['aux'], cx.c, mc, cx.device, cx.hard, row['seed'])
    sec = time.perf_counter()-t0
    old = dict(np.load(npz_mc[task['id']]))
    dP = float(np.max(np.abs(ref['P']-old['P'])/old['P']))
    dF = float(np.max(np.abs(ref['F']-old['F'])))
    out['mc_reproduces_development'] = dict(task=task['id'], paths=[info['paths'], row['paths']], max_rel_dP=dP, max_dF=dF,
                                            seconds=sec, seconds_per_path_round=sec/(info['paths']*task['H']),
                                            passed=info['paths'] == row['paths'] and info['resolved'] == row['resolved']
                                            and dP <= 1e-9 and dF <= 1e-9)
    # 4-6: the production task runner reproduces M3 (L, C1) and step 1 (F2, M0, F2mul); the frozen model verifies
    model = old_model(cfg, cx.device, inputs)
    out['model_verified'] = dict(train_job=cfg['s1']['old_train_job'], passed=True)
    task = next(t for t in dev_tasks if t['id'] == sm['deploy_task'])
    states, weights, ctx = task_context(cx, task)
    ref = dict(np.load(npz[task['id']]))
    cells, runs = run_task(cx, cfg, task, ctx, states, weights, ref, model,
                           dict(L=[sm['alpha']], C1=[sm['geo_budget']], F2=[sm['K']]))
    m3 = m3_cells(cfg, inputs)
    for method, b in (('L', sm['alpha']), ('C1', sm['geo_budget'])):
        c = next(x for x in cells if x['method'] == method)
        r = next(x for x in runs if x['method'] == method)
        o = next(x for x in m3 if x['task_id'] == task['id'] and x.get('method') == method and x.get('budget') == b)
        out['deploy_%s' % method] = dict(task=task['id'], budget=b, calls=[c['physical_calls'], o['physical_calls']],
                                         Q=[c['Q_QoI'], o['Q_QoI']], seconds=r['seconds'],
                                         passed=c['physical_calls'] == o['physical_calls'] and abs(c['Q_QoI']-o['Q_QoI']) <= 1e-9)
    for arm, method in F2_ARMS:
        c = next(x for x in cells if x['method'] == method)
        s = next(x for x in step1 if x['task_id'] == task['id'] and x['arm'] == arm and x['K'] == sm['K'])
        q = [c.get('Q_QoI'), s.get('Q_QoI')]
        same_q = q[0] == q[1] if None in q else abs(q[0]-q[1]) <= 1e-9
        out['f2_%s' % method] = dict(task=task['id'], K=sm['K'], calls=[c['physical_calls'], s['physical_calls']], Q=q,
                                     passed=c['physical_calls'] == s['physical_calls'] and same_q)
    # 7: the registered analysis reproduces 22.35 on the development grids and the dev default-alpha pass counts
    dcells, dtasks = development_cells(cfg, inputs)
    s = summarize(cfg, dtasks, dcells)
    out['analysis_reproduces_22_35'] = matches_ext(cfg, s)
    da = {g: [s['default_alpha'][g]['passed'], s['default_alpha'][g]['tasks']] for g in ('hard', 'easy')}
    out['default_alpha_development'] = dict(counts=da, expected=sm['default_alpha_development'],
                                            passed=da == sm['default_alpha_development'])
    b = s['bootstrap']['hard']['L_vs_F2']
    out['bootstrap_runs'] = dict(hard_L_vs_F2=b, passed=b['low'] <= b['ratio'] <= b['high'])
    return out

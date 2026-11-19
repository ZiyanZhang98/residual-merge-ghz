"""Step-1 deployment of HANDOFF 22.29 on extended grids, and its analysis.

Batch 'B': the 12 B-family tasks. Batch 'hard': the 9-task hard-family check, run only after batch B passes.
Per task: the step-1 model L' at the M3 alphas, plus 32/64/128 where alpha=16 passes (central Q or the reference
band's lower bound); in batch B also the old model L at 32/64/128 where its M3 alpha=16 point passed; C1 at the M3
budgets plus 0.3/1; F2 (fixed-K merge+PTA, hard_regime step-1 code path) at K=1, 2. The M3 points of L, C1 and the
step-1 F2 cells at K 4..256 are read read-only (verified by file hashes). No training; final data are never accessed.
"""
import rm_runtime as rt
from pathlib import Path
import numpy as np
from g1_core import check_deadline
from qa_sources import PTA
from hr_eval import context as task_context, scored, shard as take_shard
from hr_goal import rollout as goal_rollout
from rm_deploy import run_one, task_list, needed, gmean, compare

M3_KEYS = ('m3', 'merge', 'tolerance', 'hard_regime', 'f2')
M3_MODEL_FILES = ('rm_core.py', 'rm_labels.py', 'rm_runtime.py', 'rm_report.py', 'rm_model.py')


# ------------------------------------------------------------------ earlier artifacts (read-only)

def verified_shard(r, status, keys, files):
    """A completed shard whose relevant settings and deployment-defining sources are unchanged and whose files match
    their hashes. rt.artifact demands the whole config and every source unchanged, which no longer holds for M3
    (config 'diag' key and rm_deploy.py's diagnostics were added afterwards)."""
    m = rt.read(r/'manifest.json')
    if m['status'] != status:
        raise ValueError('Incomplete shard: '+str(r))
    now_cfg = rt.config()
    if any(m['config'][k] != now_cfg[k] for k in keys):
        raise ValueError('Settings changed since: '+str(r))
    mine = str(rt.HERE.relative_to(rt.REPO))+'/'
    keep = lambda s: {k: v for k, v in s.items() if not k.startswith(mine) or Path(k).name in files}  # noqa: E731
    if not __import__('qn_public').same_sources(keep(m['sources']), keep(rt.sources())):  # PUBLIC: see qn_public.py
        raise ValueError('Deployment-defining sources changed since: '+str(r))
    for filename, value in m['files'].items():
        if __import__('qn_public').present(r/filename) and rt.sha(r/filename) != value:
            raise ValueError('Shard content changed: '+str(r/filename))
    return m


def m3_cells(cfg, inputs):
    """Cells of the M3 deployment (verified shard by shard)."""
    roots = sorted((rt.HERE/'results').glob('deploy_%s_*' % cfg['eval2']['m3_deploy_job']))
    if not roots:
        raise ValueError('No M3 deployment shards')
    cells = []
    for r in roots:
        verified_shard(r, 'DEPLOY_SHARD_COMPLETE', M3_KEYS, M3_MODEL_FILES)
        inputs[r.name] = rt.sha(r/'manifest.json')
        cells += rt.read(r/'cells.json')
    return cells


def passes16(cells, task_id, method):
    """Whether the alpha=16 point of `method` passes by the central Q or by the reference band's lower bound."""
    for c in cells:
        if c['task_id'] == task_id and c.get('method') == method and c.get('budget') == 16 and c.get('status') == 'OK':
            return (c.get('Q_QoI') is not None and c['Q_QoI'] <= 1) or (c.get('Q_QoI_low') is not None and c['Q_QoI_low'] <= 1)
    return False


def models(cfg, train_job, device, inputs, step=1):
    """Step 1: the deployed arm of lik_<job> (LearnedCost). Step 2: the signed model of sgn_<job>. Plus the old model."""
    from rm_model import load_model
    from rm_verify import verified_model
    old_root = rt.HERE/'results'/('train_%s' % cfg['s1']['old_train_job'])
    verified_model(old_root)
    inputs[old_root.name] = rt.sha(old_root/'manifest.json')
    old = load_model(old_root/'model.pt', device)
    if step == 2:
        from rm_signed import verified_sgn, load_signed
        new_root = rt.HERE/'results'/('sgn_%s' % train_job)
        verified_sgn(new_root)
        inputs[new_root.name] = rt.sha(new_root/'manifest.json')
        if not rt.read(new_root/'decision.json')['deploy']:
            raise ValueError('Step-2 training did not pass the coverage check: '+str(new_root))
        return 'signed', load_signed(new_root/'model.pt', device), old
    from rm_lik import verified_lik
    new_root = rt.HERE/'results'/('lik_%s' % train_job)
    verified_lik(new_root)
    inputs[new_root.name] = rt.sha(new_root/'manifest.json')
    arm = rt.read(new_root/'decision.json')['deploy']
    if arm is None:
        raise ValueError('Step-1 training did not pass the deployment gate: '+str(new_root))
    old_root = rt.HERE/'results'/('train_%s' % cfg['s1']['old_train_job'])
    verified_model(old_root)
    inputs[old_root.name] = rt.sha(old_root/'manifest.json')
    return arm, load_model(new_root/('model_%s.pt' % arm), device), load_model(old_root/'model.pt', device)


# ------------------------------------------------------------------ deployment

def f2_cell(cx, cfg, task, ctx, states, weights, ref, K):
    """Fixed-K merge + additive PTA through hard_regime step 1's own code path (hr_goal.rollout, no look-ahead)."""
    check_deadline(cx.hard)
    sim, rho, ell, est = goal_rollout(K, task['word'], ctx, cx.c, states, weights, {'M_PTA': PTA()}, cx.hard)
    c = scored(task, 'M_PTA', K, est['M_PTA'], ref, cfg['tolerance'], dict(physical_calls=sim.cost['physical_calls']))
    c.update(method='F2', budget=None)
    print('RM_EVAL2 %s F2 K=%d calls=%d Q=%s' % (task['id'], K, c['physical_calls'], c.get('Q_QoI')), flush=True)
    return c


def labelled(c, run, method):
    if c is not None:
        c['method'] = method
    run['method'] = method
    return c, run


def batch_tasks(cfg, batch, tasks):
    if batch == 'B':
        return [t for t in tasks if t['family'] == cfg['eval2']['b_family']]
    ids = set(cfg['eval2']['hard_check'])
    out = [t for t in tasks if t['id'] in ids]
    if len(out) != len(ids):
        raise ValueError('Hard-check tasks missing from the resolved task list')
    return out


def deploy(cx, cfg, batch, shard, shards, train_job, inputs, save, step=1):
    e2, m3 = cfg['eval2'], cfg['m3']
    arm, new, old = models(cfg, train_job, cx.device, inputs, step)
    if step == 2:
        from rm_signed import run_signed
        new_run = lambda task, ctx, states, weights, ref, a: run_signed(cx, cfg, task, ctx, states, weights, ref, a, new)  # noqa: E731
    else:
        new_run = lambda task, ctx, states, weights, ref, a: run_one(cx, cfg, task, ctx, states, weights, ref, 'L', a, new)  # noqa: E731
    tasks, npz, _ = task_list(cfg, inputs)
    mine = take_shard(batch_tasks(cfg, batch, tasks), shard, shards)
    old_cells = m3_cells(cfg, inputs)
    save('unit.json', dict(step=step, batch=batch, index=shard, shards=shards, arm=arm, mine=[t['id'] for t in mine]))
    cells, runs = [], []

    def add(c, r):
        runs.append(r)
        if c is not None:
            cells.append(c)

    for task in mine:
        states, weights, ctx = task_context(cx, task)
        ref = dict(np.load(npz[task['id']]))
        for a in m3['alpha']:
            add(*labelled(*new_run(task, ctx, states, weights, ref, a), 'Lnew'))
        if passes16(cells, task['id'], 'Lnew'):
            for a in e2['alpha_ext']:
                add(*labelled(*new_run(task, ctx, states, weights, ref, a), 'Lnew'))
        if batch == 'B' and passes16(old_cells, task['id'], 'L'):
            for a in e2['alpha_ext']:
                add(*labelled(*run_one(cx, cfg, task, ctx, states, weights, ref, 'L', a, old), 'Lold'))
        for b in list(m3['geo_budgets'])+list(e2['geo_ext']):
            add(*labelled(*run_one(cx, cfg, task, ctx, states, weights, ref, 'C1', b, None), 'C1'))
        for K in e2['f2_ext']:
            c = f2_cell(cx, cfg, task, ctx, states, weights, ref, K)
            add(c, dict(task_id=task['id'], family=task['family'], method='F2', K=K, status=c['status'],
                        physical_calls=c['physical_calls'], Q_QoI=c.get('Q_QoI')))
        save('cells.json', cells)
        save('runs.json', runs)
    return dict(batch=batch, arm=arm, tasks=len(mine), runs=len(runs), failures=sum(r['status'] != 'OK' for r in runs))


# ------------------------------------------------------------------ analysis (CPU)

def point(c, key):
    q = c.get(key) if c.get('status') == 'OK' else None
    return c['physical_calls'], q


def needed_by(points, key):
    """rm_deploy.needed on the chosen Q (central Q_QoI, band lower bound Q_QoI_low, or upper bound Q_QoI_high)."""
    return needed([point(c, key) for c in points])


def analyze(cfg, jobs, inputs):
    """jobs: {'B': deploy array id, 'hard': deploy array id or None}."""
    e2, f2 = cfg['eval2'], cfg['f2']
    tasks, npz, step1 = task_list(cfg, inputs)
    old_cells = m3_cells(cfg, inputs)
    new_cells, arm = [], None
    for batch, job in jobs.items():
        if job is None:
            continue
        roots = sorted((rt.HERE/'results').glob('eval2_%s_*' % job))
        if not roots:
            raise ValueError('No step-1 deployment shards for job %s' % job)
        for r in roots:
            verified_shard(r, 'EVAL2_SHARD_COMPLETE', M3_KEYS+('s1', 'eval2'), M3_MODEL_FILES+('rm_lik.py', 'rm_signed.py', 'rm_deploy.py'))
            inputs[r.name] = rt.sha(r/'manifest.json')
            u = rt.read(r/'unit.json')
            if u['batch'] != batch:
                raise ValueError('Batch mismatch in '+str(r))
            arm = u['arm']
            new_cells += rt.read(r/'cells.json')
    done = {c['task_id'] for c in new_cells}
    pts = {m: {} for m in ('Lnew', 'Lold', 'C1', 'F2')}
    for c in new_cells:
        pts[c['method']].setdefault(c['task_id'], []).append(c)
    for c in old_cells:
        if c['task_id'] in done and c.get('method') == 'L':
            pts['Lold'].setdefault(c['task_id'], []).append(c)
    for c in step1:
        if c['task_id'] in done and c['arm'] == f2['arm'] and c['K'] in f2['K']:
            pts['F2'].setdefault(c['task_id'], []).append(dict(c, method='F2'))
    # C1 re-run against M3 at the shared budgets (consistency of the new deployment path)
    c1_check = []
    for c in old_cells:
        if c['task_id'] in done and c.get('method') == 'C1':
            mine = [x for x in pts['C1'].get(c['task_id'], []) if x.get('budget') == c.get('budget')]
            if mine:
                c1_check.append(dict(task_id=c['task_id'], budget=c['budget'], calls_equal=mine[0]['physical_calls'] == c['physical_calls'],
                                     dQ=None if c.get('Q_QoI') is None or mine[0].get('Q_QoI') is None else abs(mine[0]['Q_QoI']-c['Q_QoI'])))
    sel = [t for t in tasks if t['id'] in done]
    vals = {key: {m: {t['id']: needed_by(pts[m][t['id']], key) for t in sel} for m in pts}
            for key in ('Q_QoI', 'Q_QoI_low', 'Q_QoI_high')}
    table = {}
    for t in sel:
        row = dict(family=t['family'])
        for m in pts:
            p = sorted(point(c, 'Q_QoI') for c in pts[m][t['id']])
            row[m] = dict(atoms=vals['Q_QoI'][m][t['id']][0], censored=vals['Q_QoI'][m][t['id']][1],
                          band_censored=vals['Q_QoI_low'][m][t['id']][1],
                          left_censored=bool(p and p[0][1] is not None and p[0][1] <= 1 and not vals['Q_QoI'][m][t['id']][1]),
                          points=p)
        table[t['id']] = row
    out = dict(arm=arm, c1_consistency=dict(cells=len(c1_check), all_calls_equal=all(x['calls_equal'] for x in c1_check),
                                            max_dQ=max([x['dQ'] for x in c1_check if x['dQ'] is not None] or [0.])),
               tasks=table)
    Bt = [t for t in sel if t['family'] == e2['b_family']]
    if Bt:
        cmp = {key: {'Lnew_vs_F2': compare(vals[key], 'F2', 'Lnew', Bt), 'Lnew_vs_C1': compare(vals[key], 'C1', 'Lnew', Bt),
                     'Lnew_vs_Lold': compare(vals[key], 'Lold', 'Lnew', Bt), 'Lold_vs_F2': compare(vals[key], 'F2', 'Lold', Bt)}
               for key in ('Q_QoI', 'Q_QoI_high')}
        resolvable = [t['id'] for t in Bt if vals['Q_QoI_low']['Lnew'][t['id']][1]]
        main = cmp['Q_QoI']['Lnew_vs_F2']
        if resolvable:
            reading = 'S1_EARLY'
        elif main['ratio'] < e2['b_ratio'] or main['not_worse'] < e2['b_not_worse']:
            reading = 'S1_ATOMS'
        else:
            reading = 'B_PASS'
        out['B'] = dict(compare=cmp, resolvable_failures=resolvable, reading=reading,
                        rule='S1_EARLY if any B task is right-censored with pass := Q_low <= 1; else S1_ATOMS if L\'/F2 '
                             '(prep-equal geometric mean of needed atoms, central Q) < %.2f or fewer than %d of 12 '
                             'preparations not worse; else B_PASS (run the hard check)' % (e2['b_ratio'], e2['b_not_worse']))
    Ht = [t for t in sel if t['id'] in set(e2['hard_check'])]
    if Ht:
        abab = [t for t in Ht if t['family'] in ('AB', 'AABB')]
        ratio = gmean([vals['Q_QoI']['Lnew'][t['id']][0]/vals['Q_QoI']['Lold'][t['id']][0] for t in abab])
        new_cens = [t['id'] for t in abab if vals['Q_QoI']['Lnew'][t['id']][1] and not vals['Q_QoI']['Lold'][t['id']][1]]
        abb = {t['id']: {m: min([q for _, q in table[t['id']][m]['points'] if q is not None] or [None]) for m in pts}
               for t in Ht if t['family'] == 'ABB'}
        reading = 'S1_REGRESS' if ratio > e2['regress_ratio'] or new_cens else 'S1_PASS'
        out['hard'] = dict(ratio_Lnew_over_Lold=ratio, new_censoring=new_cens, abb_min_Q=abb, reading=reading,
                           rule='S1_REGRESS if the geometric mean of L\'/L needed atoms over the 6 AB/AABB tasks > %.2f '
                                'or L\' is right-censored where L was not; else S1_PASS' % e2['regress_ratio'])
    return out


# ------------------------------------------------------------------ smoke

def smoke_deploy(cx, cfg, inputs):
    """The new deployment path reproduces M3 (old model and C1 at one point) and step 1 (F2 at K=4); F2 runs at K=1."""
    from rm_model import load_model
    from rm_verify import verified_model
    tasks, npz, step1 = task_list(cfg, inputs)
    old_cells = m3_cells(cfg, inputs)
    sm = cfg['eval2']['smoke']
    task = next(t for t in tasks if t['id'] == sm['task'])
    states, weights, ctx = task_context(cx, task)
    ref = dict(np.load(npz[task['id']]))
    old_root = rt.HERE/'results'/('train_%s' % cfg['s1']['old_train_job'])
    verified_model(old_root)
    old = load_model(old_root/'model.pt', cx.device)
    out = {}
    for method, b, model in (('L', sm['alpha'], old), ('C1', sm['geo_budget'], None)):
        c, _ = run_one(cx, cfg, task, ctx, states, weights, ref, method, b, model)
        m3 = next(x for x in old_cells if x['task_id'] == task['id'] and x.get('method') == method and x.get('budget') == b)
        out['m3_%s' % method] = dict(calls=[c['physical_calls'], m3['physical_calls']], Q=[c['Q_QoI'], m3['Q_QoI']],
                                     passed=c['physical_calls'] == m3['physical_calls'] and abs(c['Q_QoI']-m3['Q_QoI']) <= 1e-9)
    c4 = f2_cell(cx, cfg, task, ctx, states, weights, ref, 4)
    s4 = next(x for x in step1 if x['task_id'] == task['id'] and x['arm'] == cfg['f2']['arm'] and x['K'] == 4)
    a, b = np.asarray(c4['P']), np.asarray(s4['P'])
    out['f2_K4'] = dict(calls=[c4['physical_calls'], s4['physical_calls']], max_rel_dP=float(np.max(np.abs(a-b)/np.abs(b))),
                        passed=c4['physical_calls'] == s4['physical_calls'] and float(np.max(np.abs(a-b)/np.abs(b))) <= 1e-9)
    c1 = f2_cell(cx, cfg, task, ctx, states, weights, ref, 1)
    out['f2_K1'] = dict(calls=c1['physical_calls'], Q=c1.get('Q_QoI'), status=c1['status'],
                        passed=c1['status'] == 'OK' and c1['physical_calls'] == task['H'])
    return out

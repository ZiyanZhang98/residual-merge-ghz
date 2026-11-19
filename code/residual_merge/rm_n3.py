"""n=3 as a second setting (HANDOFF 5j item 22.38; user decision 2026-11-03): the frozen recipe of 22.37 applied
unchanged at n=3. Same label generator (the M1 rounds and the early rounds), same network and training settings
(1 seed), same deployment (Pauli-coordinate 8-NN candidates, budget alpha*tol/40, atom cap 256, additive PTA) and
the frozen extended grids of the n=4 test. Only n differs: the config keys m1/m3 are copied in memory with n=3, so
the n=4 settings, model and artifacts stay untouched.

Evaluation set: the resolved n=3 monitor tasks of hard_regime step 1, scored against their stored Monte Carlo truth.
The labels come from the n=3 fit preparations only (monitor labels are for the label-level evaluation), and no choice
of this recipe looked at n=3. The analysis (rm_n3_report.py) uses the n=4 test's summarize. Final data are never
accessed.
"""
import rm_runtime as rt
import time
import numpy as np
import torch
from hr_eval import context as task_context, shard as take_shard
from rm_deploy import task_list
from rm_eval2 import verified_shard
from rm_test import run_task, grid, F2_ARMS

N3_KEYS = ('n3', 'labels', 'm1', 'm1_early', 'merge', 'm2', 'm3', 'tolerance', 'hard_regime', 'f2', 'test')
N3_FILES = ('rm_core.py', 'rm_labels.py', 'rm_runtime.py', 'rm_report.py', 'rm_model.py', 'rm_deploy.py', 'rm_test.py',
            'rm_n3.py')
LABEL_PASSES = (('n3', None), ('n3early', 'm1_early'))   # results/labels_<pass>_<job>_<shard>, read by rm_report.load


def config3(cfg):
    n = cfg['n3']['n']
    return dict(cfg, m1=dict(cfg['m1'], n=n), m3=dict(cfg['m3'], n=n))


def pass_config(cfg3, early):
    """run.py's labels/labels_early settings: the early pass is the same generator at the cfg m1_early rounds."""
    return cfg3 if early is None else dict(cfg3, m1=dict(cfg3['m1'], events=cfg3[early]['events']))


def rows_of(root):
    d = torch.load(root/'labels.pt', map_location='cpu', weights_only=False)
    return d['n'], {k: v.numpy() for k, v in d['rows'].items()}


# ------------------------------------------------------------------ labels (both passes per shard)

def label_shard(cx, cfg, shard, shards, tag, sources, write):
    from rm_labels import generate
    c3 = config3(cfg)
    out = {}
    for name, early in LABEL_PASSES:
        root = rt.new_root(rt.HERE/'results'/('labels_%s_%s' % (name, tag)))
        rt.write(root/'unit.json', dict(index=shard, shards=shards, mode=name, n=c3['m1']['n']))
        res = generate(cx, pass_config(c3, early), shard, shards, root, lambda f, v, r=root: write(r/f, v))
        write(root/'summary.json', res)
        m = rt.manifest(root, 'labels_'+name, 'LABELS_SHARD_COMPLETE')
        if m['sources'] != sources:
            raise RuntimeError('Executable sources changed during this job')
        rt.write(root/'manifest.json', m)
        out[name] = res
        print('RM_N3_LABELS', name, res['preparations'], res['pairs'], res['events'], flush=True)
    return out


# ------------------------------------------------------------------ training (run.py's train, n=3 labels)

def train3(cx, cfg, job, root, inputs, save):
    from rm_report import load, report, shards as label_shards
    from rm_model import train, evaluate
    n, want = cfg['n3']['n'], cfg['n3']['label_shards']
    parts = []
    for name, _ in LABEL_PASSES:
        roots = label_shards('%s_%s' % (name, job))
        if len(roots) != want or any(rows_of(r)[0] != n for r in roots):
            raise ValueError('Expected %d n=%d label shards for %s_%s' % (want, n, name, job))
        parts.append(load('%s_%s' % (name, job), inputs))
    for i, part in enumerate(parts):
        part['event'] = part['event']+100_000_000*i   # distinct capture ids across label passes (as run.py)
    d = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
    gen = torch.Generator().manual_seed(int(cfg['m2']['seed']))
    model, log = train(d, cfg, cx.device, gen)
    tc = cfg['m2']
    torch.save(dict(din=model.mu.numel(), width=tc['width'], depth=tc['depth'], state=model.state_dict()), root/'model.pt')
    save('train_log.json', log)
    ev = evaluate(model, d, cx.device, cfg)
    save('label_eval.json', ev)
    save('label_report.json', {name: report('%s_%s' % (name, job), inputs) for name, _ in LABEL_PASSES})
    for key in ('fit_all', 'monitor_all'):
        print('RM_N3_EVAL', key, ev[key]['all_blocks'], ev[key]['long_blocks'], flush=True)
    return dict(pairs=int(len(d['event'])), fit_pairs=int((~d['split']).sum()), din=int(model.mu.numel()),
                final_loss=log[-1]['loss'])


def model3(cfg, device, job, inputs):
    from rm_model import load_model
    root = rt.HERE/'results'/('n3train_%s' % job)
    verified_shard(root, 'N3_TRAIN_COMPLETE', N3_KEYS, N3_FILES)
    inputs[root.name] = rt.sha(root/'manifest.json')
    return load_model(root/'model.pt', device)


# ------------------------------------------------------------------ evaluation on the resolved n=3 monitor tasks

def eval_shard(cx, cfg, train_job, shard, shards, inputs, save):
    c3 = config3(cfg)
    tasks, npz, _ = task_list(c3, inputs)
    model = model3(cfg, cx.device, train_job, inputs)
    mine = take_shard(tasks, shard, shards)
    save('unit.json', dict(index=shard, shards=shards, mine=[t['id'] for t in mine], grid=grid(cfg), train_job=train_job))
    cells, runs = [], []
    for task in mine:
        states, weights, ctx = task_context(cx, task)
        ref = dict(np.load(npz[task['id']]))
        c, r = run_task(cx, c3, task, ctx, states, weights, ref, model, grid(cfg))
        cells += c
        runs += r
        save('cells.json', cells)
        save('runs.json', runs)
    return dict(tasks=len(mine), runs=len(runs), failures=sum(r['status'] != 'OK' for r in runs))


# ------------------------------------------------------------------ smoke

def same_rows(a, b):
    """Max relative difference over every stored label array (event ids excluded)."""
    worst = 0.
    for k in a:
        if k == 'event':
            continue
        x, y = np.asarray(a[k], dtype=np.float64), np.asarray(b[k], dtype=np.float64)
        if x.shape != y.shape:
            return float('inf')
        if x.size:
            worst = max(worst, float(np.max(np.abs(x-y)/np.maximum(np.abs(y), 1e-300))))
    return worst


def smoke(cx, cfg, root, inputs, write):
    from rm_labels import generate, pool
    from rm_model import train, evaluate, PairNet, label_inputs
    from rm_curv import old_model
    sm, n = cfg['n3']['smoke'], cfg['n3']['n']
    c3 = config3(cfg)
    out = {}
    # 1: only n differs from the frozen n=4 settings
    changed = sorted(k for k in cfg if c3[k] != cfg[k])
    out['config_only_n_differs'] = dict(changed=changed, n=[c3['m1']['n'], c3['m3']['n']],
                                        passed=changed == ['m1', 'm3'] and dict(c3['m1'], n=cfg['m1']['n']) == cfg['m1']
                                        and dict(c3['m3'], n=cfg['m3']['n']) == cfg['m3'] and c3['m1']['n'] == c3['m3']['n'] == n)
    # 2: the label generator, called as the n=3 passes call it, reproduces the stored n=4 labels of preparation 0
    for (name, early), job in zip(LABEL_PASSES, cfg['m2']['labels_jobs']):
        r = rt.new_root(root/('n4_%s' % name))
        t0 = time.perf_counter()
        generate(cx, pass_config(cfg, early), 0, sm['n4_preps_shards'], r, lambda f, v, rr=r: write(rr/f, v))
        sec = time.perf_counter()-t0
        _, mine = rows_of(r)
        _, old = rows_of(rt.HERE/'results'/('labels_%s_0' % job))
        k = len(mine['event'])
        head = {key: v[:k] for key, v in old.items()}
        diff = same_rows(mine, head)
        out['n4_labels_reproduce_%s' % name] = dict(stored='labels_%s_0' % job, pairs=k, events=int(len(np.unique(mine['event']))),
                                                    max_rel_diff=diff, seconds=sec,
                                                    passed=k > 0 and np.array_equal(mine['event'], head['event']) and diff <= 1e-9)
    # 3: n=3 labels of one fit and one monitor preparation, both passes
    entries = pool(cx, n)
    parts, timing = [], {}
    for i, (name, early) in enumerate(LABEL_PASSES):
        r = rt.new_root(root/('n3_%s' % name))
        t0 = time.perf_counter()
        generate(cx, pass_config(c3, early), 0, sm['n3_preps_shards'], r, lambda f, v, rr=r: write(rr/f, v))
        timing[name] = time.perf_counter()-t0
        got_n, rows = rows_of(r)
        rows['event'] = rows['event']+100_000_000*i
        parts.append((got_n, rows))
    d = {k: np.concatenate([p[1][k] for p in parts]) for k in parts[0][1]}
    dims = int(d['xa'].shape[1])
    finite = all(np.all(np.isfinite(d[k])) for k in ('r_psi', 'r_phi', 'se_psi', 'pta_psi', 'level_psi'))
    out['n3_labels'] = dict(n=[p[0] for p in parts], pool=len(entries), splits=sorted({e['split'] for j, e in enumerate(entries)
                                                                                      if j % sm['n3_preps_shards'] == 0}),
                            pairs=int(len(d['event'])), monitor_pairs=int(d['split'].sum()), coordinate_dims=dims,
                            finite=finite, seconds=timing,
                            passed=all(p[0] == n for p in parts) and dims == 4**n-1 and finite and len(entries) == 60
                            and 0 < int(d['split'].sum()) < len(d['event']))
    # 4: training and label-level evaluation run on n=3 labels (shortened training; the production run uses cfg m2)
    short = dict(cfg, m2=dict(cfg['m2'], steps=sm['train_steps']))
    model, log = train(d, short, cx.device, torch.Generator().manual_seed(int(cfg['m2']['seed'])))
    ev = evaluate(model, d, cx.device, cfg)
    out['n3_training'] = dict(steps=sm['train_steps'], din=int(model.mu.numel()), loss=[log[0]['loss'], log[-1]['loss']],
                              monitor_concordance=ev['monitor_all']['all_blocks'],
                              passed=int(model.mu.numel()) == int(label_inputs(d, cx.device).shape[1])
                              and np.isfinite(log[-1]['loss']))
    # 5: n=3 evaluation tasks: the resolved monitor tasks, disjoint from the fit preparations
    tasks, npz, step1 = task_list(c3, inputs)
    fit = {e['signature'] for e in entries if e['split'] == 'fit'}
    monitor = {e['signature'] for e in entries if e['split'] == 'monitor'}
    fams = sorted({t['family'] for t in tasks})
    by = {f: sum(t['family'] == f for t in tasks) for f in fams}
    out['n3_tasks'] = dict(count=len(tasks), by_family=by, preparations=len({t['prep_signature'] for t in tasks}),
                           in_monitor_pool=all(t['prep_signature'] in monitor for t in tasks),
                           shared_with_fit=sum(t['prep_signature'] in fit for t in tasks),
                           passed=len(tasks) == sm['tasks'] and all(t['n'] == n for t in tasks)
                           and all(t['prep_signature'] in monitor for t in tasks) and not any(t['prep_signature'] in fit for t in tasks))
    # 6: the evaluation runner at n=3 (smoke-trained model): F2/M0/F2mul reproduce step 1; L and C1 run
    task = next(t for t in tasks if t['id'] == sm['task'])
    states, weights, ctx = task_context(cx, task)
    ref = dict(np.load(npz[task['id']]))
    cells, runs = run_task(cx, c3, task, ctx, states, weights, ref, model, dict(L=[sm['alpha']], C1=[sm['geo_budget']], F2=[sm['K']]))
    for arm, method in F2_ARMS:
        c = next(x for x in cells if x['method'] == method)
        s = next(x for x in step1 if x['task_id'] == task['id'] and x['arm'] == arm and x['K'] == sm['K'])
        q = [c.get('Q_QoI'), s.get('Q_QoI')]
        same_q = q[0] == q[1] if None in q else abs(q[0]-q[1]) <= 1e-9
        out['n3_f2_%s' % method] = dict(task=task['id'], K=sm['K'], calls=[c['physical_calls'], s['physical_calls']], Q=q,
                                        passed=c['physical_calls'] == s['physical_calls'] and same_q)
    for method in ('L', 'C1'):
        r = next(x for x in runs if x['method'] == method)
        out['n3_deploy_%s' % method] = dict(task=task['id'], status=r['status'], calls=r.get('physical_calls'),
                                            max_atoms=r.get('max_atoms'), Q=r.get('Q_QoI'), seconds=r.get('seconds'),
                                            merge_seconds=r.get('merge_seconds'),
                                            passed=r['status'] == 'OK' and r.get('Q_QoI') is not None and np.isfinite(r['Q_QoI']))
    # 7: the frozen n=4 model still verifies (nothing it depends on was touched)
    old_model(cfg, cx.device, inputs)
    out['n4_model_verified'] = dict(train_job=cfg['s1']['old_train_job'], passed=True)
    return out

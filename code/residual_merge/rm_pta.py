"""PTA simulator on every resolved task (HANDOFF 5j item 22.41, 2026-11-05, user-approved). CPU, read-only.

The PTA simulator replaces the noise by its Pauli twirl: each state stays GHZ-diagonal and does not branch, so every
initial branch is propagated alone, P = sum_i w_i psi_PTA,i and N = sum_i w_i phi_PTA,i (the computation of
hard_regime/results/audit2/compare.py, job 28502410, which covered the 55 resolved hard tasks only). Scored with the
same hr_eval.scored as every deployment cell against the stored Monte Carlo truth. Sets: n=4 development (63 tasks),
n=4 test (90), n=3 held-out (64). No new truth, no training, no method parameter: there is nothing to select, so the
test set is used once more without selection effects. Final data are never accessed.
"""
import rm_runtime as rt
from collections import defaultdict
import numpy as np
from qa_sources import PTA
from hr_eval import context as task_context, scored
from rm_deploy import task_list

FAMILIES = ('AB', 'AABB', 'ABB', 'A', 'AAB', 'B')


def simulate(cx, cfg, task, npz_path, pta):
    states, weights, ctx = task_context(cx, task)
    v = pta(states, ctx['initial_factors'], task['word'], ctx, cx.c)
    w = weights.to(v['psi'].dtype)
    est = dict(P=(w[:, None]*v['psi']).sum(0), N=(w[:, None]*v['phi']).sum(0))
    ref = dict(np.load(npz_path))
    c = scored(task, 'PTA_sim', None, est, ref, cfg['tolerance'],
               dict(initial_branches=int(len(states)), single_state_rounds=int(len(states))*task['H'],
                    weight_sum=float(w.sum())))
    Pr, Fa = c.get('P_relative'), c.get('F_absolute')
    tol = cfg['tolerance']
    c['first_failing_round'] = None if Pr is None else next(
        (j+1 for j in range(task['H']) if Pr[j] > tol['P_relative'] or Fa[j] > tol['F_absolute']), None)
    c['max_P_relative'] = None if Pr is None else float(max(Pr))
    c['max_F_absolute'] = None if Fa is None else float(max(Fa))
    return c


def test_set(cx, cfg, pc, inputs):
    from rm_test import load_pool
    tasks = {t['id']: t for t in load_pool(cx, cfg, pc['test_pool_job'], inputs)}
    roots = sorted((rt.HERE/'results').glob('test_%s_*' % pc['test_run_job']))
    if len(roots) != cfg['test']['shards']:
        raise ValueError('Expected %d test shards, found %d' % (cfg['test']['shards'], len(roots)))
    out = []
    for r in roots:
        inputs[r.name] = rt.sha(r/'manifest.json')
        for x in rt.read(r/'mc.json'):
            if x['resolved']:
                out.append((tasks[x['task_id']], r/('mc_%s.npz' % x['task_id'])))
    return out


def dev_or_n3(cfg, n3, inputs):
    if n3:
        from rm_n3 import config3
        cfg = config3(cfg)
    tasks, npz, _ = task_list(cfg, inputs)
    return [(t, npz[t['id']]) for t in tasks]


def summary(cells):
    out = {}
    groups = dict(hard=('AB', 'AABB', 'ABB'), easy=('A', 'AAB', 'B'), **{f: (f,) for f in FAMILIES})
    for g, fams in groups.items():
        cs = [c for c in cells if c['family'] in fams]
        if not cs:
            continue
        ok = [c for c in cs if c.get('status') == 'OK' and c.get('Q_QoI') is not None]
        ff = [c['first_failing_round'] for c in ok if c['first_failing_round']]
        out[g] = dict(tasks=len(cs), scored=len(ok), passed=sum(c['Q_QoI'] <= 1 for c in ok),
                      passed_by_low=sum(c.get('Q_QoI_low') is not None and c['Q_QoI_low'] <= 1 for c in ok),
                      passed_by_high=sum(c.get('Q_QoI_high') is not None and c['Q_QoI_high'] <= 1 for c in ok),
                      median_Q=float(np.median([c['Q_QoI'] for c in ok])) if ok else None,
                      min_Q=float(min(c['Q_QoI'] for c in ok)) if ok else None,
                      median_max_P_relative=float(np.median([c['max_P_relative'] for c in ok])) if ok else None,
                      median_max_F_absolute=float(np.median([c['max_F_absolute'] for c in ok])) if ok else None,
                      median_first_failing_round=float(np.median(ff)) if ff else None)
    return out


def run_pta(cx, cfg, inputs, save):
    pc, pta = cfg['pta_sim'], PTA()
    sets = dict(dev=dev_or_n3(cfg, False, inputs), test=test_set(cx, cfg, pc, inputs), n3=dev_or_n3(cfg, True, inputs))
    cells = {}
    for name, items in sets.items():
        cells[name] = []
        for task, path in items:
            try:
                c = simulate(cx, cfg, task, path, pta)
            except (FloatingPointError, RuntimeError) as exc:
                c = dict(task_id=task['id'], n=task['n'], family=task['family'], status='NUMERIC_FAILURE', error=repr(exc))
            cells[name].append(c)
            print('RM_PTA', name, task['id'], c.get('status'), c.get('Q_QoI'), flush=True)
        save('cells.json', cells)
    # reproduction of the earlier PTA-alone numbers (job 28502410) on the tasks both cover
    old = rt.read(rt.ARCH/'hard_regime'/'results'/'audit2'/('compare_%s' % pc['reference_job'])/'pta_alone.json')
    rows = [(c['task_id'], c['Q_QoI'], old[c['task_id']]['Q']) for s in ('dev', 'n3') for c in cells[s]
            if c['task_id'] in old and c.get('Q_QoI') is not None]
    repro = dict(tasks=len(rows), max_abs_dQ=max([abs(a-b) for _, a, b in rows] or [0.]),
                 passed=bool(rows) and all(abs(a-b) <= 1e-9*max(1., abs(b)) for _, a, b in rows))
    if not repro['passed']:
        raise ValueError('PTA simulator does not reproduce job %s: %s' % (pc['reference_job'], repro))
    return dict(reproduction=repro, failures={s: [c['task_id'] for c in v if c.get('status') != 'OK'] for s, v in cells.items()},
                summary={s: summary(v) for s, v in cells.items()})

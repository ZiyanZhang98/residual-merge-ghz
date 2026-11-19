"""Tuning pilot (HANDOFF 22.22): is a full look-ahead tuning worth running?

Development set only (the 55 resolved hard tasks of hard_regime), K in config smoke.K, the learned
model of hr_goal (seed 0). Each look-ahead setting runs for the PTA and the learned source, so the
"remove learning" ablation gets the same settings; merge+PTA ("remove look-ahead", share = 0)
comes from hard_regime step 1, which is the same code path bit for bit.
"""
import lt_runtime as rt
import math
import time
from collections import defaultdict
import numpy as np
import torch
import qa_sim
from ghz_physics import coords
from g1_core import check_deadline, Deadline
from qa_sources import PTA
from hr_eval import load_tasks, context, scored
from hr_goal import LookAhead, rollout as hr_rollout
from lt_lookahead import StdLookAhead, rollout


def hard_tasks(cfg, refs):
    resolved = {r['task_id'] for r in refs if r['resolved']}
    return [t for t in load_tasks(rt.hr.config()) if t['family'] in cfg['hard_families'] and t['id'] in resolved]


def arms(cfg):
    out = [(s, c) for c in cfg['smoke']['configs'] for s in ('pta', 'learned')]
    return out+[('learned', c) for c in cfg['smoke']['learned_only']]


def look(source, c, models, task):
    return StdLookAhead(source, c['share'], c['lengths'], c['quantities'],
                        models[task['n']] if source == 'learned' else None, task['gamma'])


def checks(cx, cfg, task, models):
    """1. LTSim with the frozen look-ahead equals hr_goal bit for bit. 2. Standardized block variance."""
    states, weights, ctx = context(cx, task)
    old = LookAhead('pta', [1, 2, 4, 8, 16], 1.0)
    a = hr_rollout(16, task['word'], ctx, cx.c, states, weights, {'M_PTA': PTA()}, cx.hard, old)
    b = rollout(16, task['word'], ctx, cx.c, states, weights, {'M_PTA': PTA()}, cx.hard, LookAhead('pta', [1, 2, 4, 8, 16], 1.0))
    same = torch.equal(a[1], b[1]) and a[2] == b[2] and all(
        torch.equal(a[3][k][q], b[3][k][q]) for k in ('M0', 'M_PTA', 'M_PTA__mul') for q in ('P', 'N'))
    assert same, 'LTSim differs from hr_goal'
    sim = qa_sim.QoISim(64, task['word'], ctx, cx.c, states, weights)
    for x in task['word'][:6]:
        sim.advance(x)
    S, w = sim.states, sim.weights
    out = dict(bitwise_hr_goal=True)
    for source in ('pta', 'learned'):
        c = cfg['smoke']['configs'][1]
        la = look(source, c, models, task)
        la.weights = w
        Z = la(S, sim.factors, task['word'][6:], ctx, cx.c)
        wn = w/w.sum()
        geo = coords(S, ctx['aux']['pauli'])
        vg = float((wn @ (geo-wn @ geo)**2).sum())
        vz = float((wn @ (Z-wn @ Z)**2).sum())
        out['variance_ratio_'+source] = vz/vg
        assert abs(vz/vg-c['share']) < 1e-8*max(1, c['share']), source
    return out


def run(cx, cfg, tasks, npz, models, root):
    cells, timing = [], []
    for j, task in enumerate(tasks):
        check_deadline(cx.hard)
        ref = dict(np.load(npz[task['id']]))
        states, weights, ctx = context(cx, task)
        for K in cfg['smoke']['K']:
            for source, c in arms(cfg):
                label = '%s_%s' % (source, c['name'])
                start = time.perf_counter()
                try:
                    sim, _, _, est = rollout(K, task['word'], ctx, cx.c, states, weights, {'M_PTA': PTA()}, cx.hard,
                                             look(source, c, models, task))
                    sec = time.perf_counter()-start
                    for arm, e in est.items():
                        name = label+'_raw' if arm == 'M0' else label+'+'+arm
                        cells.append(scored(task, name, K, e, ref, cfg['tolerance'], dict(seconds=sec, lookahead_seconds=sim.lookahead_seconds)))
                except Deadline:
                    raise
                except (FloatingPointError, RuntimeError) as exc:
                    for suffix in ('_raw', '+M_PTA', '+M_PTA__mul'):
                        cells.append(dict(task_id=task['id'], n=task['n'], family=task['family'], prep_id=task['prep_signature'],
                                          arm=label+suffix, K=K, status='NUMERIC_FAILURE', error=repr(exc)))
                    sec = time.perf_counter()-start
                timing.append(dict(task_id=task['id'], n=task['n'], K=K, arm=label, seconds=sec))
        rt.write(root/'cells.json', cells); rt.write(root/'timing.json', timing)
        print('LT_SMOKE', j+1, len(tasks), task['id'], flush=True)
    return cells


# ------------------------------------------------------------------ decision

def q_table(cells, K_set):
    out = {}
    for c in cells:
        if c['K'] in K_set:
            out[(c['task_id'], c['arm'], c['K'])] = c.get('Q_QoI') if c.get('status') == 'OK' else None
    return out


def compare(Q, meta, n, arm, base, K_set):
    """Geometric mean over (task, K) of Q_base/Q_arm and the share of preparations not worse."""
    per_prep, logs, missing = defaultdict(list), [], 0
    for t, (tn, prep) in meta.items():
        if tn != n:
            continue
        for K in K_set:
            qa, qb = Q.get((t, arm, K)), Q.get((t, base, K))
            if qa is None or qb is None or qa <= 0 or qb <= 0:
                missing += 1; continue
            r = math.log(qb/qa)
            logs.append(r); per_prep[prep].append(r)
    if not logs:
        return None
    preps = [np.mean(v) for v in per_prep.values()]
    return dict(ratio=float(math.exp(np.mean(logs))), not_worse=int(sum(p >= 0 for p in preps)),
                preparations=len(preps), pairs=len(logs), missing=missing)


def decide(cells, step1_cells, step2_cells, cfg, tasks):
    K_set, d = cfg['smoke']['K'], cfg['decision']
    meta = {t['id']: (t['n'], t['prep_signature']) for t in tasks}
    names = {'M0': 'merge', 'M_PTA': 'merge+PTA', 'M_PTA__mul': 'merge+PTA__mul'}
    base = [dict(c, arm=names[c['arm']]) for c in step1_cells if c['arm'] in names and c['task_id'] in meta]
    old = [c for c in step2_cells if c['task_id'] in meta and c['arm'] in ('pta_look+M_PTA', 'learned_look+M_PTA', 'pta_look+M_PTA__mul', 'learned_look+M_PTA__mul')]
    Q = q_table(cells+base+old, K_set)
    labels = sorted({c['arm'] for c in cells})
    report = {}
    for n in sorted({v[0] for v in meta.values()}):
        rows = {}
        for arm in labels+['pta_look+M_PTA', 'learned_look+M_PTA']:
            readout_base = 'merge+PTA__mul' if arm.endswith('__mul') else 'merge+PTA'
            rows[arm] = dict(vs_merge_PTA=compare(Q, meta, n, arm, readout_base, K_set),
                             vs_merge=compare(Q, meta, n, arm, 'merge', K_set))
            if arm.startswith('learned_'):
                twin = 'pta_'+arm[len('learned_'):]
                if twin in labels:
                    rows[arm]['vs_same_setting_PTA'] = compare(Q, meta, n, arm, twin, K_set)
        best = {}
        for src in ('pta', 'learned'):
            cand = [a for a in labels if a.startswith(src+'_') and a.endswith('+M_PTA') and rows[a]['vs_merge_PTA']]
            if cand:
                best[src] = max(cand, key=lambda a: rows[a]['vs_merge_PTA']['ratio'])
        verdict = 'NO_GO'
        if 'pta' in best and 'learned' in best:
            vm = rows[best['learned']]['vs_merge_PTA']
            vp = compare(Q, meta, n, best['learned'], best['pta'], K_set)
            frac = d['preparation_fraction']
            strong = lambda r: r['ratio'] >= d['ratio'] and r['not_worse'] >= frac*r['preparations']-1e-12  # noqa: E731
            if strong(vm) and strong(vp):
                verdict = 'GO'
            elif vm['ratio'] > 1 and vp['ratio'] > 1:
                verdict = 'WEAK'
            best['learned_vs_best_PTA'] = vp
        report[str(n)] = dict(rows=rows, best=best, verdict=verdict)
    overall = 'GO' if any(r['verdict'] == 'GO' for r in report.values()) else \
        'WEAK' if any(r['verdict'] == 'WEAK' for r in report.values()) else 'NO_GO'
    return dict(K=K_set, rule='GO if, for some n, the best learned setting (additive readout, chosen on this set) beats '
                'merge+PTA and the best PTA look-ahead setting each by >= %.2f with >= %.3f of preparations not worse; '
                'WEAK if both ratios > 1; else NO_GO' % (d['ratio'], d['preparation_fraction']),
                by_n=report, verdict=overall)

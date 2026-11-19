"""Step 1 (independent references + hard-regime benchmark) and step 2 (look-ahead merging) of
HANDOFF 22.17, scored against Monte Carlo truth with error bars. No training, no final data."""
import hr_runtime as rt
import math
from collections import defaultdict
import numpy as np
import torch
import ghz_factors as gf
from g1_core import check_deadline, Deadline
from qa_sources import PTA
from qa_metrics import score, needed
from hr_mc import reference
from hr_goal import LookAhead, rollout as goal_rollout

QB_RESULTS = rt.ARCH/'qoi_learn'/'results'


def load_tasks(cfg):
    """All monitor tasks of the step-B data (never trained on), with their merge-reference status."""
    tasks = []
    for r in sorted(QB_RESULTS.glob('data_%s_*' % cfg['data_job'])):
        rt.qb.data_artifact(r)
        for t in rt.read(r/'tasks.json'):
            tasks.append(dict(t, merge_reference_path=str(r/'refs'/t['reference']), data_root=r.name))
    ids = [t['id'] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate task ids')
    return sorted(tasks, key=lambda t: (t['n'], t['id']))


def shard(items, index, count):
    return [x for j, x in enumerate(items) if j % count == index]


def context(cx, task):
    states, weights, factors = gf.prepare(task['prep'], cx.c, cx.device)
    letters, pta = cx.letters(task['n'], task['gamma'])
    ctx = dict(letters=letters, pta=pta, aux=cx.aux(task['n']), initial_factors=factors)
    return states, weights, ctx


def scored(task, arm, K, est, ref, tol, extra):
    s = score(np.zeros(task['H']), est['P'].cpu().numpy(), est['N'].cpu().numpy(), ref['P'], ref['F'], tol,
              2*float(ref['se_F'].max()), 2*float(ref['se_P_relative'].max()))
    return dict(task_id=task['id'], n=task['n'], family=task['family'], prep_id=task['prep_signature'],
                arm=arm, K=K, **extra, **s)


def step1(cx, cfg, tasks, root, grid=None, mc=None):
    grid, mc = grid or cfg['K'], mc or cfg['mc']
    cells, refs = [], []
    for j, task in enumerate(tasks):
        check_deadline(cx.hard)
        states, weights, ctx = context(cx, task)
        seed = mc['seed']+int(rt.digest(task['id'])[:8], 16) % 1_000_000
        ref, info = reference(task['prep'], task['word'], ctx['letters'], ctx['aux'], cx.c, mc, cx.device, cx.hard, seed)
        np.savez_compressed(root/('mc_%s.npz' % task['id']), **ref)
        row = dict(task_id=task['id'], n=task['n'], family=task['family'], prep_id=task['prep_signature'], H=task['H'],
                   merge_reference_accepted=task['accepted'], seed=seed, **info)
        if task['accepted']:
            z = np.load(task['merge_reference_path'])
            d = cfg['decision']
            dP = np.abs(z['P']-ref['P']); dF = np.abs(z['F']-ref['F'])
            row.update(max_dP_relative=float((dP/ref['P']).max()), max_dF=float(dF.max()),
                agrees=bool((dP <= d['agree_sigma']*ref['se_P']+d['reference_tolerance']*ref['P']).all()
                            and (dF <= d['agree_sigma']*ref['se_F']+d['reference_tolerance']).all()))
        refs.append(row)
        for K in grid:
            try:
                sources = {'M_PTA': PTA()} if K <= cfg['correction_K_max'] else {}
                sim, rho, ell, est = goal_rollout(K, task['word'], ctx, cx.c, states, weights, sources, cx.hard)
                extra = dict(physical_calls=sim.cost['physical_calls'], correction_seconds=sum(sim.correction_seconds.values()))
                for arm, e in est.items():
                    cells.append(scored(task, arm, K, e, ref, cfg['tolerance'], extra))
            except Deadline:
                raise
            except (FloatingPointError, RuntimeError) as exc:
                for arm in ('M0', 'M_PTA', 'M_PTA__mul') if K <= cfg['correction_K_max'] else ('M0',):
                    cells.append(dict(task_id=task['id'], n=task['n'], family=task['family'], prep_id=task['prep_signature'],
                                      arm=arm, K=K, status='NUMERIC_FAILURE', error=repr(exc)))
        rt.write(root/'cells.json', cells); rt.write(root/'mc.json', refs)
        print('HR_STEP1', j+1, len(tasks), task['id'], info, flush=True)
    return cells, refs


def k_of(rows, grid, metric, censored):
    k = needed(rows, grid, metric)
    return k['K'] if k['status'] == 'FINITE' else censored


def arm_grid(cfg, arm, grid):
    return [k for k in grid if k <= cfg['correction_K_max']] if arm != 'M0' and not arm.endswith('_raw') else grid


def k_table(cells, cfg, grid):
    by = defaultdict(list)
    for c in cells:
        by[(c['task_id'], c['arm'])].append(c)
    out = {}
    for (tid, arm), rows in by.items():
        g = [k for k in arm_grid(cfg, arm, grid) if any(r['K'] == k for r in rows)]
        out[(tid, arm)] = dict(QoI=k_of(rows, g, 'Q_QoI', cfg['decision']['censored_K']),
                               P=k_of(rows, g, 'Q_P', cfg['decision']['censored_K']),
                               F=k_of(rows, g, 'Q_F', cfg['decision']['censored_K']), n=rows[0]['n'],
                               family=rows[0]['family'], prep=rows[0]['prep_id'],
                               calls={r['K']: r.get('physical_calls') for r in rows})
    return out


def med(xs):
    return float(np.median(xs)) if xs else None


def decide1(cells, refs, cfg):
    d, tol = cfg['decision'], cfg['tolerance']
    resolved = {r['task_id'] for r in refs if r['resolved']}
    K = k_table([c for c in cells if c['task_id'] in resolved], cfg, cfg['K'])
    accepted = [r for r in refs if r['merge_reference_accepted']]
    agree = sum(r['agrees'] for r in accepted)
    report = dict(references=dict(tasks=len(refs), resolved=len(resolved), merge_accepted=len(accepted),
                  agree=agree, agree_fraction=agree/len(accepted) if accepted else None,
                  max_dP_relative_median=med([r['max_dP_relative'] for r in accepted]),
                  max_dF_median=med([r['max_dF'] for r in accepted]),
                  consistent=bool(accepted) and agree/len(accepted) >= d['agree_fraction']), by_n={})
    hard_meds = {}
    for n in sorted({r['n'] for r in refs}):
        rows = {}
        for arm in ('M0', 'M_PTA', 'M_PTA__mul'):
            fam = defaultdict(list)
            for (tid, a), v in K.items():
                if a == arm and v['n'] == n:
                    fam[v['family']].append(v['QoI'])
            rows[arm] = {f: med(v) for f, v in sorted(fam.items())}
            rows[arm]['hard'] = med([x for f in cfg['hard_families'] for x in fam.get(f, [])])
        # Plain Monte Carlo cost to reach the same 1% P accuracy (2 sigma), in single-state rounds,
        # against merge's physical calls at its own required K.
        mc_cost, merge_cost = [], []
        for r in refs:
            if r['n'] != n or r['family'] not in cfg['hard_families'] or r['task_id'] not in resolved:
                continue
            need = r['paths']*(r['max_se_P_relative']/(tol['P_relative']/2))**2
            mc_cost.append(need*r['H'])
            v = K.get((r['task_id'], 'M0'))
            if v and v['QoI'] in v['calls'] and v['calls'][v['QoI']]:
                merge_cost.append(v['calls'][v['QoI']])
        report['by_n'][str(n)] = dict(K_median_by_family=rows, hard_tasks_resolved=sum(
            1 for r in refs if r['n'] == n and r['family'] in cfg['hard_families'] and r['task_id'] in resolved),
            mc_rounds_for_1pct_median=med(mc_cost), merge_rounds_at_needed_K_median=med(merge_cost))
        hard_meds[n] = rows['M0']['hard']
    vals = [v for v in hard_meds.values() if v is not None]
    if vals and max(vals) >= d['room_K']:
        action = 'ROOM'
    elif vals and max(vals) <= d['no_room_K']:
        action = 'STOP_NO_ROOM'
    else:
        action = 'MARGINAL'
    report.update(merge_hard_K_median=hard_meds, action=action,
                  rule='ROOM if any n has hard-family median merge QoI K >= room_K; STOP if all <= no_room_K')
    return report


def goal_grid(cfg, n):
    """Step-2 K grid; plan B (user, 2026-10-22) caps n=4 at 256. A smoke config may set goal_K."""
    return cfg['goal_K'] if 'goal_K' in cfg else cfg['goal_K_by_n'][str(n)]


def step2(cx, cfg, tasks, refs_dir, root, models, grid=None):
    g = cfg['goal']
    cells = []
    for j, task in enumerate(tasks):
        check_deadline(cx.hard)
        ref = dict(np.load(refs_dir[task['id']]))
        states, weights, ctx = context(cx, task)
        variants = {'pta_look': LookAhead('pta', g['lengths'], g['weight']),
                    'learned_look': LookAhead('learned', g['lengths'], g['weight'], models[task['n']], task['gamma'])}
        for name, look in variants.items():
            for K in grid or goal_grid(cfg, task['n']):
                try:
                    sources = {'M_PTA': PTA()} if K <= cfg['correction_K_max'] else {}
                    sim, rho, ell, est = goal_rollout(K, task['word'], ctx, cx.c, states, weights, sources, cx.hard, look)
                    extra = dict(physical_calls=sim.cost['physical_calls'], lookahead_seconds=sim.lookahead_seconds)
                    for arm, e in est.items():
                        label = name+'_raw' if arm == 'M0' else name+'+'+arm
                        cells.append(scored(task, label, K, e, ref, cfg['tolerance'], extra))
                except Deadline:
                    raise
                except (FloatingPointError, RuntimeError) as exc:
                    for arm in [name+'_raw']+([name+'+M_PTA', name+'+M_PTA__mul'] if K <= cfg['correction_K_max'] else []):
                        cells.append(dict(task_id=task['id'], n=task['n'], family=task['family'], prep_id=task['prep_signature'],
                                          arm=arm, K=K, status='NUMERIC_FAILURE', error=repr(exc)))
        rt.write(root/'cells.json', cells)
        print('HR_STEP2', j+1, len(tasks), task['id'], flush=True)
    return cells


def decide2(cells2, cells1, refs, cfg):
    d = cfg['decision']
    hard = {r['task_id'] for r in refs if r['resolved'] and r['family'] in cfg['hard_families']}
    # Geometric merge is scored on exactly the look-ahead grid of its n (no larger-K advantage).
    base = [c for c in cells1 if c['task_id'] in hard and c['K'] in goal_grid(cfg, c['n'])]
    base = [dict(c, arm={'M0': 'geo_raw', 'M_PTA': 'geo+M_PTA', 'M_PTA__mul': 'geo+M_PTA__mul'}[c['arm']]) for c in base]
    union = sorted({k for n in (3, 4) for k in goal_grid(cfg, n)})
    K = k_table(base+[c for c in cells2 if c['task_id'] in hard], cfg, union)
    report = {}
    for n in sorted({v['n'] for v in K.values()}):
        prep = defaultdict(lambda: defaultdict(list))
        for (tid, arm), v in K.items():
            if v['n'] == n:
                prep[arm][v['prep']].append(v['QoI'])
        pm = {arm: {p: med(x) for p, x in ps.items()} for arm, ps in prep.items()}
        res = {}
        for read in ('raw', '+M_PTA', '+M_PTA__mul'):
            geo, lrn, pta = ('geo'+('_raw' if read == 'raw' else read)), ('learned_look'+('_raw' if read == 'raw' else read)), ('pta_look'+('_raw' if read == 'raw' else read))
            common = sorted(set(pm.get(geo, {})) & set(pm.get(lrn, {})) & set(pm.get(pta, {})))
            if not common:
                continue
            ratio = float(math.exp(np.mean([math.log(pm[geo][p]/pm[lrn][p]) for p in common])))
            vs_pta = float(math.exp(np.mean([math.log(pm[pta][p]/pm[lrn][p]) for p in common])))
            ok = sum(pm[lrn][p] <= pm[geo][p] for p in common)
            res[read] = dict(preparations=len(common), K_ratio_geo_over_learned=ratio, K_ratio_ptalook_over_learned=vs_pta,
                             not_worse=ok, medians={a: med(list(pm[a].values())) for a in (geo, pta, lrn)},
                             passes=ratio >= d['goal_ratio'] and ok/len(common) >= d['preparation_fraction']-1e-12 and vs_pta > 1)
        report[str(n)] = res
    action = 'PASS' if any(r.get('passes') for v in report.values() for r in v.values()) else 'FAIL'
    return dict(by_n=report, action=action, hard_tasks=len(hard))

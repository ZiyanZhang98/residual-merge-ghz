"""Analysis of the n=3 second setting (HANDOFF 5j item 22.38), written before any n=3 evaluation result exists; it
runs once. The tables, readings, bootstrap and default-alpha counts are the n=4 test's (rm_test_report.summarize) on
the resolved n=3 monitor tasks. Added: label noise statistics of the n=3 labels next to the n=4 ones (M1 report
28734447), reproduction of the step-1 F2/M0/F2mul cells at the shared K, truth resolvability, plain Monte Carlo cost,
run cost, and the n=4 test numbers for the registered prediction (L/F2 on the hard families smaller at n=3).
CPU, read-only. Final data are never accessed.
"""
import rm_runtime as rt
from collections import defaultdict
import numpy as np
from rm_deploy import task_list
from rm_eval2 import verified_shard
from rm_n3 import config3, N3_KEYS, N3_FILES
from rm_test import grid
from rm_test_report import summarize, groups

ARM_OF = {'F2': 'M_PTA', 'M0': 'M0', 'F2mul': 'M_PTA__mul'}
FAMILIES = ('A', 'B', 'AB', 'AAB', 'AABB', 'ABB')


def noise_table(report):
    """Per family: share of pairs whose residual is detected above the Monte Carlo noise; median noise share of the
    long blocks (fit pairs)."""
    out = {}
    for f in FAMILIES:
        lr = report['long_range'].get('fit_'+f, {})
        out[f] = dict(detected_fraction=report['by_family'][f].get('detected_fraction_psi'),
                      noise_share={b: lr[b]['noise_share_median'] for b in ('4-7', '8-15', '16-40') if b in lr})
    return out


def analyze(cfg, train_job, eval_job, inputs):
    nc = cfg['n3']
    tasks, npz, step1 = task_list(config3(cfg), inputs)
    troot = rt.HERE/'results'/('n3train_%s' % train_job)
    verified_shard(troot, 'N3_TRAIN_COMPLETE', N3_KEYS, N3_FILES)
    inputs[troot.name] = rt.sha(troot/'manifest.json')
    roots = sorted((rt.HERE/'results').glob('n3eval_%s_*' % eval_job))
    if len(roots) != nc['eval_shards']:
        raise ValueError('Expected %d n=3 evaluation shards, found %d' % (nc['eval_shards'], len(roots)))
    cells, runs = [], []
    for r in roots:
        verified_shard(r, 'N3_EVAL_SHARD_COMPLETE', N3_KEYS, N3_FILES)
        if rt.read(r/'unit.json')['train_job'] != train_job:
            raise ValueError('Shard evaluated another model: '+str(r))
        inputs[r.name] = rt.sha(r/'manifest.json')
        cells += rt.read(r/'cells.json')
        runs += rt.read(r/'runs.json')
    gr = grid(cfg)
    expect = dict(L=len(gr['L']), C1=len(gr['C1']), F2=len(gr['F2']), M0=len(gr['F2']), F2mul=len(gr['F2']))
    have = defaultdict(int)
    for c in cells:
        have[(c['method'], c['task_id'])] += 1
    out = summarize(cfg, tasks, cells)
    out['incomplete_grids'] = [[m, t['id'], have[(m, t['id'])]] for t in tasks for m in expect if have[(m, t['id'])] != expect[m]]
    out['failures'] = [x for x in runs if x['status'] != 'OK']
    # the F2 rollouts reproduce hard_regime step 1 wherever the K grids overlap
    old = {(c['task_id'], c['arm'], c['K']): c for c in step1}
    checked, bad = 0, []
    for c in cells:
        s = old.get((c['task_id'], ARM_OF.get(c.get('method')), c.get('K')))
        if s is None:
            continue
        checked += 1
        q = [c.get('Q_QoI'), s.get('Q_QoI')]
        same = q[0] == q[1] if None in q else abs(q[0]-q[1]) <= 1e-9
        if c['physical_calls'] != s['physical_calls'] or not same:
            bad.append([c['task_id'], c['method'], c['K'], c['physical_calls'], s['physical_calls'], q])
    out['step1_reproduction'] = dict(cells=checked, mismatches=len(bad), examples=bad[:20])
    # truth (all n=3 monitor tasks of step 1) and plain Monte Carlo at 1 % P accuracy (2 sigma), single-state rounds
    refs, _, _ = rt.references(inputs, cfg['hard_regime']['step1_job'])
    mine = [x for x in refs if x['n'] == nc['n']]
    fam = defaultdict(list)
    for x in mine:
        fam[x['task_id'].split('_')[-1]].append(x)
    out['truth'] = {f: dict(tasks=len(v), resolved=sum(x['resolved'] for x in v),
                            unresolved=[x['task_id'] for x in v if not x['resolved']],
                            median_paths=float(np.median([x['paths'] for x in v]))) for f, v in sorted(fam.items())}
    tol = cfg['tolerance']['P_relative']
    mcc = {x['task_id']: x['paths']*(x['max_se_P_relative']/(tol/2))**2*x['H'] for x in mine if x['resolved']}
    Lneed = out['tables']['Q_QoI']['needed']['L']
    out['plain_mc'] = {g: dict(median_rounds=float(np.median([mcc[t['id']] for t in tasks if t['family'] in fams])),
                               median_L_atoms=float(np.median([Lneed[t['id']]['atoms'] for t in tasks if t['family'] in fams])))
                       for g, fams in groups().items() if any(t['family'] in fams for t in tasks)}
    cost = defaultdict(list)
    for x in runs:
        if x['status'] == 'OK':
            cost[x['method']].append(x)
    out['run_cost'] = {m: dict(runs=len(v), median_seconds=float(np.median([x['seconds'] for x in v])),
                               median_merge_seconds=float(np.median([x.get('merge_seconds', 0.) for x in v])),
                               total_seconds=sum(x['seconds'] for x in v)) for m, v in cost.items()}
    # labels: n=3 against the n=4 M1 labels (same generator, M1 rounds)
    rep3 = rt.read(troot/'label_report.json')['n3']
    r4 = rt.HERE/'results'/('labels_report_%s' % nc['n4_label_report_job'])
    inputs[r4.name] = rt.sha(r4/'manifest.json')
    out['labels'] = dict(n3=noise_table(rep3), n4=noise_table(rt.read(r4/'report.json')),
                         label_eval=rt.read(troot/'label_eval.json'))
    # registered prediction: the hard-family gain over F2 is smaller at n=3 than on the n=4 test
    t4 = rt.read(rt.HERE/'results'/('testa_%s' % nc['n4_test_analysis_job'])/'analysis.json')
    inputs['testa_%s' % nc['n4_test_analysis_job']] = rt.sha(rt.HERE/'results'/('testa_%s' % nc['n4_test_analysis_job'])/'manifest.json')
    h3 = out['tables']['Q_QoI']['groups']['hard']['L_vs_F2']['ratio']
    h4 = t4['tables']['Q_QoI']['groups']['hard']['L_vs_F2']['ratio']
    out['prediction'] = dict(n3_hard_L_vs_F2=h3, n4_test_hard_L_vs_F2=h4, smaller_at_n3=h3 < h4,
                             n4_test={g: {k: t4['tables']['Q_QoI']['groups'][g][k]['ratio'] for k in ('L_vs_F2', 'L_vs_C1', 'C1_vs_F2')}
                                      for g in ('hard', 'easy')})
    out['tasks_evaluated'] = len(tasks)
    return out

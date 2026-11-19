#!/usr/bin/env python3
"""HANDOFF 22.17 entry: smoke | step1 | decide1 | step2 | decide2. Compute nodes only."""
import hr_runtime as rt
import argparse
import copy
import time
from pathlib import Path


def parts(prefix, job):
    name = '%s_%s' % (prefix, job)
    # Array shards are <prefix>_<job>_<index>; single CPU jobs are <prefix>_<job>.
    found = sorted(p for p in (rt.HERE/'results').iterdir() if p.name == name or p.name.startswith(name+'_'))
    if not found:
        raise ValueError('No %s artifacts for job %s' % (prefix, job))
    return found


def collect_step1(job, inputs):
    cells, refs, npz = [], [], {}
    for r in parts('step1', job):
        rt.artifact(r, ['STEP1_SHARD_COMPLETE']); inputs[r.name] = rt.sha(r/'manifest.json')
        cells += rt.read(r/'cells.json'); refs += rt.read(r/'mc.json')
        npz.update({x['task_id']: str(r/('mc_%s.npz' % x['task_id'])) for x in rt.read(r/'mc.json')})
    return cells, refs, npz


def load_models(cfg, device, inputs):
    from qb_model import load
    models = {}
    for n, rel in cfg['learned'].items():
        path = (rt.HERE/rel).resolve()
        inputs['model_n'+n] = rt.sha(path)
        models[int(n)] = load(path, rt.qb.config()['train'], device)[0]
    return models


def smoke(cx, cfg, root, inputs):
    """Reduced chain on two tasks; also checks that GoalSim without look-ahead is the frozen M0."""
    import numpy as np
    import torch
    import qa_sim
    from hr_eval import load_tasks, step1, decide1, step2, decide2, context
    from hr_goal import rollout as goal_rollout
    sc = copy.deepcopy(cfg)
    sc['mc'].update(initial_paths=cfg['smoke']['initial_paths'], max_paths=cfg['smoke']['max_paths'])
    tasks = load_tasks(cfg)
    pick = [next(t for t in tasks if t['n'] == 3 and t['family'] == 'AB' and t['accepted']),
            next(t for t in tasks if t['n'] == 4 and t['family'] == 'ABB')][:cfg['smoke']['tasks']]
    states, weights, ctx = context(cx, pick[0])
    for K in cfg['smoke']['K']:
        _, r1, l1, e1 = qa_sim.rollout(K, pick[0]['word'], ctx, cx.c, states, weights, None, cx.hard)
        _, r2, l2, e2 = goal_rollout(K, pick[0]['word'], ctx, cx.c, states, weights, None, cx.hard)
        if not torch.equal(r1, r2) or l1 != l2 or not torch.equal(e1['M0']['P'], e2['M0']['P']):
            raise AssertionError('GoalSim without look-ahead differs from frozen M0')
    d1, d2 = root/'step1', root/'step2'
    d1.mkdir(); d2.mkdir()
    cells1, refs = step1(cx, sc, pick, d1, grid=cfg['smoke']['K'], mc=sc['mc'])
    sc['K'] = cfg['smoke']['K']; sc['goal_K'] = cfg['smoke']['goal_K']
    dec1 = decide1(cells1, refs, sc)
    for r in refs:
        r['resolved'] = True  # smoke only: exercise step 2 regardless of the reduced path budget
    npz = {r['task_id']: str(d1/('mc_%s.npz' % r['task_id'])) for r in refs}
    hard = [t for t in pick if t['family'] in cfg['hard_families']]
    cells2 = step2(cx, sc, hard, npz, d2, load_models(cfg, cx.device, inputs), grid=cfg['smoke']['goal_K'])
    rt.write(root/'smoke.json', dict(decision1=dec1, decision2=decide2(cells2, cells1, refs, sc),
                                     references=refs, bitwise_M0=True, final_accessed=False))
    return 'SMOKE_COMPLETE'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['smoke', 'step1', 'decide1', 'step2', 'decide2'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--minutes', type=float, required=True)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--step1-job'); p.add_argument('--decide1-job'); p.add_argument('--step2-job')
    a = p.parse_args()
    cfg = rt.config(); root = rt.new_root(a.out)
    start = time.perf_counter()
    hard = time.time()+a.minutes*60-cfg['resource']['reserve_seconds']
    cx = rt.context(root, hard, a.device)
    inputs, initial = {}, rt.manifest(root, a.mode, 'RUNNING')
    rt.write(root/'manifest.json', initial)
    try:
        if a.mode == 'smoke':
            status = smoke(cx, cfg, root, inputs)
        elif a.mode == 'step1':
            from hr_eval import load_tasks, shard, step1
            tasks = load_tasks(cfg)
            mine = shard(tasks, a.index, a.shards)
            rt.write(root/'unit.json', dict(index=a.index, shards=a.shards, all=[t['id'] for t in tasks],
                                            mine=[t['id'] for t in mine]))
            step1(cx, cfg, mine, root)
            status = 'STEP1_SHARD_COMPLETE'
        elif a.mode == 'decide1':
            from hr_eval import load_tasks, decide1
            cells, refs, _ = collect_step1(a.step1_job, inputs)
            want = sorted(t['id'] for t in load_tasks(cfg))
            if sorted(r['task_id'] for r in refs) != want:
                raise ValueError('Step-1 shards do not cover every monitor task exactly once')
            dec = decide1(cells, refs, cfg)
            rt.write(root/'decision.json', dec)
            print('HR_DECIDE1', dec['action'], dec['merge_hard_K_median'], dec['references'], flush=True)
            status = 'DECIDE1_COMPLETE'
        elif a.mode == 'step2':
            from hr_eval import load_tasks, shard, step2
            dec = rt.read(parts('decide1', a.decide1_job)[0]/'decision.json')
            inputs['decide1'] = a.decide1_job
            if dec['action'] == 'STOP_NO_ROOM':
                rt.write(root/'skipped.json', dict(reason='Step 1 found no room for merge improvement', decision=dec['action']))
                status = 'STEP2_SKIPPED'
            else:
                _, refs, npz = collect_step1(a.step1_job, inputs)
                resolved = {r['task_id'] for r in refs if r['resolved']}
                hard_tasks = [t for t in load_tasks(cfg) if t['family'] in cfg['hard_families'] and t['id'] in resolved]
                mine = shard(hard_tasks, a.index, a.shards)
                rt.write(root/'unit.json', dict(index=a.index, shards=a.shards, all=[t['id'] for t in hard_tasks],
                                                mine=[t['id'] for t in mine]))
                step2(cx, cfg, mine, npz, root, load_models(cfg, cx.device, inputs))
                status = 'STEP2_SHARD_COMPLETE'
        else:
            from hr_eval import decide2
            cells1, refs, _ = collect_step1(a.step1_job, inputs)
            cells2, skipped = [], 0
            for r in parts('step2', a.step2_job):
                m = rt.artifact(r, ['STEP2_SHARD_COMPLETE', 'STEP2_SKIPPED']); inputs[r.name] = rt.sha(r/'manifest.json')
                if m['status'] == 'STEP2_SKIPPED':
                    skipped += 1
                else:
                    cells2 += rt.read(r/'cells.json')
            dec = dict(action='SKIPPED', reason='step 1 found no room') if skipped else decide2(cells2, cells1, refs, cfg)
            rt.write(root/'decision.json', dec)
            print('HR_DECIDE2', dec['action'], dec.get('by_n'), flush=True)
            status = 'DECIDE2_COMPLETE'
        m = rt.manifest(root, a.mode, status, inputs)
        if m['sources'] != initial['sources']:
            raise RuntimeError('Executable sources changed during this job')
        m['seconds'] = time.perf_counter()-start
        rt.write(root/'manifest.json', m)
    except Exception as exc:
        m = rt.manifest(root, a.mode, 'FAILED', inputs)
        m.update(error=repr(exc), seconds=time.perf_counter()-start)
        rt.write(root/'manifest.json', m)
        raise


if __name__ == '__main__':
    main()

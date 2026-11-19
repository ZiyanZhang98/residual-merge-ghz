#!/usr/bin/env python3
"""HANDOFF 22.29 step 1 entry: smoke, training (1B, then 1A only if needed), deployment batches and analysis.
Compute nodes only (rm_runtime refuses anything else); submit the sbatch scripts from the repository root."""
import rm_runtime as rt
import argparse
import math
import time
from pathlib import Path
import numpy as np
import torch


def clean(x, count):
    """JSON forbids NaN/inf (written as null and counted) and numpy/torch types (same as run.py; 'run' is ambiguous
    on the frozen import path)."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().tolist()
    elif isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if not math.isfinite(x):
            count[0] += 1
            return None
        return x
    if isinstance(x, dict):
        return {str(k): clean(v, count) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v, count) for v in x]
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['lik_smoke', 'lik_train', 'eval2_deploy', 'eval2_analyze', 'sgn_smoke', 'sgn_train', 'curv_a', 'curv_b', 'curv2_deploy', 'curv2_analyze', 'ext_deploy', 'ext_analyze',
                                    'test_smoke', 'test_pool', 'test_run', 'test_analyze',
                                    'n3_smoke', 'n3_labels', 'n3_train', 'n3_eval', 'n3_analyze', 'abl', 'figs', 'pta_sim'])
    p.add_argument('--pool-job')
    p.add_argument('--job')
    p.add_argument('--step', type=int, default=1)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--minutes', type=float, required=True)
    p.add_argument('--train-job')
    p.add_argument('--batch', choices=['B', 'hard'])
    p.add_argument('--b-job')
    p.add_argument('--hard-job')
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    a = p.parse_args()
    cfg = rt.config()
    root = rt.new_root(a.out)
    start = time.perf_counter()
    hard = time.time()+a.minutes*60-cfg['resource']['reserve_seconds']
    cx = rt.context(root, hard, a.device)
    inputs, initial, nonfinite = {}, rt.manifest(root, a.mode, 'RUNNING'), [0]
    rt.write(root/'manifest.json', initial)
    save = lambda name, value: rt.write(root/name, clean(value, nonfinite))  # noqa: E731

    def finish(status, **extra):
        m = rt.manifest(root, a.mode, status, inputs)
        if m['sources'] != initial['sources']:
            raise RuntimeError('Executable sources changed during this job')
        m.update(seconds=time.perf_counter()-start, nonfinite_values_written_as_null=nonfinite[0], **extra)
        rt.write(root/'manifest.json', m)
        print('RM2_STATUS', status, '%.0fs' % m['seconds'], flush=True)

    try:
        if a.mode == 'lik_smoke':
            from rm_lik import smoke_lik
            from rm_eval2 import smoke_deploy
            checks = dict(lik=smoke_lik(cx, cfg, inputs), deploy=smoke_deploy(cx, cfg, inputs))
            failed = [f'{g}.{k}' for g, part in checks.items() for k, v in part.items() if isinstance(v, dict) and not v.get('passed', True)]
            save('checks.json', dict(checks, failed=failed))
            for g, part in checks.items():
                for k, v in part.items():
                    print('RM2_SMOKE %s.%s %s' % (g, k, v), flush=True)
            finish('LIK_SMOKE_COMPLETE' if not failed else 'LIK_SMOKE_FAILED', failed_checks=failed)
            if failed:
                raise SystemExit(1)
            return
        if a.mode == 'lik_train':
            from rm_lik import run_training
            decision = run_training(cx, cfg, root, inputs, save)
            print('RM2_DECISION', decision, flush=True)
            return finish('LIK_COMPLETE')
        if a.mode == 'sgn_smoke':
            from rm_signed import smoke
            checks = smoke(cx, cfg, inputs)
            failed = [k for k, v in checks.items() if isinstance(v, dict) and not v.get('passed', True)]
            save('checks.json', dict(checks, failed=failed))
            for k, v in checks.items():
                print('RM2_SMOKE %s %s' % (k, v), flush=True)
            finish('SGN_SMOKE_COMPLETE' if not failed else 'SGN_SMOKE_FAILED', failed_checks=failed)
            if failed:
                raise SystemExit(1)
            return
        if a.mode == 'sgn_train':
            from rm_signed import run_training as run_sgn
            decision = run_sgn(cx, cfg, root, inputs, save)
            print('RM2_DECISION', decision, flush=True)
            return finish('SGN_COMPLETE')
        if a.mode == 'ext_deploy':
            from rm_ext import deploy_ext
            save('summary.json', deploy_ext(cx, cfg, a.index, a.shards, inputs, save))
            return finish('EXT_SHARD_COMPLETE')
        if a.mode == 'ext_analyze':
            from rm_ext import analyze_ext
            out = analyze_ext(cfg, a.job, inputs)
            save('analysis.json', out)
            print('RM2_EXT m3_rule', out['m3_rule'], 'reproduction', out['reproduction'], 'failures', len(out['failures']), flush=True)
            for g in ('hard', 'easy', 'A', 'B', 'AB', 'AAB', 'AABB', 'ABB'):
                e, m = out['extended']['Q_QoI']['groups'][g], out['m3_grid']['Q_QoI']['groups'][g]
                print('RM2_EXT %-5s L/F2 ext %.2f (%d/%d) m3 %.2f | L/C1 ext %.2f m3 %.2f' % (
                    g, e['L_vs_F2']['ratio'], e['L_vs_F2']['not_worse'], e['L_vs_F2']['preparations'], m['L_vs_F2']['ratio'],
                    e['L_vs_C1']['ratio'], m['L_vs_C1']['ratio']), flush=True)
            return finish('EXT_ANALYSIS_COMPLETE')
        if a.mode == 'test_smoke':
            from rm_test import smoke as test_smoke
            checks = test_smoke(cx, cfg, inputs)
            failed = [k for k, v in checks.items() if isinstance(v, dict) and not v.get('passed', True)]
            save('checks.json', dict(checks, failed=failed))
            for k, v in checks.items():
                print('RM2_TEST_SMOKE %s %s' % (k, v), flush=True)
            finish('TEST_SMOKE_COMPLETE' if not failed else 'TEST_SMOKE_FAILED', failed_checks=failed)
            if failed:
                raise SystemExit(1)
            return
        if a.mode == 'test_pool':
            from rm_test import write_pool
            iso, tasks = write_pool(cx, cfg, save)
            print('RM2_TEST_POOL tasks', len(tasks), 'isolation', iso, flush=True)
            finish('TEST_POOL_COMPLETE' if iso['passed'] else 'TEST_POOL_FAILED')
            if not iso['passed']:
                raise SystemExit(1)
            return
        if a.mode == 'test_run':
            from rm_test import run_shard
            save('summary.json', run_shard(cx, cfg, a.pool_job, a.index, a.shards, root, inputs, save))
            return finish('TEST_SHARD_COMPLETE')
        if a.mode == 'test_analyze':
            from rm_test_report import analyze as test_analyze
            out = test_analyze(cfg, a.pool_job, a.job, inputs)
            save('analysis.json', out)
            print('RM2_TEST reading', out['reading'], flush=True)
            print('RM2_TEST easy', out['easy_not_worse'], flush=True)
            print('RM2_TEST default_alpha', {g: out['default_alpha'][g] for g in ('hard', 'easy')}, flush=True)
            print('RM2_TEST incomplete', len(out['incomplete_grids']), 'failures', len(out['failures']), flush=True)
            return finish('TEST_ANALYSIS_COMPLETE')
        if a.mode == 'abl':
            from rm_abl import run_ablation
            out = run_ablation(cx, cfg, root, inputs, save)
            save('ablation.json', out)
            for k, v in out['reading'].items():
                print('RM2_ABL reading', k, v, flush=True)
            print('RM2_ABL checks', out['checks'], flush=True)
            return finish('ABL_COMPLETE')
        if a.mode == 'figs':
            from rm_figs import run_figs
            out = run_figs(cfg, root, inputs, save)
            save('figs_data.json', out)
            for s in ('test', 'n3', 'dev'):
                print('RM2_FIGS', s, 'crosscheck', out[s]['crosscheck']['passed'], out[s]['crosscheck']['tasks'], flush=True)
                for k, v in out[s]['breakeven'].items():
                    print('RM2_FIGS', s, k, {g: v['groups'][g] for g in ('hard', 'easy') if g in v['groups']}, flush=True)
            print('RM2_FIGS figures', out['figures'], flush=True)
            return finish('FIGS_COMPLETE')
        if a.mode == 'pta_sim':
            from rm_pta import run_pta
            out = run_pta(cx, cfg, inputs, save)
            save('pta_sim.json', out)
            print('RM2_PTA reproduction', out['reproduction'], 'failures', {k: len(v) for k, v in out['failures'].items()}, flush=True)
            for s, v in out['summary'].items():
                for g, x in v.items():
                    print('RM2_PTA', s, g, x, flush=True)
            return finish('PTA_SIM_COMPLETE')
        if a.mode == 'n3_smoke':
            from rm_n3 import smoke as n3_smoke
            checks = n3_smoke(cx, cfg, root, inputs, lambda path, v: rt.write(path, clean(v, nonfinite)))
            failed = [k for k, v in checks.items() if isinstance(v, dict) and not v.get('passed', True)]
            save('checks.json', dict(checks, failed=failed))
            for k, v in checks.items():
                print('RM2_N3_SMOKE %s %s' % (k, v), flush=True)
            finish('N3_SMOKE_COMPLETE' if not failed else 'N3_SMOKE_FAILED', failed_checks=failed)
            if failed:
                raise SystemExit(1)
            return
        if a.mode == 'n3_labels':
            from rm_n3 import label_shard
            import os
            tag = '%s_%d' % (os.environ.get('SLURM_ARRAY_JOB_ID') or os.environ.get('SLURM_JOB_ID', 'local'), a.index)
            save('summary.json', label_shard(cx, cfg, a.index, a.shards, tag, initial['sources'],
                                             lambda path, v: rt.write(path, clean(v, nonfinite))))
            return finish('N3_LABELS_COMPLETE')
        if a.mode == 'n3_train':
            from rm_n3 import train3
            save('summary.json', train3(cx, cfg, a.job, root, inputs, save))
            return finish('N3_TRAIN_COMPLETE')
        if a.mode == 'n3_eval':
            from rm_n3 import eval_shard
            save('summary.json', eval_shard(cx, cfg, a.train_job, a.index, a.shards, inputs, save))
            return finish('N3_EVAL_SHARD_COMPLETE')
        if a.mode == 'n3_analyze':
            from rm_n3_report import analyze as n3_analyze
            out = n3_analyze(cfg, a.train_job, a.job, inputs)
            save('analysis.json', out)
            print('RM2_N3 reading', out['reading'], flush=True)
            print('RM2_N3 easy', out['easy_not_worse'], flush=True)
            print('RM2_N3 prediction', {k: out['prediction'][k] for k in ('n3_hard_L_vs_F2', 'n4_test_hard_L_vs_F2', 'smaller_at_n3')}, flush=True)
            print('RM2_N3 default_alpha', {g: out['default_alpha'][g] for g in ('hard', 'easy')}, flush=True)
            print('RM2_N3 incomplete', len(out['incomplete_grids']), 'failures', len(out['failures']),
                  'step1_reproduction', {k: out['step1_reproduction'][k] for k in ('cells', 'mismatches')}, flush=True)
            return finish('N3_ANALYSIS_COMPLETE')
        if a.mode == 'curv2_deploy':
            from rm_curv import deploy_metric
            save('summary.json', deploy_metric(cx, cfg, a.index, a.shards, inputs, save))
            return finish('CURV2_SHARD_COMPLETE')
        if a.mode == 'curv2_analyze':
            from rm_curv import analyze_metric
            out = analyze_metric(cfg, a.job, inputs)
            save('analysis.json', out)
            print('RM2_CURV2', out['reading'], out['regressed_groups'], out['new_censoring_outside_ABB'], flush=True)
            print('RM2_CURV2_M3RULE', {k: v for k, v in out['m3_rule_for_Lm'].items()}, flush=True)
            return finish('CURV2_ANALYSIS_COMPLETE')
        if a.mode == 'curv_a':
            from rm_curv import candidates
            out = candidates(cx, cfg, inputs, save)
            save('curv_a.json', out)
            print('RM2_CURV_A', out['reading'], 'repro', out['repro'], flush=True)
            return finish('CURV_A_COMPLETE')
        if a.mode == 'curv_b':
            from rm_curv import label_curvature
            out = label_curvature(cx, cfg, inputs, save, root)
            save('curv_b.json', out)
            print('RM2_CURV_B', out['reading'], out['checks'], flush=True)
            return finish('CURV_B_COMPLETE')
        if a.mode == 'eval2_deploy':
            from rm_eval2 import deploy
            out = deploy(cx, cfg, a.batch, a.index, a.shards, a.train_job, inputs, save, a.step)
            save('summary.json', out)
            return finish('EVAL2_SHARD_COMPLETE')
        from rm_eval2 import analyze
        out = analyze(cfg, {'B': a.b_job, 'hard': a.hard_job}, inputs)
        save('analysis.json', out)
        for part in ('B', 'hard'):
            if part in out:
                print('RM2_READING', part, out[part]['reading'], flush=True)
        print('RM2_C1_CONSISTENCY', out['c1_consistency'], flush=True)
        return finish('EVAL2_ANALYSIS_COMPLETE')
    except SystemExit:
        raise
    except Exception as exc:
        m = rt.manifest(root, a.mode, 'FAILED', inputs)
        m.update(error=repr(exc), seconds=time.perf_counter()-start)
        rt.write(root/'manifest.json', m)
        raise


if __name__ == '__main__':
    torch.set_grad_enabled(True)
    main()

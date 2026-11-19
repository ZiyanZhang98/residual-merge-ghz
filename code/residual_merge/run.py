#!/usr/bin/env python3
"""HANDOFF 22.28 entry (M0): smoke (GPU). Compute nodes only; submit run.sbatch from the repository root."""
import rm_runtime as rt
import argparse
import math
import time
from pathlib import Path
import numpy as np
import torch


def clean(x, count):
    """JSON forbids NaN/inf (written as null and counted) and numpy/torch types."""
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
    p.add_argument('mode', choices=['smoke', 'labels', 'labels_early', 'labels_report', 'train', 'deploy', 'analyze', 'diagnose'])
    p.add_argument('--labels-job')
    p.add_argument('--train-job')
    p.add_argument('--deploy-job')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--minutes', type=float, required=True)
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

    def finish(status):
        m = rt.manifest(root, a.mode, status, inputs)
        if m['sources'] != initial['sources']:
            raise RuntimeError('Executable sources changed during this job')
        m.update(seconds=time.perf_counter()-start, nonfinite_values_written_as_null=nonfinite[0])
        rt.write(root/'manifest.json', m)
        print('RM_STATUS', status, '%.0fs' % m['seconds'], flush=True)

    try:
        if a.mode == 'train':
            from rm_report import load
            from rm_model import train, evaluate
            from rm_deploy import profile
            parts = [load(j, inputs) for j in (cfg['m2'].get('labels_jobs') or [cfg['m2']['labels_job']])]
            for i, part in enumerate(parts):
                part['event'] = part['event']+100_000_000*i   # distinct capture ids across label jobs
            d = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
            gen = torch.Generator().manual_seed(int(cfg['m2']['seed']))
            model, log = train(d, cfg, cx.device, gen)
            tc = cfg['m2']
            torch.save(dict(din=model.mu.numel(), width=tc['width'], depth=tc['depth'], state=model.state_dict()),
                       root/'model.pt')
            save('train_log.json', log)
            ev = evaluate(model, d, cx.device, cfg)
            save('label_eval.json', ev)
            for key in ('fit_all', 'monitor_all'):
                print('RM_EVAL', key, ev[key]['all_blocks'], ev[key]['long_blocks'], flush=True)
            for f in range(6):
                print('RM_EVAL monitor_fam%d' % f, ev['monitor_fam%d' % f]['long_blocks'], flush=True)
            save('profile.json', profile(cx, cfg, model, inputs))
            return finish('TRAIN_COMPLETE')
        if a.mode == 'deploy':
            from rm_model import load_model
            from rm_deploy import deploy
            tr = rt.HERE/'results'/('train_%s' % a.train_job)
            __import__('rm_verify').verified_model(tr)
            inputs[tr.name] = rt.sha(tr/'manifest.json')
            out = deploy(cx, cfg, a.index, a.shards, load_model(tr/'model.pt', cx.device), inputs, save)
            save('summary.json', out)
            return finish('DEPLOY_SHARD_COMPLETE')
        if a.mode == 'diagnose':
            from rm_model import load_model
            from rm_deploy import diagnose
            tr = rt.HERE/'results'/('train_%s' % a.train_job)
            __import__('rm_verify').verified_model(tr)
            inputs[tr.name] = rt.sha(tr/'manifest.json')
            save('diagnose.json', diagnose(cx, cfg, load_model(tr/'model.pt', cx.device), inputs))
            return finish('DIAGNOSE_COMPLETE')
        if a.mode == 'analyze':
            from rm_deploy import analyze
            out = analyze(cfg, a.deploy_job, inputs)
            save('analysis.json', out)
            print('RM_DECISION', out['decision'], flush=True)
            for name in ('hard', 'easy'):
                print('RM_COMPARE', name, out[name], flush=True)
            for fam, v in out['by_family'].items():
                print('RM_FAMILY', fam, v, flush=True)
            return finish('ANALYSIS_COMPLETE')
        if a.mode in ('labels', 'labels_early'):
            from rm_labels import generate
            rt.write(root/'unit.json', dict(index=a.index, shards=a.shards, mode=a.mode))
            # labels_early: the same generator at the first compressions (cfg m1_early), which M1 did not sample.
            lcfg = cfg if a.mode == 'labels' else dict(cfg, m1=dict(cfg['m1'], events=cfg['m1_early']['events']))
            out = generate(cx, lcfg, a.index, a.shards, root, save)
            save('summary.json', out)
            print('RM_LABELS_DONE', out, flush=True)
            m = rt.manifest(root, a.mode, 'LABELS_SHARD_COMPLETE', inputs)
            if m['sources'] != initial['sources']:
                raise RuntimeError('Executable sources changed during this job')
            m.update(seconds=time.perf_counter()-start, nonfinite_values_written_as_null=nonfinite[0])
            rt.write(root/'manifest.json', m)
            return
        if a.mode == 'labels_report':
            from rm_report import report
            out = report(a.labels_job, inputs)
            save('report.json', out)
            for key, s in list(out['by'].items())+list(out['by_family'].items()):
                print('RM_REPORT %-14s pairs=%-6d detected=%s exact=%s' % (key, s['pairs'], s.get('detected_fraction_psi'),
                                                                           s.get('exact_prefixes')), flush=True)
                for b, v in s.get('blocks', {}).items():
                    print('    %-6s rms residual %.2e  pta gap %.2e  ratio %s  noise %.2e' % (
                        b, v['rms_rel_residual_psi'], v['rms_rel_pta_gap_psi'], v['ratio_psi'], v['noise_rms_psi']), flush=True)
                if 'spearman_prefix3' in s:
                    print('    spearman(prefix 3)', s['spearman_prefix3'], flush=True)
            for key, blocks in out['long_range'].items():
                for b, v in blocks.items():
                    print('RM_LONG %-8s %-6s pairs=%-6d noise_share=%.2f  ward rho median=%s mean=%s | pta-gap rho median=%s' % (
                        key, b, v['pairs'], v['noise_share_median'], v['ward_vs_residual']['median'],
                        v['ward_vs_residual']['mean'], v['pta_gap_vs_residual']['median']), flush=True)
            m = rt.manifest(root, a.mode, 'LABELS_REPORT_COMPLETE', inputs)
            m.update(seconds=time.perf_counter()-start, nonfinite_values_written_as_null=nonfinite[0])
            rt.write(root/'manifest.json', m)
            return
        from rm_smoke import smoke
        result = smoke(cx, cfg, root, inputs, save)
        status = 'SMOKE_COMPLETE' if result['passed'] else 'SMOKE_CHECK_FAILED'
        m = rt.manifest(root, a.mode, status, inputs)
        if m['sources'] != initial['sources']:
            raise RuntimeError('Executable sources changed during this job')
        m.update(seconds=time.perf_counter()-start, nonfinite_values_written_as_null=nonfinite[0],
                 failed_checks=result['failed'])
        rt.write(root/'manifest.json', m)
        print('RM_STATUS', status, 'failed:', result['failed'], '%.0fs' % m['seconds'], flush=True)
        if not result['passed']:
            raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as exc:
        m = rt.manifest(root, a.mode, 'FAILED', inputs)
        m.update(error=repr(exc), seconds=time.perf_counter()-start)
        rt.write(root/'manifest.json', m)
        raise


if __name__ == '__main__':
    main()

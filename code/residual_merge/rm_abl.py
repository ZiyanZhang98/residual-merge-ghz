"""HANDOFF 22.39: label-level ablations of the frozen pairwise residual cost model L (train 28736643), written before
any ablation result exists; it runs once. Same n=4 labels (M1 28727358 + early 28735174), same recipe (m2, seed 0),
same label-level evaluation (rm_model.evaluate on the same monitor preparations) as the frozen model.

Variants:
  full_repro  the frozen recipe retrained through this code path (run-to-run reference);
  no_pta      without the PTA inputs (PTA gap of P and phi per prefix, PTA level, conditional fidelity; 4H columns);
  no_state    without the state coordinates (merged-state Pauli coordinates and squared differences; 2D columns);
              lambda(1-lambda) and log Ward stay;
  reg_only    full inputs, regression loss only (rank and calibration weights 0);
  no_early    the existing model 28734894 (trained without the round 0-2 labels), evaluated only.
Checks: the frozen model evaluated through this code path reproduces its stored label_eval.json exactly (a failure
stops the job before any training). Feature layout of rm_model.features: [m (D), d2 (D), lambda(1-lambda), log Ward,
asinh gap psi (H), asinh gap phi (H), F (H), log level (H), word A (H), word B (H), t, L]. Final data are never accessed.
"""
import rm_runtime as rt
import numpy as np
import torch
import rm_model
from rm_model import H, train, evaluate, load_model
from rm_report import load

VARIANTS = ('full_repro', 'no_pta', 'no_state', 'reg_only')


def labels(cfg, inputs):
    """The frozen model's training data, built exactly as run.py train does."""
    parts = [load(j, inputs) for j in cfg['m2']['labels_jobs']]
    for i, part in enumerate(parts):
        part['event'] = part['event']+100_000_000*i
    return {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}


def dropped(name, D):
    if name == 'no_pta':
        return list(range(2*D+2, 2*D+2+4*H))
    if name == 'no_state':
        return list(range(0, 2*D))
    return []


def plain(x):
    """The JSON form written by run2.clean (numpy to Python, non-finite floats to None), for exact comparison."""
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
        return x if np.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x


def same(a, b):
    """Exact equality of two nested JSON-like values (None for NaN as written by clean)."""
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    return a == b


def max_diff(a, b):
    """Largest absolute difference over numeric leaves; None if the structures differ."""
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            return None
        vals = [max_diff(a[k], b[k]) for k in a]
    elif isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            return None
        vals = [max_diff(x, y) for x, y in zip(a, b)]
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a)-float(b))
    else:
        return 0. if a == b else None
    return None if any(v is None for v in vals) else max(vals, default=0.)


def summary(ev):
    mon = ev['monitor_all']
    return dict(monitor_all=mon['all_blocks']['learned'], monitor_long=mon['long_blocks']['learned'],
                fit_all=ev['fit_all']['all_blocks']['learned'],
                monitor_family={f: ev['monitor_fam%d' % i]['all_blocks']['learned']
                                for i, f in enumerate(('A', 'B', 'AB', 'AAB', 'AABB', 'ABB'))},
                monitor_log_rmse=mon['log_rmse_by_output'])


def run_variant(cx, cfg, d, name):
    D = d['xa'].shape[1]
    drop = set(dropped(name, D))
    din = 2*D+6*H+4
    keep = [i for i in range(din) if i not in drop]
    c = dict(cfg, m2=dict(cfg['m2'], rank_weight=0.0, cal_weight=0.0)) if name == 'reg_only' else cfg
    orig = rm_model.label_inputs
    if drop:
        idx = torch.as_tensor(keep, device=cx.device)
        rm_model.label_inputs = lambda dd, device: orig(dd, device)[:, idx]  # noqa: E731
    try:
        gen = torch.Generator().manual_seed(int(cfg['m2']['seed']))
        model, log = train(d, c, cx.device, gen)
        ev = evaluate(model, d, cx.device, c)
    finally:
        rm_model.label_inputs = orig
    if model.mu.numel() != len(keep):
        raise RuntimeError('Input width mismatch for %s' % name)
    return model, log, ev, keep


def run_ablation(cx, cfg, root, inputs, save):
    ac = cfg['abl']
    d = labels(cfg, inputs)
    # check: the frozen model through this code path reproduces its stored label-level evaluation exactly
    fro = rt.HERE/'results'/('train_%s' % ac['frozen_train_job'])
    from rm_verify import verified_model
    verified_model(fro)
    inputs[fro.name] = rt.sha(fro/'manifest.json')
    stored = rt.read(fro/'label_eval.json')
    ev_frozen = plain(evaluate(load_model(fro/'model.pt', cx.device), d, cx.device, cfg))
    gap = max_diff(ev_frozen, stored)
    checks = dict(frozen_reproduced=dict(exact=bool(same(ev_frozen, stored)), max_abs_diff=gap,
                                         passed=bool(gap is not None and gap <= 1e-9),
                                         monitor_all=[ev_frozen['monitor_all']['all_blocks']['learned'],
                                                      stored['monitor_all']['all_blocks']['learned']]))
    print('RM2_ABL check', checks, flush=True)
    if not checks['frozen_reproduced']['passed']:
        save('checks.json', checks)
        raise RuntimeError('The frozen model does not reproduce its label-level evaluation through this code path')
    table = dict(frozen=summary(stored))
    baselines = {m: dict(all=stored['monitor_all']['all_blocks'][m], long=stored['monitor_all']['long_blocks'][m])
                 for m in ('C1_ward', 'pta_gap', 'linear_ward_pta')}
    logs = {}
    for name in VARIANTS:
        model, log, ev, keep = run_variant(cx, cfg, d, name)
        torch.save(dict(din=len(keep), keep=keep, width=cfg['m2']['width'], depth=cfg['m2']['depth'], state=model.state_dict()),
                   root/('model_%s.pt' % name))
        save('label_eval_%s.json' % name, ev)
        logs[name] = log
        table[name] = dict(summary(ev), inputs=len(keep))
        print('RM2_ABL', name, table[name], flush=True)
    # existing model trained without the early labels, evaluated on the same data
    ne = rt.HERE/'results'/('train_%s' % ac['no_early_train_job'])
    m = rt.read(ne/'manifest.json')
    if m['status'] != 'TRAIN_COMPLETE':
        raise ValueError('Incomplete training: '+str(ne))
    inputs[ne.name] = rt.sha(ne/'manifest.json')
    inputs[ne.name+'/model.pt'] = rt.sha(ne/'model.pt')
    ev = evaluate(load_model(ne/'model.pt', cx.device), d, cx.device, cfg)
    save('label_eval_no_early.json', ev)
    table['no_early'] = summary(ev)
    print('RM2_ABL no_early', table['no_early'], flush=True)
    ref = table['full_repro']['monitor_all']
    checks['retrain_noise'] = dict(full_repro=ref, frozen=table['frozen']['monitor_all'],
                                   difference=ref-table['frozen']['monitor_all'],
                                   within_tolerance=abs(ref-table['frozen']['monitor_all']) <= ac['repro_tolerance'])
    reading = {}
    for name in ('no_pta', 'no_state', 'reg_only', 'no_early'):
        drop = ref-table[name]['monitor_all']
        reading[name] = dict(drop_vs_full_repro=drop, drop_vs_frozen=table['frozen']['monitor_all']-table[name]['monitor_all'],
                             matters=bool(drop > ac['threshold']))
    save('train_logs.json', logs)
    save('checks.json', checks)
    return dict(table=table, baselines=baselines, reading=reading, checks=checks, rule=ac['rule'])

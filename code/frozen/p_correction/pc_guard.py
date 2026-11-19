"""Physical preparation identity, metadata-only reservation, and independent pools."""
import pc_runtime as rt
import math
import random
from collections import defaultdict
from pathlib import Path
from ghz_data import plan as physical_plan


def canonical(p, physical):
    if p['family'] not in ('depolarizing', 'phase_biased') or p['n'] not in (2, 3, 4) or p['depth'] not in (2, 4):
        raise ValueError('Unsupported preparation')
    vals = {k: float(p[k]) for k in ('F0', 'phi', 'gamma')}
    if not all(math.isfinite(v) for v in vals.values()):
        raise ValueError('Nonfinite preparation')
    if not 1/2**p['n'] < vals['F0'] < 1 or not 0 <= vals['phi'] < math.pi or not 0 < vals['gamma'] < 1:
        raise ValueError('Outside canonical strictly positive spectrum')
    history = ('AB'*p['depth'])[:p['depth']]
    if p.get('history_word', history) != history:
        raise ValueError('Unsupported history override')
    return dict(n=p['n'], depth=p['depth'], family=p['family'], generator='ghz_factors_v1',
                history=history, rotation_party=p['n']-1,
                tail=float(physical['spectra']['phase_biased_tail']).hex() if p['family'] == 'phase_biased' else None,
                **{k: v.hex() for k, v in vals.items()})


def signature(p, physical):
    return rt.digest(canonical(p, physical))


def history(p):
    return float(p['phi']), float(p['gamma'])


def near(a, b):
    return max(abs(x-y) for x, y in zip(a, b)) <= 1e-10


def metadata(cfg):
    path = (rt.HERE/cfg['g1a']).resolve()/'evaluation'/'tasks_eval_items.json'
    items = rt.read(path)
    if any(i['prep']['split'] != 'evaluation' for i in items):
        raise ValueError('Evaluation metadata contains sealed/non-evaluation data')
    return items


def stratified_preps(items, physical, seed):
    by_sig = defaultdict(list)
    for i in items:
        by_sig[signature(i['prep'], physical)].append(i)
    strata = defaultdict(list)
    for sig, group in sorted(by_sig.items()):
        p = group[0]['prep']
        strata[(p['family'], p['phi'], p['gamma'], p['depth'])].append(sig)
    rng = random.Random(seed)
    for group in strata.values():
        rng.shuffle(group)
    order = []
    while any(strata.values()):
        for key in sorted(strata):
            if strata[key]:
                order.append(strata[key].pop())
    return order, by_sig


def choose_tasks(groups, sigs, count, H):
    queues = {s: sorted(groups[s], key=lambda x: (x['family'], x['id']))[:] for s in sigs}
    result = []
    # Cycle word families as well as preparations; no outcome-based selection.
    for j, sig in enumerate(sigs):
        q = queues[sig]
        if q:
            offset = j % len(q)
            queues[sig] = q[offset:]+q[:offset]
    while len(result) < count and any(queues.values()):
        for sig in sigs:
            if queues[sig] and len(result) < count:
                i = queues[sig].pop(0)
                result.append(dict(i, id='pc_n%d_H%d__%s' % (i['n'], H, i['id']),
                    source_id=i['id'], H=H, word=i['word'][:H], prep_signature=sig))
    return result


def make_plan(cfg, physical):
    all_items = metadata(cfg)
    old = rt.read((rt.HERE/cfg['old_panel']).resolve())['items']
    if len(old) != 8 or any(i['n'] != 4 or i['H'] != 10 for i in old):
        raise ValueError('Historical L panel changed')
    old = [dict(i, prep_signature=signature(i['prep'], physical), domain='n4_H10') for i in old]
    used = {i['prep_signature'] for i in old}
    reservations, audits = {}, {}
    stage = (rt.HERE/cfg['g1a']).resolve()/'evaluation'
    for n in (3, 4):
        candidates = [i for i in all_items if i['n'] == n and i['H'] == 40
            and signature(i['prep'], physical) not in used and (stage/'refs'/(i['id']+'.npz')).exists()]
        order, groups = stratified_preps(candidates, physical, cfg['seed']+n)
        reserve = order[:cfg['split']['reserve_preparations']]
        if len(reserve) < cfg['split']['reserve_preparations']:
            raise ValueError('Insufficient metadata-only C1 reserve; needs new preparations')
        reservations[str(n)] = dict(signatures=reserve,
            items=[dict(i, prep_signature=s) for s in reserve for i in groups[s]],
            historical_exposure='G1a examined historically; held out from PC method selection')
        available = order[len(reserve):]
        if n == 3:
            take = available[:cfg['split']['audit_preparations']]
            tasks = choose_tasks(groups, take, cfg['split']['audit_tasks'], 40)
            audits['n3_H40'] = [dict(i, domain='n3_H40',
                source_reference=str(stage/'refs'/(i['source_id']+'.npz'))) for i in tasks]
    return dict(version=1, seed=cfg['seed'], K=cfg['K'], old=old, audits=audits,
        reservations=reservations, inputs={str(stage/'tasks_eval_items.json'): rt.sha(stage/'tasks_eval_items.json'),
        str((rt.HERE/cfg['old_panel']).resolve()): rt.sha((rt.HERE/cfg['old_panel']).resolve())},
        final_accessed=False, selection='metadata only, entire physical preparations reserved')


def validate_plan(plan, physical):
    reserved = {s for g in plan['reservations'].values() for s in g['signatures']}
    audit = plan['old']+[i for group in plan['audits'].values() for i in group]
    for i in audit:
        if i['prep']['split'] != 'evaluation' or signature(i['prep'], physical) in reserved:
            raise ValueError('PC0/C1 overlap, including renamed preparations')
        if signature(i['prep'], physical) != i['prep_signature']:
            raise ValueError('Preparation signature changed')
    for path, value in plan['inputs'].items():
        if rt.sha(path) != value:
            raise ValueError('Metadata used for reservation changed')
    return True


def generate_pool(cfg, physical, n, profile=False):
    # Static specifications only: no final state/reference is loaded or generated.
    forbidden = [p for p in physical_plan(physical) if p['split'] in ('evaluation', 'final')]
    forbidden += [i['prep'] for i in metadata(cfg)]
    forbidden_s = {signature(p, physical) for p in forbidden}
    forbidden_h = [history(p) for p in forbidden]
    forbidden_h += [tuple(h) for hs in physical['histories'].values() for h in hs]
    rng = random.Random(cfg['seed']+n)
    seen, split_h, out = set(), {'fit': [], 'monitor': []}, []
    for split in ('fit', 'monitor'):
        for j in range(cfg['pool'][split]):
            for _ in range(10000):
                p = dict(n=n, family=('depolarizing', 'phase_biased')[j % 2],
                    depth=(2, 4)[(j//2) % 2], split='calibration',
                    F0=rng.uniform(max(cfg['pool']['F0'][0], 1/2**n+1e-6), cfg['pool']['F0'][1]),
                    phi=rng.uniform(*cfg['pool']['phi']), gamma=rng.uniform(*cfg['pool']['gamma']))
                s, h = signature(p, physical), history(p)
                if s not in forbidden_s|seen and not any(near(h, x) for x in forbidden_h+split_h['fit']+split_h['monitor']):
                    break
            else:
                raise ValueError('Unable to generate isolated history')
            seen.add(s); split_h[split].append(h)
            p['id'] = 'pc_'+s[:20]
            if not profile or j < cfg['profile'][split]:
                out.append(dict(prep=p, split=split, signature=s))
    return out

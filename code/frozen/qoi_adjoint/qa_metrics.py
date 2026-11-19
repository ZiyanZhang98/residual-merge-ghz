"""Quantity-of-interest accuracy (P relative, F absolute, per prefix), suffix K, step-A decision."""
import qa_runtime  # noqa: F401
import math
from collections import defaultdict
import numpy as np

METRICS = {'P': 'Q_P', 'F': 'Q_F', 'QoI': 'Q_QoI', 'TD': 'Q_TD', 'old_strict': 'Q_old_strict'}


def _bounds(err, delta, sign, relative):
    e = np.maximum(0., err+sign*delta)
    return e/(1-sign*delta) if relative else e


def score(td, P, N, ref_P, ref_F, tol, dt=0., dp=0.):
    td, P, N, ref_P, ref_F = (np.asarray(x, dtype=float) for x in (td, P, N, ref_P, ref_F))
    if not (td.shape == P.shape == N.shape == ref_P.shape == ref_F.shape):
        raise ValueError('Prefix array mismatch')
    if not all(np.isfinite(x).all() for x in (td, ref_P, ref_F)):
        raise FloatingPointError('Nonfinite trajectory/reference metric')
    out = dict(status='OK', td=td.tolist(), P_reference=ref_P.tolist(), F_reference=ref_F.tolist(),
               Q_TD=float(td.max()/tol['TD']))
    for tag, s in (('low', -1), ('high', 1)):
        out['Q_TD_'+tag] = float(_bounds(td, dt, s, False).max()/tol['TD'])
    # Success probability, signed raw value; no clipping.
    if np.any(ref_P <= 0):
        out['status_P'] = 'UNDEFINED_RELATIVE_REFERENCE'
    elif not np.isfinite(P).all():
        out['status_P'] = 'NUMERIC_FAILURE'
    else:
        ep = abs(P-ref_P)/ref_P
        out.update(status_P='OK', P=P.tolist(), P_relative=ep.tolist(), Q_P=float(ep.max()/tol['P_relative']),
                   negative_P_prefixes=int((P < 0).sum()), above_one_P_prefixes=int((P > 1).sum()))
        for tag, s in (('low', -1), ('high', 1)):
            out['Q_P_'+tag] = float(_bounds(ep, dp, s, True).max()/tol['P_relative'])
    # Fidelity F = N/P needs a valid positive corrected P.
    if out['status_P'] != 'OK' or not np.isfinite(N).all():
        out['status_F'] = 'NUMERIC_FAILURE'
    elif np.any(P <= 0):
        out['status_F'] = 'UNDEFINED_NONPOSITIVE_P'
    else:
        F = N/P
        ef = abs(F-ref_F)
        out.update(status_F='OK', N=N.tolist(), F=F.tolist(), F_absolute=ef.tolist(),
                   Q_F=float(ef.max()/tol['F_absolute']),
                   F_outside_unit=int(((F < 0) | (F > 1)).sum()))
        # |F_ref - F_true| <= reference TD difference.
        for tag, s in (('low', -1), ('high', 1)):
            out['Q_F_'+tag] = float(_bounds(ef, dt, s, False).max()/tol['F_absolute'])
    for tag in ('', '_low', '_high'):
        qp, qf, qt = out.get('Q_P'+tag), out.get('Q_F'+tag), out['Q_TD'+tag]
        out['Q_QoI'+tag] = None if qp is None or qf is None else max(qp, qf)
        out['Q_old_strict'+tag] = None if qp is None else max(qp, qt)
    return out


def needed(rows, grid, metric):
    by = {r['K']: r for r in rows}
    if any(k not in by for k in grid):
        return dict(status='INCOMPLETE', K=None)
    if any(by[k]['status'] == 'NA_REFERENCE' for k in grid):
        return dict(status='NA_REFERENCE', K=None)
    passes = [by[k]['status'] == 'OK' and by[k].get(metric) is not None and by[k][metric] <= 1 for k in grid]
    for j, k in enumerate(grid):
        if all(passes[j:]):
            return dict(status='FINITE', K=k)
    return dict(status='RIGHT_CENSORED', K=None, lower_bound=max(grid),
                numerical_failure=any(by[k]['status'] != 'OK' or by[k].get(metric) is None for k in grid))


def kvalue(k, censored):
    return k['K'] if k['status'] == 'FINITE' else censored if k['status'] == 'RIGHT_CENSORED' else None


def analyze(cells, items, grid):
    by = defaultdict(list)
    for r in cells:
        by[(r['task_id'], r['arm'])].append(r)
    tasks = []
    for item in items:
        arms = sorted(a for t, a in by if t == item['id'])
        row = dict(task_id=item['id'], prep_id=item['prep_signature'], domain=item['domain'],
                   panel=item.get('panel'), family=item.get('family'), word=item['word'],
                   stratum='single_letter' if item.get('family') in ('A', 'B') else 'mixed_word', arms={})
        for a in arms:
            runs = by[(item['id'], a)]
            ks = {name: needed(runs, grid, metric) for name, metric in METRICS.items()}
            for tag in ('low', 'high'):
                ks['QoI_'+tag] = needed(runs, grid, 'Q_QoI_'+tag)
            row['arms'][a] = dict(K=ks, reference_sensitive=any(ks['QoI_'+t] != ks['QoI'] for t in ('low', 'high')),
                                  statuses=sorted({r['status'] for r in runs}))
        row['reference_usable'] = bool(row['arms']) and row['arms']['M0']['K']['TD']['status'] != 'NA_REFERENCE'
        tasks.append(row)
    return dict(tasks=tasks, grid=grid, final_accessed=False,
                scope='all usable-reference tasks; QoI = per-prefix P relative and F absolute error')


def geo(values):
    return float(math.exp(np.mean(np.log(values)))) if values else None


def decide(report, cfg):
    d = cfg['decision']; cen = d['censored_K']
    formulas = ('M_PTA', 'M_PTA__mul')
    domains = sorted({t['domain'] for t in report['tasks']})
    out = {}
    for dom in domains:
        tasks = [t for t in report['tasks'] if t['domain'] == dom and t['reference_usable']]
        excluded = sum(1 for t in report['tasks'] if t['domain'] == dom and not t['reference_usable'])
        info = dict(tasks=len(tasks), excluded_reference=excluded,
                    preparations=len({t['prep_id'] for t in tasks}), formulas={})
        for f in formulas:
            fr = dict()
            for metric in ('QoI', 'P', 'F'):
                prep = defaultdict(list); flags = 0
                for t in tasks:
                    a, b = t['arms']['M0']['K'][metric], t['arms'][f]['K'][metric]
                    x, y = kvalue(a, cen), kvalue(b, cen)
                    if x is None or y is None:
                        continue
                    flags += a['status'] != 'FINITE' or b['status'] != 'FINITE'
                    prep[t['prep_id']].append(x/y)
                fr[metric] = dict(ratio=geo([geo(v) for v in prep.values()]), censored_terms=flags,
                                  preparations=len(prep))
            info['formulas'][f] = fr
        best = max(formulas, key=lambda f: (info['formulas'][f]['QoI']['ratio'] or 0., f == 'M_PTA'))
        info['chosen_formula'] = best
        info['framework_ratio'] = info['formulas'][best]['QoI']['ratio']
        rooms = {}
        for metric in ('QoI', 'P', 'F'):
            for scope in ('all', 'mixed_word'):
                prep = defaultdict(list)
                for t in tasks:
                    if scope == 'mixed_word' and t['stratum'] != 'mixed_word':
                        continue
                    v = kvalue(t['arms'][best]['K'][metric], cen)
                    if v is not None:
                        prep[t['prep_id']].append(v)
                hit = sum(float(np.median(v)) >= d['room_K'] for v in prep.values())
                rooms[metric+'_'+scope] = dict(preparations=len(prep), with_room=hit,
                    fraction=hit/len(prep) if prep else None,
                    holds=bool(prep) and hit/len(prep) > d['room_preparation_fraction'])
        info['room_after_PTA'] = rooms
        out[dom] = info
    framework = [dom for dom in d['h10_domains'] if dom in out and (out[dom]['framework_ratio'] or 0) >= d['framework_ratio']]
    room = [dom for dom in d['h40_domains'] if dom in out and out[dom]['room_after_PTA']['QoI_all']['holds']]
    f_room = [dom for dom in out if out[dom]['room_after_PTA']['F_all']['holds']]
    action = 'STEP_B' if room else 'STEP_B_F' if f_room else 'STOP_REEVALUATE'
    return dict(domains=out, framework_value_domains=framework, framework_value=bool(framework),
                learning_room_domains=room, learning_room=bool(room), F_room_domains=f_room, action=action,
                rule='HANDOFF 22.7 step A; room = median K over a preparation\'s usable tasks after the chosen '
                     'global PTA formula >= room_K, in more than half of the preparations; censored K counts as %d' % cen)

"""M0 smoke of HANDOFF 22.28: hard checks of the residual-merge code, label throughput, F2 availability.

Development tasks only (resolved monitor tasks of hard_regime step 1); no training; final data are never
accessed. Checks (all must pass):
  f2_available                  every n=4 task has the fixed-K merge+PTA cells (K 4..256, physical_calls, Q_QoI)
  counting_matches_step1        re-running step 1's code path reproduces its cells (physical_calls exactly)
  policy_simulator_equivalence  AdaptiveSim with the frozen Lloyd partition equals the frozen simulator
  exact_adjoint_recovers_truth  with the exact adjoint, budgeted merging + correction = unmerged truth
  telescoping                   pairwise merge gaps of a merge sequence sum to the clusters' gaps
  coupled_vs_exact              coupled pair gaps agree with the exact tree on the exact prefixes
  branch_mc_vs_exact            per-branch Monte Carlo agrees with the exact tree (22.26's revised rule)
  bookkeeping                   coupled member average - representative = gap on the same paths
  c1_rollouts_numeric           C1 (geometric cost, per-compression budget) rollouts finish without failures
"""
import rm_runtime as rt
import time
from collections import defaultdict
import numpy as np
import torch
import qa_sim
from ghz_physics import coords
from g1_core import check_deadline
from qa_sources import PTA, Exact
from qa_metrics import needed
from hr_eval import load_tasks, context as task_context, scored
from hr_goal import rollout as goal_rollout
from lg_probe import exact, branch_paths, mc_summary, zcheck, summary_z
from rm_core import (GeoCost, Agglomerative, LloydPolicy, agglomerate, adaptive_rollout, Capture, knn_pairs,
                     pair_labels, telescoping)

ARMS = ('M0', 'M_PTA', 'M_PTA__mul')


def f2_summary(step1, tasks, cfg):
    """F2 = fixed-K merge+PTA (additive) straight from hard_regime step 1 (HANDOFF 22.28: no re-run)."""
    f = cfg['f2']
    by = defaultdict(dict)
    for c in step1:
        if c['arm'] == f['arm']:
            by[c['task_id']][c['K']] = c
    rows, missing = [], []
    for t in tasks:
        if t['n'] not in f['n']:
            continue
        cs = by.get(t['id'], {})
        pts = [cs[k] for k in f['K'] if k in cs]
        if len(pts) != len(f['K']) or not all(p.get('status') == 'OK' and p.get('physical_calls')
                                              and p.get('Q_QoI') is not None for p in pts):
            missing.append(t['id'])
            continue
        k = needed(pts, f['K'], 'Q_QoI')
        rows.append(dict(task_id=t['id'], n=t['n'], family=t['family'], prep_id=t['prep_signature'],
                         points=[[p['K'], p['physical_calls'], p['Q_QoI']] for p in pts], status=k['status'],
                         K_needed=k['K'], calls_needed=next((p['physical_calls'] for p in pts if p['K'] == k['K']), None)))
    fam = defaultdict(list)
    for r in rows:
        fam[r['family']].append(r)
    summary = {}
    for name, v in sorted(fam.items()):
        finite = [r['K_needed'] for r in v if r['K_needed'] is not None]
        summary[name] = dict(tasks=len(v), right_censored=len(v)-len(finite),
                             median_K=float(np.median(finite)) if finite else None,
                             median_calls=float(np.median([r['calls_needed'] for r in v if r['calls_needed']]))
                             if finite else None)
    return dict(rows=rows, missing=missing, summary=summary, passed=bool(rows) and not missing)


def check_counting(cx, cfg, task, ref, step1):
    """Step 1's code path (hr_goal.rollout without look-ahead, PTA source) re-run at one K."""
    K = cfg['smoke']['check_K']
    states, weights, ctx = task_context(cx, task)
    sim, rho, ell, est = goal_rollout(K, task['word'], ctx, cx.c, states, weights, {'M_PTA': PTA()}, cx.hard)
    out = dict(task_id=task['id'], K=K, calls_new=sim.cost['physical_calls'])
    for arm in ARMS:
        mine = scored(task, arm, K, est[arm], ref, cfg['tolerance'], {})
        old = next(c for c in step1 if c['task_id'] == task['id'] and c['arm'] == arm and c['K'] == K)
        a, b = np.asarray(mine['P']), np.asarray(old['P'])
        out[arm] = dict(Q_QoI_new=mine['Q_QoI'], Q_QoI_old=old['Q_QoI'], max_rel_dP=float(np.max(np.abs(a-b)/np.abs(b))),
                        calls_old=old.get('physical_calls'))
    out['calls_old'] = out['M_PTA']['calls_old']
    out['passed'] = out['calls_new'] == out['calls_old'] and all(out[a]['max_rel_dP'] <= 1e-9 for a in ARMS)
    return out, (states, weights, ctx, sim, rho, est)


def check_equivalence(cx, cfg, task, first):
    """AdaptiveSim + LloydPolicy(K) against the frozen simulator run in check_counting."""
    states, weights, ctx, sim0, rho0, est0 = first
    K = cfg['smoke']['check_K']
    sim, rho, ell, est = adaptive_rollout(LloydPolicy(K), task['word'], ctx, cx.c, states, weights, {'M_PTA': PTA()}, cx.hard)
    rel = lambda a, b: float(((a-b).abs()/b.abs().clamp(min=1e-300)).max())  # noqa: E731
    out = dict(task_id=task['id'], K=K, calls=[sim.cost['physical_calls'], sim0.cost['physical_calls']],
               rho_max_abs=float((rho-rho0).abs().max()),
               estimates={a: {q: rel(est[a][q], est0[a][q]) for q in ('P', 'N')} for a in ARMS})
    out['passed'] = (out['calls'][0] == out['calls'][1] and out['rho_max_abs'] <= 1e-12
                     and all(v <= 1e-12 for a in ARMS for v in out['estimates'][a].values()))
    return out


def check_exact_adjoint(cx, cfg, task):
    """Short word: the exact adjoint makes budgeted merging + additive correction equal the unmerged truth."""
    s, mg = cfg['smoke'], cfg['merge']
    word = task['word'][:s['short_word']]
    states, weights, ctx = task_context(cx, task)
    _, _, _, base = qa_sim.rollout(10**9, word, ctx, cx.c, states, weights, None, cx.hard)
    runs = []
    for b in s['exact_budgets']:
        pol = Agglomerative(GeoCost(), b, 10**9, mg['knn'], mg['max_rounds'])
        sim, rho, ell, est = adaptive_rollout(pol, word, ctx, cx.c, states, weights, {'EX': Exact()}, cx.hard)
        r = dict(budget=b, merges=sum(e.get('merges', 0)+e.get('forced', 0) for e in sim.merge_log),
                 atoms=[e['after'] for e in sim.merge_log])
        for q in ('P', 'N'):
            ref = base['M0'][q]
            r['corrected_max_rel_'+q] = float(((est['EX'][q]-ref).abs()/ref.abs()).max())
            r['raw_max_rel_'+q] = float(((est['M0'][q]-ref).abs()/ref.abs()).max())
        r['passed'] = r['corrected_max_rel_P'] <= 1e-10 and r['corrected_max_rel_N'] <= 1e-10
        runs.append(r)
    # The largest budget merges every round down to one atom, so at least that run exercises the correction.
    return dict(task_id=task['id'], word=word, runs=runs,
                passed=all(r['passed'] for r in runs) and any(r['merges'] > 0 for r in runs))


def capture(cx, task, K, at):
    states, weights, ctx = task_context(cx, task)
    sim = Capture(K, task['word'], ctx, cx.c, states, weights, cx.hard, at)
    for a in task['word'][:max(at)]:
        sim.advance(a)
    return sim.kept, ctx


def residual_summary(lab):
    """Size of the residual gaps against the members' future success and against the PTA gaps (M1 planning)."""
    level = np.maximum(np.abs(lab['level_psi']), 1e-300)
    rel_r = np.max(np.abs(lab['r_psi'])/level, 1)
    rel_pta = np.max(np.abs(lab['pta_psi'])/level, 1)
    e = lab['exact_prefixes']
    resolvable = np.any(np.abs(lab['r_psi'][:, e:]) > 2*lab['se_psi'][:, e:], 1) if lab['r_psi'].shape[1] > e else np.ones(len(level), bool)
    return dict(median_max_rel_residual_psi=float(np.median(rel_r)), median_max_rel_pta_gap_psi=float(np.median(rel_pta)),
                q90_max_rel_residual_psi=float(np.quantile(rel_r, .9)), resolvable_fraction=float(np.mean(resolvable)))


def check_events(cx, cfg, task, gen):
    """At the pre-merge branches of the frozen merge_64 trajectory: pair labels (throughput, coupled vs exact),
    per-branch Monte Carlo vs exact, and the telescoping identity of a multi-round agglomeration."""
    s, lab, mg = cfg['smoke'], cfg['labels'], cfg['merge']
    kept, ctx = capture(cx, task, s['probe_K'], s['probe_events'])
    rec = dict(task_id=task['id'], n=task['n'], family=task['family'], events=[])
    z = dict(psi=([], []), phi=([], []), branch=([], []))
    books, worst = [], []
    rec['worst_coupled_vs_exact'] = worst
    for t in sorted(kept):
        check_deadline(cx.hard)
        ev = kept[t]
        S, w, F, rest = ev['states'], ev['w'], ev['factors'], ev['rest']
        x = coords(S, ctx['aux']['pauli'])
        pairs = knn_pairs(x, 1)[:s['pairs_per_event']]
        out = pair_labels(F, w, pairs, rest, ctx, cx.c, cx.hard, lab, gen)
        e, raw = out['exact_prefixes'], out['raw']
        # Both gaps are scaled by the members' average psi: a phi gap enters the fidelity as d(phi)/psi
        # (F = phi/psi), so psi is the scale on which 1e-3 means a 0.1% effect. Smoke 28721901 used phi itself,
        # which inflates the relative difference by 1/F (2 of 1321 entries at 1.2e-3); that ratio is kept below.
        for q, g, se, ex in (('psi', 'coupled_g', 'se_g', 'exact_psi'), ('phi', 'coupled_h', 'se_h', 'exact_phi')):
            diff = raw[g][:, :e]-raw[ex][:, :e]
            zz, dd = zcheck(diff, raw[se][:, :e], raw['level_exact_psi'][:, :e])
            z[q][0].extend(zz)
            z[q][1].extend(dd)
            rel_own = np.abs(diff)/np.maximum(np.abs(raw['level_exact_'+q][:, :e]), 1e-300)
            j = np.unravel_index(np.argmax(rel_own), rel_own.shape)
            worst.append(dict(t=t, quantity=q, pair=int(j[0]), prefix=int(j[1])+1, diff=float(diff[j]),
                              se=float(raw[se][:, :e][j]), level_psi=float(raw['level_exact_psi'][:, :e][j]),
                              level_own=float(raw['level_exact_'+q][:, :e][j]), max_rel_own_scale=float(rel_own[j])))
        books.append(out['bookkeeping_max_abs'])
        steps = out['coupled_path_steps']
        sec = out['seconds']
        row = dict(t=t, branches=len(w), pairs=len(pairs), L=len(rest), seconds=sec,
                   seconds_per_pair=sum(sec.values())/len(pairs), coupled_path_steps=steps,
                   coupled_seconds_per_path_step=sec['coupled']/max(steps, 1), residual=residual_summary(out))
        if t == min(kept):
            m = s['branch_mc_paths']
            sel = torch.arange(min(s['branch_mc_branches'], len(w)), device=w.device)
            mc = mc_summary(branch_paths(F[sel], rest, ctx, cx.c, m, lab['chunk_paths'], gen, cx.hard), m)
            exb = exact(S[sel], rest[:e], ctx, cx.c, cx.hard, lab['exact_chunk'])
            zz, dd = zcheck((mc['psi'][:, :e]-exb['psi']).cpu().numpy(), mc['se_psi'][:, :e].cpu().numpy(),
                            exb['psi'].cpu().numpy())
            z['branch'][0].extend(zz)
            z['branch'][1].extend(dd)
        if t == s['telescoping_t']:
            labels, info = agglomerate(S, w, x, GeoCost(), 0., max(2, len(w)//4), mg['knn'], mg['max_rounds'], record=True)
            tel = telescoping(dict(info.pop('record'), states=S, branch_w=w, labels=labels, rest=rest), ctx, cx.c, cx.hard, lab)
            tel.update(agglomeration=info)
            tel['passed'] = tel['rounds'] >= 2 and all(tel['%s_%s_max_abs' % (a, q)] <= 1e-12
                                                     for a in ('pta', 'exact') for q in ('psi', 'phi'))
            rec['telescoping'] = tel
        rec['events'].append(row)
        print('RM_EVENT %s t=%d branches=%d pairs=%d L=%d %.2fs/pair (pta %.1fs exact %.1fs coupled %.1fs, %.2e s/path-step)' % (
            task['id'], t, len(w), len(pairs), len(rest), row['seconds_per_pair'], sec['pta'], sec['exact'], sec['coupled'],
            row['coupled_seconds_per_path_step']), flush=True)
    return rec, z, books


def c1_runs(cx, cfg, task, ref):
    """C1 = geometric Ward cost with a fixed budget per compression; one rollout per budget (cap = resource limit)."""
    mg = cfg['merge']
    states, weights, ctx = task_context(cx, task)
    cells, runs = [], []
    for b in cfg['smoke']['geo_budgets']:
        check_deadline(cx.hard)
        pol = Agglomerative(GeoCost(), b, mg['cap'], mg['knn'], mg['max_rounds'])
        t0 = time.perf_counter()
        try:
            sim, rho, ell, est = adaptive_rollout(pol, task['word'], ctx, cx.c, states, weights, {'M_PTA': PTA()}, cx.hard)
        except (FloatingPointError, RuntimeError) as exc:
            runs.append(dict(task_id=task['id'], budget=b, status_ok=False, error=repr(exc)))
            continue
        log = sim.merge_log
        extra = dict(policy='C1', budget=b, physical_calls=sim.cost['physical_calls'], seconds=time.perf_counter()-t0,
                     atoms_after=[e['after'] for e in log], forced=sum(e.get('forced', 0) for e in log),
                     merge_seconds=sum(e['seconds'] for e in log), correction_seconds=sum(sim.correction_seconds.values()))
        for arm, e in est.items():
            cells.append(scored(task, 'C1_raw' if arm == 'M0' else 'C1+'+arm, None, e, ref, cfg['tolerance'], extra))
        mine = {c['arm']: c for c in cells[-len(est):]}
        runs.append(dict(task_id=task['id'], family=task['family'], budget=b, status_ok=all(c['status'] == 'OK' for c in mine.values()),
                         physical_calls=extra['physical_calls'], max_atoms=max(extra['atoms_after'] or [0]), forced=extra['forced'],
                         seconds=extra['seconds'], merge_seconds=extra['merge_seconds'],
                         Q_QoI={a: c.get('Q_QoI') for a, c in mine.items()}))
        r = runs[-1]
        print('RM_C1 %s budget=%g calls=%d max_atoms=%d forced=%d Q_QoI(+PTA)=%s %.1fs' % (
            task['id'], b, r['physical_calls'], r['max_atoms'], r['forced'], r['Q_QoI'].get('C1+M_PTA'), r['seconds']), flush=True)
    return cells, runs


def smoke(cx, cfg, root, inputs, save):
    s = cfg['smoke']
    refs, npz, step1 = rt.references(inputs, cfg['hard_regime']['step1_job'])
    resolved = {r['task_id'] for r in refs if r['resolved']}
    tasks = [t for t in load_tasks(rt.hr.config()) if t['id'] in resolved]
    picks = [next(t for t in tasks if t['n'] == s['n'] and t['family'] == f) for f in s['families']]
    save('unit.json', dict(picks=[t['id'] for t in picks], resolved_tasks=len(tasks)))
    gen = torch.Generator(device=cx.device).manual_seed(int(s['seed']))
    checks, failed = {}, []

    def done(name, value, ok):
        checks[name] = dict(value, passed=bool(ok)) if isinstance(value, dict) else dict(value=value, passed=bool(ok))
        if not ok:
            failed.append(name)
        save('checks.json', dict(checks=checks, failed=failed))
        print('RM_CHECK', name, 'PASS' if ok else 'FAIL', flush=True)

    f2 = f2_summary(step1, tasks, cfg)
    save('f2.json', f2)
    done('f2_available', dict(tasks=len(f2['rows']), missing=f2['missing'], summary=f2['summary']), f2['passed'])

    task = picks[0]
    ref = dict(np.load(npz[task['id']]))
    c, first = check_counting(cx, cfg, task, ref, step1)
    done('counting_matches_step1', c, c['passed'])
    e = check_equivalence(cx, cfg, task, first)
    done('policy_simulator_equivalence', e, e['passed'])
    x = check_exact_adjoint(cx, cfg, task)
    done('exact_adjoint_recovers_truth', x, x['passed'])

    events, zs, books = [], dict(psi=([], []), phi=([], []), branch=([], [])), []
    for t in picks:
        rec, z, b = check_events(cx, cfg, t, gen)
        events.append(rec)
        books += b
        for k in zs:
            zs[k][0].extend(z[k][0])
            zs[k][1].extend(z[k][1])
        save('events.json', events)
    tel = [r['telescoping'] for r in events if 'telescoping' in r]
    done('telescoping', dict(runs=tel), bool(tel) and all(r['passed'] for r in tel))
    for q in ('psi', 'phi'):
        sz = summary_z(*zs[q])
        done('coupled_vs_exact_'+q, sz, sz['fail'] == 0 and sz['count']+sz['deterministic'] > 0)
    sz = summary_z(*zs['branch'])
    done('branch_mc_vs_exact', sz, sz['fail'] == 0 and sz['count']+sz['deterministic'] > 0)
    done('bookkeeping', dict(max_abs=max(books)), max(books) <= 1e-10)

    cells, runs = [], []
    for t in picks:
        r_ = dict(np.load(npz[t['id']]))
        c1c, c1r = c1_runs(cx, cfg, t, r_)
        cells += c1c
        runs += c1r
        save('cells.json', cells)
        save('c1_runs.json', runs)
    done('c1_rollouts_numeric', dict(runs=runs), bool(runs) and all(r['status_ok'] for r in runs))
    return dict(passed=not failed, failed=failed)

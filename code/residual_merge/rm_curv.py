"""Curvature smoke for the ABB plateau (HANDOFF 5j item 22.30, 2026-10-31, user request). Two parts, no training.

A  Candidate set. The old model L (train 28736643) prices only k-nearest-neighbour pairs in Pauli coordinates, and on
   ABB the atom cap forces thousands of merges among them (M3). Here the neighbours are searched in the goal-metric
   space of HANDOFF 22.18 (z = M^1/2 (x - xbar) plus its geometric tie-breaker; Hessian of the D1 or PTA response at
   the ensemble mean, once per compression). Prices, budget, cap, PTA correction and scoring are unchanged.
   Repro check: the same code with z = x reproduces the M3 point of L.
B  Curvature as a feature. On the monitor label events of chosen preparations the captured ensembles are
   regenerated (frozen geometric merge_K trajectory, as rm_labels) and the labelled pairs are matched by their
   coordinates. Per pair, the residual gap that one exact round can see is computed two ways: exactly (D1 gap minus
   PTA gap) and by the global quadratic form (1/2) lam(1-lam) e^T (H_D1 - H_PTA) e. Checks against the stored labels,
   then whether these quantities explain the old model's error on the residual targets, by family.
Final data are never accessed.
"""
import rm_runtime as rt
import time
import numpy as np
import torch
import ghz_factors as gf
from g1_core import check_deadline, Deadline
from qa_sources import PTA
from ghz_physics import coords
from ghz_merge import combine
from hr_eval import context as task_context, scored
from pg_metric import GoalMetric
from pg_sources import DExact
from rm_core import GeoCost, knn_pairs, adaptive_rollout, agglomerate, Capture
from rm_model import LearnedCost, load_model, label_inputs, targets, BLOCKS, OUT
from rm_deploy import task_list

LONG = (1, 2, 3, len(BLOCKS)+1, len(BLOCKS)+2, len(BLOCKS)+3)   # P and F blocks 4-7, 8-15, 16-40
GROUPS = {'ABB': (5,), 'AB+AABB': (2, 4), 'easy': (0, 1, 3)}     # rm_labels.FAMILIES indices


def old_model(cfg, device, inputs):
    from rm_verify import verified_model
    root = rt.HERE/'results'/('train_%s' % cfg['s1']['old_train_job'])
    verified_model(root)
    inputs[root.name] = rt.sha(root/'manifest.json')
    return load_model(root/'model.pt', device)


def metric_of(cfg, kind):
    m = cfg['curv']['metric']
    return GoalMetric(kind, cfg['tolerance'], m['hessian_chunk'], m['tie_breaker'])


# ------------------------------------------------------------------ A: candidate neighbours in the goal metric

@torch.no_grad()
def agglomerate_z(S, w, x, z, cost, budget, cap, knn, max_rounds):
    """rm_core.agglomerate with the candidate neighbours searched in z instead of x. z is a linear image of x, so a
    merged atom's z is the same weighted mean; prices still see the atoms' states and Pauli coordinates."""
    N, dev = len(w), w.device
    size = 2*N
    AS = S.new_zeros((size,)+tuple(S.shape[1:]))
    AS[:N] = S
    AW = w.new_zeros(size)
    AW[:N] = w
    AX = x.new_zeros((size, x.shape[1]))
    AX[:N] = x
    AZ = z.new_zeros((size, z.shape[1]))
    AZ[:N] = z
    parent = torch.arange(size, device=dev)
    alive = torch.arange(N, device=dev)
    top, spent, forced_cost, merges, forced, rounds = N, 0., 0., 0, 0, 0
    while len(alive) > 1 and rounds < max_rounds:
        need = len(alive)-cap
        P = knn_pairs(AZ[alive], knn)
        gi, gj = alive[P[:, 0]], alive[P[:, 1]]
        c = cost(dict(S=AS, w=AW, x=AX), gi, gj)
        if not bool(torch.isfinite(c).all()) or bool((c < 0).any()):
            raise FloatingPointError('Invalid merge cost')
        order = torch.sort(c, stable=True).indices.cpu().tolist()
        cc, pi, pj = c.cpu().tolist(), P[:, 0].cpu().tolist(), P[:, 1].cpu().tolist()
        used, take, extra = set(), [], []
        for k in order:
            i, j = pi[k], pj[k]
            if i in used or j in used:
                continue
            if spent+cc[k] <= budget:
                take.append(k)
                spent += cc[k]
            elif len(take)+len(extra) < need:
                extra.append(k)
                forced_cost += cc[k]
            else:
                break
            used.add(i)
            used.add(j)
        sel = take+extra
        if not sel:
            break
        idx = torch.as_tensor(sel, device=dev)
        I, J = gi[idx], gj[idx]
        new = torch.arange(top, top+len(sel), device=dev)
        Wn = AW[I]+AW[J]
        lam = AW[I]/Wn
        AS[new] = lam[:, None, None]*AS[I]+(1-lam)[:, None, None]*AS[J]
        AX[new] = lam[:, None]*AX[I]+(1-lam)[:, None]*AX[J]
        AZ[new] = lam[:, None]*AZ[I]+(1-lam)[:, None]*AZ[J]
        AW[new] = Wn
        parent[I] = new
        parent[J] = new
        gone = torch.zeros(size, dtype=torch.bool, device=dev)
        gone[I] = True
        gone[J] = True
        alive = torch.cat((alive[~gone[alive]], new))
        top += len(sel)
        merges += len(take)
        forced += len(extra)
        rounds += 1
    root = parent[:N]
    while True:
        nxt = parent[root]
        if torch.equal(nxt, root):
            break
        root = nxt
    labels = torch.unique(root, return_inverse=True)[1]
    return labels, dict(before=N, after=int(labels.max())+1, merges=merges, forced=forced, spent=spent,
                        forced_cost=forced_cost, rounds=rounds)


class MetricPolicy:
    """The learned cost with candidate neighbours in the goal metric (metric None: Pauli coordinates, as M3)."""
    def __init__(self, cost, budget, cap, knn, max_rounds, metric):
        self.cost, self.budget, self.cap, self.knn, self.max_rounds = cost, float(budget), int(cap), int(knn), int(max_rounds)
        self.metric, self.metric_seconds = metric, 0.

    def __call__(self, sim, states, w):
        self.cost.prepare(sim, states, w)
        x = coords(states, sim.ctx['aux']['pauli'])
        z = x
        # with at most knn+1 atoms every pair is a candidate in any space, so the metric cannot change the result
        if self.metric is not None and len(w) > self.knn+1:
            t0 = time.perf_counter()
            z, _ = self.metric(states, w, sim.word[sim.t:], sim.ctx)
            self.metric_seconds += time.perf_counter()-t0
        labels, info = agglomerate_z(states, w, x, z, self.cost, self.budget, self.cap, self.knn, self.max_rounds)
        return labels, dict(policy='learned', budget=self.budget, pairs_priced=self.cost.evaluated, **info)


def run_curv(cx, cfg, task, ctx, states, weights, ref, alpha, model, kind):
    """rm_deploy.run_one for L with neighbours `kind` ('geo', 'D1' or 'PTA'); same scoring and bookkeeping."""
    check_deadline(cx.hard)
    mg = cfg['merge']
    pol = MetricPolicy(LearnedCost(model, cfg['tolerance']), alpha/cfg['m3']['horizon'], mg['cap'], mg['knn'],
                       mg['max_rounds'], None if kind == 'geo' else metric_of(cfg, kind))
    t0 = time.perf_counter()
    try:
        sim, rho, ell, est = adaptive_rollout(pol, task['word'], ctx, cx.c, states, weights, {'M_PTA': PTA()}, cx.hard)
    except Deadline:
        raise
    except (FloatingPointError, RuntimeError) as exc:
        return None, dict(task_id=task['id'], family=task['family'], neighbours=kind, budget=alpha,
                          status='NUMERIC_FAILURE', error=repr(exc))
    log = sim.merge_log
    extra = dict(method='L', neighbours=kind, budget=alpha, physical_calls=sim.cost['physical_calls'],
                 seconds=time.perf_counter()-t0, max_atoms=max([e['after'] for e in log] or [0]),
                 forced=sum(e.get('forced', 0) for e in log), merge_seconds=sum(e['seconds'] for e in log),
                 metric_seconds=pol.metric_seconds)
    c = scored(task, 'L_%s+M_PTA' % kind, None, est['M_PTA'], ref, cfg['tolerance'], extra)
    run = dict(task_id=task['id'], family=task['family'], status=c['status'], Q_QoI=c.get('Q_QoI'),
               Q_P=c.get('Q_P'), Q_F=c.get('Q_F'), **extra)
    print('RM_CURV_RUN %s %s alpha=%g calls=%d max_atoms=%d forced=%d Q=%s %.0fs (metric %.0fs)' % (
        task['id'], kind, alpha, extra['physical_calls'], extra['max_atoms'], extra['forced'], run['Q_QoI'],
        extra['seconds'], extra['metric_seconds']), flush=True)
    return c, run


def candidates(cx, cfg, inputs, save):
    from rm_eval2 import m3_cells
    cv = cfg['curv']
    model = old_model(cfg, cx.device, inputs)
    tasks, npz, _ = task_list(cfg, inputs)
    byid = {t['id']: t for t in tasks}
    m3 = [c for c in m3_cells(cfg, inputs) if c.get('method') == 'L']
    rp = cv['repro']
    plan = [(rp['task'], rp['alpha'], 'geo')]
    plan += [(t, a, 'D1') for a in cv['alpha'] for t in cv['tasks']]
    plan += [(t, a, 'PTA') for a in cv['pta_alpha'] for t in cv['tasks']]
    runs = []
    for tid, a, kind in plan:
        task = byid[tid]
        states, weights, ctx = task_context(cx, task)
        ref = dict(np.load(npz[tid]))
        runs.append(run_curv(cx, cfg, task, ctx, states, weights, ref, a, model, kind)[1])
        save('runs.json', runs)
    mc = next(c for c in m3 if c['task_id'] == rp['task'] and c['budget'] == rp['alpha'])
    r0 = runs[0]
    repro = dict(task=rp['task'], alpha=rp['alpha'], m3_calls=mc['physical_calls'], m3_Q=mc.get('Q_QoI'),
                 calls=r0.get('physical_calls'), Q=r0.get('Q_QoI'),
                 passed=bool(r0['status'] == 'OK' and r0.get('physical_calls') == mc['physical_calls']
                             and abs(r0['Q_QoI']-mc['Q_QoI']) <= 1e-9))
    per = {}
    for tid in cv['tasks']:
        old = {c['budget']: (c['physical_calls'], c.get('Q_QoI')) for c in m3 if c['task_id'] == tid}
        pts = lambda kind: {r['budget']: (r.get('physical_calls'), r.get('Q_QoI')) for r in runs  # noqa: E731
                            if r['task_id'] == tid and r['neighbours'] == kind}
        new = pts('D1')
        old_q = [old[a][1] for a in cv['alpha'] if a in old and old[a][1] is not None]
        new_q = [q for _, q in new.values() if q is not None]
        per[tid] = dict(m3_geo={str(a): old.get(a) for a in cv['alpha']}, D1={str(a): v for a, v in new.items()},
                        PTA={str(a): v for a, v in pts('PTA').items()},
                        min_Q_m3=min(old_q) if old_q else None, min_Q_D1=min(new_q) if new_q else None,
                        passes=any(q <= 1 for q in new_q),
                        halved=bool(new_q and old_q and min(new_q) <= 0.5*min(old_q)))
    hits = sum(p['passes'] or p['halved'] for p in per.values())
    reading = 'A_CANDIDATE' if repro['passed'] and hits >= cv['reading']['a_tasks'] else ('A_REPRO_FAILED' if not repro['passed'] else 'A_NO')
    return dict(repro=repro, tasks=per, hits=hits, reading=reading,
                rule='A_CANDIDATE if the geometric repro passes and, on >= %d of the %d ABB tasks, L with D1-metric '
                     'neighbours reaches Q <= 1 at some alpha or halves the minimum Q of M3 L over the same alphas; '
                     'else A_NO' % (cv['reading']['a_tasks'], len(cv['tasks'])))


# ------------------------------------------------------------------ B: curvature of the labelled pairs

def rel_blocks(d, rp, rf):
    """Block RMS of a per-prefix gap pair (psi rp, phi rf; [pairs, H]) relative to the pair's level, with the same
    definitions as rm_model.targets (F gap = phi gap - F * psi gap)."""
    lam = d['wa']/(d['wa']+d['wb'])
    a_psi = lam[:, None]*d['pta_a_psi']+(1-lam)[:, None]*d['pta_b_psi']
    a_phi = lam[:, None]*d['pta_a_phi']+(1-lam)[:, None]*d['pta_b_phi']
    Fb = np.where(a_psi > 0, a_phi/np.where(a_psi > 0, a_psi, 1.), 0.)
    lvl = d['level_psi']
    ok = d['mask'] & (lvl > 0)
    lv = np.where(ok, lvl, 1.)
    rho = (rp/lv, (rf-Fb*rp)/lv)
    k = np.arange(1, rp.shape[1]+1)[None, :]
    out = np.zeros((len(lvl), OUT))
    for q in range(2):
        for b, (lo, hi) in enumerate(BLOCKS):
            sel = ok & (k >= lo) & (k <= hi)
            out[:, q*len(BLOCKS)+b] = np.sqrt(np.where(sel, rho[q]**2, 0.).sum(1)/np.maximum(sel.sum(1), 1))
    return out


def spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    c = np.corrcoef(ra, rb)[0, 1]
    return float(c) if np.isfinite(c) else None


def padH(v, Hh):
    out = np.zeros((v.shape[0], Hh))
    out[:, :v.shape[1]] = v
    return out


def label_curvature(cx, cfg, inputs, save, root):
    from rm_labels import pool, FAMILIES
    from rm_report import shards, verified
    cv, lc, mg = cfg['curv'], cfg['m1'], cfg['merge']
    qb = rt.hr.qb.config()
    Hh, gamma, n = qb['H'], qb['gamma'], lc['n']
    model = old_model(cfg, cx.device, inputs)
    entries = {e['signature']: e for e in pool(cx, n)}
    letters, pta = cx.letters(n, gamma)
    aux = cx.aux(n)
    gD1, gPT, D1 = metric_of(cfg, 'D1'), metric_of(cfg, 'PTA'), DExact(1)
    np_ = lambda t: t.detach().cpu().numpy()  # noqa: E731
    parts, checks, timing = [], dict(match=[], prefix1=[], d1_exact=[], pta=[], hessian_failures=0), []
    for r in shards(cfg['m2']['labels_job']):
        events = rt.read(r/'events.json')
        sigs = sorted({e['prep'] for e in events if any(e['prep'].startswith(p) for p in cv['label_preps'])})
        if not sigs:
            continue
        verified(r)
        inputs[r.name] = rt.sha(r/'manifest.json')
        rows = {k: v.numpy() for k, v in torch.load(r/'labels.pt', map_location='cpu', weights_only=False)['rows'].items()}
        for sig in sigs:
            entry = entries[sig]
            if entry['split'] != 'monitor':
                raise ValueError('Curvature smoke uses monitor preparations only: '+sig)
            tick = time.perf_counter()
            states, weights, factors = gf.prepare(entry['prep'], cx.c, cx.device)
            ctx = dict(letters=letters, pta=pta, aux=aux, initial_factors=factors)
            hsec = dsec = 0.
            for fi, family in enumerate(FAMILIES):
                evs = [e for e in events if e['prep'] == sig and e['family'] == family]
                if not evs:
                    continue
                check_deadline(cx.hard)
                word = (family*Hh)[:Hh]
                sim = Capture(evs[0]['K'], word, ctx, cx.c, states, weights, cx.hard, lc['events'])
                for a in word[:max(lc['events'])]:
                    sim.advance(a)
                for ev in evs:
                    kept = sim.kept[ev['t']]
                    S, w, rest = kept['states'], kept['w'], kept['rest']
                    L = len(rest)
                    x = coords(S, aux['pauli'])
                    lab, _ = agglomerate(S, w, x, GeoCost(), 0., max(2, len(w)//4), mg['knn'], mg['max_rounds'])
                    Sm, wm = combine(S, w, lab)
                    sets = {False: (S, w, x), True: (Sm, wm, coords(Sm, aux['pauli']))}
                    t0 = time.perf_counter()
                    try:
                        xbar = w @ x
                        HD = gD1.matrix(xbar, rest, ctx)[1]
                        HP = gPT.matrix(xbar, rest, ctx)[1]
                    except (FloatingPointError, RuntimeError):
                        HD = HP = None
                        checks['hessian_failures'] += 1
                    hsec += time.perf_counter()-t0
                    for merged, (Ss, ws, xs) in sets.items():
                        idx = np.nonzero((rows['event'] == ev['event']) & (rows['kind'] == merged))[0]
                        if len(idx) == 0:
                            continue
                        sub = {k: v[idx] for k, v in rows.items()}
                        xa = torch.as_tensor(sub['xa'], device=xs.device)
                        xb = torch.as_tensor(sub['xb'], device=xs.device)
                        da, I = torch.cdist(xa, xs, compute_mode='donot_use_mm_for_euclid_dist').min(1)
                        db, J = torch.cdist(xb, xs, compute_mode='donot_use_mm_for_euclid_dist').min(1)
                        dw = max(float((ws[I]-torch.as_tensor(sub['wa'], device=ws.device)).abs().max()),
                                 float((ws[J]-torch.as_tensor(sub['wb'], device=ws.device)).abs().max()))
                        checks['match'].append(dict(event=int(ev['event']), merged=merged, pairs=len(idx),
                                                    max_dx=float(torch.maximum(da, db).max()), max_dw=dw))
                        lam = ws[I]/(ws[I]+ws[J])
                        reps = lam[:, None, None]*Ss[I]+(1-lam)[:, None, None]*Ss[J]
                        used = torch.unique(torch.cat((I, J)))
                        pos = torch.full((len(ws),), -1, dtype=torch.long, device=ws.device)
                        pos[used] = torch.arange(len(used), device=ws.device)
                        nodes = torch.cat((Ss[used], reps))
                        nu = len(used)
                        gap = lambda v: np_(lam[:, None]*v[pos[I]]+(1-lam)[:, None]*v[pos[J]]-v[nu:])  # noqa: E731
                        t1 = time.perf_counter()
                        vD, vP = D1(nodes, None, rest, ctx, cx.c, cx.hard), PTA()(nodes, None, rest, ctx, cx.c, cx.hard)
                        dsec += time.perf_counter()-t1
                        gD = {q: gap(vD[q]) for q in ('psi', 'phi')}
                        gP = {q: gap(vP[q]) for q in ('psi', 'phi')}
                        scale = lambda a: max(float(np.abs(a).max()), 1e-300)  # noqa: E731
                        # D1 is exact for psi up to prefix 2 (PTA reproduces the next acceptance) and for phi on prefix 1
                        e2 = min(2, L, int(sub['exact_prefixes'][0]))
                        checks['d1_exact'].append(max(float(np.abs(gD[q][:, :k]-sub['g_'+q][:, :k]).max())/scale(sub['g_'+q][:, :k])
                                                      for q, k in (('psi', e2), ('phi', 1))))
                        checks['pta'].append(max(float(np.abs(gP[q]-sub['pta_'+q][:, :L]).max())/scale(sub['pta_'+q][:, :L])
                                                 for q in ('psi', 'phi')))
                        if HD is not None:
                            e = xs[I]-xs[J]
                            fac = 0.5*lam*(1-lam)
                            qD = np_(fac[:, None]*torch.einsum('pi,kij,pj->pk', e, HD, e))
                            qP = np_(fac[:, None]*torch.einsum('pi,kij,pj->pk', e, HP, e))
                            checks['prefix1'].append(max(float(np.abs(qq[:, 0]-sub['g_psi'][:, 0]).max())/scale(sub['g_psi'][:, 0])
                                                         for qq in (qD, qP)))
                            hp, hf = qD[:, :L]-qP[:, :L], qD[:, L:]-qP[:, L:]
                        else:
                            hp = hf = np.full((len(idx), L), np.nan)
                        sub['d1_psi'], sub['d1_phi'] = padH(gD['psi']-gP['psi'], Hh), padH(gD['phi']-gP['phi'], Hh)
                        sub['h_psi'], sub['h_phi'] = padH(hp, Hh), padH(hf, Hh)
                        parts.append(sub)
            timing.append(dict(prep=sig[:16], seconds=time.perf_counter()-tick, hessian_seconds=hsec, d1_seconds=dsec))
            print('RM_CURV_PREP %s %.0fs (hessian %.0fs, d1 %.0fs) pairs=%d' % (
                sig[:16], timing[-1]['seconds'], hsec, dsec, sum(len(p['event']) for p in parts)), flush=True)
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez_compressed(root/'curv_pairs.npz', **{k: d[k] for k in ('event', 'kind', 'family', 't', 'd1_psi', 'd1_phi', 'h_psi', 'h_phi')})
    y, valid = targets(d)
    with torch.no_grad():
        pred = model(label_inputs(d, cx.device)).cpu().numpy()
    logy = np.log(np.maximum(y, 1e-300))
    f_d1 = np.log(np.maximum(rel_blocks(d, d['d1_psi'], d['d1_phi']), 1e-300))
    hb = rel_blocks(d, np.nan_to_num(d['h_psi']), np.nan_to_num(d['h_phi']))
    f_h = np.log(np.maximum(hb, 1e-300))
    hok = np.isfinite(d['h_psi']).all(1)
    tol = cv['checks']
    ck = dict(match_max_dx=max(m['max_dx'] for m in checks['match']), match_max_dw=max(m['max_dw'] for m in checks['match']),
              prefix1_max_rel=max(checks['prefix1']) if checks['prefix1'] else None,
              d1_exact_max_rel=max(checks['d1_exact']), pta_max_rel=max(checks['pta']),
              hessian_failures=checks['hessian_failures'], pairs=int(len(logy)))
    ck['passed'] = bool(ck['match_max_dx'] <= tol['match'] and ck['match_max_dw'] <= tol['match']
                        and ck['prefix1_max_rel'] is not None and ck['prefix1_max_rel'] <= tol['prefix1']
                        and ck['d1_exact_max_rel'] <= tol['exact'] and ck['pta_max_rel'] <= tol['exact'])
    names = ['P%d-%d' % b for b in BLOCKS]+['F%d-%d' % b for b in BLOCKS]
    analysis = {}
    for gname, fams in GROUPS.items():
        g = np.isin(d['family'], fams)
        res = {}
        for j in range(OUT):
            s = g & valid[:, j] & (y[:, j] > 0) & np.isfinite(pred[:, j])
            sh = s & hok
            err = logy[:, j]-pred[:, j]
            res[names[j]] = dict(
                n=int(s.sum()), n_hessian=int(sh.sum()),
                rho_target_model=spearman(logy[s, j], pred[s, j]),
                rho_target_d1=spearman(logy[s, j], f_d1[s, j]),
                rho_target_hess=spearman(logy[sh, j], f_h[sh, j]),
                rho_error_d1=spearman(err[s], f_d1[s, j]),
                rho_error_hess=spearman(err[sh], f_h[sh, j]),
                rho_d1_hess=spearman(f_d1[sh, j], f_h[sh, j]),
                median_d1_over_target=float(np.median(np.exp(f_d1[s, j]-logy[s, j]))) if s.any() else None)
        analysis[gname] = res
    rb = cv['reading']
    hits = [names[j] for j in LONG
            if analysis['ABB'][names[j]]['n'] >= rb['b_min_pairs']
            and max(abs(analysis['ABB'][names[j]]['rho_error_d1'] or 0.), abs(analysis['ABB'][names[j]]['rho_error_hess'] or 0.)) >= rb['b_rho']]
    reading = 'B_CHECK_FAILED' if not ck['passed'] else ('B_INFO' if len(hits) >= rb['b_outputs'] else 'B_NO_INFO')
    return dict(checks=ck, analysis=analysis, timing=timing, abb_hits=hits, reading=reading,
                rule='B_INFO if the checks pass and, on ABB monitor pairs, |Spearman(old-model log error, curvature '
                     'feature)| >= %.2f (exact D1 or Hessian form) on >= %d of the 6 long P/F outputs with >= %d pairs; '
                     'else B_NO_INFO' % (rb['b_rho'], rb['b_outputs'], rb['b_min_pairs']))


# ------------------------------------------------------------------ C2: other families with goal-metric neighbours

def deploy_metric(cx, cfg, shard, shards, inputs, save):
    """L with goal-metric candidate neighbours (cfg curv2.kind) at the M3 alphas on every resolved n=4 M3 task."""
    from hr_eval import shard as take_shard
    c2 = cfg['curv2']
    model = old_model(cfg, cx.device, inputs)
    tasks, npz, _ = task_list(cfg, inputs)
    mine = take_shard(tasks, shard, shards)
    save('unit.json', dict(index=shard, shards=shards, kind=c2['kind'], mine=[t['id'] for t in mine]))
    cells, runs = [], []
    for task in mine:
        states, weights, ctx = task_context(cx, task)
        ref = dict(np.load(npz[task['id']]))
        for a in cfg['m3']['alpha']:
            c, r = run_curv(cx, cfg, task, ctx, states, weights, ref, a, model, c2['kind'])
            runs.append(r)
            if c is not None:
                c['method'] = 'Lm'
                cells.append(c)
        save('cells.json', cells)
        save('runs.json', runs)
    return dict(tasks=len(mine), runs=len(runs), failures=sum(r['status'] != 'OK' for r in runs))


def analyze_metric(cfg, job, inputs):
    """Lm against M3's L (same alphas), C1 and F2, by family, with the rule written before the results."""
    from collections import defaultdict
    from rm_eval2 import m3_cells, verified_shard, M3_KEYS, M3_MODEL_FILES
    from rm_deploy import needed, compare, HARD
    c2, f2 = cfg['curv2'], cfg['f2']
    tasks, npz, step1 = task_list(cfg, inputs)
    pts = defaultdict(lambda: defaultdict(list))
    failures = []
    roots = sorted((rt.HERE/'results').glob('curv2_%s_*' % job))
    if len(roots) != c2['shards']:
        raise ValueError('Expected %d C2 shards, found %d' % (c2['shards'], len(roots)))
    for r in roots:
        verified_shard(r, 'CURV2_SHARD_COMPLETE', M3_KEYS+('curv', 'curv2'), M3_MODEL_FILES+('rm_curv.py', 'rm_deploy.py'))
        inputs[r.name] = rt.sha(r/'manifest.json')
        for c in rt.read(r/'cells.json'):
            pts['Lm'][c['task_id']].append((c['physical_calls'], c.get('Q_QoI') if c['status'] == 'OK' else None))
        for x in rt.read(r/'runs.json'):
            if x['status'] == 'NUMERIC_FAILURE':
                failures.append(x)
                pts['Lm'][x['task_id']].append((0, None))   # a failed run counts as not passing
    for c in m3_cells(cfg, inputs):
        if c.get('method') in ('L', 'C1'):
            pts[c['method']][c['task_id']].append((c['physical_calls'], c.get('Q_QoI') if c['status'] == 'OK' else None))
    for c in step1:
        if c['arm'] == f2['arm'] and c['K'] in f2['K']:
            pts['F2'][c['task_id']].append((c['physical_calls'], c.get('Q_QoI') if c['status'] == 'OK' else None))
    expect = {'Lm': len(cfg['m3']['alpha']), 'L': len(cfg['m3']['alpha']), 'C1': len(cfg['m3']['geo_budgets']), 'F2': len(f2['K'])}
    missing = [(m, t['id']) for m in expect for t in tasks if len(pts[m][t['id']]) != expect[m]]
    if missing:
        raise ValueError('Incomplete grid: %s' % missing[:10])
    values = {m: {t['id']: needed(pts[m][t['id']]) for t in tasks} for m in expect}
    groups = {'AB+AABB': ('AB', 'AABB'), 'ABB': ('ABB',), 'easy': ('A', 'AAB', 'B')}
    groups.update({f: (f,) for f in ('A', 'B', 'AB', 'AAB', 'AABB')})
    out = dict(tasks=len(tasks), failures=failures, groups={})
    for g, fams in groups.items():
        ts = [t for t in tasks if t['family'] in fams]
        out['groups'][g] = dict(
            tasks=len(ts),
            Lm_vs_L=compare(values, 'L', 'Lm', ts), Lm_vs_F2=compare(values, 'F2', 'Lm', ts),
            Lm_vs_C1=compare(values, 'C1', 'Lm', ts), L_vs_F2=compare(values, 'F2', 'L', ts),
            new_censoring=[t['id'] for t in ts if values['Lm'][t['id']][1] and not values['L'][t['id']][1]],
            fixed_censoring=[t['id'] for t in ts if values['L'][t['id']][1] and not values['Lm'][t['id']][1]])
    hard = [t for t in tasks if t['family'] in HARD]
    vs = {b: compare(values, b, 'Lm', hard) for b in ('F2', 'C1')}
    best = min(vs, key=lambda b: vs[b]['ratio'])
    d = cfg['m3']['decision']
    m = vs[best]
    m3_reading = ('EXPAND' if m['ratio'] >= d['ratio'] and m['not_worse'] >= d['fraction']*m['preparations']-1e-9
                  else 'ITERATE' if m['ratio'] > 1.0 else 'DIAGNOSE')
    out['m3_rule_for_Lm'] = dict(stronger_baseline=best, **m, reading=m3_reading)
    out['abb_min_Q'] = {t['id']: {k: min([q for _, q in pts[k][t['id']] if q is not None] or [None]) for k in expect}
                        for t in tasks if t['family'] == 'ABB'}
    lim = 1/c2['regress_ratio']
    bad = [g for g in ('AB+AABB', 'easy') if out['groups'][g]['Lm_vs_L']['ratio'] < lim]
    newc = [i for g in ('AB+AABB', 'easy') for i in out['groups'][g]['new_censoring']]
    out['reading'] = 'C2_ADOPT' if not bad and not newc and not failures else 'C2_REGRESS'
    out['regressed_groups'], out['new_censoring_outside_ABB'] = bad, newc
    out['rule'] = ('C2_ADOPT if, against M3 L at the same alphas, Lm needs at most %.2fx the atoms (prep-equal geometric '
                   'mean) on AB+AABB and on the easy families (A, AAB, B), with no new right-censoring there and no '
                   'numeric failure; else C2_REGRESS' % c2['regress_ratio'])
    out['points'] = {k: {t: sorted(v) for t, v in pm.items()} for k, pm in pts.items()}
    out['needed'] = {k: {t: dict(atoms=v[0], censored=v[1]) for t, v in vm.items()} for k, vm in values.items()}
    return out

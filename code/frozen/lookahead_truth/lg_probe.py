"""Ground truth for look-ahead merge decisions (HANDOFF 22.26, part 2). No training; final data are never accessed.

At sampled compressions of the frozen geometric merge trajectory (M0 of hard_regime step 1), the same
pre-merge branches are grouped by geometry alone and by geometry plus a look-ahead block whose future
responses come from PTA, from the learned model (PTA x exp(r), seed 0) or from the truth: the exact
outcome tree for the first rounds and unbiased per-branch path sampling beyond; two independent halves
of the paths give the Monte Carlo noise floor. Settings reproduce the look-ahead of hard_regime step 2
('orig') and the standardized one of lookahead_tuning / lookahead_oracle ('s1_long', 's1_k3').

Read-outs per compression:
  partitions  agreement of every grouping with the truth grouping (pairs, mass-weighted pairs, ARI) and
              how many of its departures from the geometric grouping the truth grouping shares;
  features    accuracy of the branch-to-branch part of the PTA and learned predictions (the only part
              clustering sees) against the truth, with the Monte Carlo noise removed;
  costs       the true error that this one compression puts on every later prefix, raw and after the
              additive PTA correction: exact for the first rounds; for every prefix a coupled estimator.
The coupled estimator draws outcome sequences along each merged representative's own path measure and
lets every member follow the same outcomes with its likelihood ratio R; per path it equals
W_k (sum_i p_i R_ik - 1), whose first-order part cancels because sum_i p_i (rho_i - rho_c) = 0.
"""
import math
import time
import numpy as np
import torch
import ghz_factors as gf
import qa_sim
from ghz_physics import coords, ghz_diagonal
from ghz_merge import combine
from g1_core import lloyd_async, check_deadline
from qa_sources import PTA
from qb_model import Learned
from pg_sources import exact_rounds
from hr_eval import context

KEYS = ('w', 'w2', 'n', 'n2', 'wn')
SOURCES = ('pta', 'ml', 'truth', 'truthA', 'truthB')
COST_SOURCES = ('pta', 'ml', 'truth')


# ------------------------------------------------------------------ look-ahead blocks

def orig_block(v, L, spec):
    """hr_goal.LookAhead: weight * [log psi_k / k, F_k] for k = min(k, L), duplicates kept."""
    cols = []
    for k in [min(k, L) for k in spec['lengths']]:
        psi = torch.clamp(v['psi'][:, k-1], min=1e-300)
        cols += [psi.log()/k, v['phi'][:, k-1]/psi]
    return spec['weight']*torch.stack(cols, 1)


def std_block(v, L, spec, w, geo):
    """lt_lookahead.StdLookAhead ('psiF') = lookahead_oracle.OracleLookAhead: columns standardized over the
    current branches, block scaled to share x the variance of the Pauli coordinates."""
    ks = sorted({min(k, L) for k in spec['lengths']})
    cols = []
    for k in ks:
        cols.append(torch.clamp(v['psi'][:, k-1], min=1e-300).log())
        cols.append(v['phi'][:, k-1]/torch.clamp(v['psi'][:, k-1], min=1e-300))
    X = torch.stack(cols, 1)
    w = w/w.sum()
    mean = w @ X
    sd = torch.sqrt(w @ (X-mean)**2)
    live = sd > 1e-12
    Z = torch.where(live, (X-mean)/torch.where(live, sd, torch.ones_like(sd)), torch.zeros_like(X))
    vg = float((w @ (geo-w @ geo)**2).sum())
    m = int(live.sum())
    return Z*(math.sqrt(spec['share']*vg/m) if m and vg > 0 else 0.)


def block(spec, v, L, w, geo):
    return orig_block(v, L, spec) if spec['kind'] == 'orig' else std_block(v, L, spec, w, geo)


# ------------------------------------------------------------------ true future responses

def exact(states, word, ctx, c, hard, chunk):
    parts = [exact_rounds(states[i:i+chunk], word, ctx, c, hard)[:2] for i in range(0, len(states), chunk)]
    return dict(psi=torch.cat([p[0] for p in parts]), phi=torch.cat([p[1] for p in parts]))


@torch.no_grad()
def branch_paths(factors, word, ctx, c, per_branch, chunk, gen, hard):
    """hr_mc's unbiased path sampler started from every branch, per_branch paths each; sums [L, N]."""
    N, L, dev = len(factors), len(word), factors.device
    B, letters = ctx['aux']['ghz'], ctx['letters']
    S = {k: torch.zeros(L, N, dtype=torch.float64, device=dev) for k in KEYS}
    total = N*per_branch
    for start in range(0, total, chunk):
        check_deadline(hard)
        owner = torch.arange(start, min(start+chunk, total), device=dev)//per_branch
        f = factors[owner]
        W = torch.ones(len(owner), dtype=torch.float64, device=dev)
        rows = torch.arange(len(owner), device=dev)
        for t, a in enumerate(word):
            cf, p = gf.children(f, letters[a], c['chunk'])
            acc = p.sum(1)
            W = W*acc
            draw = torch.where((acc > 0)[:, None], p, torch.ones_like(p))
            f = cf[rows, torch.multinomial(draw, 1, generator=gen).squeeze(1)]
            WF = W*ghz_diagonal(gf.density(f), B)[:, 0]
            S['w'][t].index_add_(0, owner, W); S['w2'][t].index_add_(0, owner, W*W)
            S['n'][t].index_add_(0, owner, WF); S['n2'][t].index_add_(0, owner, WF*WF)
            S['wn'][t].index_add_(0, owner, W*WF)
    return S


def mc_summary(S, m):
    psi, phi = S['w']/m, S['n']/m
    var_w = torch.clamp(S['w2']/m-psi*psi, min=0.)/m
    var_n = torch.clamp(S['n2']/m-phi*phi, min=0.)/m
    cov = (S['wn']/m-psi*phi)/m
    F = phi/torch.clamp(psi, min=1e-300)
    var_F = torch.clamp(var_n-2*F*cov+F*F*var_w, min=0.)/torch.clamp(psi*psi, min=1e-300)
    return dict(psi=psi.T.contiguous(), phi=phi.T.contiguous(), se_psi=var_w.sqrt().T, se_F=var_F.sqrt().T)


def mixed(ex, mc, e):
    """Exact for the first e prefixes, Monte Carlo beyond."""
    return {q: torch.cat((ex[q][:, :e], mc[q][:, e:]), 1) for q in ('psi', 'phi')}


# ------------------------------------------------------------------ coupled merge cost

@torch.no_grad()
def coupled(F, w, clusters, word, ctx, c, paths, chunk, gen, hard):
    """Per cluster (member indices, >= 2 members): path sums [L] of g = W(sum p R - 1), h = W(sum p R F_i - F_c),
    their squares and product, m = W sum p R (member average of psi) and W (psi of the representative)."""
    dev, L = F.device, len(word)
    B, letters = ctx['aux']['ghz'], ctx['letters']
    out, batch, size = [], [], 0
    for idx in clusters:
        if batch and size+(len(idx)+1)*paths > chunk:
            out += _coupled_batch(F, w, batch, word, letters, B, c, paths, gen, hard)
            batch, size = [], 0
        batch.append(idx); size += (len(idx)+1)*paths
    if batch:
        out += _coupled_batch(F, w, batch, word, letters, B, c, paths, gen, hard)
    return out


def _coupled_batch(F, w, batch, word, letters, B, c, paths, gen, hard):
    dev, L, nc = F.device, len(word), len(batch)
    zero = lambda m: torch.zeros(m, dtype=torch.long, device=dev)  # noqa: E731
    f_c = torch.cat([gf.combine(F[idx], w[idx], zero(len(idx))) for idx in batch]).repeat_interleave(paths, 0)
    sizes = torch.tensor([len(idx) for idx in batch], device=dev)
    mem = torch.cat(batch)
    p_mem = torch.cat([w[idx]/w[idx].sum() for idx in batch])
    cl = torch.repeat_interleave(torch.arange(nc, device=dev), sizes)
    Pc, Pm = nc*paths, len(mem)*paths
    owner = torch.arange(Pc, device=dev)//paths
    parent = (cl*paths).repeat_interleave(paths)+torch.arange(paths, device=dev).repeat(len(mem))
    f_m, p_m = F[mem].repeat_interleave(paths, 0), p_mem.repeat_interleave(paths)
    W = torch.ones(Pc, dtype=torch.float64, device=dev)
    R = torch.ones(Pm, dtype=torch.float64, device=dev)
    rc, rm = torch.arange(Pc, device=dev), torch.arange(Pm, device=dev)
    acc = {k: torch.zeros(L, nc, dtype=torch.float64, device=dev) for k in ('g', 'g2', 'h', 'h2', 'gh', 'm', 'm2', 'c')}
    for t, a in enumerate(word):
        check_deadline(hard)
        cc, pc = gf.children(f_c, letters[a], c['chunk'])
        A = pc.sum(1)
        W = W*A
        y = torch.multinomial(torch.where((A > 0)[:, None], pc, torch.ones_like(pc)), 1, generator=gen).squeeze(1)
        f_c = cc[rc, y]
        cm, pm = gf.children(f_m, letters[a], c['chunk'])
        ym = y[parent]
        pcy = pc[parent, ym]
        R = R*torch.where(pcy > 0, pm[rm, ym]/torch.where(pcy > 0, pcy, torch.ones_like(pcy)), torch.zeros_like(pcy))
        f_m = cm[rm, ym]
        Fc = ghz_diagonal(gf.density(f_c), B)[:, 0]
        Fm = ghz_diagonal(gf.density(f_m), B)[:, 0]
        s_psi = torch.zeros(Pc, dtype=torch.float64, device=dev).index_add_(0, parent, p_m*R)
        s_phi = torch.zeros(Pc, dtype=torch.float64, device=dev).index_add_(0, parent, p_m*R*Fm)
        g, h = W*(s_psi-1.), W*(s_phi-Fc)
        for k, v in (('g', g), ('g2', g*g), ('h', h), ('h2', h*h), ('gh', g*h), ('m', W*s_psi), ('m2', (W*s_psi)**2), ('c', W)):
            acc[k][t].index_add_(0, owner, v)
    acc = {k: (v/paths).T.cpu().numpy() for k, v in acc.items()}
    return [{k: v[j] for k, v in acc.items()} for j in range(nc)]


def clusters_of(labels):
    out = []
    for lab in range(int(labels.max())+1):
        idx = torch.nonzero(labels == lab).flatten()
        if len(idx):
            out.append(idx)
    return out


def key(idx):
    return tuple(int(i) for i in idx.tolist())


# ------------------------------------------------------------------ partitions and features

def ari(a, b):
    a, b = a.cpu().numpy(), b.cpu().numpy()
    ct = np.zeros((a.max()+1, b.max()+1))
    np.add.at(ct, (a, b), 1.)
    c2 = lambda x: x*(x-1)/2  # noqa: E731
    sij, sa, sb, n2 = c2(ct).sum(), c2(ct.sum(1)).sum(), c2(ct.sum(0)).sum(), c2(len(a))
    expected, top = sa*sb/n2, (sa+sb)/2
    return float((sij-expected)/(top-expected)) if top != expected else 1.


def partitions(labels, w):
    """labels: name -> labels over the same branches ('geo', 'pta', 'ml', 'truth', 'truthA', 'truthB')."""
    N = len(w)
    i, j = torch.triu_indices(N, N, 1, device=w.device)
    same = {k: (v[:, None] == v[None, :])[i, j] for k, v in labels.items()}
    ww = (w[:, None]*w[None, :])[i, j]
    ww = ww/ww.sum()
    T, G = same['truth'], same['geo']
    moved_T = T != G
    out = dict(pairs=int(len(T)), truth_departs=int(moved_T.sum()), truth_departs_mass=float((ww*moved_T).sum()))
    for k in ('geo', 'pta', 'ml', 'truthA', 'truthB'):
        d, moved = same[k] != T, same[k] != G
        out[k] = dict(disagree=float(d.double().mean()), disagree_mass=float((ww*d).sum()), ari=ari(labels[k], labels['truth']),
                      departs=int(moved.sum()), departs_mass=float((ww*moved).sum()),
                      precision=float((moved & ~d).sum()/moved.sum()) if bool(moved.any()) else None,
                      recall=float((moved_T & ~d).sum()/moved_T.sum()) if bool(moved_T.any()) else None)
    for a, b in (('pta', 'ml'), ('truthA', 'truthB')):
        d = same[a] != same[b]
        out[a+'_vs_'+b] = dict(disagree=float(d.double().mean()), disagree_mass=float((ww*d).sum()), ari=ari(labels[a], labels[b]))
    return out


def features(preds, truth, A, B, w):
    """Branch-to-branch accuracy against the truth per prefix (weighted, centered). The noise variance of the
    truth (zero on exact prefixes) is estimated from the two Monte Carlo halves and removed."""
    w = w/w.sum()
    fns = (('logpsi', lambda v: torch.clamp(v['psi'], min=1e-300).log()),
           ('F', lambda v: v['phi']/torch.clamp(v['psi'], min=1e-300)))
    out = {}
    for q, fn in fns:
        y, ya, yb = fn(truth), fn(A), fn(B)
        noise = w @ ((ya-yb)/2)**2
        yc = y-w @ y
        spread2 = w @ yc**2
        signal2 = torch.clamp(spread2-noise, min=1e-300)
        out[q] = dict(spread=spread2.sqrt().tolist(), noise=noise.sqrt().tolist())
        for name, v in preds.items():
            x = fn(v)
            xc = x-w @ x
            err2 = w @ (xc-yc)**2
            out[q][name] = dict(nrmse=(err2/torch.clamp(spread2, min=1e-300)).sqrt().tolist(),
                                nrmse_debiased=(torch.clamp(err2-noise, min=0.)/signal2).sqrt().tolist(),
                                corr=((w @ (xc*yc))/torch.sqrt(torch.clamp(w @ xc**2, min=1e-300)*signal2)).tolist(),
                                common=(w @ x-w @ y).tolist())
    return out


# ------------------------------------------------------------------ one compression

class Setup:
    def __init__(self, cfg, part, model, gamma, gen):
        self.cfg, self.part, self.gen = cfg, part, gen
        self.learned = Learned(model, gamma, PTA())
        self.checks = None   # set to a dict to run the feature checks at the next compression


def costs(S, F, w, labels, ex, vp, truth, truth_se, rest, ctx, c, hard, part, setup, e):
    """One-compression error of every grouping on prefixes 1..L, raw and after the additive PTA correction."""
    L = len(rest)
    tot_psi, tot_phi = w @ truth['psi'], w @ truth['phi']
    Ft = tot_phi/tot_psi
    pta_m = {q: w @ vp[q] for q in ('psi', 'phi')}
    ex_m = {q: w @ ex[q] for q in ('psi', 'phi')}
    uniq = {}
    for lab in labels.values():
        for idx in clusters_of(lab):
            if len(idx) > 1:
                uniq.setdefault(key(idx), idx)
    keys = sorted(uniq)
    path_steps = sum(len(uniq[k])+1 for k in keys)*part['coupled_paths']*L
    start = time.perf_counter()
    est = coupled(F, w, [uniq[k] for k in keys], rest, ctx, c, part['coupled_paths'], part['chunk_paths'], setup.gen, hard)
    seconds = time.perf_counter()-start
    est = dict(zip(keys, est))
    m = part['coupled_paths']
    groups, z_exact, d_exact, z_long, d_long, seen, ident = {}, [], [], [], [], set(), 0.
    for name, lab in labels.items():
        reps, mass = combine(S, w, lab)
        vr = PTA()(reps, None, rest, ctx, c, hard)
        er = exact(reps, rest[:e], ctx, c, hard, setup.cfg['probe']['exact_chunk'])
        g_pta = {q: pta_m[q]-mass @ vr[q] for q in ('psi', 'phi')}
        g_ex = {q: ex_m[q]-mass @ er[q] for q in ('psi', 'phi')}
        g = np.zeros((2, L)); var = np.zeros((3, L))
        member, member_var, members = np.zeros(L), np.zeros(L), []
        for c_i, idx in enumerate(clusters_of(lab)):
            if len(idx) < 2:
                continue
            r = est[key(idx)]
            Wc = float(mass[c_i])
            vg = np.maximum(r['g2']-r['g']**2, 0.)/m
            vh = np.maximum(r['h2']-r['h']**2, 0.)/m
            cv = (r['gh']-r['g']*r['h'])/m
            g[0] += Wc*r['g']; g[1] += Wc*r['h']
            var[0] += Wc*Wc*vg; var[1] += Wc*Wc*vh; var[2] += Wc*Wc*cv
            member += Wc*r['m']; member_var += Wc*Wc*np.maximum(r['m2']-r['m']**2, 0.)/m; members.append(idx)
            if key(idx) not in seen:
                # The coupled gap against the exact tree on the exact prefixes (each cluster once).
                seen.add(key(idx))
                p = w[idx]/w[idx].sum()
                level = (p @ ex['psi'][idx]).cpu().numpy()
                ex_c = level-er['psi'][c_i].cpu().numpy()
                zz, dd = zcheck(r['g'][:e]-ex_c, np.sqrt(vg[:e]), level)
                z_exact += zz; d_exact += dd
                ident = max(ident, float(np.max(np.abs(r['m']-r['c']-r['g']))))
        if name == 'geo' and members:
            # Long-horizon check of the members' likelihood ratios: sum_c W_c (member average of psi) from the
            # coupled paths against the independent per-branch truth of the same members.
            ns = torch.cat(members)
            ref = (w[ns] @ truth['psi'][ns]).cpu().numpy()
            ref_var = ((w[ns]**2) @ truth_se[ns]**2).cpu().numpy()
            z_long, d_long = zcheck((member-ref)[e:], np.sqrt(member_var+ref_var)[e:], ref[e:])
        gp = torch.as_tensor(g[0], device=w.device); gn = torch.as_tensor(g[1], device=w.device)
        # Exact prefixes use the exact gaps; Monte Carlo variance only beyond them.
        gp[:e], gn[:e] = g_ex['psi'], g_ex['phi']
        var[:, :e] = 0.
        vF = var[1]-2*Ft.cpu().numpy()*var[2]+Ft.cpu().numpy()**2*var[0]
        tp = tot_psi.cpu().numpy()
        res = dict(
            P_raw=(-gp/tot_psi).tolist(), P_cor=(-(gp-g_pta['psi'])/tot_psi).tolist(),
            F_raw=(-(gn-Ft*gp)/tot_psi).tolist(), F_cor=(-((gn-g_pta['phi'])-Ft*(gp-g_pta['psi']))/tot_psi).tolist(),
            P_se=(np.sqrt(var[0])/tp).tolist(), F_se=(np.sqrt(np.maximum(vF, 0.))/tp).tolist(),
            P_pta=(-g_pta['psi']/tot_psi).tolist(), clusters=len(clusters_of(lab)),
            coupled_P_exact_prefixes=(-torch.as_tensor(g[0][:e], device=w.device)/tot_psi[:e]).tolist())
        groups[name] = res
    # Variance of the difference between two groupings: clusters they share cancel exactly.
    pairs = {}
    for a, b in part['cost_pairs']:
        ka = {key(i) for i in clusters_of(labels[a]) if len(i) > 1}
        kb = {key(i) for i in clusters_of(labels[b]) if len(i) > 1}
        va, mass_a = np.zeros((3, L)), dict(zip(map(key, clusters_of(labels[a])), combine(S, w, labels[a])[1].tolist()))
        mass_b = dict(zip(map(key, clusters_of(labels[b])), combine(S, w, labels[b])[1].tolist()))
        for kk in ka ^ kb:
            r, Wc = est[kk], (mass_a if kk in ka else mass_b)[kk]
            va[0] += Wc*Wc*np.maximum(r['g2']-r['g']**2, 0.)/m
            va[1] += Wc*Wc*np.maximum(r['h2']-r['h']**2, 0.)/m
            va[2] += Wc*Wc*(r['gh']-r['g']*r['h'])/m
        va[:, :e] = 0.
        Fn, tp = Ft.cpu().numpy(), tot_psi.cpu().numpy()
        pairs['%s|%s' % (a, b)] = dict(shared=len(ka & kb), differing=len(ka ^ kb),
                                       P_se=(np.sqrt(va[0])/tp).tolist(),
                                       F_se=(np.sqrt(np.maximum(va[1]-2*Fn*va[2]+Fn**2*va[0], 0.))/tp).tolist())
    # bookkeeping: member average minus representative equals the gap estimate (same paths).
    return dict(groups=groups, pairs=pairs, unique_clusters=len(keys), coupled_seconds=seconds, coupled_path_steps=path_steps,
                checks=dict(coupled_vs_exact_z=summary_z(z_exact, d_exact), coupled_vs_branch_mc_z=summary_z(z_long, d_long),
                            bookkeeping_max_abs=ident))


def zcheck(diff, se, scale):
    """z = diff/se where the Monte Carlo error is resolvable, and the relative difference |diff|/|scale| of every
    entry. With few paths a rare outcome may go unsampled: the sample se is then ~0 although the estimate differs
    slightly, so the z-test alone over-flags; an entry fails only if |z| > 4 and the relative difference > 1e-3."""
    diff, se, scale = (np.asarray(x, dtype=float).ravel() for x in (diff, se, scale))
    rel = np.abs(diff)/np.maximum(np.abs(scale), 1e-300)
    live = se > 1e-10*np.maximum(np.abs(scale), 1e-300)
    z = np.where(live, diff/np.where(live, se, 1.), 0.)
    return list(zip(z[live], rel[live])), list(rel[~live])


def summary_z(zr, det=()):
    det = np.asarray(det, dtype=float)
    z = np.asarray([a for a, _ in zr], dtype=float)
    rel = np.concatenate((np.asarray([b for _, b in zr], dtype=float), det))
    out = dict(count=int(len(z)), deterministic=int(len(det)),
               max_relative=float(rel.max()) if len(rel) else 0.,
               fail=int(sum(1 for a, b in zr if abs(a) > 4 and b > 1e-3)+int((det > 1e-3).sum())))
    if len(z):
        out.update(mean=float(z.mean()), rms=float(np.sqrt((z*z).mean())), median_abs=float(np.median(np.abs(z))),
                   frac_abs_gt4=float((np.abs(z) > 4).mean()), max_abs=float(np.abs(z).max()))
    return out


def inspect(sim, S, w, setup, task):
    cfg, part = setup.cfg, setup.part
    ctx, c, hard, K = sim.ctx, sim.c, sim.hard, sim.K
    F, rest = sim.factors, sim.word[sim.t:]
    L = len(rest)
    e = min(cfg['probe']['exact_rounds'], L)
    timing = {}
    t0 = time.perf_counter()
    geo = coords(S, ctx['aux']['pauli'])
    vp = PTA()(S, F, rest, ctx, c, hard)
    vl = setup.learned(S, F, rest, ctx, c, hard)
    ex = exact(S, rest[:e], ctx, c, hard, cfg['probe']['exact_chunk'])
    timing['predict_exact'] = time.perf_counter()-t0
    t0 = time.perf_counter()
    half = part['truth_paths']//2
    A = mc_summary(branch_paths(F, rest, ctx, c, half, part['chunk_paths'], setup.gen, hard), half)
    Bh = mc_summary(branch_paths(F, rest, ctx, c, half, part['chunk_paths'], setup.gen, hard), half)
    timing['truth_mc'] = time.perf_counter()-t0
    full = {q: (A[q]+Bh[q])/2 for q in ('psi', 'phi')}
    truth, tA, tB = mixed(ex, full, e), mixed(ex, A, e), mixed(ex, Bh, e)
    truth_se = torch.sqrt(A['se_psi']**2+Bh['se_psi']**2)/2
    truth_se[:, :e] = 0.
    # Monte Carlo against the exact tree on the exact prefixes (both halves).
    z_mc, d_mc = [], []
    for h in (A, Bh):
        zz, dd = zcheck((h['psi'][:, :e]-ex['psi'][:, :e]).cpu().numpy(), h['se_psi'][:, :e].cpu().numpy(),
                        ex['psi'][:, :e].cpu().numpy())
        z_mc += zz; d_mc += dd
    preds = dict(pta=vp, ml=vl, truth=truth, truthA=tA, truthB=tB)
    t0 = time.perf_counter()
    labels = {'geo': lloyd_async(geo, w, K, c['clustering_iterations'])}
    blocks = {}
    for s, spec in cfg['settings'].items():
        for src in SOURCES:
            blocks[(s, src)] = block(spec, preds[src], L, w, geo)
            labels[(s, src)] = lloyd_async(torch.cat((geo, blocks[(s, src)]), 1), w, K, c['clustering_iterations'])
    timing['cluster'] = time.perf_counter()-t0
    if setup.checks is not None and not setup.checks:
        setup.checks.update(feature_checks(S, w, F, rest, ctx, c, hard, setup, task, blocks, ex))
    rec = dict(task_id=task['id'], n=task['n'], family=task['family'], prep_id=task['prep_signature'], K=K, t=sim.t, L=L,
               branches=len(w), letter=sim.word[sim.t-1], ell=sim.ell, exact_prefixes=e,
               truth_se=dict(logpsi_median=[float(x) for x in torch.median(A['se_psi']/torch.clamp(A['psi'], min=1e-300), 0).values/math.sqrt(2)],
                             F_median=[float(x) for x in torch.median(A['se_F'], 0).values/math.sqrt(2)]),
               partitions={s: partitions(dict(geo=labels['geo'], **{src: labels[(s, src)] for src in SOURCES}), w)
                           for s in cfg['settings']},
               features=features(dict(pta=vp, ml=vl), truth, tA, tB, w))
    t0 = time.perf_counter()
    cost_labels = dict(geo=labels['geo'], **{'%s:%s' % (s, src): labels[(s, src)]
                                           for s in part['cost_settings'] for src in COST_SOURCES})
    rec['costs'] = costs(S, F, w, cost_labels, ex, vp, truth, truth_se, rest, ctx, c, hard, part, setup, e)
    timing['costs'] = time.perf_counter()-t0
    rec['checks'] = dict(mc_vs_exact_z=summary_z(z_mc, d_mc), **rec['costs'].pop('checks'))
    rec['timing'] = timing
    rec['_geo_labels'] = labels['geo']
    return rec


def feature_checks(S, w, F, rest, ctx, c, hard, setup, task, blocks, ex):
    """The look-ahead blocks equal those of the frozen classes: hr_goal.LookAhead ('orig'),
    lt_lookahead.StdLookAhead ('s1_long'), lookahead_oracle.OracleLookAhead ('s1_k3'; exact tree for the truth)."""
    from hr_goal import LookAhead
    from lt_lookahead import StdLookAhead
    from lo_eval import OracleLookAhead
    cfg, out = setup.cfg, {}
    model = setup.learned.model
    o = cfg['settings']['orig']
    a = LookAhead('pta', o['lengths'], o['weight'])(S, F, rest, ctx, c, hard)
    b = LookAhead('learned', o['lengths'], o['weight'], model, task['gamma'])(S, F, rest, ctx, c, hard)
    out['orig_pta_bitwise'] = bool(torch.equal(a, blocks[('orig', 'pta')]))
    out['orig_learned_bitwise'] = bool(torch.equal(b, blocks[('orig', 'ml')]))
    s = cfg['settings']['s1_long']
    for src, name in (('pta', 'pta'), ('learned', 'ml')):
        la = StdLookAhead(src, s['share'], s['lengths'], 'psiF', model if src == 'learned' else None, task['gamma'])
        la.weights = w
        out['s1_long_%s_bitwise' % name] = bool(torch.equal(la(S, F, rest, ctx, c, hard), blocks[('s1_long', name)]))
    k = cfg['settings']['s1_k3']
    for src, name in (('pta', 'pta'), ('exact', 'truth')):
        la = OracleLookAhead(src, k['share'], k['lengths'], chunk=cfg['probe']['exact_chunk'])
        la.weights = w
        diff = float((la(S, F, rest, ctx, c, hard)-blocks[('s1_k3', name)]).abs().max())
        out['s1_k3_%s_max_abs' % name] = diff
    out['task_id'], out['t'] = task['id'], len(task['word'])-len(rest)
    ok = all(out[x] for x in ('orig_pta_bitwise', 'orig_learned_bitwise', 's1_long_pta_bitwise', 's1_long_ml_bitwise'))
    if not ok or out['s1_k3_pta_max_abs'] != 0. or out['s1_k3_truth_max_abs'] > 1e-10:
        raise AssertionError('Look-ahead blocks differ from the frozen classes: %s' % out)
    return out


class Probe(qa_sim.QoISim):
    """Frozen M0 trajectory; at the chosen compressions `inspect` runs first on the pre-merge branches and the
    geometric grouping it computed is checked against the merge that M0 then performs."""
    def __init__(self, K, word, ctx, c, states, weights, hard, events, setup, task):
        # QoISim keeps its own compression log in self.events; the probed rounds live in self.probe_at.
        self.probe_at, self.setup, self.task, self.records = set(events), setup, task, []
        super().__init__(K, word, ctx, c, states, weights, None, hard)

    def compress(self, states, w):
        if self.t not in self.probe_at or len(w) <= self.K:
            return super().compress(states, w)
        before = self.factors
        rec = inspect(self, states, w, self.setup, self.task)
        labels = rec.pop('_geo_labels')
        out = super().compress(states, w)
        rec['checks']['geo_equals_M0_merge'] = bool(torch.equal(gf.combine(before, w, labels), self.factors))
        if not rec['checks']['geo_equals_M0_merge']:
            raise AssertionError('Geometric grouping differs from the M0 merge')
        self.records.append(rec)
        return out


def plan(cfg, smoke):
    """Probe settings (the smoke overrides K, events and path counts); the grouping pairs whose cost
    differences get their own standard errors."""
    part = dict(cfg['probe'])
    if smoke:
        part.update({k: v for k, v in cfg['smoke'].items() if k != 'picks'})
    pairs = []
    for s in part['cost_settings']:
        g = ['%s:%s' % (s, x) for x in COST_SOURCES]
        pairs += [('geo', x) for x in g]+[(g[0], g[1]), (g[0], g[2]), (g[1], g[2])]
    part['cost_pairs'] = pairs
    return part


def run_task(cx, cfg, part, task, model, K, seed, checks=None):
    states, weights, ctx = context(cx, task)
    gen = torch.Generator(device=cx.device).manual_seed(int(seed))
    setup = Setup(cfg, part, model, task['gamma'], gen)
    setup.checks = checks
    sim = Probe(K, task['word'], ctx, cx.c, states, weights, cx.hard, part['events'], setup, task)
    for a in task['word'][:max(part['events'])]:
        sim.advance(a)
    return sim.records

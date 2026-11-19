"""Residual merge (HANDOFF 5j item 22.28, M0): pairwise merge gaps, budgeted agglomeration and the adaptive
simulator. No training; final data are never accessed.

Merging atoms a, b (weights w_a, w_b) into their weighted mean m changes every later prefix k by the gap
w_a f_k(a) + w_b f_k(b) - (w_a+w_b) f_k(m) of the true future response f (psi: success; phi: success and GHZ).
A merged atom is just a new state, so merging it again telescopes: any sequence of pairwise merges costs
exactly the sum of its pairwise gaps (checked in the smoke). The additive PTA correction adds back the same
gap of the PTA response; what remains is the gap of f - f_PTA, the residual gap the model will learn (M1-M2).

The simulator is the frozen qa_sim.QoISim (exact physics, mean-preserving merge, PTA correction bookkeeping,
physical_calls counted per propagated atom); only the partition, and with it the number of atoms kept,
comes from a policy.
"""
import rm_runtime  # noqa: F401  (paths, require_compute)
import time
import numpy as np
import torch
import ghz_factors as gf
import qa_sim
from ghz_physics import coords, ghz_diagonal
from ghz_merge import sqdist, features, combine
from g1_core import lloyd_async, check_deadline
from qa_sources import PTA
from lg_probe import exact, coupled

QUANTITIES = ('psi', 'phi')


# ------------------------------------------------------------------ candidate pairs and costs

@torch.no_grad()
def knn_pairs(x, k, chunk=2048):
    """Unique unordered pairs (i < j) of every row of x with its k nearest neighbours."""
    n = len(x)
    if n < 2:
        return torch.empty((0, 2), dtype=torch.long, device=x.device)
    k = min(k, n-1)
    parts = []
    for s in range(0, n, chunk):
        d = sqdist(x[s:s+chunk], x)
        rows = torch.arange(len(d), device=x.device)
        d[rows, s+rows] = float('inf')
        nb = torch.topk(d, k, dim=1, largest=False).indices
        i = (s+rows).repeat_interleave(k)
        j = nb.flatten()
        parts.append(torch.stack((torch.minimum(i, j), torch.maximum(i, j)), 1))
    return torch.unique(torch.cat(parts), dim=0)


class GeoCost:
    """C1: Ward increment of merging two atoms in Pauli coordinates, W_a W_b/(W_a+W_b) |x_a - x_b|^2."""
    name = 'geo'

    def __call__(self, atoms, I, J):
        wI, wJ = atoms['w'][I], atoms['w'][J]
        return wI*wJ/(wI+wJ)*((atoms['x'][I]-atoms['x'][J])**2).sum(1)


# ------------------------------------------------------------------ budgeted agglomeration

@torch.no_grad()
def agglomerate(S, w, x, cost, budget, cap, knn, max_rounds, record=False):
    """Greedy pairwise merging of the N pre-merge branches (states S, weights w, Pauli coordinates x).

    Each round pairs every atom with its k nearest neighbours in x (a merged atom is the weighted mean, so its
    state and coordinates are means too), prices the candidate pairs with `cost` and takes disjoint pairs in
    increasing cost while the compression's accumulated cost stays within `budget`. If more than `cap` atoms
    would remain (resource limit), the cheapest further disjoint pairs are merged regardless of the budget and
    counted separately. Stops when a round merges nothing. Returns contiguous labels over the N branches and a
    log; with record=True the log also holds the atom table and the merge sequence (telescoping check)."""
    N, dev = len(w), w.device
    size = 2*N
    AS = S.new_zeros((size,)+tuple(S.shape[1:]))
    AS[:N] = S
    AW = w.new_zeros(size)
    AW[:N] = w
    AX = x.new_zeros((size, x.shape[1]))
    AX[:N] = x
    parent = torch.arange(size, device=dev)
    alive = torch.arange(N, device=dev)
    top, spent, forced_cost, merges, forced, rounds, seq = N, 0., 0., 0, 0, 0, []
    while len(alive) > 1 and rounds < max_rounds:
        need = len(alive)-cap
        P = knn_pairs(AX[alive], knn)
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
                break   # costs only grow from here and the cap is met
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
        AW[new] = Wn
        parent[I] = new
        parent[J] = new
        if record:
            seq.append((I, J, new))
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
    info = dict(before=N, after=int(labels.max())+1, merges=merges, forced=forced, spent=spent,
                forced_cost=forced_cost, rounds=rounds)
    if record:
        info['record'] = dict(S=AS[:top], w=AW[:top], seq=seq)
    return labels, info


class Agglomerative:
    """Policy: budgeted agglomeration with a fixed budget per compression (HANDOFF 22.28, rule 'per round')."""
    def __init__(self, cost, budget, cap, knn, max_rounds, record_at=()):
        self.cost, self.budget, self.cap, self.knn, self.max_rounds = cost, float(budget), int(cap), int(knn), int(max_rounds)
        self.record_at, self.records = set(record_at), {}

    def __call__(self, sim, states, w):
        x = coords(states, sim.ctx['aux']['pauli'])
        labels, info = agglomerate(states, w, x, self.cost, self.budget, self.cap, self.knn, self.max_rounds,
                                   record=sim.t in self.record_at)
        if 'record' in info:
            self.records[sim.t] = dict(info.pop('record'), states=states, branch_w=w, labels=labels,
                                       rest=sim.word[sim.t:])
        return labels, dict(policy=self.cost.name, budget=self.budget, **info)


class LloydPolicy:
    """The frozen geometric merge_K partition (for the equivalence check of the simulator)."""
    def __init__(self, K):
        self.K = K

    def __call__(self, sim, states, w):
        if len(w) <= self.K:
            return torch.arange(len(w), device=w.device), dict(policy='lloyd', before=len(w), after=len(w))
        labels = lloyd_async(features('geo', states, sim.ctx['aux']), w, self.K, sim.c['clustering_iterations'])
        return labels, dict(policy='lloyd', before=len(w), after=int(labels.max())+1)


# ------------------------------------------------------------------ adaptive simulator

class AdaptiveSim(qa_sim.QoISim):
    """qa_sim.QoISim whose partition comes from `policy` (K is not fixed). The frozen compression and
    correction code runs unchanged: its single clustering call is answered with the policy's labels."""
    def __init__(self, policy, word, ctx, c, states, weights, sources=None, hard=None):
        # QoISim keeps its own compression log in self.events; the policy log is self.merge_log.
        self.policy, self.merge_log = policy, []
        super().__init__(0, word, ctx, c, states, weights, sources, hard)

    def compress(self, states, w):
        if len(w) <= 1:
            return self.validate(states, w)
        check_deadline(self.hard)
        start = time.perf_counter()
        labels, info = self.policy(self, states, w)
        self.merge_log.append(dict(t=self.t, seconds=time.perf_counter()-start, **info))
        if int(labels.max())+1 == len(w):
            return self.validate(states, w)
        frozen = qa_sim.lloyd_async
        qa_sim.lloyd_async = lambda *args, **kwargs: labels
        try:
            return super().compress(states, w)
        finally:
            qa_sim.lloyd_async = frozen


def estimates(sim, word, weights, rho, ell, ctx):
    """Readouts of hr_goal.rollout / qa_sim.rollout, operation for operation."""
    log_raw = weights.new_tensor(ell)
    raw = {'P': log_raw.exp()}
    raw['N'] = raw['P']*ghz_diagonal(rho, ctx['aux']['ghz'])[:, 0]
    out = {'M0': dict(raw)}
    nan = torch.full_like(log_raw, float('nan'))
    for name in sim.sources:
        for method in qa_sim.ESTIMATORS:
            a, est = qa_sim.arm(name, method), {}
            for q, _ in qa_sim.QUANTITIES:
                if (a, q) in sim.failed:
                    est[q] = nan.clone()
                    continue
                if method == 'add':
                    value = raw[q]+sim.add[name][q]
                elif q == 'P':
                    value = (log_raw+sim.logm[name][q]).exp()
                else:
                    value = raw[q]*sim.logm[name][q].exp()
                if not bool(torch.isfinite(value).all()):
                    sim.fail(name, method, q, len(word), 'Nonfinite corrected value')
                    value = nan.clone()
                est[q] = value
            out[a] = est
    return out


def adaptive_rollout(policy, word, ctx, c, states, weights, sources=None, hard=None):
    """Returns sim, rho[H,d,d], ell, estimates[arm] = dict(P, N)."""
    sim = AdaptiveSim(policy, word, ctx, c, states, weights, sources, hard)
    rho, ell = [], []
    for a in word:
        r, l = sim.advance(a)
        rho.append(r)
        ell.append(l)
    rho = torch.stack(rho)
    return sim, rho, ell, estimates(sim, word, weights, rho, ell, ctx)


class Capture(qa_sim.QoISim):
    """Frozen M0 trajectory (geometric merge_K) that keeps the pre-merge branches at chosen rounds."""
    def __init__(self, K, word, ctx, c, states, weights, hard, at):
        self.at, self.kept = set(at), {}
        super().__init__(K, word, ctx, c, states, weights, None, hard)

    def compress(self, states, w):
        if self.t in self.at:
            self.kept[self.t] = dict(states=states, w=w, factors=self.factors, rest=self.word[self.t:])
        return super().compress(states, w)


# ------------------------------------------------------------------ pairwise gaps (labels)

@torch.no_grad()
def pair_labels(F, w, pairs, rest, ctx, c, hard, lab, gen):
    """Gaps of merging each pair (i, j) of branches (square-root factors F, weights w) into its weighted mean,
    per unit merged weight, on prefixes 1..L of `rest`.

    True gaps are exact on the first e = lab['exact_rounds'] prefixes (exact outcome tree) and coupled
    likelihood-ratio Monte Carlo beyond (lg_probe.coupled, lab['coupled_paths'] paths per pair); PTA gaps
    are exact. Returns numpy arrays [pairs, L]: g_psi, g_phi (true), se_psi, se_phi (0 on exact prefixes),
    pta_psi, pta_phi, r_psi = g_psi - pta_psi, r_phi (the residual gaps), level_psi (members' average psi),
    plus the raw exact and coupled parts on every prefix for the checks, and timings."""
    L, e, dev = len(rest), min(lab['exact_rounds'], len(rest)), w.device
    I, J = pairs[:, 0], pairs[:, 1]
    W = w[I]+w[J]
    lam = w[I]/W
    S = gf.density(F)
    reps = lam[:, None, None]*S[I]+(1-lam)[:, None, None]*S[J]
    used = torch.unique(torch.cat((I, J)))
    pos = torch.full((len(w),), -1, dtype=torch.long, device=dev)
    pos[used] = torch.arange(len(used), device=dev)
    nodes = torch.cat((S[used], reps))
    nu = len(used)

    def avg(v):
        return lam[:, None]*v[pos[I]]+(1-lam)[:, None]*v[pos[J]]

    def gap(v):
        return avg(v)-v[nu:]

    t0 = time.perf_counter()
    vp = PTA()(nodes, None, rest, ctx, c, hard)
    t1 = time.perf_counter()
    ex = exact(nodes, rest[:e], ctx, c, hard, lab['exact_chunk'])
    t2 = time.perf_counter()
    est = coupled(F, w, [pairs[k] for k in range(len(pairs))], rest, ctx, c, lab['coupled_paths'],
                  lab['chunk_paths'], gen, hard)
    t3 = time.perf_counter()
    m = lab['coupled_paths']

    def stack(key):
        return np.stack([r[key] for r in est])

    g, h = stack('g'), stack('h')
    se_g = np.sqrt(np.maximum(stack('g2')-g**2, 0.)/m)
    se_h = np.sqrt(np.maximum(stack('h2')-h**2, 0.)/m)
    level = stack('m')
    bookkeeping = float(np.max(np.abs(stack('m')-stack('c')-g)))
    npy = lambda t: t.cpu().numpy()  # noqa: E731
    ex_psi, ex_phi = npy(gap(ex['psi'])), npy(gap(ex['phi']))
    lvl_psi, lvl_phi = npy(avg(ex['psi'])), npy(avg(ex['phi']))
    gp, gn, sp, sn, lv = g.copy(), h.copy(), se_g.copy(), se_h.copy(), level.copy()
    gp[:, :e], gn[:, :e], sp[:, :e], sn[:, :e], lv[:, :e] = ex_psi, ex_phi, 0., 0., lvl_psi
    tp, tn = npy(gap(vp['psi'])), npy(gap(vp['phi']))
    pta_values = {'%s_%s' % (who, q): npy(v) for q in QUANTITIES
                  for who, v in (('a', vp[q][pos[I]]), ('b', vp[q][pos[J]]), ('rep', vp[q][nu:]))}
    return dict(g_psi=gp, g_phi=gn, se_psi=sp, se_phi=sn, pta_psi=tp, pta_phi=tn, r_psi=gp-tp, r_phi=gn-tn,
                level_psi=lv, lam=npy(lam), W=npy(W), exact_prefixes=e, pta_values=pta_values,
                raw=dict(coupled_g=g, coupled_h=h, se_g=se_g, se_h=se_h, exact_psi=ex_psi, exact_phi=ex_phi,
                         level_exact_psi=lvl_psi, level_exact_phi=lvl_phi),
                bookkeeping_max_abs=bookkeeping,
                seconds=dict(pta=t1-t0, exact=t2-t1, coupled=t3-t2),
                coupled_path_steps=int(len(pairs)*3*m*L))


@torch.no_grad()
def telescoping(rec, ctx, c, hard, lab):
    """Sum of the pairwise gaps of every merge in the recorded sequence against the gap of the final clusters
    computed directly from the original branches, for the exact (first prefixes) and the PTA responses."""
    AS, AW, seq = rec['S'], rec['w'], rec['seq']
    S, w, labels, rest = rec['states'], rec['branch_w'], rec['labels'], rec['rest']
    N, e = len(w), min(lab['exact_rounds'], len(rest))
    m = int(labels.max())+1
    Wc = torch.zeros(m, dtype=w.dtype, device=w.device).index_add_(0, labels, w)
    reps = combine(S, w, labels)[0]
    out = dict(merges=sum(len(x[0]) for x in seq), rounds=len(seq), clusters=m, branches=N)
    for name, fn in (('pta', lambda s: PTA()(s, None, rest, ctx, c, hard)),
                     ('exact', lambda s: exact(s, rest[:e], ctx, c, hard, lab['exact_chunk']))):
        va, vr = fn(AS), fn(reps)
        for q in QUANTITIES:
            v = va[q]
            left = sum((AW[I][:, None]*v[I]+AW[J][:, None]*v[J]-AW[new][:, None]*v[new]).sum(0) for I, J, new in seq)
            right = w @ v[:N]-Wc @ vr[q]
            out['%s_%s_max_abs' % (name, q)] = float((left-right).abs().max())
            out['%s_%s_scale' % (name, q)] = float(right.abs().max())
    return out

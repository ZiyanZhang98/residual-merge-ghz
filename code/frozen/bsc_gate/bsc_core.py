"""BSC core (HANDOFF 5i): exact expansion, unbiased branch-sampling compression and the K scan.

Reused, not re-implemented: the exact same-history step and its numerical rules (ghz_physics),
the G0/G1a clustering (g1_core.lloyd_async, bit-identical to ghz_merge.lloyd), Pauli coordinates,
the PTA letters, the strict scorer, the K_needed scan and the deadline rule (g1_core).
New here: `expand` (ghz_physics.step that also returns parent indices; the check stage requires
identical output), unbiased representative selection, the FRI-style heavy-branch rule (Lim &
Weare 2017), systematic resampling, the groupings (geometric, sorted by a scalar key, exact future
response) and the exact future-response and PTA keys.
"""
import hashlib
import math
import time
import torch
from ghz_physics import C, dag, trace, check_ensemble, coords
from ghz_merge import onehot
from g1_core import lloyd_async, scored_run, scan_K, check_deadline

ARMS = ('S0', 'S1', 'S2', 'S3', 'S4', 'O1', 'O2')
HEAVY = ('S3', 'S4', 'O1', 'O2')


def seed_of(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:15], 16)


def generator(device, *parts):
    """Independent, reproducible random stream per (task, arm, K, replicate, ...)."""
    g = torch.Generator(device=device)
    g.manual_seed(seed_of(*parts))
    return g


def expand(states, weights, letter, c):
    """ghz_physics.step (same operations, same order, same numerical rules) that also returns the
    parent index of every kept child."""
    ch, d = c['chunk'], states.shape[-1]
    raw = torch.cat([letter.raw(states[i:i+ch]) for i in range(0, len(states), ch)])
    p = trace(raw)
    if float(p.min()) < -c['physical_tolerance']:
        raise FloatingPointError('Negative outcome probability %.3e' % float(p.min()))
    w = (weights[:, None]*p).reshape(-1)
    parent = torch.arange(len(states), device=states.device).repeat_interleave(raw.shape[1])
    keep = w > 0
    raw, w, parent = raw.reshape(-1, d, d)[keep], w[keep], parent[keep]
    mass = float(w.sum())
    if not mass > 0:
        raise FloatingPointError('Zero acceptance')
    anti = float((raw-dag(raw)).abs().max())/2
    if not anti <= c['antihermitian_max']:
        raise FloatingPointError('Anti-Hermitian part %.3e exceeds round-off bound; not cleaned' % anti)
    raw = (raw+dag(raw))/2
    return raw/trace(raw)[:, None, None], w/mass, mass, parent


def representatives(w, labels, gen, score=None, mix=0.):
    """One real branch per group, unbiased: E[sum_g W'_g delta_{rho_J(g)}] = sum_j w_j delta_{rho_j}.

    Untwisted (score None): J ~ w_j/W_g and W'_g = W_g, so the total weight is conserved.
    Twisted: J ~ q_j = (1-mix) w_j s_j / sum_g(w s) + mix w_j/W_g (defensive mixture; plain w_j/W_g
    where the group's sum of w s is 0), W'_g = w_J/q_J. Returns (indices into w, new weights).
    """
    labels = torch.unique(labels, return_inverse=True)[1]
    m = int(labels.max())+1
    order = torch.argsort(labels, stable=True)
    lab = labels[order]
    oh = onehot(labels, m, w.dtype)
    W = oh @ w
    q = w[order]/W[lab]
    if score is not None:
        sw = w*score
        S = oh @ sw
        ok = S[lab] > 0
        q = torch.where(ok, (1-mix)*sw[order]/torch.where(ok, S[lab], torch.ones_like(S[lab])) + mix*q, q)
    cs = torch.cumsum(q, 0)
    groups = torch.arange(m, device=w.device)
    start = torch.searchsorted(lab, groups)
    end = torch.searchsorted(lab, groups, right=True)
    base = cs[start]-q[start]
    u = torch.rand(m, generator=gen, device=w.device, dtype=w.dtype)
    pos = torch.searchsorted(cs, base+u*(cs[end-1]-base), right=True)
    pos = torch.minimum(torch.maximum(pos, start), end-1)
    J = order[pos]
    return J, (W if score is None else w[J]/q[pos])


def systematic(w, K, gen):
    """Systematic resampling to K equal-weight copies (unbiased counts); the total weight is conserved."""
    cs = torch.cumsum(w, 0)
    total = float(cs[-1])
    u = torch.rand(1, generator=gen, device=w.device, dtype=w.dtype)
    pos = (u+torch.arange(K, device=w.device, dtype=w.dtype))*(total/K)
    idx = torch.clamp(torch.searchsorted(cs, pos, right=True), max=len(w)-1)
    return idx, torch.full((K,), total/K, dtype=w.dtype, device=w.device)


def heavy(w, K):
    """FRI-style rule (Lim & Weare 2017): keep exactly every branch whose weight is at least the
    average weight per remaining slot, leaving at least one slot for the rest. Boolean mask."""
    ws, order = torch.sort(w, descending=True)
    slots = K-torch.arange(len(ws), device=w.device)
    rest = w.sum()-(torch.cumsum(ws, 0)-ws)
    ok = (ws*slots >= rest) & (slots > 1)
    m = int(torch.cumprod(ok.to(torch.int64), 0).sum())
    mask = torch.zeros(len(w), dtype=torch.bool, device=w.device)
    mask[order[:m]] = True
    return mask


def sorted_groups(key, w, m):
    """At most m contiguous groups of nearly equal weight along the order of a scalar key."""
    order = torch.argsort(key, stable=True)
    ws = w[order]
    start = torch.cumsum(ws, 0)-ws
    g = torch.clamp(torch.floor(start/(ws.sum()/m)).to(torch.int64), min=0, max=m-1)
    labels = torch.empty_like(g)
    labels[order] = g
    return torch.unique(labels, return_inverse=True)[1]


def future_response(states, word, depths, letters, aux, c, chunk_leaves, factors=None):
    """Exact unnormalized future responses F_{word[:k]}(rho) (all accepted-outcome trees, same-history
    pairing) for k in depths, as [N, 1+(4^n-1)] blocks [tr F, Pauli coordinates of F]; and the number
    of exact one-round evaluations spent."""
    ks = [k for k in depths if k <= len(word)]
    if not ks:
        return {}, 0
    kmax, per = max(ks), 1
    for a in word[:kmax]:
        per *= len(letters[a].accept)
    ch, d, evals = max(1, chunk_leaves//per), states.shape[-1], 0
    out = {k: [] for k in ks}
    for i in range(0, len(states), ch):
        cur = states[i:i+ch]
        factor = None if factors is None else factors[i:i+ch]
        m = len(cur)
        prob = torch.ones(m, 1, dtype=torch.float64, device=states.device)
        cur = cur[:, None]
        for j, a in enumerate(word[:kmax], 1):
            flat = cur.reshape(-1, d, d)
            evals += len(flat)
            if factor is None:
                raw = torch.cat([letters[a].raw(flat[s:s+c['chunk']]) for s in range(0, len(flat), c['chunk'])])
                p = torch.clamp(trace(raw), min=0.)
                child = raw/torch.where(p > 0, p, torch.ones_like(p))[..., None, None]
            else:
                from ghz_factors import children, density
                factor, p = children(factor.reshape(-1, d, d), letters[a], c['chunk'])
                child = density(factor)
            prob = (prob.reshape(-1, 1)*p).reshape(m, -1)
            cur = child.reshape(m, -1, d, d)
            if j in out:
                F = torch.einsum('ml,mlij->mij', prob.to(C), cur)
                out[j].append(torch.cat([trace(F)[:, None], coords(F, aux['pauli'])], 1))
    return {k: torch.cat(v) for k, v in out.items()}, evals


def la_features(R, w):
    """Future-response grouping key: per look-ahead length, [tr F, Pauli(F)] divided by the ensemble's
    weighted mean success probability at that length (one physical scalar per block, no per-feature
    variance normalisation; HANDOFF 5f numerical lesson)."""
    blocks = []
    for k in sorted(R):
        psi = float((w*R[k][:, 0]).sum()/w.sum())
        blocks.append(R[k]/(psi if psi > 0 else 1.))
    return torch.cat(blocks, 1)


def pta_logpsi(states, word, pta, c):
    """log of the PTA success probability of the whole remaining word (closed-form physics prior)."""
    logp = torch.zeros(len(states), dtype=torch.float64, device=states.device)
    s = states
    for a in word:
        raw = torch.cat([pta[a].raw(s[i:i+c['chunk']]) for i in range(0, len(s), c['chunk'])])[:, 0]
        p = trace(raw)
        logp = logp+torch.log(torch.clamp(p, min=1e-300))
        s = raw/torch.where(p > 0, p, torch.ones_like(p))[:, None, None]
    return logp


class BSC:
    """Branch-sampling compression on the exact same-history step (HANDOFF 5i); interface of g1_core.Sim.

    advance(a) returns the estimate at the new prefix BEFORE compression (S1-O2: the exact mean over all
    expanded children, unbiased given the parents; S0: the standard bootstrap estimate after sampling
    one child per particle), then compresses to <= K atoms:
      S0 ordinary bootstrap SMC (one outcome per particle from all outcomes, rejection kills it;
         systematic resampling to K particles)
      S1 exact expansion + systematic resampling
      S2 groups = lloyd on Pauli coordinates (the merge_K clustering) + random representative
      S3 S2 + heavy branches kept exactly
      S4 heavy + groups contiguous along log psi_PTA(remaining word)
      O1 heavy + lloyd on the exact future response (oracle)
      O2 O1 + selection twisted by the exact look-ahead success probability (oracle)
    O1/O2 fall back to geometric groups at the last step (no remaining protocol).
    """
    def __init__(self, K, arm, word, ctx, c, b, gen, states, weights, factors=None):
        if arm not in ARMS:
            raise ValueError('Unknown arm '+arm)
        self.K, self.arm, self.word, self.c, self.b, self.gen = K, arm, word, c, b, gen
        self.L, self.pta, self.aux = ctx['letters'], ctx['pta'], ctx['aux']
        self.t, self.ell, self.conservation = 0, 0., 0.
        self.calls = dict(atoms=0, lookahead=0)
        self.factors = factors
        self.states, self.weights = self.compress(states, weights/weights.sum())

    def advance(self, a):
        if a != self.word[self.t]:
            raise ValueError('Simulator was built for a different protocol')
        self.calls['atoms'] += len(self.weights)
        if self.factors is None:
            children, w, mass, parent = expand(self.states, self.weights, self.L[a], self.c)
        else:
            from ghz_factors import expand as factor_expand
            children, w, mass, parent, self.factors = factor_expand(self.factors, self.weights, self.L[a], self.c)
        if not math.isfinite(mass) or not 0 < mass <= 1+1e-10:
            raise FloatingPointError('Invalid exact acceptance %r' % mass)
        self.t += 1
        self.ell += math.log(mass)
        if self.arm == 'S0':
            # ordinary bootstrap: each particle draws one outcome from ALL outcomes; a rejected outcome
            # kills it. Equivalently: survive with its acceptance a, then one accepted child ~ p_y, weight W/a.
            J, W = representatives(w, parent, self.gen)
            a = torch.clamp(W*mass/self.weights[torch.unique(parent)], max=1.)
            alive = torch.rand(len(W), generator=self.gen, device=W.device, dtype=W.dtype) < a
            if not bool(alive.any()):
                raise FloatingPointError('All bootstrap particles rejected')
            children, w = children[J[alive]], (W/a)[alive]
            if self.factors is not None:
                self.factors = self.factors[J[alive]]
            total = float(w.sum())
            self.ell += math.log(total)
            w = w/total
        estimate = torch.einsum('n,nij->ij', w.to(C), children), self.ell
        self.states, self.weights = self.compress(children, w)
        return estimate

    def compress(self, states, w):
        K, arm = self.K, self.arm
        if arm in ('S0', 'S1'):
            if arm == 'S1' and len(w) <= K:
                return self.finish(states, w, w)
            idx, nw = systematic(w, K, self.gen)
            if self.factors is not None:
                self.factors = self.factors[idx]
            return self.finish(states[idx], nw, w)
        if len(w) <= K:
            return self.finish(states, w, w)
        kept = heavy(w, K) if arm in HEAVY else torch.zeros(len(w), dtype=torch.bool, device=w.device)
        k_idx, r_idx = kept.nonzero()[:, 0], (~kept).nonzero()[:, 0]
        slots = K-len(k_idx)
        J, nw = r_idx, w[r_idx]
        if len(r_idx) > slots:
            j, nw = self.group_and_pick(states[r_idx], w[r_idx], slots,
                                        factors=None if self.factors is None else self.factors[r_idx])
            J = r_idx[j]
        idx = torch.cat([k_idx, J])
        if self.factors is not None:
            self.factors = self.factors[idx]
        return self.finish(states[idx], torch.cat([w[k_idx], nw]), w)

    def group_and_pick(self, sr, wr, slots, factors=None):
        it, la = self.c['clustering_iterations'], self.b['lookahead']
        remaining = self.word[self.t:]
        score = None
        if self.arm == 'S4':
            labels = sorted_groups(pta_logpsi(sr, remaining, self.pta, self.c), wr, slots)
        elif self.arm in ('O1', 'O2') and remaining:
            R, ev = future_response(sr, remaining, la['depths'], self.L, self.aux, self.c, la['chunk_leaves'],
                                    factors=factors)
            self.calls['lookahead'] += ev
            labels = lloyd_async(la_features(R, wr), wr, slots, it)
            if self.arm == 'O2':
                score = R[max(R)][:, 0]
        else:
            labels = lloyd_async(coords(sr, self.aux['pauli']), wr, slots, it)
        return representatives(wr, labels, self.gen, score, la['twist_mix'])

    def finish(self, states, nw, w_before):
        total = float(nw.sum())
        if not (math.isfinite(total) and total > 0):
            raise FloatingPointError('Invalid compressed weight %r' % total)
        if self.arm != 'O2':
            self.conservation = max(self.conservation, abs(total-float(w_before.sum())))
        self.ell += math.log(total)
        nw = nw/total
        check_ensemble(states, nw, self.c['physical_tolerance'])
        return states, nw


def evaluate_K(make, word, ref, eps, M, need, hard):
    """Independent replicates at one K, stopped as soon as the decision is known: passes iff at least
    `need` of M replicates pass the strict all-prefix criterion (g1_core.scored_run, physical failure
    = not passed)."""
    runs, passes = [], 0
    for r in range(M):
        check_deadline(hard)
        box = {}

        def build(r=r):
            box['sim'] = make(r)
            return box['sim']
        out = scored_run(build, word, ref, eps)
        sim = box.get('sim')
        out.update(atoms=sim.calls['atoms'] if sim else 0, lookahead=sim.calls['lookahead'] if sim else 0,
                   conservation=sim.conservation if sim else 0.)
        runs.append(out)
        passes += int(out['passed'])
        if passes >= need or len(runs)-passes > M-need:
            break
    return passes >= need, runs


def summarize_runs(runs, ok):
    n = len(runs)
    return dict(passed=bool(ok), replicates=n, passes=sum(int(r['passed']) for r in runs),
                fail_steps=[r['fail_step'] for r in runs if not r['passed']],
                max_td=max(r['max_td'] for r in runs), max_ep=max(r['max_ep'] for r in runs),
                seconds=sum(r['seconds'] for r in runs)/n, atoms=sum(r['atoms'] for r in runs)/n,
                lookahead=sum(r['lookahead'] for r in runs)/n, conservation=max(r['conservation'] for r in runs),
                errors=sorted({r['error'] for r in runs if r.get('error')})[:3])


def scan_arm(arm, states, weights, word, ref, ctx, c, b, g1, task_id, device, hard):
    """K_needed(eps, 95%) of one stochastic arm with the G0 K grid and the G1a window rule (g1_core.scan_K)."""
    M, need, records = b['replicates']['M'], b['replicates']['need'], {}

    def joint(K):
        t0 = time.perf_counter()
        make = lambda r, K=K: BSC(K, arm, word, ctx, c, b, generator(device, b['seed'], task_id, arm, K, r),  # noqa: E731
                                  states, weights)
        ok, runs = evaluate_K(make, word, ref, g1['eps'], M, need, hard)
        records[K] = dict(summarize_runs(runs, ok), wall=time.perf_counter()-t0)
        return ok
    K, _ = scan_K(b.get('K_grid') or c['K_grid'], joint, g1['K_window'], g1['K_unreached'])   # smoke: coarse grid
    return dict(K_needed=K, runs=[dict(K=k, **records[k]) for k in sorted(records)])

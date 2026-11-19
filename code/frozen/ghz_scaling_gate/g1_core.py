"""G1a core (HANDOFF 5h, G1精简版): merge simulators, references, strict scoring and the K_needed scan.

Physics, merge rule, features and numerical rules are the G0 ones (ghz_physics.step,
ghz_merge.combine/features, check_ensemble). The only new numerical piece is `lloyd_async`:
the G0 torch clustering port with its farthest-point seeding written without a host
synchronisation per seed (the G0 bottleneck). It performs the same tensor operations on the
same values and must reproduce `ghz_merge.lloyd` partitions bit for bit (check stage).
"""
import math
import time
from pathlib import Path
import numpy as np
import torch
from ghz_physics import C, Letter, dag, step, check_ensemble, perturb, ghz_letters, ghz_diagonal, pauli_stack, ghz_basis
from ghz_merge import sqdist, onehot, combine, features
from ghz_data import prepare

SEED_STOP = 1e-28   # ghz_merge.lloyd: seeding stops when the best weighted distance is <= this
# Numerical failures of a run (counted as "not passed" / reference failed). Anything else, e.g. a
# CUDA out-of-memory error, is an ERROR row (infrastructure, not physics).
NUMERIC = (FloatingPointError, OverflowError, torch.linalg.LinAlgError)


class Deadline(Exception):
    """The hard deadline passed inside a task: the task is recorded as NOT_FINISHED_DEADLINE."""


def check_deadline(hard):
    if hard is not None and time.time() > hard:
        raise Deadline('hard deadline passed inside the task')


def lloyd_async(x, w, budget, iterations):
    """ghz_merge.lloyd with the seeding loop kept on the device.

    lloyd appends argmax(w*dist with taken points set to -1) and stops before appending the
    first pick whose score is <= 1e-28. Here every pick and its score are queued without a
    host read; the stop rule is applied once afterwards (picks after it are discarded, they
    never influence earlier picks). The Lloyd rounds are lloyd's, line for line.
    """
    if len(w) <= budget:
        return torch.arange(len(w), device=x.device)
    center = w @ x
    first = torch.argmin(((x-center)**2).sum(1)).reshape(1)
    taken = torch.zeros(len(w), dtype=torch.bool, device=x.device)
    taken[first] = True
    dist = ((x-x[first])**2).sum(1)
    picks, scores = [first], []
    for _ in range(1, budget):
        score = (w*dist).masked_fill(taken, -1.)
        j = torch.argmax(score).reshape(1)
        picks.append(j)
        scores.append(score[j])
        taken[j] = True
        dist = torch.minimum(dist, ((x-x[j])**2).sum(1))
    m = len(picks)
    if scores:
        stop = torch.nonzero(torch.cat(scores) <= SEED_STOP)
        if len(stop):
            m = 1+int(stop[0, 0])
    centers, last = x[torch.cat(picks[:m])].clone(), None
    for _ in range(iterations):
        labels = sqdist(x, centers).argmin(1)
        if last is not None and torch.equal(labels, last):
            break
        last = labels.clone()
        oh = onehot(labels, len(centers), x.dtype)
        mass = oh @ w
        keep = mass > 0
        centers = (oh @ (w[:, None]*x))[keep]/mass[keep, None]
    return torch.unique(sqdist(x, centers).argmin(1), return_inverse=True)[1]


class Sim:
    """merge_K on the exact same-history step: ghz_merge.MergeK with lloyd_async, without timers/hooks.
    K >= ensemble size never merges (exact propagation; with PTA letters: the PTA simulator)."""
    def __init__(self, K, kind, letters, aux, c, states, weights, factors=None):
        self.K, self.kind, self.letters, self.aux, self.c = K, kind, letters, aux, c
        self.t, self.ell = 0, 0.
        self.factors = factors
        self.states, self.weights = self.compress(states, weights)

    def compress(self, states, weights):
        if len(weights) > self.K:
            labels = lloyd_async(features(self.kind, states, self.aux), weights, self.K, self.c['clustering_iterations'])
            if self.factors is not None:
                from ghz_factors import combine as combine_factors
                self.factors = combine_factors(self.factors, weights, labels)
            states, weights = combine(states, weights, labels)
            if self.factors is not None:
                from ghz_factors import density
                states = density(self.factors)
        weights = weights/weights.sum()
        check_ensemble(states, weights, self.c['physical_tolerance'])
        return states, weights

    def advance(self, a):
        if self.factors is None:
            states, weights, mass, _, _ = step(self.states, self.weights, self.letters[a], self.c)
        else:
            from ghz_factors import expand as factor_expand
            states, weights, mass, _, self.factors = factor_expand(self.factors, self.weights, self.letters[a], self.c)
        if not math.isfinite(mass) or not 0 < mass <= 1+1e-10:
            raise FloatingPointError('Invalid exact acceptance %r' % mass)
        self.t += 1
        self.ell += math.log(mass)
        self.states, self.weights = self.compress(states, weights)
        return torch.einsum('n,nij->ij', self.weights.to(C), self.states), self.ell


class PTALetter:
    """State-level PTA of one noisy round, the d=4 convention (dejmps_experiment pta_initial_density_matrix:
    the noisy single-copy state is dephased in the Bell basis, here the GHZ basis, before the
    purification kernel). A GHZ-diagonal input makes every accepted outcome identical, so the
    outcomes are summed into one child per parent: PTA erases branching."""
    def __init__(self, letter, B):
        self.kraus, self.B = letter.kraus, B
        self.kernel = Letter(np.eye(letter.d)[None], letter.basis, letter.accept, device=letter.kraus.device)

    def raw(self, states):
        k = self.kraus
        s = (k @ states[:, None] @ dag(k)).sum(1)
        lam = ghz_diagonal(s, self.B).to(C)
        s = torch.einsum('ia,na,ja->nij', self.B, lam, self.B.conj())
        return self.kernel.raw(s).sum(1, keepdim=True)


def pta_letters(letters, B):
    return {a: PTALetter(L, B) for a, L in letters.items()}


def tdist(a, b):
    return torch.linalg.eigvalsh(a-b).abs().sum(-1)/2


def trajectory(sim, word):
    """Full rollout: mean states [H, d, d] and cumulative log acceptances."""
    rhos, ells = [], []
    for a in word:
        rho, ell = sim.advance(a)
        rhos.append(rho)
        ells.append(ell)
    return torch.stack(rhos), ells


def compare(r1, l1, r2, l2):
    """Largest prefix trace distance and cumulative-probability relative error between two trajectories."""
    return float(tdist(r1, r2).max()), max(abs(math.expm1(a-b)) for a, b in zip(l1, l2))


def scored_run(make, word, ref, eps, early_stop=True, B=None):
    """One strict run: passes iff physical with TD <= eps and P-relative error <= eps at every prefix.
    early_stop ends the run at the first violated prefix (pass/fail unchanged; G1精简版 scan)."""
    t0 = time.perf_counter()
    out = dict(passed=False, steps=0, max_td=0., max_ep=0., fail_step=None)
    try:
        sim = make()
        for h, a in enumerate(word, 1):
            rho, ell = sim.advance(a)
            out['steps'] = h
            td = float(tdist(rho, ref['rho'][h-1]))
            ep = abs(math.expm1(ell-ref['ell'][h-1]))
            if not math.isfinite(td+ep):
                raise FloatingPointError('Nonfinite metric')
            out['max_td'], out['max_ep'] = max(out['max_td'], td), max(out['max_ep'], ep)
            if out['fail_step'] is None and (td > eps or ep > eps):
                out['fail_step'] = h
                if early_stop:
                    break
        out['passed'] = out['fail_step'] is None and out['steps'] == len(word)
        if B is not None and out['steps'] == len(word):
            out['final_fidelity'] = float(ghz_diagonal(rho[None], B)[0, 0])
    except NUMERIC as exc:
        out.update(passed=False, error=str(exc), fail_step=out['fail_step'] or out['steps']+1)
    out['seconds'] = time.perf_counter()-t0
    return out


def scan_K(grid, joint, window, unreached):
    """K_needed: the smallest grid K such that K and the grid values above it (at most window-1 of
    them, fewer at the top of the grid) all pass jointly; `unreached` if none. Ascending scan that
    stops at the first qualifying K, evaluating each K at most once (equivalent to the definition)."""
    passes = {}

    def ok(j):
        if j not in passes:
            passes[j] = bool(joint(grid[j]))
        return passes[j]
    for i in range(len(grid)):
        if all(ok(j) for j in range(i, min(i+window, len(grid)))):
            return grid[i], passes
    return unreached, passes


def k_needed_definition(grid, passes, window, unreached):
    """The definition evaluated on a complete pass table (used only by the check stage)."""
    for i in range(len(grid)):
        if all(passes[j] for j in range(i, min(i+window, len(grid)))):
            return grid[i]
    return unreached


def guard(prep):
    if prep['split'] not in ('calibration', 'evaluation'):
        raise RuntimeError('G1a never prepares the sealed GHZ final or unknown splits: '+prep['id'])


def judge(d1, d2, g):
    """Two-tier reference acceptance (g1_config v2, user option C, 2026-10-14).

    d1: check difference merge_R vs merge_{R/2}; d2: merge_{2R} vs merge_R if the upgrade ran, else None.
    The upgrade is triggered by d1 > tolerance (1e-4, unchanged); the finest reference computed is used,
    and its last check difference decides: accepted iff <= accept_tolerance (1e-3), and 'strict' iff
    <= tolerance, which is exactly the original HANDOFF 5h rule (kept as a sensitivity verdict)."""
    last = d1 if d2 is None else d2
    return last <= g['reference']['accept_tolerance'], last <= g['reference']['tolerance'], last


def reference(n, word, states, weights, letters, aux, c, g, hard=None):
    """Near-exact merge_R reference (HANDOFF 5h, two-tier acceptance of g1_config v2): checked against
    merge_{R/2} (prefix max TD and P error); one R x 2 upgrade if that exceeds 1e-4; the finest reference is
    used and accepted if its last difference is <= 1e-3, else excluded as 'unconverged'; info['strict'] records
    the original 1e-4 rule. Excluded as 'unstable' if a 1e-12 perturbation of the initial ensemble moves the
    reference by more than 1e-4; 'failed' on a physical failure."""
    R, R2 = c['reference'][str(n)]
    tol = g['reference']['tolerance']
    info = dict(R=R, check_R=R2, upgraded=False, runs=[], excluded=None)

    def roll(K, st):
        check_deadline(hard)
        t0 = time.perf_counter()
        sim = Sim(K, 'geo', letters, aux, c, st, weights)
        r, l = trajectory(sim, word)
        info['runs'].append(dict(K=K, steps=len(word), seconds=time.perf_counter()-t0))
        return r, l, sim
    try:
        r, l, sim = roll(R, states)
        r2, l2, _ = roll(R2, states)
        info['check_td'], info['check_ep'] = compare(r, l, r2, l2)
        d1, d2 = max(info['check_td'], info['check_ep']), None
        if d1 > tol and g['reference']['upgrades'] > 0:
            r3, l3, sim3 = roll(2*R, states)
            info['upgrade_td'], info['upgrade_ep'] = compare(r3, l3, r, l)
            info.update(upgraded=True, R=2*R, check_R=R)
            d2 = max(info['upgrade_td'], info['upgrade_ep'])
            r, l, sim = r3, l3, sim3
        ok, info['strict'], info['last_difference'] = judge(d1, d2, g)
        info.update(final_fidelity=float(ghz_diagonal(r[-1:], aux['ghz'])[0, 0]), final_atoms=len(sim.weights),
                    participation=float(1/(sim.weights**2).sum()), log_acceptance=l[-1])
        if not ok:
            info['excluded'] = 'unconverged'
            return None, info
        rp, lp, _ = roll(info['R'], perturb(states, c['perturbation']))
        info['sensitivity_td'], info['sensitivity_ep'] = compare(r, l, rp, lp)
        if max(info['sensitivity_td'], info['sensitivity_ep']) > g['reference']['sensitivity_tolerance']:
            info['excluded'] = 'unstable'
            return None, info
    except NUMERIC as exc:
        info.update(excluded='failed', error=str(exc))
        return None, info
    return dict(rho=r, ell=l), info


def arms(states, weights, word, ref, letters, pta, aux, c, g, hard=None):
    """geo_K and diag_K: K_needed by the ascending scan, every K run on the initial ensemble and, only
    if that passes, on its 1e-12 perturbation (joint pass). K=1 and PTA: one full run each."""
    eps, grid = g['eps'], c['K_grid']
    pert = perturb(states, c['perturbation'])
    out = {}
    for kind in ('geo', 'diag'):
        runs = {}

        def joint(K):
            check_deadline(hard)
            base = scored_run(lambda: Sim(K, kind, letters, aux, c, states, weights), word, ref, eps)
            moved = scored_run(lambda: Sim(K, kind, letters, aux, c, pert, weights), word, ref, eps) if base['passed'] else None
            runs[K] = dict(base=base, perturbed=moved)
            return base['passed'] and moved['passed']
        K, _ = scan_K(grid, joint, g['K_window'], g['K_unreached'])
        out[kind] = dict(K_needed=K, runs=[dict(K=k, **runs[k]) for k in sorted(runs)])
    out['K1'] = scored_run(lambda: Sim(1, 'geo', letters, aux, c, states, weights), word, ref, eps, False, aux['ghz'])
    out['PTA'] = scored_run(lambda: Sim(10**9, 'geo', pta, aux, c, states, weights), word, ref, eps, False, aux['ghz'])
    return out


class Context:
    """Per-worker caches of letters and features (one device, one G0 config, one G1a protocol)."""
    def __init__(self, c, g, device, out, hard=None):
        self.c, self.g, self.device, self.out, self.hard = c, g, device, Path(out), hard
        self._aux, self._letters = {}, {}

    def aux(self, n):
        if n not in self._aux:
            self._aux[n] = dict(pauli=pauli_stack(n, self.device),
                                ghz=torch.as_tensor(ghz_basis(n), dtype=C, device=self.device))
        return self._aux[n]

    def letters(self, n, gamma):
        if (n, gamma) not in self._letters:
            L = ghz_letters(n, gamma, self.c['noise']['overrotation'], self.device)
            self._letters[(n, gamma)] = (L, pta_letters(L, self.aux(n)['ghz']))
        return self._letters[(n, gamma)]


def meta(item):
    return {k: v for k, v in item.items() if k != 'word'}


def run_scan(item, ctx):
    """Threshold scan point: endpoint GHZ fidelity of the merge_{scan_K} (AB)^20 rollout."""
    prep = item['prep']
    guard(prep)
    n, row, sim = prep['n'], meta(item), None
    try:
        states, weights = prepare(prep, ctx.c, ctx.device)
        sim = Sim(ctx.g['threshold']['scan_K'], 'geo', ctx.letters(n, item['gamma'])[0], ctx.aux(n), ctx.c, states, weights)
        for a in item['word']:
            rho, ell = sim.advance(a)
        row.update(status='OK', fidelity=float(ghz_diagonal(rho[None], ctx.aux(n)['ghz'])[0, 0]), log_acceptance=ell)
    except NUMERIC as exc:
        row.update(status='FAILED', error=str(exc), step=sim.t+1 if sim is not None else 0)
    return row


def run_task(item, ctx):
    """One gate task: reference, then (if the reference is kept) geo/diag K_needed, K=1 and PTA."""
    prep = item['prep']
    guard(prep)
    n, c, g, row = prep['n'], ctx.c, ctx.g, meta(item)
    letters, pta = ctx.letters(n, item['gamma'])
    aux = ctx.aux(n)
    t0 = time.perf_counter()
    try:
        states, weights = prepare(prep, c, ctx.device)
    except NUMERIC as exc:
        row.update(status='PREPARE_FAILED', error=str(exc))
        return row
    row['initial_atoms'] = len(weights)
    try:
        ref, row['reference'] = reference(n, item['word'], states, weights, letters, aux, c, g, ctx.hard)
    except Deadline as exc:
        row.update(status='NOT_FINISHED_DEADLINE', error=str(exc))
        return row
    row['seconds_reference'] = time.perf_counter()-t0
    if ref is None:
        row['status'] = 'REFERENCE_EXCLUDED'
        return row
    np.savez_compressed(ctx.out/'refs'/(item['id']+'.npz'), rho=ref['rho'].cpu().numpy(),
                        ell=np.asarray(ref['ell']), R=row['reference']['R'])
    t1 = time.perf_counter()
    try:
        row['arms'] = arms(states, weights, item['word'], ref, letters, pta, aux, c, g, ctx.hard)
    except Deadline as exc:
        row.update(status='NOT_FINISHED_DEADLINE', error=str(exc))
        return row
    row['seconds_arms'] = time.perf_counter()-t1
    row['status'] = 'OK'
    return row


HANDLERS = dict(scan=run_scan, task=run_task)

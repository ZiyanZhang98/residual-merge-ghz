"""Look-ahead merging (HANDOFF 22.17, step 2): the future response only decides which branches merge.

Clustering features = geometric Pauli coordinates (+) weight * [log psi_k / k, F_k] for k in the
configured look-ahead lengths (capped at the remaining length). Merges stay mass- and mean-
preserving and physics is unchanged, so readout corrections (merge+PTA) remain exact identities.
With no look-ahead the simulator is the frozen M0 (qa_sim.QoISim) operation for operation.
"""
import hr_runtime  # noqa: F401
import time
import torch
import qa_sim
from qa_sources import PTA
from qb_model import Learned


class LookAhead:
    def __init__(self, kind, lengths, weight, model=None, gamma=None):
        if kind not in ('pta', 'learned') or (kind == 'learned' and model is None):
            raise ValueError('Look-ahead source must be pta or a learned model')
        self.kind, self.lengths, self.weight = kind, list(lengths), float(weight)
        self.learned = Learned(model, gamma, PTA()) if kind == 'learned' else None

    @torch.no_grad()
    def __call__(self, states, factors, word, ctx, c, hard=None):
        L = len(word)
        ks = [min(k, L) for k in self.lengths]
        if self.kind == 'pta':
            v = PTA()(states, factors, word[:max(ks)], ctx, c, hard)
        else:
            # The learned model was trained on full remaining words; never truncate its context.
            v = self.learned(states, factors, word, ctx, c, hard)
        cols = []
        for k in ks:
            psi = torch.clamp(v['psi'][:, k-1], min=1e-300)
            cols += [psi.log()/k, v['phi'][:, k-1]/psi]
        return self.weight*torch.stack(cols, 1)


class GoalSim(qa_sim.QoISim):
    def __init__(self, K, word, ctx, c, states, weights, sources=None, hard=None, lookahead=None):
        self.lookahead, self.lookahead_seconds = lookahead, 0.
        super().__init__(K, word, ctx, c, states, weights, sources, hard)

    def compress(self, states, w):
        if self.lookahead is None or len(w) <= self.K:
            return super().compress(states, w)
        start = time.perf_counter()
        extra = self.lookahead(states, self.factors, self.word[self.t:], self.ctx, self.c, self.hard)
        self.lookahead_seconds += time.perf_counter()-start
        base = qa_sim.features
        # QoISim.compress computes its clustering features exactly once through qa_sim.features;
        # extend that single call instead of duplicating the frozen compression/correction code.
        qa_sim.features = lambda kind, s, aux: torch.cat((base(kind, s, aux), extra), 1)
        try:
            return super().compress(states, w)
        finally:
            qa_sim.features = base


def rollout(K, word, ctx, c, states, weights, sources=None, hard=None, lookahead=None):
    """qa_sim.rollout with GoalSim; returns sim, rho, ell, estimates[arm] = dict(P, N)."""
    sim = GoalSim(K, word, ctx, c, states, weights, sources, hard, lookahead)
    rho, ell = [], []
    for a in word:
        r, l = sim.advance(a)
        rho.append(r); ell.append(l)
    rho = torch.stack(rho)
    log_raw = weights.new_tensor(ell)
    raw = {'P': log_raw.exp()}
    raw['N'] = raw['P']*qa_sim.ghz_diagonal(rho, ctx['aux']['ghz'])[:, 0]
    out = {'M0': dict(raw)}
    nan = torch.full_like(log_raw, float('nan'))
    for name in sim.sources:
        for method in qa_sim.ESTIMATORS:
            a, est = qa_sim.arm(name, method), {}
            for q, _ in qa_sim.QUANTITIES:
                if (a, q) in sim.failed:
                    est[q] = nan.clone(); continue
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
    return sim, rho, ell, out

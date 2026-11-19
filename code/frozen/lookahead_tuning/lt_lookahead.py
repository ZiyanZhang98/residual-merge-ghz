"""Standardized look-ahead features for merging (HANDOFF 22.22).

At every compression the look-ahead columns (log psi_k and F_k for the chosen horizons) are
centered and scaled to unit weighted standard deviation across the current branches, then the
whole block is scaled so that its total weighted variance equals `share` times that of the Pauli
coordinates. share = 0 is merge+PTA; the frozen hr_goal setting (w = 1, k <= 16, log psi/k) puts
about 0.1-0.2 of the geometric variance into the look-ahead block (job 28554135).
Quantities: 'psiF' (psi and F from the source), 'psi' (psi only), 'psiL_Fpta' (learned psi with
PTA F; the learned F is less accurate than PTA's). Merging, physics and corrections are those of
hr_goal.GoalSim; only the clustering features change.
"""
import lt_runtime  # noqa: F401  (paths)
import math
import torch
import qa_sim
from ghz_physics import coords, ghz_diagonal
from qa_sources import PTA
from qb_model import Learned
from hr_goal import GoalSim

QUANTITIES = ('psiF', 'psi', 'psiL_Fpta')


class StdLookAhead:
    def __init__(self, source, share, lengths, quantities, model=None, gamma=None):
        if source not in ('pta', 'learned') or quantities not in QUANTITIES or (quantities == 'psiL_Fpta' and source != 'learned'):
            raise ValueError('Unsupported look-ahead: %s/%s' % (source, quantities))
        self.source, self.share, self.lengths, self.quantities = source, float(share), list(lengths), quantities
        self.learned = Learned(model, gamma, PTA()) if source == 'learned' else None
        self.weights, self.last = None, None

    @torch.no_grad()
    def __call__(self, states, factors, word, ctx, c, hard=None):
        L = len(word)
        ks = sorted({min(k, L) for k in self.lengths})
        need_pta = self.source == 'pta' or self.quantities == 'psiL_Fpta'
        vp = PTA()(states, factors, word[:ks[-1]], ctx, c, hard) if need_pta else None
        vl = self.learned(states, factors, word, ctx, c, hard) if self.source == 'learned' else None
        vpsi = vl if vl is not None else vp
        vF = vp if self.quantities == 'psiL_Fpta' else vpsi
        cols = []
        for k in ks:
            cols.append(torch.clamp(vpsi['psi'][:, k-1], min=1e-300).log())
            if self.quantities != 'psi':
                cols.append(vF['phi'][:, k-1]/torch.clamp(vF['psi'][:, k-1], min=1e-300))
        X = torch.stack(cols, 1)
        w = self.weights/self.weights.sum()
        mean = w @ X
        sd = torch.sqrt(w @ (X-mean)**2)
        live = sd > 1e-12
        Z = torch.where(live, (X-mean)/torch.where(live, sd, torch.ones_like(sd)), torch.zeros_like(X))
        geo = coords(states, ctx['aux']['pauli'])
        vg = float((w @ (geo-w @ geo)**2).sum())
        m = int(live.sum())
        scale = math.sqrt(self.share*vg/m) if m and vg > 0 else 0.
        self.last = dict(columns=X.shape[1], live=m, geometric_variance=vg, scale=scale)
        return Z*scale


class LTSim(GoalSim):
    """GoalSim that hands the current branch weights to the look-ahead (for weighted scaling)."""
    def compress(self, states, w):
        if self.lookahead is not None:
            self.lookahead.weights = w
        return super().compress(states, w)


def rollout(K, word, ctx, c, states, weights, sources=None, hard=None, lookahead=None):
    """hr_goal.rollout with LTSim; returns sim, rho, ell, estimates[arm] = dict(P, N)."""
    sim = LTSim(K, word, ctx, c, states, weights, sources, hard, lookahead)
    rho, ell = [], []
    for a in word:
        r, l = sim.advance(a)
        rho.append(r); ell.append(l)
    rho = torch.stack(rho)
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

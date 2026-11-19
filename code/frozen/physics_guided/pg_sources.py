"""Analytic future responses without learning (HANDOFF 22.18), for corrections and merge metrics.

D_m  exact all-outcome branching for the first m rounds of the remaining word, then the PTA tail
     (qa_sources.PTA) from every child. D_0 is PTA. PTA reproduces the next-round acceptance
     exactly (kernel check, job 28503397), so psi of D_m is exact up to prefix m+1.
MF   branch-summed exact kernel without the GHZ twirl: keeps every coherence, erases branching.
Values follow qa_sources: psi[:, k-1] = P(word[:k] succeeds), phi[:, k-1] = P(succeeds, |GHZ>).
Children are kept parent-major in dense blocks (zero-mass children carry the maximally mixed
filler), so every sum over children is a deterministic reshape-sum, never a scatter.
"""
import pg_runtime  # noqa: F401  (paths)
import torch
from ghz_physics import C, trace, ghz_diagonal
from qa_sources import PTA
from g1_core import check_deadline


class Summed:
    """A letter whose accepted outcomes are summed into one unnormalized child."""
    def __init__(self, letter):
        self.letter = letter

    def raw(self, states):
        return self.letter.raw(states).sum(1, keepdim=True)


class MF:
    def __init__(self):
        self.inner = PTA()

    @torch.no_grad()
    def __call__(self, states, roots, word, ctx, c, hard=None):
        summed = {a: Summed(letter) for a, letter in ctx['letters'].items()}
        return self.inner(states, roots, word, dict(ctx, pta=summed), c, hard)


def exact_rounds(states, word, ctx, c, hard=None):
    """Exact branching through `word`; returns per-prefix psi, phi [N, len(word)] and the
    normalized children [N*prod(Y), d, d] with their absolute masses (parent-major)."""
    N, d = len(states), states.shape[-1]
    B = ctx['aux']['ghz']
    filler = torch.eye(d, dtype=C, device=states.device)/d
    s, mass, psi, phi = states, states.real.new_ones(N), [], []
    for a in word:
        check_deadline(hard)
        letter = ctx['letters'][a]
        raw = torch.cat([letter.raw(s[i:i+c['chunk']]) for i in range(0, len(s), c['chunk'])])
        p = trace(raw)
        if not bool(torch.isfinite(p).all()) or float(p.min()) < -c['physical_tolerance']:
            raise FloatingPointError('Invalid exact outcome probability')
        Y = p.shape[1]
        g = ghz_diagonal(raw.flatten(0, 1), B)[:, 0].reshape(-1, Y)
        phi.append((mass[:, None]*g).reshape(N, -1).sum(1))
        ok = p > 0
        child = torch.where(ok, mass[:, None]*p, torch.zeros_like(p))
        psi.append((mass[:, None]*p).reshape(N, -1).sum(1))
        norm = torch.where(ok, p, torch.ones_like(p))
        s = torch.where(ok.flatten()[:, None, None], (raw/norm[..., None, None]).flatten(0, 1), filler)
        mass = child.flatten()
    return torch.stack(psi, 1), torch.stack(phi, 1), s, mass


class DExact:
    def __init__(self, m):
        self.m, self.tail = int(m), PTA()

    @torch.no_grad()
    def __call__(self, states, roots, word, ctx, c, hard=None):
        if self.m == 0 or not word:
            return self.tail(states, roots, word, ctx, c, hard)
        N = len(states)
        psi, phi, s, mass = exact_rounds(states, word[:self.m], ctx, c, hard)
        rest = word[self.m:]
        if rest:
            t = self.tail(s, None, rest, ctx, c, hard)
            psi = torch.cat((psi, (mass[:, None]*t['psi']).reshape(N, -1, len(rest)).sum(1)), 1)
            phi = torch.cat((phi, (mass[:, None]*t['phi']).reshape(N, -1, len(rest)).sum(1)), 1)
        return dict(psi=psi, phi=phi)


SOURCES = {'M_PTA': PTA, 'D1': lambda: DExact(1), 'D2': lambda: DExact(2), 'MF': MF}

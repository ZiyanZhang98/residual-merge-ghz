"""Absolute future responses for the two quantities of interest.

psi[:, k-1]: probability that word[:k] succeeds from the node.
phi[:, k-1]: probability that word[:k] succeeds AND the output is |GHZ_00>,
             i.e. Tr(Pi_GHZ F_k(rho)); 0 <= phi <= psi.
No floors, clipping or rescaling.
"""
import qa_runtime  # noqa: F401  (paths)
import numpy as np
import torch
from ghz_physics import trace, ghz_diagonal, coords
from g1_core import check_deadline
from rr_features import responses


def projector(ctx):
    v = ctx['aux']['ghz'][:, 0]
    return torch.outer(v, v.conj())


class PTA:
    """Analytic adjoint: noise twirled to GHZ-diagonal before every round, branching erased.

    psi reproduces pc_sources.pta_all operation by operation; phi reads the GHZ
    fidelity element of the same unnormalized child, so it costs no extra rounds.
    """
    @torch.no_grad()
    def __call__(self, states, roots, word, ctx, c, hard=None):
        B = ctx['aux']['ghz']
        s, product, psi, phi = states, states.real.new_ones(len(states)), [], []
        for a in word:
            check_deadline(hard)
            raw = torch.cat([ctx['pta'][a].raw(s[i:i+c['chunk']])[:, 0]
                             for i in range(0, len(s), c['chunk'])])
            p = trace(raw)
            if not bool(torch.isfinite(p).all()) or bool((p < 0).any()) or bool((p > 1+1e-10).any()):
                raise FloatingPointError('Invalid PTA acceptance')
            phi.append(product*ghz_diagonal(raw, B)[:, 0])
            product = product*p
            psi.append(product)
            s = torch.where((p > 0)[:, None, None], raw/torch.where(p > 0, p, torch.ones_like(p))[:, None, None], s)
        if not psi:
            empty = states.real.new_empty((len(states), 0))
            return dict(psi=empty, phi=empty.clone())
        return dict(psi=torch.stack(psi, 1), phi=torch.stack(phi, 1))


class Exact:
    """Exact all-outcome tree (short words only; correctness checks)."""
    @torch.no_grad()
    def __call__(self, states, roots, word, ctx, c, hard=None, chunk=4096):
        if not word:
            empty = states.real.new_empty((len(states), 0))
            return dict(psi=empty, phi=empty.clone())
        R, _ = responses(states, word, len(word), dict(ctx, response_factors=roots), c,
                         {'features': {'chunk_leaves': chunk}}, hard=hard)
        g = coords(projector(ctx)[None], ctx['aux']['pauli'])[0].cpu().numpy()
        d = states.shape[-1]
        psi = np.stack([R[k][:, 0] for k in sorted(R)], 1)
        # F = (tr F I + sum_k tr(P_k F) P_k)/d, so tr(Pi F) = (tr F + sum_k tr(P_k F) tr(P_k Pi))/d.
        phi = np.stack([(R[k][:, 0]+R[k][:, 1:] @ g)/d for k in sorted(R)], 1)
        as_t = lambda x: torch.as_tensor(x, dtype=torch.float64, device=states.device)  # noqa: E731
        return dict(psi=as_t(psi), phi=as_t(phi))

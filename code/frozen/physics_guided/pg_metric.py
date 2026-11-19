"""Goal-oriented merge metric (HANDOFF 22.18, method A).

Merging a cluster (w_i, rho_i) into its mean changes every future response psi_k by
sum_i w_i [psi_k(rho_i) - psi_k(rho_bar)] = (1/2) sum_i w_i e_i^T H_k e_i + O(e^3), with e_i the
Pauli-coordinate deviation; exact for k = 1 because psi_1 is quadratic. Dividing by the tolerated
error of each prefix (P relative, F = N/P absolute) and bounding each signed form by its absolute
value gives one PSD matrix M = (1/2) sum_k (|A_k^P| + |A_k^F|). Lloyd on z = M^(1/2) x minimizes
the summed bound. The Hessian is taken once per compression at the ensemble mean (global metric).

The response is differentiable code identical in operations to pg_sources (PTA: D_0; D1: one
exact round, then PTA; MF: branch-summed kernel without the twirl, added after the smoke check
of job 28516826 showed MF to be the most accurate short-range response). A geometric
tie-breaker at 1e-6 of the largest curvature only orders points the metric cannot distinguish.
"""
import pg_runtime  # noqa: F401  (paths)
import torch
from ghz_physics import C, trace, coords
from pg_sources import Summed

# kind -> (exact rounds, tail letters)
KINDS = {'PTA': (0, 'pta'), 'D1': (1, 'pta'), 'MF': (0, 'summed')}


def response_fn(kind, word, ctx):
    """x (Pauli coordinates of one normalized state) -> [psi_1..psi_L, phi_1..phi_L]."""
    m, tail = KINDS[kind]
    P, B = ctx['aux']['pauli'], ctx['aux']['ghz']
    d = P.shape[-1]
    eye, g = torch.eye(d, dtype=C, device=P.device), B[:, 0]
    after = (lambda a: ctx['pta'][a]) if tail == 'pta' else (lambda a: Summed(ctx['letters'][a]))
    letters = [ctx['letters'][a] if j < m else after(a) for j, a in enumerate(word)]

    def f(x):
        s = ((eye+torch.einsum('k,kij->ij', x.to(C), P))/d)[None]
        mass, psi, phi = x.new_ones(1), [], []
        for letter in letters:
            raw = letter.raw(s)
            p = trace(raw)
            gg = torch.einsum('i,nyij,j->ny', g.conj(), raw, g).real
            phi.append((mass[:, None]*gg).sum())
            mass = (mass[:, None]*p).flatten()
            psi.append(mass.sum())
            safe = torch.where(p > 0, p, torch.ones_like(p))
            s = (raw/safe[..., None, None]).flatten(0, 1)
        return torch.stack(psi+phi)
    return f


def hessian(f, x, chunk):
    """Forward-over-reverse Hessians of every output, [outputs, dim, dim], symmetrized.
    `chunk` tangents are batched; on GPU memory exhaustion the batch is halved (same values)."""
    jac = torch.func.jacrev(f)
    E = torch.eye(len(x), dtype=x.dtype, device=x.device)
    while True:
        try:
            rows = [torch.func.vmap(lambda v: torch.func.jvp(jac, (x,), (v,))[1])(E[i:i+chunk])
                    for i in range(0, len(x), chunk)]
            break
        except torch.cuda.OutOfMemoryError:
            if chunk == 1:
                raise
            torch.cuda.empty_cache()
            chunk = max(1, chunk//2)
    H = torch.cat(rows).permute(1, 0, 2)
    return (H+H.transpose(1, 2))/2, chunk


def absm(A):
    lam, V = torch.linalg.eigh(A)
    return (V*lam.abs()[..., None, :]) @ V.transpose(-1, -2)


class GoalMetric:
    def __init__(self, kind, tol, chunk, tie):
        if kind not in KINDS:
            raise ValueError('Unknown metric response: '+kind)
        self.kind, self.tol, self.chunk, self.tie = kind, tol, int(chunk), float(tie)

    def matrix(self, xbar, word, ctx):
        f = response_fn(self.kind, word, ctx)
        v = f(xbar)
        H, self.chunk = hessian(f, xbar, self.chunk)
        L = len(word)
        psi, phi = v[:L], v[L:]
        if not bool(torch.isfinite(H).all()) or not bool(torch.isfinite(v).all()) or float(psi.min()) <= 0:
            raise FloatingPointError('Invalid goal-metric response or Hessian')
        F = phi/psi
        AP = H[:L]/(self.tol['P_relative']*psi)[:, None, None]
        AF = (H[L:]-F[:, None, None]*H[:L])/(self.tol['F_absolute']*psi)[:, None, None]
        return 0.5*(absm(AP).sum(0)+absm(AF).sum(0)), H, v

    def __call__(self, states, w, word, ctx):
        x = coords(states, ctx['aux']['pauli'])
        xbar = w @ x
        M, _, _ = self.matrix(xbar, word, ctx)
        lam, V = torch.linalg.eigh(M)
        lam = lam.clamp(min=0)
        dx = x-xbar
        z = torch.cat(((dx @ V)*lam.sqrt(), self.tie*float(lam.max().sqrt())*dx), 1)
        total = float(lam.sum())
        top = torch.sort(lam, descending=True).values
        info = dict(trace=total, rank99=int((torch.cumsum(top, 0) < 0.99*total).sum())+1 if total > 0 else 0,
                    predicted_if_one_cluster=float(w @ ((dx @ V)**2*lam).sum(1)))
        return z, info

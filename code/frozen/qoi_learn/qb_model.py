"""Two-head future-response model (HANDOFF 22.5; revision 2 of 22.12, 2026-10-21).

kind mlp | kernel | quad (quad = smooth residual, exactly quadratic in the state; HANDOFF 22.14).
pta param:    psi = psi_PTA * exp(r_P),   F = F_PTA * exp(r_F),   phi = psi * F   (zero init == PTA)
direct param: psi = s_P * exp(r_P),       F = s_F * exp(r_F),     phi = psi * F   (no online PTA)
The fidelity head corrects PTA's conditional fidelity F_PTA = phi_PTA/psi_PTA, so errors of the
success head no longer enter F. Outputs are positive by construction; no clipping.
"""
import qb_runtime  # noqa: F401
import torch
from torch import nn
from ghz_physics import coords
from qa_sources import PTA
from qb_data import word_code, pad


class Model(nn.Module):
    def __init__(self, n, H, train_cfg, kind, param, scale=None, landmarks=None):
        super().__init__()
        self.n, self.H, self.kind, self.param = n, H, kind, param
        dim = 4**n-1+2*H+1
        self.register_buffer('scale', torch.ones(2, H) if scale is None else scale.clone())
        if kind == 'mlp':
            w = train_cfg['width']
            self.net = nn.Sequential(nn.Linear(dim, w), nn.SiLU(), nn.Linear(w, w), nn.SiLU(), nn.Linear(w, 2*H))
            nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        elif kind == 'quad':
            # Smooth residual (HANDOFF 22.14): r(x; ctx) = a(ctx) + sum_m beta_m(ctx) (v_m.x)
            # + sum_r s_r(ctx) (u_r.x)^2. Coefficients depend only on the protocol context, so r is
            # exactly quadratic in the state coordinates x: affine parts cancel in mean-preserving
            # merges and the learned part of a cluster gap is curvature x cluster covariance.
            w, d2 = train_cfg['width'], 4**n-1
            self.d2, self.M, self.R = d2, train_cfg['quad_linear'], train_cfg['quad_rank']
            self.directions = nn.Linear(d2, self.M+self.R, bias=False)
            self.context = nn.Sequential(nn.Linear(2*H+1, w), nn.SiLU(), nn.Linear(w, w), nn.SiLU(),
                                         nn.Linear(w, 2*H*(1+self.M+self.R)))
            nn.init.zeros_(self.context[-1].weight); nn.init.zeros_(self.context[-1].bias)
        elif kind in ('rate', 'rate_state'):
            # HANDOFF 22.16. Rate term a_k(ctx): context only, per remaining length (L*Delta+c is a
            # special case); it scales every node of a cluster alike, so it adds no gap noise and
            # cannot diverge in state space. Optional bounded state term s = b*tanh(q/b), q linear +
            # quadratic in the state with context-only coefficients.
            w, d2 = train_cfg['width'], 4**n-1
            self.d2 = d2
            self.rate = nn.Sequential(nn.Linear(2*H+1, w), nn.SiLU(), nn.Linear(w, w), nn.SiLU(), nn.Linear(w, 2*H))
            nn.init.zeros_(self.rate[-1].weight); nn.init.zeros_(self.rate[-1].bias)
            self.state_on = kind == 'rate_state'
            if kind == 'rate_state':
                self.M, self.R, self.bound = train_cfg['quad_linear'], train_cfg['quad_rank'], train_cfg['state_bound']
                self.directions = nn.Linear(d2, self.M+self.R, bias=False)
                self.state_ctx = nn.Sequential(nn.Linear(2*H+1, w), nn.SiLU(), nn.Linear(w, w), nn.SiLU(),
                                               nn.Linear(w, 2*H*(self.M+self.R)))
                nn.init.zeros_(self.state_ctx[-1].weight); nn.init.zeros_(self.state_ctx[-1].bias)
        elif kind == "kernel":
            if landmarks is None:
                raise ValueError('Kernel landmarks must come from fit nodes')
            self.register_buffer('landmarks', landmarks.clone())
            self.bandwidth = train_cfg['kernel_bandwidth']
            self.net = nn.Linear(len(landmarks)+1, 2*H, bias=False)
            nn.init.zeros_(self.net.weight)
        else:
            raise ValueError(kind)
        self.double()

    def forward(self, x, psi_pta, phi_pta):
        if self.kind in ('rate', 'rate_state'):
            coords, z = x[:, :self.d2], x[:, self.d2:]
            uniq, inv = torch.unique(z, dim=0, return_inverse=True)
            r = self.rate(uniq)[inv]
            if self.kind == 'rate_state' and self.state_on:
                coef = self.state_ctx(uniq).view(len(uniq), 2*self.H, self.M+self.R)[inv]
                proj = self.directions(coords)
                q = torch.einsum('nkm,nm->nk', coef[..., :self.M], proj[:, :self.M]) \
                    + torch.einsum('nkr,nr->nk', coef[..., self.M:], proj[:, self.M:].square())
                r = r+self.bound*torch.tanh(q/self.bound)
        elif self.kind == 'quad':
            coords, z = x[:, :self.d2], x[:, self.d2:]
            uniq, inv = torch.unique(z, dim=0, return_inverse=True)
            coef = self.context(uniq).view(len(uniq), 2*self.H, 1+self.M+self.R)[inv]
            proj = self.directions(coords)
            lin, quad = proj[:, :self.M], proj[:, self.M:].square()
            r = coef[..., 0]+torch.einsum('nkm,nm->nk', coef[..., 1:1+self.M], lin) \
                + torch.einsum('nkr,nr->nk', coef[..., 1+self.M:], quad)
        else:
            if self.kind == 'kernel':
                k = torch.exp(-torch.cdist(x, self.landmarks).square()/(2*self.bandwidth**2))
                x = torch.cat((k, k.new_ones(len(k), 1)), 1)
            r = self.net(x)
        rP, rF = r[:, :self.H], r[:, self.H:]
        if self.param == 'pta':
            psi = psi_pta*torch.exp(rP)
            F = torch.where(psi_pta > 0, phi_pta/torch.clamp(psi_pta, min=1e-300), torch.zeros_like(psi_pta))
        else:
            psi, F = self.scale[0]*torch.exp(rP), self.scale[1]
        return psi, psi*F*torch.exp(rF)

    def curvature(self, z):
        """Mean over contexts/outputs of sum_r |s_r| |u_r|^2 (state term only; penalized in training)."""
        uniq = torch.unique(z, dim=0)
        coef = self.state_ctx(uniq).view(len(uniq), 2*self.H, self.M+self.R)[..., self.M:]
        return (coef.abs()*self.directions.weight[self.M:].square().sum(1)).sum(-1).mean()


class CachedPTA:
    """PTA responses shared by every provider at one compression event (same nodes tensor)."""
    def __init__(self):
        self.key, self.value, self.inner = None, None, PTA()

    def __call__(self, states, roots, word, ctx, c, hard=None):
        if self.key is not None and self.key[0] is states and self.key[1] == word:
            return self.value
        self.key, self.value = (states, word), self.inner(states, roots, word, ctx, c, hard)
        return self.value


class Learned:
    def __init__(self, model, gamma, prior):
        self.model, self.gamma, self.prior = model.eval(), gamma, prior

    @torch.no_grad()
    def __call__(self, states, roots, word, ctx, c, hard=None):
        H, L = self.model.H, len(word)
        if states.shape[-1] != 2**self.model.n or L > H:
            raise ValueError('Deployment scope mismatch')
        if L == 0:
            empty = states.real.new_empty((len(states), 0))
            return dict(psi=empty, phi=empty.clone())
        x = torch.cat((coords(states, ctx['aux']['pauli']), word_code(word, self.gamma, H, states.device).expand(len(states), -1)), 1)
        if self.model.param == 'pta':
            p = self.prior(states, roots, word, ctx, c, hard)
            pp, pf = pad(p['psi'], H), pad(p['phi'], H)
        else:
            pp = pf = states.real.new_zeros((len(states), H))
        psi, phi = self.model(x, pp, pf)
        return dict(psi=psi[:, :L], phi=phi[:, :L])


def save(model, path, **meta):
    torch.save(dict(n=model.n, H=model.H, kind=model.kind, param=model.param, state_dict=model.state_dict(),
                    scale=model.scale, landmarks=getattr(model, 'landmarks', None), **meta), path)


def load(path, train_cfg, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = Model(ck['n'], ck['H'], train_cfg, ck['kind'], ck['param'], ck['scale'], ck['landmarks']).to(device)
    m.load_state_dict(ck['state_dict'])
    return m.eval(), ck

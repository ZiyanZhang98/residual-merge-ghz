"""M2 of HANDOFF 22.28: the pairwise residual-merge cost model, its training and the deployment cost.

The model predicts, for merging atoms a and b into their weighted mean, the residual gap that the additive PTA
correction cannot remove, relative to the pair's own future success psi, as an RMS over the horizon blocks
2-3, 4-7, 8-15, 16-40 for P and for F (8 outputs, log scale). Inputs are symmetric in (a, b): the merged state's
Pauli coordinates, the squared coordinate differences, lambda(1-lambda), log Ward cost, the PTA gap of the pair
relative to its PTA level per prefix (P and phi), the PTA level and conditional fidelity, and the remaining word.
`features` is the single feature map used by training and deployment.

Deployment cost of a merge, in tolerance units: max over blocks and quantities of
W * max_{k in block} (psi_PTA,pair,k / Psi_PTA,ensemble,k) * predicted relative residual / tolerance,
i.e. the predicted change of the ensemble's P (or F) at the worst prefix of the block.
"""
import rm_runtime  # noqa: F401
import math
import numpy as np
import torch
from torch import nn
from qa_sources import PTA

H = 40
BLOCKS = ((2, 3), (4, 7), (8, 15), (16, 40))
OUT = 2*len(BLOCKS)   # [P blocks..., F blocks...]


def features(xa, xb, wa, wb, pta, rest, t, L):
    """pta: dict a_psi, b_psi, rep_psi, a_phi, b_phi, rep_phi of [P, H] (zero beyond L); rest [P, H] (0 A, 1 B, -1 pad)."""
    dev = xa.device
    lam = wa/(wa+wb)
    m = lam[:, None]*xa+(1-lam)[:, None]*xb
    d2 = (xa-xb)**2
    ward = lam*(1-lam)*d2.sum(1)
    mask = (torch.arange(H, device=dev)[None, :] < L[:, None]).to(xa.dtype)
    avg_psi = lam[:, None]*pta['a_psi']+(1-lam)[:, None]*pta['b_psi']
    avg_phi = lam[:, None]*pta['a_phi']+(1-lam)[:, None]*pta['b_phi']
    live = mask > 0
    lvl = torch.where(live & (avg_psi > 0), avg_psi, torch.ones_like(avg_psi))
    gpsi = torch.where(live, (avg_psi-pta['rep_psi'])/lvl, torch.zeros_like(lvl))
    gphi = torch.where(live, (avg_phi-pta['rep_phi'])/lvl, torch.zeros_like(lvl))
    Fb = torch.where(live, avg_phi/lvl, torch.zeros_like(lvl))
    logl = torch.where(live, torch.log(lvl), torch.zeros_like(lvl))
    rest = rest.to(xa.dtype)
    return torch.cat([m, d2, (lam*(1-lam))[:, None], torch.log(ward+1e-30)[:, None],
                      torch.asinh(gpsi/1e-4), torch.asinh(gphi/1e-4), Fb, logl,
                      (rest == 0).to(xa.dtype), (rest == 1).to(xa.dtype),
                      (t.to(xa.dtype)/H)[:, None], (L.to(xa.dtype)/H)[:, None]], 1)


def targets(d):
    """Debiased block RMS of the relative residual (P and F) per pair, and validity; numpy in, numpy out."""
    lam = d['wa']/(d['wa']+d['wb'])
    a_psi = lam[:, None]*d['pta_a_psi']+(1-lam)[:, None]*d['pta_b_psi']
    a_phi = lam[:, None]*d['pta_a_phi']+(1-lam)[:, None]*d['pta_b_phi']
    Fb = np.where(a_psi > 0, a_phi/np.where(a_psi > 0, a_psi, 1.), 0.)
    lvl = d['level_psi']
    ok = d['mask'] & (lvl > 0)
    lv = np.where(ok, lvl, 1.)
    rho = (d['r_psi']/lv, (d['r_phi']-Fb*d['r_psi'])/lv)
    se = (d['se_psi']/lv, np.sqrt(d['se_phi']**2+(Fb*d['se_psi'])**2)/lv)
    k = np.arange(1, H+1)[None, :]
    y, valid = np.zeros((len(lvl), OUT)), np.zeros((len(lvl), OUT), bool)
    for q in range(2):
        for b, (lo, hi) in enumerate(BLOCKS):
            sel = ok & (k >= lo) & (k <= hi)
            n = sel.sum(1)
            raw2 = np.where(sel, rho[q]**2, 0.).sum(1)/np.maximum(n, 1)
            noise2 = np.where(sel, se[q]**2, 0.).sum(1)/np.maximum(n, 1)
            j = q*len(BLOCKS)+b
            y[:, j] = np.sqrt(np.maximum(raw2-noise2, 0.05*raw2))
            valid[:, j] = (n > 0) & (raw2 > 0)
    return y, valid


def label_inputs(d, device):
    t = lambda key, dt=torch.float64: torch.as_tensor(d[key], dtype=dt, device=device)  # noqa: E731
    pta = {key[4:]: t(key) for key in d if key.startswith('pta_') and key[4:].split('_')[0] in ('a', 'b', 'rep')}
    return features(t('xa'), t('xb'), t('wa'), t('wb'), pta, t('rest', torch.int64), t('t'), t('L'))


class PairNet(nn.Module):
    def __init__(self, din, width, depth):
        super().__init__()
        layers, w = [], din
        for _ in range(depth):
            layers += [nn.Linear(w, width), nn.SiLU()]
            w = width
        layers.append(nn.Linear(w, OUT))
        self.net = nn.Sequential(*layers)
        self.register_buffer('mu', torch.zeros(din))
        self.register_buffer('sd', torch.ones(din))

    def forward(self, f):
        return self.net((f-self.mu)/self.sd)


def scalar(logy, valid):
    """log of the summed predicted residual over valid outputs (the ranking quantity)."""
    return torch.logsumexp(torch.where(valid, logy, torch.full_like(logy, -1e30)), 1)


def train(d, cfg, device, gen):
    tc = cfg['m2']
    fit = ~d['split']
    sub = {k: v[fit] for k, v in d.items()}
    X = label_inputs(sub, device)
    y, valid = targets(sub)
    Y = torch.as_tensor(np.log(np.maximum(y, 1e-30)), device=device)
    V = torch.as_tensor(valid, device=device)
    group = torch.as_tensor(sub['event']*2+sub['kind'].astype(np.int64), device=device)
    uniq, gidx = torch.unique(group, return_inverse=True)
    members = [torch.nonzero(gidx == g).flatten() for g in range(len(uniq))]
    torch.manual_seed(tc['seed'])
    model = PairNet(X.shape[1], tc['width'], tc['depth']).to(device).double()
    model.mu.copy_(X.mean(0))
    model.sd.copy_(X.std(0).clamp(min=1e-8))
    opt = torch.optim.Adam(model.parameters(), lr=tc['lr'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, tc['steps'], eta_min=tc['lr']*1e-2)
    log = []
    for step in range(tc['steps']):
        pick = torch.randint(len(members), (tc['batch_groups'],), generator=gen, device='cpu')
        idx = torch.cat([members[int(g)] for g in pick])
        out = model(X[idx])
        v, yt = V[idx], Y[idx]
        reg = torch.nn.functional.huber_loss(out[v], yt[v], delta=1.0)
        c, cs = scalar(out, v), scalar(yt, v)
        g = gidx[idx]
        same = g[:, None] == g[None, :]
        diff = cs[:, None]-cs[None, :]
        pm = same & (diff > tc['rank_margin'])
        rank = torch.nn.functional.softplus(-(c[:, None]-c[None, :]))[pm].mean() if bool(pm.any()) else c.new_zeros(())
        G = int(g.max())+1
        cm, csm = c.detach().max(), cs.max()
        ev = torch.zeros(G, dtype=c.dtype, device=device).index_add_(0, g, torch.exp(c-cm))
        evs = torch.zeros(G, dtype=c.dtype, device=device).index_add_(0, g, torch.exp(cs-csm))
        live = (ev > 0) & (evs > 0)
        cal = ((torch.log(ev[live])+cm-torch.log(evs[live])-csm)**2).mean()
        loss = reg+tc['rank_weight']*rank+tc['cal_weight']*cal
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step == tc['steps']-1:
            log.append(dict(step=step, loss=float(loss), reg=float(reg), rank=float(rank), cal=float(cal)))
            print('RM_TRAIN step %d loss %.4f reg %.4f rank %.4f cal %.4f' % (step, *[log[-1][k] for k in ('loss', 'reg', 'rank', 'cal')]), flush=True)
    return model, log


# ------------------------------------------------------------------ label-level evaluation

def concordance(score, target, group, margin):
    """Fraction of within-group pairs of pairs ordered like the target (target gap > margin)."""
    num = den = 0
    for g in np.unique(group):
        s = group == g
        if s.sum() < 2:
            continue
        a, b = score[s], target[s]
        dt = b[:, None]-b[None, :]
        m = dt > margin
        num += int(((a[:, None]-a[None, :]) > 0)[m].sum())
        den += int(m.sum())
    return dict(pairs=den, concordance=num/den if den else None)


@torch.no_grad()
def evaluate(model, d, device, cfg):
    """Label-level ranking of the residual (all blocks; long blocks 8-40 only) by the model, by C1 (Ward), by the
    PTA gap size, and by a linear fit of log residual on log Ward and log PTA gap (fitted on fit pairs)."""
    margin = cfg['m2']['rank_margin']
    X = label_inputs(d, device)
    y, valid = targets(d)
    logy = np.log(np.maximum(y, 1e-30))
    pred = model(X).cpu().numpy()
    lam = d['wa']/(d['wa']+d['wb'])
    ward = np.log(lam*(1-lam)*((d['xa']-d['xb'])**2).sum(1)+1e-30)
    a_psi = lam[:, None]*d['pta_a_psi']+(1-lam)[:, None]*d['pta_b_psi']
    gp = np.where(d['mask'] & (a_psi > 0), (a_psi-d['pta_rep_psi'])/np.where(a_psi > 0, a_psi, 1.), 0.)
    pta = np.log(np.sqrt((gp**2).sum(1)/np.maximum(d['mask'].sum(1), 1))+1e-30)
    lse = lambda v, m: np.log(np.maximum(np.where(m, np.exp(np.clip(v, -700, 700)), 0.).sum(1), 1e-300))  # noqa: E731
    long_cols = np.zeros(OUT, bool)
    long_cols[[2, 3, 6, 7]] = True
    tgt_all, tgt_long = lse(logy, valid), lse(logy, valid & long_cols[None, :])
    mod_all, mod_long = lse(pred, valid), lse(pred, valid & long_cols[None, :])
    fit = ~d['split']
    A = np.stack([np.ones(len(ward)), ward, pta], 1)
    coef = np.linalg.lstsq(A[fit], tgt_all[fit], rcond=None)[0]
    lin = A @ coef
    group = d['event']*2+d['kind'].astype(np.int64)
    out = dict(linear_coefficients=coef.tolist())
    for split, sel in (('fit', fit), ('monitor', ~fit)):
        for fam in range(6):
            for name, s in (('all', sel), ('fam%d' % fam, sel & (d['family'] == fam))):
                if name == 'all' and fam > 0:
                    continue
                key = '%s_%s' % (split, name)
                res = {}
                for horizon, tg, md in (('all_blocks', tgt_all, mod_all), ('long_blocks', tgt_long, mod_long)):
                    ok = s & np.isfinite(tg) & (tg > -600)
                    res[horizon] = {m: concordance(sc[ok], tg[ok], group[ok], margin)['concordance']
                                    for m, sc in (('learned', md), ('C1_ward', ward), ('pta_gap', pta), ('linear_ward_pta', lin))}
                v = valid & s[:, None]
                res['log_rmse_by_output'] = [float(np.sqrt(np.mean((pred[v[:, j], j]-logy[v[:, j], j])**2))) if v[:, j].any() else None
                                             for j in range(OUT)]
                out[key] = res
    return out


# ------------------------------------------------------------------ deployment

def load_model(path, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    model = PairNet(blob['din'], blob['width'], blob['depth']).to(device).double()
    model.load_state_dict(blob['state'])
    model.eval()
    return model


def pad(v):
    out = v.new_zeros((v.shape[0], H))
    out[:, :v.shape[1]] = v
    return out


class LearnedCost:
    """Pair cost in tolerance units from the learned relative residual (see the module docstring)."""
    name = 'learned'

    def __init__(self, model, tol):
        self.model, self.tol = model, tol

    def prepare(self, sim, states, w):
        self.ctx, self.c, self.hard, self.t = sim.ctx, sim.c, sim.hard, sim.t
        self.rest = sim.word[sim.t:]
        L = len(self.rest)
        vp = PTA()(states, None, self.rest, self.ctx, self.c, self.hard)
        self.Psi = pad((w @ vp['psi'])[None])[0]
        self.cache, self.pairs = {}, {}
        self.evaluated = 0
        code = torch.full((H,), -1, dtype=torch.int64, device=w.device)
        code[:L] = torch.as_tensor([0 if x == 'A' else 1 for x in self.rest], device=w.device)
        self.code, self.L = code, L
        lo = torch.arange(1, H+1, device=w.device)
        self.block_masks = [(lo >= a) & (lo <= b) & (lo <= L) for a, b in BLOCKS]

    def pta_of(self, S, ids):
        ids_list = ids.tolist()
        missing = sorted({i for i in ids_list if i not in self.cache})
        if missing:
            m = torch.as_tensor(missing, device=S.device)
            v = PTA()(S[m], None, self.rest, self.ctx, self.c, self.hard)
            for j, i in enumerate(missing):
                self.cache[i] = (v['psi'][j], v['phi'][j])
        return pad(torch.stack([self.cache[i][0] for i in ids_list])), pad(torch.stack([self.cache[i][1] for i in ids_list]))

    @torch.no_grad()
    def __call__(self, atoms, I, J):
        """Atoms never change once created within a compression, so a pair's cost is computed once and cached."""
        keys = list(zip(I.tolist(), J.tolist()))
        new = [n for n, k in enumerate(keys) if k not in self.pairs]
        if new:
            sel = torch.as_tensor(new, device=I.device)
            fresh = self.price(atoms, I[sel], J[sel]).tolist()
            for n, v in zip(new, fresh):
                self.pairs[keys[n]] = v
            self.evaluated += len(new)
        return torch.as_tensor([self.pairs[k] for k in keys], dtype=atoms['w'].dtype, device=I.device)

    def price(self, atoms, I, J):
        S, w, x = atoms['S'], atoms['w'], atoms['x']
        P = len(I)
        a_psi, a_phi = self.pta_of(S, I)
        b_psi, b_phi = self.pta_of(S, J)
        lam = w[I]/(w[I]+w[J])
        reps = lam[:, None, None]*S[I]+(1-lam)[:, None, None]*S[J]
        vr = PTA()(reps, None, self.rest, self.ctx, self.c, self.hard)
        pta = dict(a_psi=a_psi, b_psi=b_psi, rep_psi=pad(vr['psi']), a_phi=a_phi, b_phi=b_phi, rep_phi=pad(vr['phi']))
        f = features(x[I], x[J], w[I], w[J], pta, self.code[None].expand(P, H),
                     torch.full((P,), self.t, device=w.device), torch.full((P,), self.L, device=w.device))
        rho = torch.exp(self.model(f))
        avg = lam[:, None]*a_psi+(1-lam)[:, None]*b_psi
        share = torch.where(self.Psi[None] > 0, avg/torch.where(self.Psi > 0, self.Psi, torch.ones_like(self.Psi))[None], torch.zeros_like(avg))
        W = w[I]+w[J]
        cost = torch.zeros(P, dtype=w.dtype, device=w.device)
        for b, bm in enumerate(self.block_masks):
            if not bool(bm.any()):
                continue
            sh = share[:, bm].max(1).values
            cost = torch.maximum(cost, W*sh*rho[:, b]/self.tol['P_relative'])
            cost = torch.maximum(cost, W*sh*rho[:, len(BLOCKS)+b]/self.tol['F_absolute'])
        return torch.nan_to_num(cost, nan=1e30, posinf=1e30)

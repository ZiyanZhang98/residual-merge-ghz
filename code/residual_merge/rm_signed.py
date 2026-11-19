"""Step 2 of HANDOFF 22.29: signed expected residual plus uncertainty, and a budget on the signed sum.

Per merge (pair a, b) the model outputs, for every later prefix k of P and of F, a signed expected relative residual
mu_k = sigma_b(k) m_k, and per horizon block b a log uncertainty log sigma_b (88 outputs). Training: the known-noise
Gaussian likelihood r_k ~ N(mu_k, sigma_b(k)^2 + se_k^2) on the existing labels (fit preparations only).

Budget: on the scale of the deployment cost (rm_model.LearnedCost), merge m changes the ensemble's P (or F) at
prefix k by c_k = W share_k mu_k / tolerance with variance v_k = (W share_k sigma_k / tolerance)^2. Within one
compression the accepted merges accumulate M_k = sum c and V_k = sum v; candidates are taken in increasing
single-pair risk max_k (|c_k| + kappa sqrt(v_k)) while max_k (|M_k| + kappa sqrt(V_k)) stays within the round's
budget (kappa = 2, fixed). The atom cap, forced merges, mean-preserving merges and the PTA correction are unchanged
(rm_core.agglomerate otherwise). No new labels; final data are never accessed.
"""
import rm_runtime as rt
from pathlib import Path
import numpy as np
import torch
from torch import nn
from qa_sources import PTA
from rm_core import knn_pairs, Agglomerative, adaptive_rollout
from rm_model import features, label_inputs, LearnedCost, BLOCKS, H
from rm_lik import load_labels, lik_terms, FAMILIES, _t

NB = len(BLOCKS)
OUT2 = 2*H+2*NB
MODEL_FILES = ('rm_signed.py', 'rm_lik.py', 'rm_model.py', 'rm_core.py', 'rm_labels.py', 'rm_runtime.py', 'rm_report.py')
MODEL_KEYS = ('labels', 'm1', 'merge', 'm2', 'tolerance', 's1', 's2')


def block_index():
    k = np.arange(1, H+1)
    bidx = np.full(H, -1)
    for b, (lo, hi) in enumerate(BLOCKS):
        bidx[(k >= lo) & (k <= hi)] = b
    return bidx


class SignedNet(nn.Module):
    """PairNet's body with 2H standardized signed outputs m and 2*NB log uncertainties."""
    def __init__(self, din, width, depth):
        super().__init__()
        layers, w = [], din
        for _ in range(depth):
            layers += [nn.Linear(w, width), nn.SiLU()]
            w = width
        layers.append(nn.Linear(w, OUT2))
        self.net = nn.Sequential(*layers)
        self.register_buffer('mu', torch.zeros(din))
        self.register_buffer('sd', torch.ones(din))

    def forward(self, f):
        return self.net((f-self.mu)/self.sd)


def split_out(out, bidx_t):
    """out [n, OUT2] -> mu, sigma [n, 2, H] (zero on prefixes outside the blocks)."""
    n = out.shape[0]
    m = out[:, :2*H].reshape(n, 2, H)
    logs = out[:, 2*H:]
    cols = torch.stack([q*NB+bidx_t.clamp(min=0) for q in range(2)])
    live = (bidx_t >= 0)[None, None, :]
    sig = torch.where(live, torch.exp(logs[:, cols]), torch.zeros_like(m))
    return sig*m, sig


# ------------------------------------------------------------------ training

def train_signed(d, T, cfg, device, gen):
    tc = cfg['s2']
    fit = ~d['split']
    sub = {k: v[fit] for k, v in d.items()}
    X = label_inputs(sub, device)
    r, se, wt = _t(T['r'][fit], device), _t(T['se'][fit], device), _t(T['wt'][fit], device)
    nvalid = _t(T['valid'][fit].sum(1), device)
    bidx_t = torch.as_tensor(block_index(), device=device)
    group = torch.as_tensor(sub['event']*2+sub['kind'].astype(np.int64), device=device)
    uniq, gidx = torch.unique(group, return_inverse=True)
    members = [torch.nonzero(gidx == g).flatten() for g in range(len(uniq))]
    torch.manual_seed(tc['seed'])
    model = SignedNet(X.shape[1], tc['width'], tc['depth']).to(device).double()
    model.mu.copy_(X.mean(0))
    model.sd.copy_(X.std(0).clamp(min=1e-8))
    with torch.no_grad():
        last = model.net[-1]
        last.weight[:2*H].mul_(1e-2)
        last.bias[:2*H].zero_()
        lograw = np.log(np.sqrt(np.maximum(T['raw2'][fit], 1e-300)))
        valid = T['valid'][fit]
        last.bias[2*H:].copy_(_t([np.median(lograw[valid[:, j], j]) if valid[:, j].any() else 0. for j in range(2*NB)], device))
    opt = torch.optim.Adam(model.parameters(), lr=tc['lr'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, tc['steps'], eta_min=tc['lr']*1e-2)
    log = []
    for step in range(tc['steps']):
        pick = torch.randint(len(members), (tc['batch_groups'],), generator=gen, device='cpu')
        idx = torch.cat([members[int(g)] for g in pick])
        mu, sig = split_out(model(X[idx]), bidx_t)
        w = wt[idx]
        live = w > 0
        var = torch.where(live, sig**2+se[idx]**2, torch.ones_like(sig))
        term = 0.5*(torch.log(var)+(r[idx]-mu)**2/var)
        loss = torch.where(live, w*term, torch.zeros_like(term)).sum()/nvalid[idx].sum().clamp(min=1)
        if not bool(torch.isfinite(loss)):
            return model, log, 'NONFINITE_LOSS'
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step == tc['steps']-1:
            log.append(dict(step=step, loss=float(loss)))
            print('RM_SGN step %d loss %.4f' % (step, log[-1]['loss']), flush=True)
    return model, log, 'OK'


# ------------------------------------------------------------------ label-level coverage

def scaled(d, tol):
    """W * share_k / tolerance per pair, quantity and prefix (zero beyond L): the budget scale of the deployment."""
    lam = d['wa']/(d['wa']+d['wb'])
    W = d['wa']+d['wb']
    avg = lam[:, None]*d['pta_a_psi']+(1-lam)[:, None]*d['pta_b_psi']
    Psi = d['Psi']
    share = np.where(Psi > 0, avg/np.where(Psi > 0, Psi, 1.), 0.)
    k = np.arange(1, H+1)
    live = (k[None, :] <= d['L'][:, None]) & (block_index() >= 0)[None, :]
    base = np.where(live, W[:, None]*share, 0.)
    return np.stack([base/tol['P_relative'], base/tol['F_absolute']], 1)


@torch.no_grad()
def predict(model, d, device, chunk=8192):
    X = label_inputs(d, device)
    bidx_t = torch.as_tensor(block_index(), device=device)
    mus, sigs = [], []
    for s in range(0, len(X), chunk):
        mu, sig = split_out(model(X[s:s+chunk]), bidx_t)
        mus.append(mu.cpu().numpy())
        sigs.append(sig.cpu().numpy())
    return np.concatenate(mus), np.concatenate(sigs)


def coverage(d, T, mu, sig, tol, kappa, rho=0.):
    """Per capture group (event x atom set) and prefix: |sum x - sum c| <= kappa sqrt(sum v + sum se_x^2 + rho * cross),
    x, c, v on the budget scale; cross = (sum sqrt v)^2 - sum v (intra-group correlation term). Returns coverage per
    split and family and the per-cell standardized errors."""
    g = scaled(d, tol)
    ok = T['wt'] > 0
    x, c = np.where(ok, g*T['r'], 0.), np.where(ok, g*mu, 0.)
    v, sx2 = np.where(ok, (g*sig)**2, 0.), np.where(ok, (g*T['se'])**2, 0.)
    group = d['event']*2+d['kind'].astype(np.int64)
    uniq, inv = np.unique(group, return_inverse=True)
    G = len(uniq)

    def gs(a):
        out = np.zeros((G,)+a.shape[1:])
        np.add.at(out, inv, a)
        return out
    X, C, V, SX, SV, N = gs(x), gs(c), gs(v), gs(sx2), gs(np.sqrt(v)), gs(ok.astype(float))
    cross = np.maximum(SV**2-V, 0.)
    den = np.sqrt(V+SX+rho*cross)
    live = (N > 0) & (den > 0)
    z = np.where(live, np.abs(X-C)/np.where(den > 0, den, 1.), np.nan)
    first = np.zeros(G, dtype=np.int64)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    fam, split = d['family'][first], d['split'][first]
    out = {}

    def cover(zz):   # only live cells: (NaN <= kappa) is False, so the mean must be taken over finite z
        fin = np.isfinite(zz)
        return float(np.mean(zz[fin] <= kappa)) if fin.any() else None
    for spname, sp in (('fit', False), ('monitor', True)):
        sel = split == sp
        out[spname] = dict(all=cover(z[sel]), cells=int(np.isfinite(z[sel]).sum()))
        for i, f in enumerate(FAMILIES):
            out[spname][f] = cover(z[sel & (fam == i)])
    return out


def risk(d, mu, sig, tol, kappa):
    g = scaled(d, tol)
    return (np.abs(g*mu)+kappa*g*sig).reshape(len(mu), -1).max(1)


def fit_rho(d, T, mu, sig, tol, kappa, target):
    """Smallest intra-group correlation rho in [0, 1] (bisection) whose fit coverage reaches the target."""
    if coverage(d, T, mu, sig, tol, kappa, 0.)['fit']['all'] >= target:
        return 0.
    lo, hi = 0., 1.
    if coverage(d, T, mu, sig, tol, kappa, hi)['fit']['all'] < target:
        return None
    for _ in range(20):
        mid = 0.5*(lo+hi)
        if coverage(d, T, mu, sig, tol, kappa, mid)['fit']['all'] >= target:
            hi = mid
        else:
            lo = mid
    return hi


# ------------------------------------------------------------------ deployment cost and agglomeration

class SignedCost(LearnedCost):
    """Per candidate merge: signed contributions c and variances v [2, H] on the budget scale (cached per pair)."""
    name = 'signed'

    def __init__(self, model, tol, kappa):
        super().__init__(model, tol)
        self.kappa = kappa
        dev = next(model.parameters()).device
        self.bidx_t = torch.as_tensor(block_index(), device=dev)
        self.tolv = torch.tensor([tol['P_relative'], tol['F_absolute']], dtype=torch.float64, device=dev)[None, :, None]

    @torch.no_grad()
    def contrib(self, atoms, I, J):
        keys = list(zip(I.tolist(), J.tolist()))
        new = [n for n, k in enumerate(keys) if k not in self.pairs]
        if new:
            sel = torch.as_tensor(new, device=I.device)
            c, v = self.compute(atoms, I[sel], J[sel])
            for n, a, b in zip(new, c, v):
                self.pairs[keys[n]] = (a, b)
            self.evaluated += len(new)
        c = np.stack([self.pairs[k][0] for k in keys])
        v = np.stack([self.pairs[k][1] for k in keys])
        return c, v

    def compute(self, atoms, I, J):
        S, w, x = atoms['S'], atoms['w'], atoms['x']
        P = len(I)
        a_psi, a_phi = self.pta_of(S, I)
        b_psi, b_phi = self.pta_of(S, J)
        lam = w[I]/(w[I]+w[J])
        reps = lam[:, None, None]*S[I]+(1-lam)[:, None, None]*S[J]
        vr = PTA()(reps, None, self.rest, self.ctx, self.c, self.hard)
        from rm_model import pad
        pta = dict(a_psi=a_psi, b_psi=b_psi, rep_psi=pad(vr['psi']), a_phi=a_phi, b_phi=b_phi, rep_phi=pad(vr['phi']))
        f = features(x[I], x[J], w[I], w[J], pta, self.code[None].expand(P, H),
                     torch.full((P,), self.t, device=w.device), torch.full((P,), self.L, device=w.device))
        mu, sig = split_out(self.model(f), self.bidx_t)
        avg = lam[:, None]*a_psi+(1-lam)[:, None]*b_psi
        share = torch.where(self.Psi[None] > 0, avg/torch.where(self.Psi > 0, self.Psi, torch.ones_like(self.Psi))[None],
                            torch.zeros_like(avg))
        k = torch.arange(1, H+1, device=w.device)
        live = ((k <= self.L) & (self.bidx_t >= 0))[None, None, :]
        base = (w[I]+w[J])[:, None, None]*share[:, None, :]/self.tolv
        c = torch.where(live, base*mu, torch.zeros_like(mu))
        v = torch.where(live, (base*sig)**2, torch.zeros_like(sig))
        if not (bool(torch.isfinite(c).all()) and bool(torch.isfinite(v).all())):
            raise FloatingPointError('Invalid signed merge cost')
        return c.cpu().numpy(), v.cpu().numpy()


@torch.no_grad()
def agglomerate_signed(S, w, x, cost, budget, cap, knn, max_rounds, kappa):
    """rm_core.agglomerate with the signed budget: accept disjoint candidate pairs in increasing single-pair risk while
    max_k (|M_k + c_k| + kappa sqrt(V_k + v_k)) <= budget; beyond the atom cap the cheapest further pairs are merged
    regardless (forced, outside the budget, as in rm_core)."""
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
    M, V = np.zeros((2, H)), np.zeros((2, H))
    top, merges, forced, rounds = N, 0, 0, 0
    while len(alive) > 1 and rounds < max_rounds:
        need = len(alive)-cap
        P = knn_pairs(AX[alive], knn)
        gi, gj = alive[P[:, 0]], alive[P[:, 1]]
        c, v = cost.contrib(dict(S=AS, w=AW, x=AX), gi, gj)
        rk = (np.abs(c)+kappa*np.sqrt(v)).reshape(len(c), -1).max(1)
        order = np.argsort(rk, kind='stable').tolist()
        pi, pj = P[:, 0].cpu().tolist(), P[:, 1].cpu().tolist()
        used, take, extra = set(), [], []
        for k in order:
            i, j = pi[k], pj[k]
            if i in used or j in used:
                continue
            Mn, Vn = M+c[k], V+v[k]
            if float((np.abs(Mn)+kappa*np.sqrt(Vn)).max()) <= budget:
                take.append(k)
                M, V = Mn, Vn
            elif len(take)+len(extra) < need:
                extra.append(k)
            else:
                break
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
    used_budget = float((np.abs(M)+kappa*np.sqrt(V)).max())
    return labels, dict(before=N, after=int(labels.max())+1, merges=merges, forced=forced, spent=used_budget, rounds=rounds)


class SignedPolicy(Agglomerative):
    def __init__(self, cost, budget, cap, knn, max_rounds, kappa):
        super().__init__(cost, budget, cap, knn, max_rounds)
        self.kappa = kappa

    def __call__(self, sim, states, w):
        from ghz_physics import coords
        self.cost.prepare(sim, states, w)
        x = coords(states, sim.ctx['aux']['pauli'])
        labels, info = agglomerate_signed(states, w, x, self.cost, self.budget, self.cap, self.knn, self.max_rounds, self.kappa)
        info['pairs_priced'] = self.cost.evaluated
        return labels, dict(policy='signed', budget=self.budget, **info)


def run_signed(cx, cfg, task, ctx, states, weights, ref, alpha, model):
    """rm_deploy.run_one for the step-2 model (same scoring and bookkeeping)."""
    import time
    from g1_core import check_deadline, Deadline
    from hr_eval import scored
    check_deadline(cx.hard)
    mg, kappa = cfg['merge'], cfg['s2']['kappa']
    pol = SignedPolicy(SignedCost(model, cfg['tolerance'], kappa), alpha/cfg['m3']['horizon'], mg['cap'], mg['knn'],
                       mg['max_rounds'], kappa)
    t0 = time.perf_counter()
    try:
        sim, rho, ell, est = adaptive_rollout(pol, task['word'], ctx, cx.c, states, weights, {'M_PTA': PTA()}, cx.hard)
    except Deadline:
        raise
    except (FloatingPointError, RuntimeError) as exc:
        return None, dict(task_id=task['id'], family=task['family'], method='S', budget=alpha, status='NUMERIC_FAILURE',
                          error=repr(exc))
    log = sim.merge_log
    extra = dict(method='S', budget=alpha, physical_calls=sim.cost['physical_calls'], seconds=time.perf_counter()-t0,
                 max_atoms=max([e['after'] for e in log] or [0]), forced=sum(e.get('forced', 0) for e in log),
                 merge_seconds=sum(e['seconds'] for e in log))
    c = scored(task, 'S+M_PTA', None, est['M_PTA'], ref, cfg['tolerance'], extra)
    run = dict(task_id=task['id'], family=task['family'], prep_id=task['prep_signature'], status=c['status'],
               Q_QoI=c.get('Q_QoI'), **extra)
    print('RM_SGN_RUN %s alpha=%g calls=%d max_atoms=%d forced=%d Q=%s %.1fs' % (
        task['id'], alpha, extra['physical_calls'], extra['max_atoms'], extra['forced'], run['Q_QoI'], extra['seconds']), flush=True)
    return c, run


# ------------------------------------------------------------------ artifacts and jobs

def save_model(model, tc, path):
    torch.save(dict(din=model.mu.numel(), width=tc['width'], depth=tc['depth'], state=model.state_dict()), path)


def load_signed(path, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    model = SignedNet(blob['din'], blob['width'], blob['depth']).to(device).double()
    model.load_state_dict(blob['state'])
    model.eval()
    return model


def verified_sgn(root):
    m = rt.read(root/'manifest.json')
    if m['status'] != 'SGN_COMPLETE':
        raise ValueError('Incomplete step-2 training artifact: '+str(root))
    cfg = rt.config()
    if any(m['config'][k] != cfg[k] for k in MODEL_KEYS):
        raise ValueError('Model settings changed since: '+str(root))
    mine = str(rt.HERE.relative_to(rt.REPO))+'/'
    keep = lambda s: {k: v for k, v in s.items() if not k.startswith(mine) or Path(k).name in MODEL_FILES}  # noqa: E731
    if not __import__('qn_public').same_sources(keep(m['sources']), keep(rt.sources())):  # PUBLIC: see qn_public.py
        raise ValueError('Model-defining sources changed since: '+str(root))
    for filename, value in m['files'].items():
        if __import__('qn_public').present(root/filename) and rt.sha(root/filename) != value:
            raise ValueError('Step-2 training artifact content changed: '+str(root/filename))
    return m


def run_training(cx, cfg, root, inputs, save):
    """Train the signed model, then the pre-registered coverage check (monitor >= gate; otherwise one intra-group
    correlation fitted on fit groups, re-checked on monitor). Writes model.pt, checks.json, decision.json."""
    from rm_model import load_model
    from rm_verify import verified_model
    from rm_lik import deploy_cost
    tc = cfg['s2']
    d = load_labels(cfg['s1']['labels_jobs'], inputs)
    T = lik_terms(d, cfg['s1']['resolved_ratio'])
    gen = torch.Generator().manual_seed(int(tc['seed']))
    model, log, status = train_signed(d, T, cfg, cx.device, gen)
    save('train_log.json', log)
    if status != 'OK':
        save('decision.json', dict(deploy=False, status=status))
        return dict(deploy=False, status=status)
    save_model(model, tc, root/'model.pt')
    mu, sig = predict(model, d, cx.device)
    kappa = tc['kappa']
    cov0 = coverage(d, T, mu, sig, cfg['tolerance'], kappa)
    out = dict(coverage=cov0, rho=0.)
    deploy = cov0['monitor']['all'] is not None and cov0['monitor']['all'] >= tc['coverage_gate']
    if not deploy:
        rho = fit_rho(d, T, mu, sig, cfg['tolerance'], kappa, tc['coverage_fit_target'])
        out['rho'] = rho
        if rho is not None:
            cov1 = coverage(d, T, mu, sig, cfg['tolerance'], kappa, rho)
            out['coverage_rho'] = cov1
            deploy = cov1['monitor']['all'] >= tc['coverage_gate']
    # information: single-pair risk against the old model's cost, per family (monitor), and |mu|/sigma
    old_root = rt.HERE/'results'/('train_%s' % cfg['s1']['old_train_job'])
    verified_model(old_root)
    inputs[old_root.name] = rt.sha(old_root/'manifest.json')
    old = load_model(old_root/'model.pt', cx.device)
    with torch.no_grad():
        s_old = np.exp(old(label_inputs(d, cx.device)).cpu().numpy())
    old_cost = deploy_cost(d, s_old, cfg['tolerance']).max(1)
    rk = risk(d, mu, sig, cfg['tolerance'], kappa)
    fam, mon = d['family'], d['split']
    out['risk_over_old_cost_median'] = {f: float(np.median(rk[mon & (fam == i)]/np.maximum(old_cost[mon & (fam == i)], 1e-300)))
                                        for i, f in enumerate(FAMILIES)}
    ratio = np.abs(mu)/np.where(sig > 0, sig, np.nan)
    out['median_abs_mu_over_sigma'] = {f: float(np.nanmedian(ratio[mon & (fam == i)])) for i, f in enumerate(FAMILIES)}
    decision = dict(deploy=bool(deploy), status='OK', rho=out['rho'], monitor_coverage=cov0['monitor']['all'])
    save('checks.json', out)
    save('decision.json', decision)
    print('RM_SGN coverage', cov0, 'rho', out['rho'], flush=True)
    print('RM_SGN risk/old', out['risk_over_old_cost_median'], flush=True)
    print('RM_SGN |mu|/sigma', out['median_abs_mu_over_sigma'], flush=True)
    return decision


def smoke(cx, cfg, inputs):
    """(1) with c = +GeoCost on one prefix and v = 0 the signed rule selects the same merges as rm_core.agglomerate;
    (2) short training is finite and the coverage check runs; (3) one signed rollout runs end to end."""
    from rm_core import agglomerate, GeoCost
    from ghz_physics import coords
    from hr_eval import context as task_context
    from rm_deploy import task_list
    out = {}
    tasks, npz, _ = task_list(cfg, inputs)
    task = next(t for t in tasks if t['id'] == cfg['eval2']['smoke']['task'])
    states, weights, ctx = task_context(cx, task)
    x = coords(states, ctx['aux']['pauli'])

    class GeoAsSigned:
        def __init__(self):
            self.geo, self.evaluated = GeoCost(), 0

        def contrib(self, atoms, I, J):
            g = self.geo(atoms, I, J).cpu().numpy()
            c = np.zeros((len(g), 2, H))
            c[:, 0, 5] = g
            return c, np.zeros_like(c)
    mg = cfg['merge']
    res = []
    for budget in (1e-4, 1e-3, 1e-2):
        la, ia = agglomerate(states, weights, x, GeoCost(), budget, 8, mg['knn'], mg['max_rounds'])
        lb, ib = agglomerate_signed(states, weights, x, GeoAsSigned(), budget, 8, mg['knn'], mg['max_rounds'], 2.)
        res.append(dict(budget=budget, after=[ia['after'], ib['after']], same=bool(torch.equal(la, lb))))
    out['rule_reduces_to_original'] = dict(runs=res, passed=all(r['same'] for r in res))
    d = load_labels(cfg['s1']['labels_jobs'], inputs)
    T = lik_terms(d, cfg['s1']['resolved_ratio'])
    short = dict(cfg, s2=dict(cfg['s2'], steps=50))
    model, log, status = train_signed(d, T, short, cx.device, torch.Generator().manual_seed(0))
    mu, sig = predict(model, d, cx.device)
    cov = coverage(d, T, mu, sig, cfg['tolerance'], cfg['s2']['kappa'])
    out['short_training'] = dict(status=status, last=log[-1] if log else None, monitor_coverage=cov['monitor']['all'],
                                 passed=status == 'OK' and cov['monitor']['all'] is not None)
    ref = dict(np.load(npz[task['id']]))
    c, run = run_signed(cx, cfg, task, ctx, states, weights, ref, 16, model)
    out['signed_rollout'] = dict(status=run['status'], calls=run.get('physical_calls'), Q=run.get('Q_QoI'),
                                 seconds=run.get('seconds'), passed=run['status'] == 'OK')
    return out

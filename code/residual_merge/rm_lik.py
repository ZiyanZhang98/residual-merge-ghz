"""Step 1 of HANDOFF 22.29: the known-noise likelihood replaces the noise floor of the M2 targets.

M2 (rm_model.targets) fits log y with y = sqrt(max(raw2 - noise2, 0.05 raw2)) per horizon block, so a pair whose
Monte Carlo residual is pure noise gets a target of about 0.22-1 times the noise level. Here the labels enter as
they are: the relative residual r_k of every prefix k is modelled as r_k ~ N(0, s_b^2 + se_k^2), with s_b the
model output (the block RMS of the true relative residual, the same quantity as before) and se_k the label's own
Monte Carlo standard error (0 on exact prefixes). The network, features, output meaning and deployment cost
(rm_model.LearnedCost) are unchanged.

Arms: 1B = likelihood only (trained first); 1A = likelihood + the M2 rank and calibration terms on floor-free
estimates (only if 1B fails the label-level progress check). No new labels; fit preparations only; final data are
never accessed.
"""
import rm_runtime as rt
from pathlib import Path
import numpy as np
import torch
from rm_model import PairNet, label_inputs, evaluate, scalar, BLOCKS, OUT, H
from rm_report import shards, verified

FAMILIES = ('A', 'B', 'AB', 'AAB', 'AABB', 'ABB')
HARD = ('AB', 'AABB', 'ABB')
EXACT_OUTPUTS = (0, len(BLOCKS))   # block 2-3 of P and of F: exact-tree prefixes only
MODEL_FILES = ('rm_lik.py', 'rm_model.py', 'rm_core.py', 'rm_labels.py', 'rm_runtime.py', 'rm_report.py')
MODEL_KEYS = ('labels', 'm1', 'merge', 'm2', 'tolerance', 's1')


# ------------------------------------------------------------------ data

def load_labels(jobs, inputs):
    """M1 and early label shards (verified as in rm_report.load, same event offsets as run.py train), plus the
    ensemble PTA level Psi of each row's capture (events.json), which the deployment cost needs."""
    parts = []
    for i, job in enumerate(jobs):
        for r in shards(job):
            verified(r)
            inputs[r.name] = rt.sha(r/'manifest.json')
            d = torch.load(r/'labels.pt', map_location='cpu', weights_only=False)
            rows = {k: v.numpy() for k, v in d['rows'].items()}
            ev = {int(e['event']): e for e in rt.read(r/'events.json')}
            rows['Psi'] = np.stack([np.asarray(ev[int(e)]['Psi_pta'], dtype=np.float64) for e in rows['event']])
            rows['event'] = rows['event']+1_000_000*int(r.name.split('_')[-1])+100_000_000*i
            parts.append(rows)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def lik_terms(d, resolved_ratio):
    """Per prefix: relative residual r and its standard error se for P and F ([pairs, 2, H], same definitions as
    rm_model.targets); per output: block weights (1/n on the block's valid prefixes), validity, raw and noise mean
    squares, whether the block is exact, and the floor-free estimate shat = sqrt(max(raw2 - noise2, 0))."""
    lam = d['wa']/(d['wa']+d['wb'])
    a_psi = lam[:, None]*d['pta_a_psi']+(1-lam)[:, None]*d['pta_b_psi']
    a_phi = lam[:, None]*d['pta_a_phi']+(1-lam)[:, None]*d['pta_b_phi']
    Fb = np.where(a_psi > 0, a_phi/np.where(a_psi > 0, a_psi, 1.), 0.)
    lvl = d['level_psi']
    ok = d['mask'] & (lvl > 0)
    lv = np.where(ok, lvl, 1.)
    r = np.stack([d['r_psi']/lv, (d['r_phi']-Fb*d['r_psi'])/lv], 1)
    se = np.stack([d['se_psi']/lv, np.sqrt(d['se_phi']**2+(Fb*d['se_psi'])**2)/lv], 1)
    r, se = np.where(ok[:, None, :], r, 0.), np.where(ok[:, None, :], se, 0.)
    P = len(lvl)
    k = np.arange(1, H+1)
    wt = np.zeros((P, 2, H))
    valid, exact = np.zeros((P, OUT), bool), np.zeros((P, OUT), bool)
    raw2, noise2 = np.zeros((P, OUT)), np.zeros((P, OUT))
    bidx = np.full(H, -1)
    for b, (lo, hi) in enumerate(BLOCKS):
        bidx[(k >= lo) & (k <= hi)] = b
    for q in range(2):
        for b, (lo, hi) in enumerate(BLOCKS):
            sel = ok & (k >= lo) & (k <= hi)
            n = sel.sum(1)
            j = q*len(BLOCKS)+b
            raw2[:, j] = np.where(sel, r[:, q]**2, 0.).sum(1)/np.maximum(n, 1)
            noise2[:, j] = np.where(sel, se[:, q]**2, 0.).sum(1)/np.maximum(n, 1)
            valid[:, j] = (n > 0) & (raw2[:, j] > 0)
            exact[:, j] = valid[:, j] & ~np.any(sel & (se[:, q] > 0), 1)
            wt[:, q] += np.where(sel & valid[:, j][:, None], 1./np.maximum(n, 1)[:, None], 0.)
    shat = np.sqrt(np.maximum(raw2-noise2, 0.))
    resolved = np.all(~valid | (raw2 >= resolved_ratio*noise2), 1)
    return dict(r=r, se=se, wt=wt, valid=valid, exact=exact, raw2=raw2, noise2=noise2, shat=shat,
                resolved=resolved, bidx=bidx)


def _t(a, device, dtype=torch.float64):
    return torch.as_tensor(a, dtype=dtype, device=device)


def nll_terms(out, r, se, wt, bidx):
    """Sum over prefixes of wt * 0.5 (log(s^2 + se^2) + r^2/(s^2 + se^2)), s = exp(out[block of the prefix])."""
    cols = torch.stack([q*len(BLOCKS)+bidx.clamp(min=0) for q in range(2)])   # [2, H]
    var = torch.exp(2*out[:, cols])+se**2
    live = wt > 0
    var = torch.where(live, var, torch.ones_like(var))
    term = 0.5*(torch.log(var)+r**2/var)
    return torch.where(live, wt*term, torch.zeros_like(term)).sum()


# ------------------------------------------------------------------ training

def train_lik(d, T, cfg, device, gen, arm, exact_huber):
    """Arm '1B' (likelihood) or '1A' (likelihood + floor-free rank and calibration), on fit pairs. With
    exact_huber the exact outputs (block 2-3) keep the M2 log-Huber fit on their exact RMS (pre-registered
    remedy for divergence or a bad exact-block calibration)."""
    tc = cfg['s1']
    fit = ~d['split']
    sub = {k: v[fit] for k, v in d.items()}
    X = label_inputs(sub, device)
    r, se, wt = _t(T['r'][fit], device), _t(T['se'][fit], device), T['wt'][fit].copy()
    valid, exact = T['valid'][fit], T['exact'][fit]
    huber_out = np.zeros_like(valid)
    if exact_huber:
        huber_out[:, list(EXACT_OUTPUTS)] = exact[:, list(EXACT_OUTPUTS)]
        k = np.arange(1, H+1)
        lo, hi = BLOCKS[0]
        blk = (k >= lo) & (k <= hi)
        for q, j in enumerate(EXACT_OUTPUTS):
            wt[:, q, :] = np.where(huber_out[:, j][:, None] & blk[None, :], 0., wt[:, q, :])
    wt = _t(wt, device)
    n_lik = _t((valid & ~huber_out).sum(1), device)
    HU = _t(huber_out, device, torch.bool)
    logy_exact = _t(np.log(np.sqrt(np.maximum(T['raw2'][fit], 1e-300))), device)
    V = _t(valid, device, torch.bool)
    bidx = _t(T['bidx'], device, torch.long)
    shat_sum = T['shat'][fit].sum(1, where=valid)
    cs_all = _t(np.where(shat_sum > 0, np.log(np.maximum(shat_sum, 1e-300)), -np.inf), device)
    ts_all = _t(shat_sum, device)
    res_all = _t(T['resolved'][fit], device, torch.bool)
    group = torch.as_tensor(sub['event']*2+sub['kind'].astype(np.int64), device=device)
    uniq, gidx = torch.unique(group, return_inverse=True)
    members = [torch.nonzero(gidx == g).flatten() for g in range(len(uniq))]
    torch.manual_seed(tc['seed'])
    model = PairNet(X.shape[1], tc['width'], tc['depth']).to(device).double()
    model.mu.copy_(X.mean(0))
    model.sd.copy_(X.std(0).clamp(min=1e-8))
    with torch.no_grad():   # start each output at the median log raw RMS (convergence only)
        lograw = np.log(np.sqrt(np.maximum(T['raw2'][fit], 1e-300)))
        model.net[-1].bias.copy_(_t([np.median(lograw[valid[:, j], j]) if valid[:, j].any() else 0. for j in range(OUT)], device))
    opt = torch.optim.Adam(model.parameters(), lr=tc['lr'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, tc['steps'], eta_min=tc['lr']*1e-2)
    log = []
    for step in range(tc['steps']):
        pick = torch.randint(len(members), (tc['batch_groups'],), generator=gen, device='cpu')
        idx = torch.cat([members[int(g)] for g in pick])
        out = model(X[idx])
        count = n_lik[idx].sum()+HU[idx].sum()
        lik = nll_terms(out, r[idx], se[idx], wt[idx], bidx)
        hub = torch.nn.functional.huber_loss(out[HU[idx]], logy_exact[idx][HU[idx]], delta=1.0, reduction='sum') \
            if bool(HU[idx].any()) else out.new_zeros(())
        loss = (lik+hub)/count.clamp(min=1)
        rank = cal = out.new_zeros(())
        if arm == '1A':
            c, cs, res = scalar(out, V[idx]), cs_all[idx], res_all[idx]
            g = gidx[idx]
            same = g[:, None] == g[None, :]
            pm = same & res[:, None] & res[None, :] & torch.isfinite(cs)[:, None] & torch.isfinite(cs)[None, :]
            pm = pm & ((cs[:, None]-cs[None, :]) > tc['rank_margin'])
            if bool(pm.any()):
                rank = torch.nn.functional.softplus(-(c[:, None]-c[None, :]))[pm].mean()
            G = int(g.max())+1
            cm = c.detach().max()
            ev = torch.zeros(G, dtype=c.dtype, device=device).index_add_(0, g, torch.exp(c-cm))
            evs = torch.zeros(G, dtype=c.dtype, device=device).index_add_(0, g, ts_all[idx])
            live = (ev > 0) & (evs > 0)
            if bool(live.any()):
                cal = ((torch.log(ev[live])+cm-torch.log(evs[live]))**2).mean()
            loss = loss+tc['rank_weight']*rank+tc['cal_weight']*cal
        if not bool(torch.isfinite(loss)):
            return model, log, 'NONFINITE_LOSS'
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step == tc['steps']-1:
            log.append(dict(step=step, loss=float(loss), lik=float(lik/count.clamp(min=1)), rank=float(rank), cal=float(cal)))
            print('RM_LIK %s step %d loss %.4f rank %.4f cal %.4f' % (arm, step, log[-1]['loss'], log[-1]['rank'],
                                                                     log[-1]['cal']), flush=True)
    return model, log, 'OK'


# ------------------------------------------------------------------ label-level checks

def deploy_cost(d, s, tol):
    """Per pair and output, the deployment cost of rm_model.LearnedCost.price in tolerance units:
    W * max_{k in block, k <= L} (psi_PTA,pair,k / Psi_PTA,ensemble,k) * s / tolerance."""
    lam = d['wa']/(d['wa']+d['wb'])
    W = d['wa']+d['wb']
    avg = lam[:, None]*d['pta_a_psi']+(1-lam)[:, None]*d['pta_b_psi']
    Psi = d['Psi']
    share = np.where(Psi > 0, avg/np.where(Psi > 0, Psi, 1.), 0.)
    k = np.arange(1, H+1)
    cost = np.zeros((len(W), OUT))
    for b, (lo, hi) in enumerate(BLOCKS):
        live = ((k >= lo) & (k <= hi))[None, :] & (k[None, :] <= d['L'][:, None])
        sh = np.where(live, share, 0.).max(1)
        cost[:, b] = W*sh*s[:, b]/tol['P_relative']
        cost[:, len(BLOCKS)+b] = W*sh*s[:, len(BLOCKS)+b]/tol['F_absolute']
    return cost


def z2(T, s):
    """Block-averaged r^2/(s^2 + se^2) per pair and output (noise-aware calibration; about 1 if calibrated)."""
    out = np.full(s.shape, np.nan)
    k = np.arange(1, H+1)
    for q in range(2):
        for b, (lo, hi) in enumerate(BLOCKS):
            j = q*len(BLOCKS)+b
            blk = (k >= lo) & (k <= hi)
            w = np.where(blk[None, :], T['wt'][:, q], 0.)
            v = s[:, j][:, None]**2+T['se'][:, q]**2
            zz = (w*np.where(w > 0, T['r'][:, q]**2/np.where(v > 0, v, 1.), 0.)).sum(1)
            out[:, j] = np.where(T['valid'][:, j], zz, np.nan)
    return out


@torch.no_grad()
def predict(model, X):
    return model(X).cpu().numpy()


def checks(models, d, T, cfg, device):
    """Label-level comparison of the models (old M2 model and the step-1 arms) on the same labels."""
    X = label_inputs(d, device)
    out = dict(cost={}, z2={}, argmax={}, concordance={}, finite={})
    fam = d['family']
    costs = {}
    for name, model in models.items():
        logs = predict(model, X)
        out['finite'][name] = bool(np.isfinite(logs).all())
        s = np.exp(logs)
        c = deploy_cost(d, s, cfg['tolerance'])
        costs[name] = c.max(1)
        zz = z2(T, s)
        out['z2'][name], out['argmax'][name], out['cost'][name] = {}, {}, {}
        for split, sel in (('fit', ~d['split']), ('monitor', d['split'])):
            out['cost'][name][split] = {f: float(np.median(costs[name][sel & (fam == i)])) for i, f in enumerate(FAMILIES)}
            out['z2'][name][split] = {f: [float(np.nanmean(zz[sel & (fam == i), j])) if np.isfinite(zz[sel & (fam == i), j]).any()
                                          else None for j in range(OUT)] for i, f in enumerate(FAMILIES)}
            out['argmax'][name][split] = {f: np.bincount(c[sel & (fam == i)].argmax(1), minlength=OUT).tolist()
                                          for i, f in enumerate(FAMILIES)}
        ev = evaluate(model, d, device, cfg)
        out['concordance'][name] = {key: ev[key]['long_blocks']['learned'] for key in ev if key.startswith('monitor_')}
        out['concordance'][name].update({key+'_all_blocks': ev[key]['all_blocks']['learned'] for key in ev if key.startswith('monitor_')})
    out['cost_ratio_vs_old'] = {}
    for name in models:
        if name == 'old':
            continue
        ratio = costs[name]/np.maximum(costs['old'], 1e-300)
        out['cost_ratio_vs_old'][name] = {split: {f: float(np.median(ratio[sel & (fam == i)])) for i, f in enumerate(FAMILIES)}
                                          for split, sel in (('fit', ~d['split']), ('monitor', d['split']))}
    return out


def needs_remedy(chk, name, cfg):
    """Pre-registered remedy trigger: non-finite predictions, or a hard family's exact-block z2 mean outside the range."""
    lo, hi = cfg['s1']['gate']['z2_exact_range']
    if not chk['finite'][name]:
        return True
    for f in HARD:
        for j in EXACT_OUTPUTS:
            v = chk['z2'][name]['monitor'][f][j]
            if v is not None and not lo <= v <= hi:
                return True
    return False


def progress(chk, name, cfg):
    """Pre-registered label-level progress check (HANDOFF 22.29): the B-family deployment cost falls to at most half
    of the old model's (monitor pairs, median of the per-pair ratio), and the long-block concordance does not drop
    by more than the margin overall or in any hard family."""
    g = cfg['s1']['gate']
    ratio = chk['cost_ratio_vs_old'][name]['monitor']['B']
    keys = ['monitor_all']+['monitor_fam%d' % FAMILIES.index(f) for f in HARD]
    drops = {k: chk['concordance'][name][k]-chk['concordance']['old'][k] for k in keys}
    ok_cost = ratio <= g['b_cost_ratio']
    ok_conc = all(v >= -g['concordance_drop'] for v in drops.values())
    return dict(arm=name, b_cost_ratio=ratio, concordance_change=drops, cost_ok=bool(ok_cost), concordance_ok=bool(ok_conc),
                passed=bool(ok_cost and ok_conc))


def save_model(model, tc, path):
    torch.save(dict(din=model.mu.numel(), width=tc['width'], depth=tc['depth'], state=model.state_dict()), path)


def verified_lik(root):
    """A completed step-1 training artifact whose model-defining sources and settings are unchanged."""
    m = rt.read(root/'manifest.json')
    if m['status'] != 'LIK_COMPLETE':
        raise ValueError('Incomplete step-1 training artifact: '+str(root))
    cfg = rt.config()
    if any(m['config'][k] != cfg[k] for k in MODEL_KEYS):
        raise ValueError('Model settings changed since: '+str(root))
    mine = str(rt.HERE.relative_to(rt.REPO))+'/'
    keep = lambda s: {k: v for k, v in s.items() if not k.startswith(mine) or Path(k).name in MODEL_FILES}  # noqa: E731
    if not __import__('qn_public').same_sources(keep(m['sources']), keep(rt.sources())):  # PUBLIC: see qn_public.py
        raise ValueError('Model-defining sources changed since: '+str(root))
    for filename, value in m['files'].items():
        if __import__('qn_public').present(root/filename) and rt.sha(root/filename) != value:
            raise ValueError('Step-1 training artifact content changed: '+str(root/filename))
    return m


# ------------------------------------------------------------------ smoke

def fit_scale(r, se, device):
    """Maximum-likelihood s of r ~ N(0, s^2 + se^2) through nll_terms (one prefix of block 2-3 per sample)."""
    N = len(r)
    R, S, W = (torch.zeros(N, 2, H, dtype=torch.float64, device=device) for _ in range(3))
    R[:, 0, 1], S[:, 0, 1], W[:, 0, 1] = _t(r, device), _t(se, device), 1.
    bidx = torch.full((H,), -1, dtype=torch.long, device=device)
    bidx[1:3] = 0
    o = torch.tensor(float(np.log(np.sqrt(np.mean(r**2))))+1., dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([o], max_iter=200, tolerance_grad=1e-14, tolerance_change=1e-16, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = nll_terms(o.expand(N, OUT), R, S, W, bidx)/N
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.exp(o))


def smoke_lik(cx, cfg, inputs):
    """Likelihood checks on synthetic data, label loading (Psi), and short training runs of both arms."""
    rng = np.random.default_rng(20261030)
    N, out = 100_000, {}
    t = rng.normal(0., 1e-3, N)
    s_exact = fit_scale(t, np.zeros(N), cx.device)
    rms = float(np.sqrt(np.mean(t**2)))
    out['exact_equals_rms'] = dict(s=s_exact, rms=rms, passed=abs(s_exact/rms-1) <= 1e-6)
    se = np.full(N, 1e-3)
    noise_only = rng.normal(0., 1e-3, N)
    s0 = fit_scale(noise_only, se, cx.device)
    floor = np.sqrt(np.maximum(noise_only**2-se**2, 0.05*noise_only**2))
    out['pure_noise'] = dict(s=s0, se=1e-3, old_target_median=float(np.median(floor)), passed=s0 <= 0.2e-3)
    mixed = rng.normal(0., 1e-3, N)+rng.normal(0., 1e-3, N)
    s1 = fit_scale(mixed, se, cx.device)
    out['signal_plus_noise'] = dict(s=s1, truth=1e-3, passed=abs(s1/1e-3-1) <= 0.05)
    tc = cfg['s1']
    d = load_labels(tc['labels_jobs'], inputs)
    T = lik_terms(d, tc['resolved_ratio'])
    k = np.arange(1, H+1)
    live = d['mask'] & (k[None, :] <= d['L'][:, None])
    out['labels'] = dict(pairs=int(len(d['split'])), psi_positive=float(np.mean(np.all(~live | (d['Psi'] > 0), 1))),
                         valid_fraction=[float(v) for v in T['valid'].mean(0)], exact_fraction=[float(v) for v in T['exact'].mean(0)],
                         unresolved_fraction_B=[float(v) for v in np.mean((T['raw2'] < tc['resolved_ratio']*T['noise2'])[d['family'] == 1], 0)])
    out['labels']['passed'] = out['labels']['psi_positive'] == 1.0
    short = dict(cfg, s1=dict(tc, steps=50))
    for arm in ('1B', '1A'):
        for remedy in (False, True):
            gen = torch.Generator().manual_seed(0)
            model, log, status = train_lik(d, T, short, cx.device, gen, arm, remedy)
            out['train_%s%s' % (arm, '_remedy' if remedy else '')] = dict(status=status, last=log[-1] if log else None,
                                                                           passed=status == 'OK')
    return out


# ------------------------------------------------------------------ job

def run_training(cx, cfg, root, inputs, save):
    """1B first; 1A only if 1B fails the progress check. Each arm is retrained once with the exact-block remedy if
    it diverges or its exact-block calibration is off. Writes model_<arm>.pt, checks.json, decision.json."""
    from rm_model import load_model
    from rm_verify import verified_model
    tc = cfg['s1']
    d = load_labels(tc['labels_jobs'], inputs)
    T = lik_terms(d, tc['resolved_ratio'])
    old_root = rt.HERE/'results'/('train_%s' % tc['old_train_job'])
    verified_model(old_root)
    inputs[old_root.name] = rt.sha(old_root/'manifest.json')
    old = load_model(old_root/'model.pt', cx.device)
    print('RM_LIK pairs %d (fit %d, monitor %d)' % (len(d['split']), int((~d['split']).sum()), int(d['split'].sum())), flush=True)
    decision, logs, arms = dict(deploy=None, arms={}), {}, {}
    for arm in ('1B', '1A'):
        status, remedy = None, False
        for remedy in (False, True):
            gen = torch.Generator().manual_seed(int(tc['seed']))
            model, log, status = train_lik(d, T, cfg, cx.device, gen, arm, remedy)
            logs['%s%s' % (arm, '_remedy' if remedy else '')] = log
            chk = checks({'old': old, arm: model}, d, T, cfg, cx.device) if status == 'OK' else None
            if status == 'OK' and not needs_remedy(chk, arm, cfg):
                break
            print('RM_LIK %s remedy needed (status %s)' % (arm, status), flush=True)
        if status != 'OK' or needs_remedy(chk, arm, cfg):
            decision['arms'][arm] = dict(status='REMEDY_FAILED', training=status, remedy=remedy)
            save('checks_%s.json' % arm, chk or {})
            continue
        save_model(model, tc, root/('model_%s.pt' % arm))
        save('checks_%s.json' % arm, chk)
        p = progress(chk, arm, cfg)
        decision['arms'][arm] = dict(status='OK', remedy=remedy, **p)
        print('RM_LIK %s progress %s' % (arm, p), flush=True)
        arms[arm] = model
        if p['passed']:
            decision['deploy'] = arm
            break
    save('train_log.json', logs)
    save('decision.json', decision)
    return decision

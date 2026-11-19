"""Noise-aware summary of the M1 labels (HANDOFF 22.28). CPU; reads the label shards read-only.

All gaps are per unit merged weight and scaled by the members' average psi of the same prefix (a gap g in
psi changes P_{t+k} by the fraction g/psi of this pair's own future success). Prefixes 1..e are exact; beyond,
the true gap is coupled Monte Carlo with standard error se, so squared residuals are debiased by se^2 and a
pair's residual counts as detected only if its chi-square over the Monte Carlo prefixes is far above its
degrees of freedom (a max over prefixes would mostly pick noise).
"""
import rm_runtime as rt
from pathlib import Path
import numpy as np
import torch

BLOCKS = (('2-3', 2, 3), ('4-7', 4, 7), ('8-15', 8, 15), ('16-40', 16, 40))
FAMILIES = ('A', 'B', 'AB', 'AAB', 'AABB', 'ABB')
DATA_FILES = ('rm_core.py', 'rm_labels.py', 'rm_runtime.py')
DATA_KEYS = ('labels', 'm1', 'merge')


def shards(job):
    found = sorted((rt.HERE/'results').glob('labels_%s_*' % job))
    if not found:
        raise ValueError('No label shards for job %s' % job)
    return found


def verified(root):
    """A completed shard whose data-generation sources, settings and files are unchanged."""
    m = rt.read(root/'manifest.json')
    if m['status'] != 'LABELS_SHARD_COMPLETE':
        raise ValueError('Incomplete label shard: '+str(root))
    now_cfg, now_src = rt.config(), rt.sources()
    if any(m['config'][k] != now_cfg[k] for k in DATA_KEYS):
        raise ValueError('Label settings changed since: '+str(root))
    mine = str(rt.HERE.relative_to(rt.REPO))+'/'
    keep = lambda s: {k: v for k, v in s.items() if not k.startswith(mine) or Path(k).name in DATA_FILES}  # noqa: E731
    if not __import__('qn_public').same_sources(keep(m['sources']), keep(now_src)):  # PUBLIC: see qn_public.py
        raise ValueError('Label-generation sources changed since: '+str(root))
    for filename, value in m['files'].items():
        if __import__('qn_public').present(root/filename) and rt.sha(root/filename) != value:
            raise ValueError('Label shard content changed: '+str(root/filename))
    return m


def load(job, inputs):
    parts = []
    for r in shards(job):
        m = verified(r)
        inputs[r.name] = rt.sha(r/'manifest.json')
        d = torch.load(r/'labels.pt', map_location='cpu', weights_only=False)
        rows = {k: v.numpy() for k, v in d['rows'].items()}
        rows['event'] = rows['event']+1_000_000*int(r.name.split('_')[-1])
        parts.append(rows)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def ranks(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind='stable')] = np.arange(len(x))
    return r


def spearman_by_event(event, a, b, min_pairs=8):
    vals = []
    for e in np.unique(event):
        s = event == e
        if s.sum() >= min_pairs and np.std(a[s]) > 0 and np.std(b[s]) > 0:
            vals.append(np.corrcoef(ranks(a[s]), ranks(b[s]))[0, 1])
    return dict(events=len(vals), mean=float(np.mean(vals)) if vals else None,
                median=float(np.median(vals)) if vals else None)


def stats(d, sel):
    if not sel.any():
        return dict(pairs=0)
    H = d['mask'].shape[1]
    k = np.arange(1, H+1)[None, :]
    lvl = np.abs(d['level_psi'][sel])
    ok = d['mask'][sel] & (lvl > 0)
    lv = np.where(ok, lvl, 1.)
    rr, sr, pg = d['r_psi'][sel]/lv, d['se_psi'][sel]/lv, d['pta_psi'][sel]/lv
    rf, sf, pf = d['r_phi'][sel]/lv, d['se_phi'][sel]/lv, d['pta_phi'][sel]/lv
    e = d['exact_prefixes'][sel][:, None]
    mc = ok & (k > e) & (d['se_psi'][sel] > 0)
    out = dict(pairs=int(sel.sum()), blocks={})
    for name, lo, hi in BLOCKS:
        b = ok & (k >= lo) & (k <= hi)
        if not b.any():
            continue
        def rms(x, s=None):
            m2 = float(np.mean(x[b]**2))-(float(np.mean(s[b]**2)) if s is not None else 0.)
            return float(np.sqrt(max(m2, 0.)))
        res, pta = rms(rr, sr), rms(pg)
        out['blocks'][name] = dict(entries=int(b.sum()), exact_fraction=float(np.mean(np.broadcast_to(k <= e, b.shape)[b])),
                                   rms_rel_residual_psi=res, rms_rel_pta_gap_psi=pta,
                                   ratio_psi=res/pta if pta > 0 else None,
                                   rms_rel_residual_phi=rms(rf, sf), rms_rel_pta_gap_phi=rms(pf),
                                   noise_rms_psi=float(np.sqrt(np.mean(sr[b]**2))))
    z2 = np.where(mc, (d['r_psi'][sel]/np.where(mc, d['se_psi'][sel], 1.))**2, 0.)
    dof = mc.sum(1)
    X = z2.sum(1)
    live = dof > 0
    thr = dof+3*np.sqrt(2*np.maximum(dof, 1))
    out['detected_fraction_psi'] = float(np.mean(X[live] > thr[live])) if live.any() else None
    ex = ok & (k <= e) & (k >= 2)
    if ex.any():
        a, p = np.abs(rr[ex]), np.abs(pg[ex])
        out['exact_prefixes'] = dict(entries=int(ex.sum()), median_rel_residual=float(np.median(a)),
                                     median_rel_pta_gap=float(np.median(p)), fraction_residual_larger=float(np.mean(a > p)))
    # Label-level preview for M2 (noise-free prefix 3): does the geometric Ward cost rank the residual?
    j = 2
    x2 = ((d['xa'][sel]-d['xb'][sel])**2).sum(1)
    lam = d['wa'][sel]/(d['wa'][sel]+d['wb'][sel])
    ward = lam*(1-lam)*x2
    live3 = ok[:, j]
    if live3.sum() > 0:
        ev = d['event'][sel][live3]
        out['spearman_prefix3'] = dict(
            ward_vs_abs_residual=spearman_by_event(ev, ward[live3], np.abs(rr[live3, j])),
            ward_vs_abs_pta_gap=spearman_by_event(ev, ward[live3], np.abs(pg[live3, j])),
            pta_gap_vs_abs_residual=spearman_by_event(ev, np.abs(pg[live3, j]), np.abs(rr[live3, j])))
    return out


def long_range(d, sel):
    """How well C1 (geometric Ward cost) and the PTA gap rank the residual per horizon block, within each capture
    and atom set. A pair's residual size in a block is its RMS relative residual over the block's prefixes with the
    Monte Carlo variance subtracted; noise_share = median over pairs of (noise variance / raw mean square)
    tells how much the remaining Monte Carlo noise can lower the correlation (0 on exact prefixes)."""
    H = d['mask'].shape[1]
    k = np.arange(1, H+1)[None, :]
    lvl = np.abs(d['level_psi'][sel])
    ok = d['mask'][sel] & (lvl > 0)
    lv = np.where(ok, lvl, 1.)
    rr, sr, pg = d['r_psi'][sel]/lv, d['se_psi'][sel]/lv, d['pta_psi'][sel]/lv
    lam = d['wa'][sel]/(d['wa'][sel]+d['wb'][sel])
    ward = lam*(1-lam)*((d['xa'][sel]-d['xb'][sel])**2).sum(1)
    grp = d['event'][sel]*2+d['kind'][sel].astype(np.int64)
    out = {}
    for name, lo, hi in BLOCKS:
        b = ok & (k >= lo) & (k <= hi)
        n = b.sum(1)
        live = n > 0
        if not live.any():
            continue
        raw2 = np.where(b, rr**2, 0.).sum(1)/np.maximum(n, 1)
        noise2 = np.where(b, sr**2, 0.).sum(1)/np.maximum(n, 1)
        size = np.sqrt(np.maximum(raw2-noise2, 0.))
        pta = np.sqrt(np.where(b, pg**2, 0.).sum(1)/np.maximum(n, 1))
        share = noise2[live]/np.maximum(raw2[live], 1e-300)
        out[name] = dict(pairs=int(live.sum()), noise_share_median=float(np.median(share)),
                         ward_vs_residual=spearman_by_event(grp[live], ward[live], size[live]),
                         pta_gap_vs_residual=spearman_by_event(grp[live], pta[live], size[live]))
    return out


def report(job, inputs):
    d = load(job, inputs)
    out = dict(pairs=int(len(d['event'])), events=int(len(np.unique(d['event']))), by={}, by_family={},
               long_range={'fit_all': long_range(d, d['split'] == False)})  # noqa: E712
    for fi, f in enumerate(FAMILIES):
        out['long_range']['fit_'+f] = long_range(d, (d['split'] == False) & (d['family'] == fi))  # noqa: E712
    for split, sv in (('fit', False), ('monitor', True)):
        for kind, kv in (('branch', False), ('merged', True)):
            out['by']['%s_%s' % (split, kind)] = stats(d, (d['split'] == sv) & (d['kind'] == kv))
    for fi, f in enumerate(FAMILIES):
        out['by_family'][f] = stats(d, d['family'] == fi)
    return out

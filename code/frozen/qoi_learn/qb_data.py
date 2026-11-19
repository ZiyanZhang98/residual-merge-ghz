"""Step-B data on isolated fit/monitor preparations (never evaluation or final).

Clusters are sampled from real M0 compressions in proportion to cluster mass (the
deployment weighting), not only the most scattered one. Every node stores:
coordinates, remaining-word encoding, PTA responses (psi, phi), exact d<=2 labels and
its one-letter children (probabilities, coordinates, GHZ fidelity, PTA responses of
word[1:]) for Bellman targets. Monitor preparations also get merge references for
sampled long-horizon nodes and for full H40 tasks.
"""
import qb_runtime as rt
import time
import numpy as np
import torch
import ghz_factors as gf
from ghz_physics import ghz_diagonal, coords
from g1_core import check_deadline, tdist
from rr_features import responses
from pc_guard import generate_pool
from pc_sim import rollout as m0_rollout
from qa_sources import PTA, projector
from qa_sim import rollout as qa_rollout


def word_code(word, gamma, H, device):
    """Identical to pc_model.encode's protocol block."""
    z = torch.zeros(2*H+1, dtype=torch.float64, device=device)
    for j, a in enumerate(word):
        z[2*j+(a == 'B')] = 1
    z[-1] = gamma
    return z


def exact_short(nodes, roots, word, ctx, c, depth, hard):
    d = min(depth, len(word))
    R, _ = responses(nodes, word, d, dict(ctx, response_factors=roots), c,
                     {'features': {'chunk_leaves': 4096}}, hard=hard)
    g = coords(projector(ctx)[None], ctx['aux']['pauli'])[0].cpu().numpy()
    dim = nodes.shape[-1]
    psi = np.stack([R[k][:, 0] for k in sorted(R)], 1)
    phi = np.stack([(R[k][:, 0]+R[k][:, 1:] @ g)/dim for k in sorted(R)], 1)
    t = lambda x: torch.as_tensor(x, dtype=torch.float64, device=nodes.device)  # noqa: E731
    return t(psi), t(phi)


def pad(x, H):
    y = x.new_zeros(x.shape[:-1]+(H,))
    y[..., :x.shape[-1]] = x
    return y


class Collector:
    def __init__(self, cfg, cx, ctx, n, H, entry, word, K, rng, long_visits, sink, longs):
        self.cfg, self.cx, self.ctx, self.n, self.H = cfg, cx, ctx, n, H
        self.entry, self.word, self.K, self.rng = entry, word, K, rng
        self.long_visits, self.sink, self.longs = long_visits, sink, longs
        d = cfg['data']
        self.visits = set(range(0, H, d['visit_stride']))
        self.diagnose = entry['split'] == 'monitor'  # exact d<=4 on the first sampled cluster of each visit

    def __call__(self, sim, s, w, f, labels, out, nw, nf):
        t, cx, ctx, H = sim.t, self.cx, self.ctx, self.H
        remaining = self.word[t:]
        if t in self.long_visits:
            j = int(self.rng.choice(len(w), p=(w/w.sum()).cpu().numpy()))
            self.longs.append(dict(prep=self.entry['signature'], split=self.entry['split'], t=t, K=self.K,
                                   word=remaining, state=s[j].cpu(), factor=f[j].cpu()))
        if t not in self.visits:
            return
        check_deadline(cx.hard)
        d = self.cfg['data']
        sizes = torch.bincount(labels, minlength=len(nw)).cpu().numpy()
        eligible = [k for k in range(len(nw)) if sizes[k] >= d['min_cluster']]
        if not eligible:
            return
        mass = nw[eligible].cpu().numpy()
        chosen = self.rng.choice(eligible, size=min(d['clusters_per_visit'], len(eligible)),
                                 replace=False, p=mass/mass.sum())
        states, factors, dcoef, gid, gmeta = [], [], [], [], []
        for g, k in enumerate(chosen):
            use = labels == int(k)
            ww = w[use]
            states += [s[use], out[int(k)][None]]; factors += [f[use], nf[int(k)][None]]
            dcoef += [ww/ww.sum(), ww.new_tensor([-1.])]
            gid.append(torch.full((int(use.sum())+1,), g, dtype=torch.long, device=s.device))
            gmeta.append(dict(prep=self.entry['signature'], split=self.entry['split'], K=self.K, t=t,
                              word=remaining, size=int(use.sum()), cluster_weight=float(nw[int(k)]),
                              mass=float(nw[int(k)])*float(np.exp(sim.ell))))
        nodes, roots = torch.cat(states), torch.cat(factors)
        L = len(remaining)
        prior = PTA()(nodes, roots, remaining, ctx, cx.c, cx.hard)
        ex_psi, ex_phi = exact_short(nodes, roots, remaining, ctx, cx.c, d['exact_depth'], cx.hard)
        cf, p = gf.children(roots, ctx['letters'][remaining[0]], cx.c['chunk'])
        N, O = p.shape
        flat = cf.flatten(0, 1)
        zero = p.flatten() == 0
        if bool(zero.any()):
            flat = flat.clone(); flat[zero] = roots.repeat_interleave(O, 0)[zero]
        cs = gf.density(flat)
        cprior = PTA()(cs, flat, remaining[1:], ctx, cx.c, cx.hard)
        pauli = ctx["aux"]["pauli"]
        omax = max(len(x.accept) for x in ctx["letters"].values())

        def wide(x):  # pad the outcome axis to the largest letter; padded outcomes have p = 0
            y = x.new_zeros((N, omax)+x.shape[2:])
            y[:, :O] = x
            return y
        batch = dict(
            coords=coords(nodes, pauli), L=torch.full((N,), L, dtype=torch.long, device=s.device),
            z=word_code(remaining, self.cfg['gamma'], H, s.device).expand(N, -1).clone(),
            psi_pta=pad(prior['psi'], H), phi_pta=pad(prior['phi'], H),
            ex_psi=pad(ex_psi, H), ex_phi=pad(ex_phi, H),
            dcoef=torch.cat(dcoef), gid=torch.cat(gid),
            child_coords=wide(coords(cs, pauli).reshape(N, O, -1)), child_p=wide(p),
            child_F=wide(ghz_diagonal(cs, ctx["aux"]["ghz"])[:, 0].reshape(N, O)),
            child_psi_pta=wide(pad(cprior["psi"], H).reshape(N, O, H)),
            child_phi_pta=wide(pad(cprior["phi"], H).reshape(N, O, H)))
        if self.diagnose and L >= d['diagnostic_depth']:
            J = torch.nonzero(batch['gid'] == 0).flatten()
            J = torch.unique(torch.stack([J[0], J[-2], J[-1]]))
            dpsi, dphi = exact_short(nodes[J], roots[J], remaining, ctx, cx.c, d['diagnostic_depth'], cx.hard)
            gmeta[0].update(diagnostic_index=(J-J[0]).cpu().tolist(), diagnostic_psi=dpsi.cpu().tolist(),
                            diagnostic_phi=dphi.cpu().tolist())
        self.sink.append(({k: v.cpu() for k, v in batch.items()}, gmeta))


def merge_reference(R, word, ctx, c, states, weights, hard):
    _, rho, ell, est = qa_rollout(R, word, ctx, c, states, weights, None, hard)
    return rho, np.asarray(ell), est['M0']['P'].cpu().numpy(), (est['M0']['N']/est['M0']['P']).cpu().numpy()


def accepted_reference(cfg, word, ctx, c, states, weights, hard):
    r = cfg['reference']
    for R in (r['R'], 2*r['R']):
        hi = merge_reference(R, word, ctx, c, states, weights, hard)
        lo = merge_reference(R//2, word, ctx, c, states, weights, hard)
        dp = float(np.max(np.abs(hi[2]-lo[2])/hi[2]))
        dF = float(np.max(np.abs(hi[3]-lo[3])))
        dt = float(tdist(hi[0], lo[0]).max())
        ok = dp <= r['P_relative'] and dF <= r['F_absolute'] and dt <= r['TD']
        info = dict(R=R, P_difference=dp, F_difference=dF, td_difference=dt, accepted=ok)
        if ok:
            return hi, info
    return hi, info


def generate(cx, cfg, pc_cfg, n, shard, shards, root):
    H, d = cfg['H'], cfg['data']
    pool = generate_pool(pc_cfg, cx.c, n)
    mine = [(j, e) for j, e in enumerate(pool) if j % shards == shard]
    sink, longs, tasks, timings = [], [], [], []
    letters, pta = cx.letters(n, cfg['gamma'])
    aux = cx.aux(n)
    (root/'refs').mkdir()
    for j, entry in mine:
        tick = time.perf_counter()
        states, weights, factors = gf.prepare(entry['prep'], cx.c, cx.device)
        ctx = dict(letters=letters, pta=pta, aux=aux, initial_factors=factors)
        rng = np.random.default_rng(1000*n+j)
        ks = [cfg['K'][(j*d['K_per_preparation']+k) % len(cfg['K'])] for k in range(d['K_per_preparation'])]
        for family in d['words']:
            word = (family*H)[:H]
            for i, K in enumerate(ks):
                lv = set(d['long_visits']) if entry['split'] == 'monitor' and i == 0 else set()
                obs = Collector(cfg, cx, ctx, n, H, entry, word, K, rng, lv, sink, longs)
                m0_rollout(K, word, ctx, cx.c, states, weights, observer=obs, hard=cx.hard)
            if entry['split'] == 'monitor':
                ref, info = accepted_reference(cfg, word, ctx, cx.c, states, weights, cx.hard)
                tid = 'qb_n%d_%s_%s' % (n, entry['signature'][:16], family)
                path = root/'refs'/(tid+'.npz')
                np.savez_compressed(path, rho=ref[0].cpu().numpy(), ell=ref[1], P=ref[2], F=ref[3])
                tasks.append(dict(id=tid, n=n, H=H, gamma=cfg['gamma'], family=family, word=word,
                                  prep=entry['prep'], prep_signature=entry['signature'],
                                  reference=str(path.name), **info))
        timings.append(dict(prep=entry['signature'], split=entry['split'], seconds=time.perf_counter()-tick))
        rt.write(root/'progress.json', dict(done=len(timings), total=len(mine), timings=timings))
        print('QB_DATA', n, j, entry['split'], len(sink), len(longs), flush=True)
    # Long-horizon node references (monitor only), from the single sampled node.
    for k, node in enumerate(longs):
        check_deadline(cx.hard)
        st, fa = node['state'].to(cx.device)[None], node['factor'].to(cx.device)[None]
        ctx = dict(letters=letters, pta=pta, aux=aux, initial_factors=fa)
        ref, info = accepted_reference(cfg, node['word'], ctx, cx.c, st, st.real.new_ones(1), cx.hard)
        node.update(P_reference=ref[2].tolist(), F_reference=ref[3].tolist(), **info)
    flat = {key: torch.cat([b[key] for b, _ in sink]) for key in sink[0][0]} if sink else {}
    offset, groups, gid = 0, [], []
    for b, meta in sink:
        gid.append(b['gid']+offset); groups += meta; offset += len(meta)
    if flat:
        flat['gid'] = torch.cat(gid)
    torch.save(dict(n=n, H=H, nodes=flat, groups=groups, longs=longs), root/'data.pt')
    rt.write(root/'tasks.json', tasks)
    rt.write(root/'summary.json', dict(n=n, shard=shard, shards=shards, preparations=len(mine),
        groups=len(groups), nodes=int(flat['L'].numel()) if flat else 0, long_nodes=len(longs),
        long_accepted=sum(x['accepted'] for x in longs), tasks=len(tasks),
        tasks_accepted=sum(t['accepted'] for t in tasks), timings=timings))

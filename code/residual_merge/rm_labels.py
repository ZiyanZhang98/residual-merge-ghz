"""M1 of HANDOFF 22.28: pairwise residual-merge labels on n=4 fit and monitor preparations.

Preparations are the step-B pool (pc_guard.generate_pool, the same fit/monitor isolation as qoi_learn); final
and evaluation preparations are excluded by construction. For every preparation and word family the frozen
geometric merge_K trajectory (K rotating over cfg) is captured at the configured rounds. At each capture two
atom sets are labelled, as deployment will see both:
  'branch'  the pre-merge branches themselves;
  'merged'  the same ensemble after a geometric agglomeration to a quarter of its size (atoms are weighted means).
Candidate pairs are k-nearest-neighbour pairs in Pauli coordinates (the candidates agglomeration prices); a random
subset of them is labelled with rm_core.pair_labels (exact tree on the first prefixes, coupled Monte Carlo beyond,
PTA and residual gaps). Model inputs are stored with the labels: both atoms' Pauli coordinates and weights, the
remaining word, and the PTA responses of both atoms and of their mean.
"""
import rm_runtime as rt
import time
import numpy as np
import torch
import ghz_factors as gf
from ghz_physics import coords
from ghz_merge import combine
from g1_core import check_deadline
from pc_guard import generate_pool
from qa_sources import PTA
from rm_core import GeoCost, agglomerate, Capture, knn_pairs, pair_labels

FAMILIES = ('A', 'B', 'AB', 'AAB', 'AABB', 'ABB')
PER_PAIR = ('g_psi', 'g_phi', 'se_psi', 'se_phi', 'pta_psi', 'pta_phi', 'r_psi', 'r_phi', 'level_psi')


def pool(cx, n):
    pc_cfg = rt.hr.qb.qa.prt.config()
    return [e for e in generate_pool(pc_cfg, cx.c, n) if e['split'] in ('fit', 'monitor')]


def pad(a, H):
    a = np.asarray(a, dtype=np.float64)
    out = np.zeros((a.shape[0], H))
    out[:, :a.shape[1]] = a
    return out


def code(rest, H):
    z = np.full(H, -1, dtype=np.int8)
    z[:len(rest)] = [0 if x == 'A' else 1 for x in rest]
    return z


def sample(P, m, gen):
    if len(P) <= m:
        return P
    return P[torch.randperm(len(P), generator=gen, device=P.device)[:m]]


def summary(rows):
    """Median over pairs of max_k |gap| / (members' average psi): residual against the PTA-visible gap."""
    if not rows:
        return {}
    lvl = np.maximum(np.abs(np.concatenate([r['level_psi'] for r in rows])), 1e-300)
    mask = np.concatenate([r['mask'] for r in rows])
    res = np.where(mask, np.abs(np.concatenate([r['r_psi'] for r in rows]))/lvl, 0.).max(1)
    pta = np.where(mask, np.abs(np.concatenate([r['pta_psi'] for r in rows]))/lvl, 0.).max(1)
    se = np.concatenate([r['se_psi'] for r in rows])
    resolvable = np.any(mask & (np.abs(np.concatenate([r['r_psi'] for r in rows])) > 2*se) & (se > 0), 1)
    return dict(pairs=int(len(res)), median_rel_residual=float(np.median(res)), median_rel_pta_gap=float(np.median(pta)),
                median_ratio_residual_over_pta=float(np.median(res/np.maximum(pta, 1e-300))),
                fraction_residual_larger=float(np.mean(res > pta)), fraction_resolvable=float(np.mean(resolvable)))


def generate(cx, cfg, shard, shards, root, save):
    lc, lab, mg = cfg['m1'], cfg['labels'], cfg['merge']
    qb = rt.hr.qb.config()
    H, gamma, n = qb['H'], qb['gamma'], lc['n']
    entries = pool(cx, n)
    mine = [(j, e) for j, e in enumerate(entries) if j % shards == shard]
    letters, pta = cx.letters(n, gamma)
    aux = cx.aux(n)
    rows, events, timings = [], [], []
    for j, entry in mine:
        tick = time.perf_counter()
        states, weights, factors = gf.prepare(entry['prep'], cx.c, cx.device)
        ctx = dict(letters=letters, pta=pta, aux=aux, initial_factors=factors)
        for fi, family in enumerate(FAMILIES):
            check_deadline(cx.hard)
            word = (family*H)[:H]
            K = lc['K'][(j+fi) % len(lc['K'])]
            seed = lc['seed']+int(rt.hr.digest([entry['signature'], family])[:8], 16) % 1_000_000
            gen = torch.Generator(device=cx.device).manual_seed(seed)
            sim = Capture(K, word, ctx, cx.c, states, weights, cx.hard, lc['events'])
            for a in word[:max(lc['events'])]:
                sim.advance(a)
            for t in sorted(sim.kept):
                ev = sim.kept[t]
                S, w, F, rest = ev['states'], ev['w'], ev['factors'], ev['rest']
                if len(w) < 4:
                    continue
                vp = PTA()(S, None, rest, ctx, cx.c, cx.hard)
                eid = len(events)
                events.append(dict(event=eid, prep=entry['signature'], split=entry['split'], family=family, K=K, t=t,
                                   L=len(rest), branches=len(w), Psi_pta=pad((w @ vp['psi'])[None].cpu().numpy(), H)[0].tolist(),
                                   Phi_pta=pad((w @ vp['phi'])[None].cpu().numpy(), H)[0].tolist()))
                m = lc['pairs_per_set'][entry['split']]
                x = coords(S, aux['pauli'])
                labels, _ = agglomerate(S, w, x, GeoCost(), 0., max(2, len(w)//4), mg['knn'], mg['max_rounds'])
                Sm, wm = combine(S, w, labels)
                sets = (('branch', F, w, x), ('merged', gf.combine(F, w, labels), wm, coords(Sm, aux['pauli'])))
                for kind, Fa, wa, xa in sets:
                    P = sample(knn_pairs(xa, mg['knn']), m, gen)
                    if len(P) == 0:
                        continue
                    out = pair_labels(Fa, wa, P, rest, ctx, cx.c, cx.hard, lab, gen)
                    I, J = P[:, 0], P[:, 1]
                    Lr = len(rest)
                    mask = np.zeros((len(P), H), dtype=bool)
                    mask[:, :Lr] = True
                    row = dict(event=np.full(len(P), eid), kind=np.full(len(P), kind == 'merged'),
                               split=np.full(len(P), entry['split'] == 'monitor'), t=np.full(len(P), t),
                               L=np.full(len(P), Lr), K=np.full(len(P), K), family=np.full(len(P), fi),
                               xa=xa[I].cpu().numpy(), xb=xa[J].cpu().numpy(),
                               wa=wa[I].cpu().numpy(), wb=wa[J].cpu().numpy(),
                               rest=np.tile(code(rest, H), (len(P), 1)), mask=mask,
                               exact_prefixes=np.full(len(P), out['exact_prefixes']))
                    for key in PER_PAIR:
                        row[key] = pad(out[key], H)
                    for key, v in out['pta_values'].items():
                        row['pta_'+key] = pad(v, H)
                    rows.append(row)
        timings.append(dict(prep=entry['signature'], split=entry['split'], seconds=time.perf_counter()-tick))
        flat = {k: torch.as_tensor(np.concatenate([r[k] for r in rows])) for k in rows[0]} if rows else {}
        torch.save(dict(n=n, H=H, gamma=gamma, rows=flat, families=FAMILIES), root/'labels.pt')
        save('events.json', events)
        save('progress.json', dict(done=len(timings), total=len(mine), timings=timings,
                                   summary={s: summary([r for r in rows if r['split'][0] == (s == 'monitor')])
                                            for s in ('fit', 'monitor')}))
        print('RM_LABELS prep %d/%d %s pairs=%d events=%d %.0fs' % (
            len(timings), len(mine), entry['split'], sum(len(r['event']) for r in rows), len(events), timings[-1]['seconds']),
            flush=True)
    by = {}
    for s in ('fit', 'monitor'):
        for kind in ('branch', 'merged'):
            by['%s_%s' % (s, kind)] = summary([r for r in rows if r['split'][0] == (s == 'monitor') and r['kind'][0] == (kind == 'merged')])
    return dict(preparations=len(mine), pairs=sum(len(r['event']) for r in rows), events=len(events), summary=by)

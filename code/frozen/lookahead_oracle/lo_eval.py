"""Step 1 of HANDOFF 22.23: upper-bound check for better look-ahead predictors.

The look-ahead block is the standardized one of lookahead_tuning (share x geometric variance);
only the predictor changes:
  pta      the PTA response (the non-learned look-ahead)
  exact    the full outcome tree for the next 3 rounds (perfect information within that horizon)
  d2       exact branching for 2 rounds, then the PTA tail (exact up to 3 rounds, better beyond)
  mf       branch-summed kernel without the GHZ twirl (most accurate short-range response, job 28516826)
  learned  the learned model of hr_goal (PTA x exp(r), seed 0), for reference
Every predictor runs with the same settings, tasks, K and PTA correction. If no better predictor
improves merging over PTA, no learned predictor can either and step 2 (an ML redesign) is not run.
"""
import lo_runtime as rt
import math
import time
import numpy as np
import torch
from ghz_physics import coords
from g1_core import check_deadline, Deadline
from qa_sources import PTA, Exact
from qb_model import Learned
from pg_sources import exact_rounds, DExact, MF
from hr_eval import context, scored
from lt_lookahead import StdLookAhead, rollout
from lt_smoke import q_table, compare
import qa_sim

SOURCES = ('pta', 'exact', 'd2', 'mf', 'learned')


class OracleLookAhead:
    def __init__(self, source, share, lengths, model=None, gamma=None, chunk=128):
        if source not in SOURCES or (source == 'learned' and model is None) or (source == 'exact' and max(lengths) > 3):
            raise ValueError('Unsupported look-ahead predictor: %s %s' % (source, lengths))
        self.source, self.share, self.lengths, self.chunk = source, float(share), list(lengths), int(chunk)
        self.learned = Learned(model, gamma, PTA()) if source == 'learned' else None
        self.weights = None

    def predict(self, states, factors, word, ctx, c, hard, kmax):
        w = word[:kmax]
        if self.source == 'pta':
            return PTA()(states, factors, w, ctx, c, hard)
        if self.source == 'mf':
            return MF()(states, factors, w, ctx, c, hard)
        if self.source == 'd2':
            return DExact(2)(states, factors, w, ctx, c, hard)
        if self.source == 'exact':
            parts = [exact_rounds(states[i:i+self.chunk], w, ctx, c, hard)[:2] for i in range(0, len(states), self.chunk)]
            return dict(psi=torch.cat([p[0] for p in parts]), phi=torch.cat([p[1] for p in parts]))
        return self.learned(states, factors, word, ctx, c, hard)

    @torch.no_grad()
    def __call__(self, states, factors, word, ctx, c, hard=None):
        """Same columns and scaling as lt_lookahead.StdLookAhead with quantities 'psiF'."""
        L = len(word)
        ks = sorted({min(k, L) for k in self.lengths})
        v = self.predict(states, factors, word, ctx, c, hard, ks[-1])
        cols = []
        for k in ks:
            cols.append(torch.clamp(v['psi'][:, k-1], min=1e-300).log())
            cols.append(v['phi'][:, k-1]/torch.clamp(v['psi'][:, k-1], min=1e-300))
        X = torch.stack(cols, 1)
        w = self.weights/self.weights.sum()
        mean = w @ X
        sd = torch.sqrt(w @ (X-mean)**2)
        live = sd > 1e-12
        Z = torch.where(live, (X-mean)/torch.where(live, sd, torch.ones_like(sd)), torch.zeros_like(X))
        geo = coords(states, ctx['aux']['pauli'])
        vg = float((w @ (geo-w @ geo)**2).sum())
        m = int(live.sum())
        return Z*(math.sqrt(self.share*vg/m) if m and vg > 0 else 0.)


def make(arm, models, task, cfg):
    return OracleLookAhead(arm['source'], arm['share'], arm['lengths'],
                           models[task['n']] if arm['source'] == 'learned' else None, task['gamma'], cfg['exact_chunk'])


def checks(cx, cfg, task, models):
    """1. The PTA predictor gives the lookahead_tuning features bit for bit. 2. 'exact' equals the
    exact tree of qa_sources for k <= 3. 3. The scaled block variance equals the share."""
    states, weights, ctx = context(cx, task)
    sim = qa_sim.QoISim(64, task['word'], ctx, cx.c, states, weights)
    for x in task['word'][:6]:
        sim.advance(x)
    S, w, R, rest = sim.states, sim.weights, sim.factors, task['word'][6:]
    out = {}
    a = StdLookAhead('pta', 1.0, [1, 2, 4, 8, 16, 32, 40], 'psiF')
    b = OracleLookAhead('pta', 1.0, [1, 2, 4, 8, 16, 32, 40])
    a.weights = b.weights = w
    out['pta_matches_lookahead_tuning'] = bool(torch.equal(a(S, R, rest, ctx, cx.c), b(S, R, rest, ctx, cx.c)))
    assert out['pta_matches_lookahead_tuning']
    ex = Exact()(S[:8], R[:8], rest[:3], ctx, cx.c)
    o = OracleLookAhead('exact', 1.0, [1, 2, 3]).predict(S[:8], R[:8], rest, ctx, cx.c, None, 3)
    out['exact_psi_relative'] = float(((o['psi']-ex['psi']).abs()/ex['psi']).max())
    out['exact_phi_absolute'] = float((o['phi']-ex['phi']).abs().max())
    assert out['exact_psi_relative'] < 1e-9 and out['exact_phi_absolute'] < 1e-10
    wn = w/w.sum()
    geo = coords(S, ctx['aux']['pauli'])
    vg = float((wn @ (geo-wn @ geo)**2).sum())
    for arm in cfg['arms']:
        la = make(arm, models, task, cfg)
        la.weights = w
        Z = la(S, R, rest, ctx, cx.c)
        out['variance_ratio_'+arm['name']] = float((wn @ (Z-wn @ Z)**2).sum())/vg
        assert abs(out['variance_ratio_'+arm['name']]-arm['share']) < 1e-8*max(1., arm['share']), arm['name']
    return out


def run(cx, cfg, tasks, npz, models, root):
    cells, timing = [], []
    for j, task in enumerate(tasks):
        check_deadline(cx.hard)
        ref = dict(np.load(npz[task['id']]))
        states, weights, ctx = context(cx, task)
        for K in cfg['K']:
            for arm in cfg['arms']:
                start = time.perf_counter()
                try:
                    sim, _, _, est = rollout(K, task['word'], ctx, cx.c, states, weights, {'M_PTA': PTA()}, cx.hard,
                                             make(arm, models, task, cfg))
                    sec = time.perf_counter()-start
                    for name, e in est.items():
                        label = arm['name']+'_raw' if name == 'M0' else arm['name']+'+'+name
                        cells.append(scored(task, label, K, e, ref, cfg['tolerance'], dict(seconds=sec, lookahead_seconds=sim.lookahead_seconds)))
                except Deadline:
                    raise
                except (FloatingPointError, RuntimeError) as exc:
                    sec = time.perf_counter()-start
                    for suffix in ('_raw', '+M_PTA', '+M_PTA__mul'):
                        cells.append(dict(task_id=task['id'], n=task['n'], family=task['family'], prep_id=task['prep_signature'],
                                          arm=arm['name']+suffix, K=K, status='NUMERIC_FAILURE', error=repr(exc)))
                timing.append(dict(task_id=task['id'], n=task['n'], K=K, arm=arm['name'], seconds=sec))
        rt.write(root/'cells.json', cells); rt.write(root/'timing.json', timing)
        print('LO_RUN', j+1, len(tasks), task['id'], flush=True)
    return cells


def decide(cells, step1_cells, cfg, tasks):
    """Pre-registered gate of HANDOFF 22.23 (additive readout primary; multiplicative reported)."""
    K_set, g = cfg['K'], cfg['gate']
    meta = {t['id']: (t['n'], t['prep_signature']) for t in tasks}
    names = {'M0': 'merge', 'M_PTA': 'merge+PTA', 'M_PTA__mul': 'merge+PTA__mul'}
    base = [dict(c, arm=names[c['arm']]) for c in step1_cells if c['arm'] in names and c['task_id'] in meta]
    Q = q_table(cells+base, K_set)
    report, verdicts = {}, []
    for n in sorted({v[0] for v in meta.values()}):
        rows = {}
        for arm in cfg['arms']:
            for suffix, b in (('+M_PTA', 'merge+PTA'), ('+M_PTA__mul', 'merge+PTA__mul'), ('_raw', 'merge')):
                rows[arm['name']+suffix] = dict(vs_merge=compare(Q, meta, n, arm['name']+suffix, b, K_set))
        pairs = {}
        for better, plain in g['pairs']+g['reference_pairs']:
            r = compare(Q, meta, n, better+'+M_PTA', plain+'+M_PTA', K_set)
            rm = compare(Q, meta, n, better+'+M_PTA__mul', plain+'+M_PTA__mul', K_set)
            vm = rows[better+'+M_PTA']['vs_merge']
            head = bool(r and vm and r['ratio'] >= g['ratio'] and r['not_worse'] >= g['preparation_fraction']*r['preparations']-1e-12
                        and vm['ratio'] >= 1.0)
            pairs['%s vs %s' % (better, plain)] = dict(additive=r, multiplicative=rm, headroom=head,
                                                       gate=[better, plain] in g['pairs'])
            if [better, plain] in g['pairs']:
                verdicts.append('HEADROOM' if head else 'MIXED' if r and r['ratio'] > 1.0 else 'NONE')
        report[str(n)] = dict(rows=rows, pairs=pairs)
    verdict = 'HEADROOM' if 'HEADROOM' in verdicts else 'MIXED' if 'MIXED' in verdicts else 'NO_HEADROOM'
    return dict(K=K_set, by_n=report, verdict=verdict,
                rule='HEADROOM if, for some n, a better predictor beats the matched PTA look-ahead by >= %.2f (additive, '
                     'geometric mean over task x K of fixed-K QoI error) with >= %.3f of preparations not worse and is not '
                     'worse than merge+PTA; MIXED if some ratio > 1; else NO_HEADROOM' % (g['ratio'], g['preparation_fraction']))

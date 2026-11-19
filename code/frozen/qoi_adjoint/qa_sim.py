"""Frozen M0 trajectory with observer-only corrections of P and of the fidelity numerator N.

P_h = exp(ell_h) and N_h = exp(ell_h) <GHZ|rho_h|GHZ> are linear functionals of the
branch measure, so the PC telescoping identity applies to both (HANDOFF 22.3).
Compression, weights, log mass and states are bit-identical to pc_sim/M0.
"""
import qa_runtime  # noqa: F401
import math
import time
import numpy as np
import torch
import ghz_factors as gf
from ghz_merge import features, combine
from ghz_physics import ghz_diagonal
from g1_core import lloyd_async, check_deadline
from rr_features import sync
from rr_sim import ResponseSim

QUANTITIES = (('P', 'psi'), ('N', 'phi'))
ESTIMATORS = ('add', 'mul')


def arm(source, method):
    return source if method == 'add' else source+'__mul'


class QoISim(ResponseSim):
    def __init__(self, K, word, ctx, c, states, weights, sources=None, hard=None):
        self.sources = sources or {}
        for name in self.sources:
            if name == 'M0' or name.endswith('__mul'):
                raise ValueError('Reserved source name: '+name)
        H = len(word)
        self.add = {s: {q: weights.new_zeros(H) for q, _ in QUANTITIES} for s in self.sources}
        self.logm = {s: {q: weights.new_zeros(H) for q, _ in QUANTITIES} for s in self.sources}
        self.failed = {}   # (arm, quantity) -> first failure
        self.correction_seconds = {s: 0. for s in self.sources}
        self.calls = {s: 0 for s in self.sources}
        self.trace = []
        super().__init__(K, {'family': 'M0', 'geometry': 'geo'}, word, ctx, c,
                         {'physics_backend': 'sqrt'}, np.random.default_rng(0), states, weights, hard)

    def fail(self, name, method, q, t, error, **extra):
        self.failed.setdefault((arm(name, method), q), dict(t=t, error=error, **extra))

    def compress(self, states, w):
        if len(w) <= self.K:
            return self.validate(states, w)
        check_deadline(self.hard)
        sync(states.device); start = time.perf_counter()
        labels = lloyd_async(features('geo', states, self.ctx['aux']), w, self.K,
                             self.c['clustering_iterations'])
        sync(states.device)
        self.cost['partition_seconds'] += time.perf_counter()-start
        start = time.perf_counter()
        # Same operation order as pc_sim.CorrectionSim (dense combine, then factors).
        out, nw = combine(states, w, labels)
        before_f = self.factors
        self.factors = gf.combine(before_f, w, labels)
        out = gf.density(self.factors)
        sync(states.device)
        self.cost['elimination_seconds'] += time.perf_counter()-start
        self.events.append(dict(t=self.t, before=len(w), after=len(nw),
                                mass_drift=abs(float(nw.sum()-w.sum()))))
        if self.sources:
            nodes, roots = torch.cat((states, out)), torch.cat((before_f, self.factors))
            signed = torch.cat((w, -nw))
            L = len(self.word)-self.t
            for name, provider in self.sources.items():
                live = [(m, q, key) for m in ESTIMATORS for q, key in QUANTITIES
                        if (arm(name, m), q) not in self.failed]
                if not live:
                    continue
                sync(states.device); start = time.perf_counter()
                record = dict(t=self.t, source=name, ell=self.ell)
                try:
                    with torch.no_grad():
                        values = provider(nodes, roots, self.word[self.t:], self.ctx, self.c, self.hard)
                    self.calls[name] += 1
                    for q, key in QUANTITIES:
                        v = values[key]
                        if v.shape != (len(nodes), L):
                            raise AssertionError('Invalid response shape: %s/%s' % (name, key))
                        if not bool(torch.isfinite(v).all()):
                            raise FloatingPointError('Nonfinite %s response' % key)
                    for q, key in QUANTITIES:
                        v = values[key]
                        before, after = w @ v[:len(w)], nw @ v[len(w):]
                        record[q] = dict(before=before.cpu().tolist(), after=after.cpu().tolist())
                        if (arm(name, 'add'), q) not in self.failed:
                            updated = self.add[name][q][self.t:]+math.exp(self.ell)*(signed @ v)
                            if bool(torch.isfinite(updated).all()):
                                self.add[name][q][self.t:] = updated
                            else:
                                self.fail(name, 'add', q, self.t, 'Nonfinite additive correction')
                        if (arm(name, 'mul'), q) not in self.failed:
                            valid = torch.isfinite(before) & torch.isfinite(after) & (before > 0) & (after > 0)
                            if not bool(valid.all()):
                                self.fail(name, 'mul', q, self.t,
                                    'Multiplicative correction requires finite positive before/after sums',
                                    invalid_prefixes=(torch.nonzero(~valid).flatten()+self.t+1).cpu().tolist())
                            else:
                                updated = self.logm[name][q][self.t:]+(before.log()-after.log())
                                if bool(torch.isfinite(updated).all()):
                                    self.logm[name][q][self.t:] = updated
                                else:
                                    self.fail(name, 'mul', q, self.t, 'Nonfinite log multiplier')
                except FloatingPointError as exc:
                    for m, q, _ in live:
                        self.fail(name, m, q, self.t, str(exc))
                    record['error'] = str(exc)
                finally:
                    sync(states.device)
                    self.correction_seconds[name] += time.perf_counter()-start
                self.trace.append(record)
        return self.validate(out, nw)


def rollout(K, word, ctx, c, states, weights, sources=None, hard=None):
    """Returns sim, rho[H,d,d], ell, and estimates[arm] = dict(P=..., N=...) (NaN when failed)."""
    sim = QoISim(K, word, ctx, c, states, weights, sources, hard)
    rho, ell = [], []
    for a in word:
        r, l = sim.advance(a)
        rho.append(r); ell.append(l)
    rho = torch.stack(rho)
    log_raw = weights.new_tensor(ell)
    raw = {'P': log_raw.exp()}
    raw['N'] = raw['P']*ghz_diagonal(rho, ctx['aux']['ghz'])[:, 0]
    out = {'M0': dict(raw)}
    nan = torch.full_like(log_raw, float('nan'))
    for name in sim.sources:
        for method in ESTIMATORS:
            a, est = arm(name, method), {}
            for q, _ in QUANTITIES:
                if (a, q) in sim.failed:
                    est[q] = nan.clone(); continue
                if method == 'add':
                    value = raw[q]+sim.add[name][q]
                elif q == 'P':
                    value = (log_raw+sim.logm[name][q]).exp()  # pc_sim order, bitwise continuity
                else:
                    value = raw[q]*sim.logm[name][q].exp()
                if not bool(torch.isfinite(value).all()):
                    sim.fail(name, method, q, len(word), 'Nonfinite corrected value')
                    value = nan.clone()
                est[q] = value
            out[a] = est
    return sim, rho, ell, out

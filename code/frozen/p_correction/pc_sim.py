"""The frozen M0 trajectory, with paired observer-only P corrections."""
import pc_runtime as rt
import math
import time
import numpy as np
import torch
import ghz_factors as gf
from ghz_merge import features, combine
from g1_core import lloyd_async, check_deadline
from rr_features import sync
from rr_sim import ResponseSim
from pc_estimators import ESTIMATORS, arm_name, normalize_estimators


class CorrectionSim(ResponseSim):
    def __init__(self, K, word, ctx, c, states, weights, sources=None, observer=None, hard=None,
                 estimators=ESTIMATORS):
        self.sources, self.observer = sources or {}, observer
        self.estimators = normalize_estimators(estimators)
        # Validate source names before any physical work. Additive names retain
        # the original API; "__mul" is reserved for the ratio estimator.
        for name in self.sources:
            for method in self.estimators:
                arm_name(name, method)
        self.failed_sources = {}
        self.correction = {name: weights.new_zeros(len(word)) for name in self.sources
                           if 'add' in self.estimators}
        self.log_multiplier = {name: weights.new_zeros(len(word)) for name in self.sources
                               if 'mul' in self.estimators}
        # Cost is provider-keyed: one psi call supplies both estimators.
        self.correction_cost = {name: 0. for name in self.sources}
        self.trace = []
        super().__init__(K, {'family': 'M0', 'geometry': 'geo'}, word, ctx, c,
                         {'physics_backend': 'sqrt'}, np.random.default_rng(0), states, weights, hard)

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
        # Preserve the original M0 operation ordering, including dense combine.
        out, nw = combine(states, w, labels)
        before_f = self.factors
        self.factors = gf.combine(before_f, w, labels)
        out = gf.density(self.factors)
        sync(states.device)
        self.cost['elimination_seconds'] += time.perf_counter()-start
        self.events.append(dict(t=self.t, before=len(w), after=len(nw),
                                mass_drift=abs(float(nw.sum()-w.sum()))))
        # All responses see the same concatenated nodes; corrections never feed back.
        if self.sources:
            nodes, roots = torch.cat((states, out)), torch.cat((before_f, self.factors))
            signed = torch.cat((w, -nw))
            # One cache lives for this shared accuracy event only. Independent
            # deployment always computes its own prior and charges that cost.
            scoring_ctx = dict(self.ctx, pc_shared_prior={}) if len(self.sources) > 1 else self.ctx
            for name, provider in self.sources.items():
                active = [method for method in self.estimators
                          if arm_name(name, method) not in self.failed_sources]
                if not active:
                    continue
                sync(states.device); start = time.perf_counter()
                record = dict(t=self.t, source=name, arm=name, ell=self.ell, estimators={})
                try:
                    with torch.no_grad():
                        values = provider(nodes, roots, self.word[self.t:], scoring_ctx, self.c, self.hard)
                        if values.shape != (len(nodes), len(self.word)-self.t):
                            raise AssertionError('Invalid absolute psi interface shape: '+name)
                        if not bool(torch.isfinite(values).all()):
                            raise FloatingPointError('Nonfinite absolute psi output')
                        before = w @ values[:len(w)]
                        after = nw @ values[len(w):]
                        record.update(psi_before=before.cpu().tolist(), psi_after=after.cpu().tolist())
                        if 'add' in active:
                            # Retain the original signed-matmul operation order.
                            delta = math.exp(self.ell)*(signed @ values)
                            updated = self.correction[name][self.t:]+delta
                            if not bool(torch.isfinite(updated).all()):
                                self.failed_sources[name] = dict(t=self.t, error='Nonfinite additive correction')
                                record['estimators']['add'] = 'NUMERIC_FAILURE'
                            else:
                                self.correction[name][self.t:] = updated
                                record.update(delta=delta.cpu().tolist())
                                record['estimators']['add'] = 'OK'
                        if 'mul' in active:
                            arm = arm_name(name, 'mul')
                            valid = torch.isfinite(before)&torch.isfinite(after)&(before > 0)&(after > 0)
                            if not bool(valid.all()):
                                self.failed_sources[arm] = dict(t=self.t,
                                    error='Multiplicative correction requires finite positive before/after psi sums',
                                    invalid_prefixes=(torch.nonzero(~valid).flatten()+self.t+1).cpu().tolist())
                                record['estimators']['mul'] = 'NUMERIC_FAILURE'
                            else:
                                # Absolute mass cancels in this ratio. Log space
                                # avoids forming a product of possibly large ratios.
                                increment = before.log()-after.log()
                                updated = self.log_multiplier[name][self.t:]+increment
                                if not bool(torch.isfinite(updated).all()):
                                    self.failed_sources[arm] = dict(t=self.t, error='Nonfinite log multiplier')
                                    record['estimators']['mul'] = 'NUMERIC_FAILURE'
                                else:
                                    self.log_multiplier[name][self.t:] = updated
                                    record.update(log_ratio=increment.cpu().tolist())
                                    record['estimators']['mul'] = 'OK'
                except FloatingPointError as exc:
                    # Invalid psi invalidates both estimators, but a nonpositive
                    # ratio above invalidates only the multiplicative estimator.
                    for method in active:
                        self.failed_sources[arm_name(name, method)] = dict(t=self.t, error=str(exc))
                        record['estimators'][method] = 'NUMERIC_FAILURE'
                    record['error'] = str(exc)
                finally:
                    sync(states.device)
                    self.correction_cost[name] += time.perf_counter()-start
                self.trace.append(record)
        if self.observer:
            self.observer(self, states, w, before_f, labels, out, nw, self.factors)
        return self.validate(out, nw)


def rollout(K, word, ctx, c, states, weights, sources=None, observer=None, hard=None,
            estimators=ESTIMATORS):
    sim = CorrectionSim(K, word, ctx, c, states, weights, sources, observer, hard, estimators)
    rho, ell = [], []
    for a in word:
        r, l = sim.advance(a)
        rho.append(r); ell.append(l)
    log_raw = weights.new_tensor(ell)
    raw = log_raw.exp()
    corrected = {name: raw+C if name not in sim.failed_sources else torch.full_like(raw, float('nan'))
                 for name, C in sim.correction.items()}
    for name, log_multiplier in sim.log_multiplier.items():
        arm = arm_name(name, 'mul')
        value = (log_raw+log_multiplier).exp()
        if arm not in sim.failed_sources and not bool(torch.isfinite(value).all()):
            sim.failed_sources[arm] = dict(t=len(word), error='Nonfinite multiplicative corrected probability')
        corrected[arm] = value if arm not in sim.failed_sources else torch.full_like(raw, float('nan'))
    return sim, torch.stack(rho), ell, dict(M0=raw, **corrected)

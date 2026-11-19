"""Exact branch propagation with interchangeable moment constraints; no learned physical rollout."""
import rr_runtime  # noqa: F401
import math
import time
import numpy as np
import torch
from bsc_core import expand
from ghz_physics import C, check_ensemble
from ghz_merge import combine, features
from g1_core import lloyd_async, check_deadline
from rr_compress import compress
from rr_features import build_features, sync
import ghz_factors


class ResponseSim:
    """Prefix estimates are read after expansion, before compression, including the final prefix.

    Initial t=0 compression IS performed. No unnecessary compression follows the
    terminal observable. P uses a separately accumulated log mass. Only unnormalized
    observables/P, not rho or log(P), inherit conditional unbiasedness.
    """
    def __init__(self, K, arm, word, ctx, c, b, rng, states, weights, hard=None, provider=build_features):
        self.K, self.arm, self.word, self.ctx, self.c, self.b = K, arm, word, ctx, c, b
        self.rng, self.hard, self.provider = rng, hard, provider
        self.factors = None
        if b.get('physics_backend') == 'sqrt':
            self.factors = ctx.get('initial_factors')
            if self.factors is None:
                # Strictly positive test inputs may be factored once. Near-pure
                # preparations must supply factors; no clipping/jitter fallback.
                self.factors = torch.linalg.cholesky(states)
            if self.factors.shape != states.shape or float((ghz_factors.density(self.factors)-states).abs().max()) > 1e-12:
                raise FloatingPointError('Initial density/factor mismatch')
        self.t, self.ell = 0, 0.
        self.events = []
        self.cost = dict(physics_seconds=0., response_seconds=0., partition_seconds=0.,
                         projection_seconds=0., elimination_seconds=0., validation_seconds=0.,
                         physical_calls=0, response_calls=0)
        self.states, self.weights = self.compress(states, weights)

    def validate(self, states, weights):
        start = time.perf_counter()
        check_ensemble(states, weights, self.c['physical_tolerance'])
        self.cost['validation_seconds'] += time.perf_counter()-start
        return states, weights

    def compress(self, states, w):
        if len(w) <= self.K:
            return self.validate(states, w)
        check_deadline(self.hard)
        if self.arm['family'] == 'M0':
            sync(states.device)
            start = time.perf_counter()
            labels = lloyd_async(features(self.arm['geometry'], states, self.ctx['aux']), w, self.K,
                                 self.c['clustering_iterations'])
            sync(states.device)
            self.cost['partition_seconds'] += time.perf_counter()-start
            start = time.perf_counter()
            out, nw = combine(states, w, labels)
            if self.factors is not None:
                self.factors = ghz_factors.combine(self.factors, w, labels)
                out = ghz_factors.density(self.factors)
            sync(states.device)
            self.cost['elimination_seconds'] += time.perf_counter()-start
            info = dict(before=len(w), after=len(nw), rank=None, eliminations=0,
                        mass_drift=abs(float(nw.sum()-w.sum())), moment_drift=None)
        else:
            feature_ctx = dict(self.ctx, response_factors=self.factors)
            A, meta = self.provider(states, w, self.word[self.t:], self.K, self.arm, feature_ctx,
                                    self.c, self.b, self.hard)
            for key in ('response_seconds', 'partition_seconds', 'projection_seconds', 'response_calls'):
                self.cost[key] += meta.get(key, 0)
            diagnostic = meta.pop('diagnostic_response', None)
            start = time.perf_counter()
            J, nw, info = compress(A, w.cpu().numpy(), self.K, self.rng,
                                  self.b['features']['rank_rtol'], self.b['features']['moment_tolerance'],
                                  response=diagnostic,
                                  checkpoint=lambda: check_deadline(self.hard))
            if diagnostic is not None:
                delta = float(nw @ diagnostic[J]-w.cpu().numpy() @ diagnostic)
                info.update(diagnostic_delta=delta, diagnostic_mass_scaled_delta=math.exp(self.ell)*delta)
            out = states[torch.as_tensor(J, device=states.device)]
            if self.factors is not None:
                self.factors = self.factors[torch.as_tensor(J, device=states.device)]
            nw = torch.as_tensor(nw, dtype=w.dtype, device=w.device)
            sync(states.device)
            self.cost['elimination_seconds'] += time.perf_counter()-start
            info.update(meta)
            info.pop('feature_names', None)  # names follow the documented fixed construction; avoid huge logs
        self.events.append(dict(t=self.t, **info))
        return self.validate(out, nw)

    def advance(self, a):
        check_deadline(self.hard)
        if a != self.word[self.t]:
            raise ValueError('Different protocol from the constraint provider')
        sync(self.states.device)
        start = time.perf_counter()
        self.cost['physical_calls'] += len(self.weights)
        if self.factors is None:
            states, weights, mass, _ = expand(self.states, self.weights, self.ctx['letters'][a], self.c)
        else:
            states, weights, mass, _, self.factors = ghz_factors.expand(
                self.factors, self.weights, self.ctx['letters'][a], self.c)
        if not math.isfinite(mass) or not 0 < mass <= 1+1e-10:
            raise FloatingPointError('Invalid acceptance')
        self.t += 1
        self.ell += math.log(mass)
        rho = torch.einsum('n,nij->ij', weights.to(C), states)
        sync(states.device)
        self.cost['physics_seconds'] += time.perf_counter()-start
        if self.t < len(self.word):
            self.states, self.weights = self.compress(states, weights)
        else:
            self.states, self.weights = self.validate(states, weights)
        return rho, self.ell

"""B0 constraint providers. All state response coordinates are UNNORMALIZED Pauli(F).

Learning can replace `build_features` through ResponseSim's provider argument; it
cannot replace physical propagation or the endpoint randomization probabilities.
"""
import rr_runtime  # noqa: F401
import time
import numpy as np
import torch
from bsc_core import future_response
from g1_core import lloyd_async, check_deadline
from ghz_physics import coords


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def pca(x, w, q):
    """Weighted, centered PCA scores; no fitted predictor or evaluation labels."""
    if q <= 0 or len(x) <= 1 or x.shape[1] == 0:
        return np.empty((len(x), 0))
    wn = w/w.sum()
    xc = x-wn @ x
    _, s, vh = np.linalg.svd(np.sqrt(wn[:, None])*xc, full_matrices=False)
    rank = int(np.sum(s > s[0]*1e-12)) if len(s) and s[0] else 0
    return xc @ vh[:min(q, rank)].T


class PTAAdapter:
    """The existing PTALetter already sums outcomes; expose its singleton to the tree routine."""
    accept = (0,)

    def __init__(self, letter):
        self.letter = letter

    def raw(self, states):
        return self.letter.raw(states)


def responses(states, remaining, depth, ctx, c, b, prior=False, hard=None):
    L = {a: PTAAdapter(v) for a, v in ctx['pta'].items()} if prior else ctx['letters']
    d = min(depth, len(remaining))
    if d == 0:
        return {}, 0
    per = int(np.prod([len(L[a].accept) for a in remaining[:d]]))
    ch = max(1, b['features']['chunk_leaves']//per)
    blocks, calls = {k: [] for k in range(1, d+1)}, 0
    factors = None if prior else ctx.get('response_factors')
    # Check deadline between bounded root batches; reuse the tested exact tree response.
    for i in range(0, len(states), ch):
        check_deadline(hard)
        R, cost = future_response(states[i:i+ch], remaining, list(blocks), L, ctx['aux'], c,
                                  b['features']['chunk_leaves'],
                                  factors=None if factors is None else factors[i:i+ch])
        calls += cost
        for k, values in R.items():
            blocks[k].append(values.cpu().numpy())
    return {k: np.concatenate(v) for k, v in blocks.items()}, calls


def build_features(states, weights, remaining, K, arm, ctx, c, b, hard=None):
    """Return A[rows,N] with constant row and metadata, fixed throughout elimination.

    Partition rule is common to P1/R2psi/R2; it reserves max d=4 global
    success rows and seven local rows (including the q=4 state candidate). We never
    silently discard requested constraints to fit K: infeasible rank is reported.
    """
    sync(states.device)
    start = time.perf_counter()
    f, family = b['features'], arm['family']
    w = weights.cpu().numpy()
    x_tensor = coords(states, ctx['aux']['pauli'])
    reserve = min(f['partition_reserve'], len(remaining))
    m = max(1, (K-reserve)//f['local_row_budget'])
    labels = lloyd_async(x_tensor, weights, m, c['clustering_iterations']).cpu().numpy()
    x = x_tensor.cpu().numpy()
    partition_seconds = time.perf_counter()-start
    t0 = time.perf_counter()
    R, calls = responses(states, remaining, arm.get('depth', 0), ctx, c, b,
                         prior=family == 'P1', hard=hard) if family in ('P1', 'R2psi', 'R2') else ({}, 0)
    response_seconds = time.perf_counter()-t0
    t0 = time.perf_counter()
    rows, names = [np.ones(len(w))], ['mass']
    # Success constraints are GLOBAL; rescaling by a fixed nonzero scalar does not change their span.
    scaled = {}
    for k, block in sorted(R.items()):
        scale = float(w @ block[:, 0]/w.sum())
        if not scale > 0:
            raise FloatingPointError('Nonpositive predicted future success')
        scaled[k] = block/scale
        rows.append(scaled[k][:, 0])
        names.append('psi_%d' % k)
    state_values = np.concatenate([v[:, 1:] for v in scaled.values()], axis=1) if scaled else None
    for label in np.unique(labels):
        J = np.flatnonzero(labels == label)
        local = [np.ones(len(J))]
        z = pca(x[J], w[J], f['base_q'])
        local.extend(z.T)
        if family == 'R1q':
            local.extend(z[:, a]*z[:, bb] for a in range(z.shape[1]) for bb in range(a, z.shape[1]))
        if family in ('P1', 'R2') and state_values is not None:
            local.extend(pca(state_values[J], w[J], arm.get('state_q', f['state_q'])).T)
        for i, values in enumerate(local):
            row = np.zeros(len(w))
            row[J] = values
            rows.append(row)
            names.append('cluster_%d_feature_%d' % (label, i))
    return np.stack(rows), dict(partitions=int(len(np.unique(labels))), feature_names=names,
                                depth=len(R), response_calls=calls, partition_seconds=partition_seconds,
                                response_seconds=response_seconds, projection_seconds=time.perf_counter()-t0)


def all_success_features(states, weights, remaining, K, arm, ctx, c, b, hard=None):
    """Only for short exact stage-1 tasks: mass + ALL true remaining prefix success responses."""
    start = time.perf_counter()
    R, calls = responses(states, remaining, len(remaining), ctx, c, b, hard=hard)
    rows = [np.ones(len(weights))]+[v[:, 0] for _, v in sorted(R.items())]
    return np.stack(rows), dict(partitions=1, depth=len(R), response_calls=calls,
                                response_seconds=time.perf_counter()-start,
                                partition_seconds=0., projection_seconds=0.)

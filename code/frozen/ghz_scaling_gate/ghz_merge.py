"""merge_K for the GHZ gate: the frozen merge rule on the exact same-history step.

Clustering is either the frozen numpy engine.clustering (CPU) or its line-by-line torch port
(`lloyd`, any device); representatives are convex means, so mass and mean are conserved.
Sums use one-hot matmuls (deterministic on GPU, unlike scatter/atomic adds).
"""
import math
import time
import torch
from ghz_physics import C, step, check_ensemble, coords, ghz_diagonal


def sqdist(x, centers):
    return torch.clamp((x*x).sum(1)[:, None]+(centers*centers).sum(1)[None, :]-2*x@centers.T, min=0.)


def onehot(labels, m, dtype):
    return (labels[None, :] == torch.arange(m, device=labels.device)[:, None]).to(dtype)


def lloyd(x, w, budget, iterations):
    """Torch port of steps_3_5/engine.clustering(kind='merge'): weighted farthest-point seeding,
    at most `iterations` Lloyd rounds, empty clusters dropped. Returns contiguous labels."""
    if len(w) <= budget:
        return torch.arange(len(w), device=x.device)
    center = w @ x
    ids = [int(torch.argmin(((x-center)**2).sum(1)))]
    dist = ((x-x[ids[0]])**2).sum(1)
    for _ in range(1, budget):
        score = w*dist
        score[ids] = -1.
        j = int(torch.argmax(score))
        if float(score[j]) <= 1e-28:
            break
        ids.append(j)
        dist = torch.minimum(dist, ((x-x[j])**2).sum(1))
    centers, last = x[ids].clone(), None
    for _ in range(iterations):
        labels = sqdist(x, centers).argmin(1)
        if last is not None and torch.equal(labels, last):
            break
        last = labels.clone()
        oh = onehot(labels, len(centers), x.dtype)
        mass = oh @ w
        keep = mass > 0
        centers = (oh @ (w[:, None]*x))[keep]/mass[keep, None]
    return torch.unique(sqdist(x, centers).argmin(1), return_inverse=True)[1]


def combine(states, weights, labels):
    """Convex representatives of a partition; total mass and mean state are conserved."""
    m, d = int(labels.max())+1, states.shape[-1]
    oh = onehot(labels, m, torch.float64)
    mass = oh @ weights
    if not bool((mass > 0).all()):
        raise FloatingPointError('Empty compression group')
    reps = (oh.to(C) @ (weights[:, None, None]*states).reshape(len(weights), -1)).reshape(m, d, d)
    return reps/mass[:, None, None], mass/mass.sum()


def features(kind, states, aux):
    """'geo': Pauli coordinates (= merge_K); 'diag': GHZ-diagonal coefficients."""
    return coords(states, aux['pauli']) if kind == 'geo' else ghz_diagonal(states, aux['ghz'])


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


class MergeK:
    """merge_K simulator. cluster='torch' uses `lloyd`; cluster='numpy' calls the frozen engine.clustering.

    `record(step, x, w)` (optional) receives the features of every pre-merge ensemble.
    Seconds spent in physics and in compression are accumulated (with device synchronization).
    """
    def __init__(self, K, kind, letters, aux, c, cluster='torch', engine=None, record=None):
        self.K, self.kind, self.letters, self.aux, self.c = K, kind, letters, aux, c
        self.cluster, self.engine, self.record = cluster, engine, record
        self.t, self.ell, self.removed, self.min_eig, self.min_p = 0, 0., 0., math.inf, math.inf
        self.seconds = dict(physics=0., compress=0.)
        self.atoms = []

    def start(self, states, weights):
        self.states, self.weights = self.compress(states, weights)
        return self

    def compress(self, states, weights):
        device = states.device
        sync(device)
        t0 = time.perf_counter()
        if len(weights) > self.K:
            x = features(self.kind, states, self.aux)
            if self.record is not None:
                self.record(self.t, x, weights)
            if self.cluster == 'numpy':
                labels = torch.as_tensor(self.engine.clustering(x.cpu().numpy(), weights.cpu().numpy(), self.K)[2], device=device)
            else:
                labels = lloyd(x, weights, self.K, self.c['clustering_iterations'])
            states, weights = combine(states, weights, labels)
        weights = weights/weights.sum()
        self.min_eig = min(self.min_eig, check_ensemble(states, weights, self.c['physical_tolerance']))
        sync(device)
        self.seconds['compress'] += time.perf_counter()-t0
        return states, weights

    def advance(self, a):
        sync(self.states.device)
        t0 = time.perf_counter()
        states, weights, mass, anti, pmin = step(self.states, self.weights, self.letters[a], self.c)
        sync(self.states.device)
        self.seconds['physics'] += time.perf_counter()-t0
        if not math.isfinite(mass) or not 0 < mass <= 1+1e-10:
            raise FloatingPointError('Invalid exact acceptance %r' % mass)
        self.t += 1
        self.ell += math.log(mass)
        self.removed, self.min_p = max(self.removed, anti), min(self.min_p, pmin)
        self.atoms.append(len(weights))
        self.states, self.weights = self.compress(states, weights)
        return torch.einsum('n,nij->ij', self.weights.to(C), self.states), self.ell

"""Square-root evaluation of the existing Letter instrument: rho = L L^dagger.

Kraus sums concatenate factors; the Schur product takes row-wise tensor products.
Reduced QR bounds factor width without deleting singular values. There is no
eigenvalue clipping, jitter, probability floor, or change to the physical map.
Dense states are readouts only: feeding their rounding error back through a rare
postselection can amplify O(eps) indefiniteness by 1/p (smoke 28115797).
"""
import torch
from ghz_physics import C, dag, history_letters


def reduced(factor):
    """L -> R^dagger where L^dagger = Q R; preserve L L^dagger, including rank deficiency."""
    return dag(torch.linalg.qr(dag(factor), mode='r').R)


def density(factor):
    rho = factor @ dag(factor)
    return (rho+dag(rho))/2


def raw_factors(factors, letter):
    """Outcome-resolved square roots, using the SAME Kraus/operators/outcomes as Letter.raw."""
    n, d, _ = factors.shape
    s = (letter.kraus @ factors[:, None]).transpose(1, 2).reshape(n, d, -1)
    if letter.pre is not None:
        s = letter.pre @ s
    if letter.basis == 'X':
        s = letter.H @ s
    s = reduced(s)
    shifted = s[:, letter.perm]
    out = (s[:, None, :, :, None]*shifted[:, :, :, None, :]).flatten(-2)
    if letter.basis == 'X':
        out = letter.H @ out
    if letter.post is not None:
        out = letter.post @ out
    return reduced(out)


def children(factors, letter, chunk):
    """Normalized factors and nonnegative probabilities, including exact zero outcomes."""
    raw = torch.cat([raw_factors(factors[i:i+chunk], letter) for i in range(0, len(factors), chunk)])
    p = raw.abs().square().sum((-2, -1))
    if not bool(torch.isfinite(p).all()):
        raise FloatingPointError('Nonfinite square-root outcome probability')
    normalizer = torch.where(p > 0, p.sqrt(), torch.ones_like(p))
    return raw/normalizer[..., None, None], p


def expand(factors, weights, letter, c):
    f, p = children(factors, letter, c['chunk'])
    w = (weights[:, None]*p).flatten()
    keep = w > 0  # no pruning of positive mass, however small
    parent = torch.arange(len(weights), device=weights.device).repeat_interleave(len(letter.accept))
    f, w, parent = f.flatten(0, 1)[keep], w[keep], parent[keep]
    mass = float(w.sum())
    if not mass > 0:
        raise FloatingPointError('Zero square-root acceptance')
    return density(f), w/mass, mass, parent, f


def combine(factors, weights, labels):
    """Square root of each existing convex merge representative, without factoring its dense readout."""
    out = []
    for label in range(int(labels.max())+1):
        use = labels == label
        w = weights[use]
        f = factors[use]*torch.sqrt(w/w.sum())[:, None, None]
        out.append(reduced(f.transpose(0, 1).flatten(1)))
    return torch.stack(out)


def prepare(prep, c, device):
    """Reuse GHZ initial/history definitions; retain factors from preparation onward."""
    from ghz_data import initial
    if prep['split'] == 'final':
        raise RuntimeError('Sealed GHZ final preparation: '+prep['id'])
    rho = torch.as_tensor(initial(prep, c)[None], dtype=C, device=device)
    # Preparations have strictly positive spectra. Failure is not repaired or projected.
    factors = torch.linalg.cholesky(rho)
    weights = torch.ones(1, dtype=torch.float64, device=device)
    letters = history_letters(prep['n'], prep['phi'], prep['gamma'], device)
    for a in ('AB'*prep['depth'])[:prep['depth']]:
        rho, weights, _, _, factors = expand(factors, weights, letters[a], c)
    return density(factors), weights, factors


def perturb(factors, eps):
    """Existing depolarizing sensitivity perturbation, applied as a positive mixture."""
    d = factors.shape[-1]
    identity = torch.eye(d, dtype=C, device=factors.device).expand_as(factors)
    return reduced(torch.cat([(1-eps)**.5*factors, (eps/d)**.5*identity], -1))

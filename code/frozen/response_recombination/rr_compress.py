"""Positive randomized nullspace elimination (Gonoskov), not a new resampling algorithm.

The feature matrix is fixed for the entire call. CPU float64 SVD on small blocks;
only endpoint coordinates are zeroed. No positive-weight pruning or renormalization.
"""
import rr_runtime  # noqa: F401
import time
import numpy as np
from scipy.linalg import svd as scipy_svd


class RankBudgetError(ValueError):
    def __init__(self, rank, budget):
        self.rank, self.budget = rank, budget
        super().__init__('constraint rank %d > K=%d' % (rank, budget))


def _checked_direction(block, vh, rtol):
    """Validate a returned basis before accepting the solver attempt."""
    checks = dict(vh_shape=list(vh.shape))

    def invalid(kind, message):
        exc = FloatingPointError(message)
        exc.svd_diagnostics = dict(checks, failure_kind=kind)
        raise exc

    if vh.shape != (block.shape[1], block.shape[1]):
        invalid('basis_shape', 'Invalid full right singular basis shape')
    finite = np.isfinite(vh)
    if not finite.all():
        checks.update(vh_nonfinite=int(np.count_nonzero(~finite)),
                      vh_nan=int(np.count_nonzero(np.isnan(vh))),
                      vh_inf=int(np.count_nonzero(np.isinf(vh))))
        invalid('nonfinite_basis', 'Nonfinite full right singular basis')
    u = vh[-1]
    if not np.any(u):
        invalid('zero_direction', 'Zero nullspace direction')
    norm = float(np.linalg.norm(u))
    checks['direction_norm'] = norm
    if not np.isfinite(norm) or abs(norm-1.) > rtol:
        invalid('nonunit_direction', 'Invalid nullspace direction norm: %g' % norm)
    scale = max(1., float(np.linalg.norm(block, ord=np.inf)*np.linalg.norm(u, ord=np.inf)))
    raw_residual = float(np.max(np.abs(block @ u)))
    residual = raw_residual/scale
    checks.update(max_residual=raw_residual, relative_residual=residual)
    if not np.isfinite(scale) or not np.isfinite(residual) or residual > rtol:
        invalid('residual', 'Nullspace residual exceeds tolerance: %g' % residual)
    return u, checks


def null_direction(block, rtol=1e-12):
    """Retry gesvd if gesdd raises OR returns an invalid direction/basis.

    A solver attempt includes its output checks. Both attempts use the unchanged
    finite matrix; neither consumes randomness or changes a valid usual direction.
    """
    block = np.asarray(block, dtype=np.float64)
    if block.ndim != 2 or not 0 < block.shape[0] < block.shape[1]:
        raise ValueError('Nullspace block must be a nonempty wide matrix')
    if not np.isfinite(block).all():
        raise FloatingPointError('Nonfinite nullspace block')
    attempts, invalid_outputs, last_error = [], {}, None
    solvers = (
        ('numpy_gesdd', lambda: np.linalg.svd(block, full_matrices=True)),
        ('scipy_gesvd', lambda: scipy_svd(block, full_matrices=True, lapack_driver='gesvd', check_finite=True)))
    for name, solve in solvers:
        start = time.perf_counter()
        try:
            _, _, vh = solve()
        except np.linalg.LinAlgError as exc:
            last_error = exc
            attempts.append(dict(solver=name, status='FAILED', failure_kind='exception',
                                 error=repr(exc), seconds=time.perf_counter()-start))
            continue
        try:
            u, checks = _checked_direction(block, vh, rtol)
        except FloatingPointError as exc:
            last_error = exc
            invalid_outputs[name] = vh.copy()
            attempts.append(dict(solver=name, status='FAILED', error=repr(exc),
                                 seconds=time.perf_counter()-start, **exc.svd_diagnostics))
            continue
        seconds = time.perf_counter()-start
        attempts.append(dict(solver=name, status='OK', seconds=seconds, **checks))
        fallback = name == 'scipy_gesvd'
        meta = dict(solver=name, fallback_used=fallback,
                    fallback_seconds=seconds if fallback else 0.,
                    relative_residual=checks['relative_residual'], attempts=attempts)
        if fallback:
            meta['primary_error'] = attempts[0]['error']
        return u, meta
    failure = np.linalg.LinAlgError('Both nullspace SVD attempts failed: '+
        '; '.join('%s: %s' % (a['solver'], a['error']) for a in attempts))
    failure.solver_attempts = attempts
    failure.solver_outputs = invalid_outputs
    raise failure from last_error


def row_basis(A, rtol):
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or not np.isfinite(A).all():
        raise ValueError('A must be a finite matrix')
    scales = np.max(np.abs(A), axis=1)
    scales[scales == 0] = 1.
    scaled = A/scales[:, None]
    _, s, vh = np.linalg.svd(scaled, full_matrices=False)
    rank = int(np.sum(s > rtol*s[0])) if len(s) and s[0] else 0
    return vh[:rank], scaled, rank


def endpoints(w, u):
    """Return maximal feasible steps and the exact endpoint indices (no pruning tolerance)."""
    pos, neg = np.flatnonzero(u > 0), np.flatnonzero(u < 0)
    if not len(pos) or not len(neg):
        raise FloatingPointError('Null direction has no two-sided feasible endpoint')
    plus = w[neg]/(-u[neg])
    minus = w[pos]/u[pos]
    tp, tm = float(plus.min()), float(minus.min())
    if not (np.isfinite(tp+tm) and tp > 0 and tm > 0):
        raise FloatingPointError('Invalid endpoint steps')
    return tp, tm, neg[plus == tp], pos[minus == tm]


def compress(A, weights, K, rng, rtol=1e-12, tolerance=1e-10, response=None, trace=False, checkpoint=None):
    """Return original indices, final positive weights and numerical diagnostics.

    `response` is optional scalar/vector F[N,...] for conditional-variance diagnostics.
    Directions depend on current weights/support/features, never on the next random coin.
    The first feature row must be exactly one (local mass rows may follow).
    Rank truncation is numerical, and drift is checked against ALL original scaled rows.
    """
    A, w0 = np.asarray(A, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    if w0.ndim != 1 or A.ndim != 2 or A.shape[1] != len(w0) or K < 1:
        raise ValueError('Shape/budget mismatch')
    if not np.isfinite(w0).all() or np.any(w0 < 0) or not w0.sum() > 0:
        raise ValueError('Weights must be finite, nonnegative and nonzero')
    if not len(A) or not np.array_equal(A[0], np.ones(len(w0))):
        raise ValueError('An explicit constant first row is required')
    active = np.flatnonzero(w0 > 0)  # pre-existing exact zeros only
    Q, scaled, rank = row_basis(A[:, active], rtol)
    if rank > K:
        raise RankBudgetError(rank, K)
    original = active.copy()
    w = w0[active].copy()
    target = scaled @ w
    # Shuffle once: avoid privileging the order in which the physics kernel lists outcomes.
    live = list(rng.permutation(len(w)))
    steps, variance, history, boundary_roundoff = 0, 0., [], 0.
    svd_fallbacks, svd_fallback_seconds, max_null_residual = 0, 0., 0.
    svd_fallback_details = []
    while len(live) > K:
        if checkpoint is not None and steps % 32 == 0:
            checkpoint()
        J = np.asarray(live[:min(rank+1, len(live))])
        try:
            u, solver = null_direction(Q[:, J], rtol)
        except (np.linalg.LinAlgError, FloatingPointError) as exc:
            # The caller persists this snapshot before aborting; no failed replicate is scored.
            failure = np.linalg.LinAlgError('Nullspace solver failed: %s' % exc)
            arrays = dict(block=Q[:, J], A=A, Q=Q, scaled=scaled, target=target,
                          J=J, w=w, w0=w0, original=original, live=np.asarray(live))
            arrays.update({'invalid_vh_'+name: value
                           for name, value in getattr(exc, 'solver_outputs', {}).items()})
            failure.compression_diagnostics = dict(
                arrays=arrays, solver_attempts=getattr(exc, 'solver_attempts', []),
                rank=rank, K=K, elimination=steps, rtol=rtol, tolerance=tolerance,
                rng_state=rng.bit_generator.state)
            raise failure from exc
        svd_fallbacks += int(solver['fallback_used'])
        svd_fallback_seconds += solver['fallback_seconds']
        max_null_residual = max(max_null_residual, solver['relative_residual'])
        if solver['fallback_used']:
            svd_fallback_details.append(dict(elimination=steps, attempts=solver['attempts']))
        tp, tm, bp, bm = endpoints(w[J], u)
        prob = tm/(tp+tm)
        if response is not None:
            d = np.tensordot(u, np.asarray(response)[original[J]], axes=(0, 0))
            variance += tp*tm*float(np.sum(d*d))
        positive = bool(rng.random() < prob)
        step = tp if positive else -tm
        updated = w[J]+step*u
        boundary = bp if positive else bm
        boundary_roundoff = max(boundary_roundoff, float(np.max(np.abs(updated[boundary]))))
        updated[boundary] = 0.  # exactly the active constraints that defined this endpoint
        if np.any(updated < 0) or not np.isfinite(updated).all():
            raise FloatingPointError('Negative/nonfinite non-boundary weight; no clipping allowed')
        w[J] = updated
        dead = set(J[boundary].tolist())
        live = [j for j in live if j not in dead]
        steps += 1
        if trace:
            history.append(dict(t_plus=tp, t_minus=tm, probability_plus=prob,
                                positive=positive, removed=len(dead)))
    drift = float(np.max(np.abs(scaled @ w-target)))
    mass_drift = abs(float(w.sum()-w0.sum()))
    if drift > tolerance*w0.sum() or mass_drift > tolerance*w0.sum():
        raise FloatingPointError('Moment drift exceeds tolerance: %g, mass %g' % (drift, mass_drift))
    keep = np.flatnonzero(w > 0)
    info = dict(rank=rank, rows=len(A), before=len(active), after=len(keep), eliminations=steps,
                moment_drift=drift, mass_drift=mass_drift, boundary_roundoff=boundary_roundoff,
                svd_fallbacks=svd_fallbacks, svd_fallback_seconds=svd_fallback_seconds,
                svd_fallback_details=svd_fallback_details,
                max_null_residual=max_null_residual,
                conditional_variance_sum=variance if response is not None else None)
    if trace:
        info['steps'] = history
    return original[keep], w[keep], info

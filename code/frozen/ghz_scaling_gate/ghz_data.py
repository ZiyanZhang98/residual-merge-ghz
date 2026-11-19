"""Preparations, splits and the sealed GHZ final (HANDOFF 5h). G0 computes no reference truth.

A preparation = GHZ-diagonal initial spectrum -> `depth` exact history rounds (AB)... with
history noise (R_y(phi) on party n, AD(gamma_h) on every qubit). Splits are by history;
the final split is sealed: `prepare` refuses it, and its specification hash is recorded in G0.
"""
import hashlib
import json
import numpy as np
import torch
from ghz_physics import C, I2, PAULI, kron, ghz_basis, ghz_state, history_letters, step


def depolarized_spectrum(n, F0):
    """GHZ-diagonal coefficients of D_p on every qubit of |GHZ>, p chosen by bisection so that F = F0."""
    B = ghz_basis(n)
    rho0 = np.outer(B[:, 0], B[:, 0]).astype(complex)
    paulis = [[kron([I2]*q+[P]+[I2]*(n-q-1)) for P in PAULI[1:]] for q in range(n)]

    def spectrum(p):
        r = rho0
        for ops in paulis:
            r = (1-3*p/4)*r + (p/4)*sum(o @ r @ o.conj().T for o in ops)
        return np.real(np.diag(B.T @ r @ B))
    lo, hi = 0., 1.
    for _ in range(100):
        mid = (lo+hi)/2
        lo, hi = (mid, hi) if spectrum(mid)[0] > F0 else (lo, mid)
    return spectrum((lo+hi)/2)


def phase_biased_spectrum(n, F0, tail):
    """F0 on |GHZ>, a fraction `tail` of the rest on the pure phase flip (s=1, j=0), the remainder uniform."""
    d = 2**n
    lam = np.full(d, (1-F0)*(1-tail)/(d-2))
    lam[0], lam[d//2] = F0, (1-F0)*tail
    return lam


def plan(c):
    preps = []
    s = c['spectra']
    for split, histories in c['histories'].items():
        for n in c['n']:
            for family in s['families']:
                for F0 in s['F0']:
                    for phi, gamma in histories:
                        for depth in c['history_depths']:
                            preps.append(dict(id='ghz%d__%s%.2f__h%.2f_%.2f__d%d' % (n, family, F0, phi, gamma, depth),
                                              split=split, n=n, family=family, F0=F0, phi=phi, gamma=gamma, depth=depth))
    return preps


def seal(c):
    """Hash of the final-split specification (recorded in G0; later stages verify it is unchanged)."""
    final = [p for p in plan(c) if p['split'] == 'final']
    return dict(final_preparations=len(final),
                final_sha256=hashlib.sha256(json.dumps(final, sort_keys=True).encode()).hexdigest())


def initial(prep, c):
    n = prep['n']
    if prep['family'] == 'depolarizing':
        lam = depolarized_spectrum(n, prep['F0'])
    else:
        lam = phase_biased_spectrum(n, prep['F0'], c['spectra']['phase_biased_tail'])
    return ghz_state(lam, ghz_basis(n))


def prepare(prep, c, device, final_access=False):
    """Initial branch ensemble of a preparation (exact, no merging). The sealed final split raises."""
    if prep['split'] == 'final' and not final_access:
        raise RuntimeError('Sealed GHZ final preparation: '+prep['id'])
    states = torch.as_tensor(initial(prep, c)[None], dtype=C, device=device)
    weights = torch.ones(1, dtype=torch.float64, device=device)
    letters = history_letters(prep['n'], prep['phi'], prep['gamma'], device)
    for a in ('AB'*prep['depth'])[:prep['depth']]:
        states, weights, _, _, _ = step(states, weights, letters[a], c)
    return states, weights

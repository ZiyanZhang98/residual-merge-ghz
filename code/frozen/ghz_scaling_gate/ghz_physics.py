"""Generic n-party physics for GHZ_n recurrence purification (HANDOFF section 5h).

Same-history pairing: every branch rho is purified against an identical copy, rho (x) rho.
A letter = single-copy noise (Kraus) -> optional local unitary on each copy -> transversal
CNOT -> measurement of copy 2 -> each accepted outcome pattern is a separate branch.

Structured kernel, no two-copy matrix is formed. For CNOT copy1->copy2 and the Z-basis
outcome y of copy 2:  child_y[a, c] = s[a, c] * s[a^y, c^y]  (s = noisy single-copy state).
The X-basis subprotocol (CNOT copy2->copy1, X-measured copy 2) is the same kernel in the
Hadamard frame:  child_x = H (t * t[x-shifted]) H  with  t = H s H.
Qubit 0 (most significant) is party 1, as in the frozen d=4 code.
"""
import itertools
from functools import reduce
import numpy as np
import torch

C = torch.complex128
I2 = np.eye(2, dtype=complex)
PAULI = [I2, np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.diag([1., -1.]).astype(complex)]
HAD = np.array([[1, 1], [1, -1]], complex)/np.sqrt(2)


def kron(mats):
    return reduce(np.kron, mats)


def dag(a):
    return a.conj().transpose(-1, -2)


def trace(a):
    return a.diagonal(dim1=-2, dim2=-1).sum(-1).real


def ry(theta):
    """exp(-i theta Y/2): the rotation convention of research_review/exp2/run_exp2.py."""
    return np.cos(theta/2)*I2 - 1j*np.sin(theta/2)*PAULI[2]


def amplitude_damping(gamma):
    return [np.array([[1, 0], [0, np.sqrt(1-gamma)]], complex), np.array([[0, np.sqrt(gamma)], [0, 0]], complex)]


def local_kraus(angles, gamma):
    """Kraus operators of R_y(angles[q]) on every qubit q, then amplitude damping AD(gamma) on every qubit."""
    rot = kron([ry(t) for t in angles])
    ks = [kron(k) @ rot for k in itertools.product(amplitude_damping(gamma), repeat=len(angles))]
    return np.stack([k for k in ks if np.abs(k).max() > 0])


def patterns(n, basis):
    """Accepted outcomes of copy 2: Z basis all-equal (P1, 2 patterns); X basis even parity (P2, 2^(n-1))."""
    d = 2**n
    return [0, d-1] if basis == 'Z' else [x for x in range(d) if bin(x).count('1') % 2 == 0]


class Letter:
    """One noisy purification round as a quantum instrument with outcome-resolved branches."""
    def __init__(self, kraus, basis, accept, pre=None, post=None, device='cpu'):
        t = lambda a: None if a is None else torch.as_tensor(np.asarray(a), dtype=C, device=device)  # noqa: E731
        self.d = kraus.shape[-1]
        self.n = self.d.bit_length()-1
        self.basis, self.accept = basis, list(accept)
        self.kraus, self.pre, self.post = t(kraus), t(pre), t(post)
        self.H = t(kron([HAD]*self.n))
        idx = torch.arange(self.d, device=device)
        self.perm = torch.stack([idx ^ y for y in self.accept])

    def raw(self, states):
        """Unnormalized accepted children [N, outcomes, d, d]; their traces are the outcome probabilities."""
        k = self.kraus
        s = (k @ states[:, None] @ dag(k)).sum(1)
        if self.pre is not None:
            s = self.pre @ s @ dag(self.pre)
        if self.basis == 'X':
            s = self.H @ s @ self.H
        p = self.perm
        out = s[:, None]*s[:, p[:, :, None], p[:, None, :]]
        if self.basis == 'X':
            out = self.H @ out @ self.H
        if self.post is not None:
            out = self.post @ out @ dag(self.post)
        return out


def ghz_letters(n, gamma, overrotation, device):
    """A = noise + P1, B = noise + P2; noise = R_y(overrotation) then AD(gamma) on every qubit."""
    k = local_kraus([overrotation]*n, gamma)
    return {'A': Letter(k, 'Z', patterns(n, 'Z'), device=device), 'B': Letter(k, 'X', patterns(n, 'X'), device=device)}


def history_letters(n, phi, gamma, device):
    """Preparation rounds: R_y(phi) on party n only, AD(gamma) on every qubit (mirrors the d=4 histories)."""
    k = local_kraus([0.]*(n-1)+[phi], gamma)
    return {'A': Letter(k, 'Z', patterns(n, 'Z'), device=device), 'B': Letter(k, 'X', patterns(n, 'X'), device=device)}


def dejmps_letters(gamma, eta, angle, device):
    """The frozen d=4 DEJMPS menu in this backend; used only to cross-check against run_exp2.Operation."""
    root = lambda s: (I2 + s*1j*PAULI[1])/np.sqrt(2)  # noqa: E731
    pre, post = np.kron(root(-1), root(1)), np.kron(I2, np.diag([1., np.sqrt(eta)]))
    return {a: Letter(local_kraus([0., th], gamma), 'Z', [0, 3], pre, post, device) for a, th in (('A', 0.), ('B', angle))}


def step(states, weights, letter, c):
    """Exact branching step: frozen g02.step semantics (every positive-mass leaf kept, weights
    renormalized) plus the HANDOFF 5f numerical rule, applied to the unnormalized (probability-
    scaled) children: an anti-Hermitian part above c['antihermitian_max'] is an error, below it
    is removed, then real-trace normalization. (Job 28050733: checking after normalization
    amplifies round-off by 1/p on rare branches; scaled by p its effect on any ensemble average
    is bounded by the same 1e-12.) Returns children, weights, acceptance, removed part, smallest p."""
    ch, d = c['chunk'], states.shape[-1]
    raw = torch.cat([letter.raw(states[i:i+ch]) for i in range(0, len(states), ch)])
    p = trace(raw)
    if float(p.min()) < -c['physical_tolerance']:
        raise FloatingPointError('Negative outcome probability %.3e' % float(p.min()))
    w = (weights[:, None]*p).reshape(-1)
    keep = w > 0
    raw, w = raw.reshape(-1, d, d)[keep], w[keep]
    mass = float(w.sum())
    if not mass > 0:
        raise FloatingPointError('Zero acceptance')
    anti = float((raw-dag(raw)).abs().max())/2
    if not anti <= c['antihermitian_max']:
        raise FloatingPointError('Anti-Hermitian part %.3e exceeds round-off bound; not cleaned' % anti)
    raw = (raw+dag(raw))/2
    return raw/trace(raw)[:, None, None], w/mass, mass, anti, float(p.reshape(-1)[keep].min())


def check_ensemble(states, weights, tol):
    """Physical ensemble or FloatingPointError (no repair, no projection). Returns the smallest eigenvalue."""
    herm = float((states-dag(states)).abs().max())
    tr = float((trace(states)-1).abs().max())
    eig = float(torch.linalg.eigvalsh(states).min())
    finite = bool(torch.isfinite(torch.view_as_real(states)).all() and torch.isfinite(weights).all())
    if not (finite and float(weights.min()) >= 0 and abs(float(weights.sum())-1) <= tol
            and herm <= tol and tr <= tol and eig >= -tol):
        raise FloatingPointError('Invalid ensemble: hermiticity %.2e, trace %.2e, min eigenvalue %.2e' % (herm, tr, eig))
    return eig


def perturb(states, eps):
    d = states.shape[-1]
    return (1-eps)*states + eps*torch.eye(d, dtype=C, device=states.device)/d


def pauli_stack(n, device):
    """All non-identity Pauli strings, IXYZ order, qubit 0 first (n=2 equals the frozen PAULI_MATS)."""
    return torch.as_tensor(np.stack([kron(p) for p in itertools.product(PAULI, repeat=n)][1:]), dtype=C, device=device)


def coords(states, stack):
    """Geometric features tr(P rho), as the frozen engine.pauli."""
    return torch.einsum('kij,nji->nk', stack, states).real


def ghz_basis(n):
    """Column s*2^(n-1)+j is |Psi_{s,j}> = (|0,j> + (-1)^s |1,~j>)/sqrt2; s = X^n bit, j_k = Z_1 Z_k bits."""
    d, h = 2**n, 2**(n-1)
    B = np.zeros((d, d))
    for s in (0, 1):
        for j in range(h):
            B[j, s*h+j] = 1/np.sqrt(2)
            B[h+(h-1-j), s*h+j] = (-1)**s/np.sqrt(2)
    return B


def ghz_diagonal(states, B):
    """GHZ-diagonal coefficients <Psi_{s,j}|rho|Psi_{s,j}> (B as a torch tensor); column 0 is the GHZ fidelity."""
    return torch.einsum('ia,nij,ja->na', B.conj(), states, B).real


def ghz_state(lam, B):
    return (B*np.asarray(lam)[None, :]) @ B.T


def recursion(lam, basis):
    """Analytic GHZ-diagonal map of the ideal subprotocols, unnormalized (its sum is the acceptance).
    P1 (Z): accept j1 = j2, s -> s1 xor s2.  P2 (X): accept s1 = s2, j -> j1 xor j2."""
    h = len(lam)//2
    L = np.asarray(lam).reshape(2, h)
    if basis == 'Z':
        out = np.stack([L[0]**2+L[1]**2, 2*L[0]*L[1]])
    else:
        j = np.arange(h)
        out = np.array([[np.sum(L[s]*L[s][j ^ k]) for k in range(h)] for s in (0, 1)])
    return out.reshape(-1)

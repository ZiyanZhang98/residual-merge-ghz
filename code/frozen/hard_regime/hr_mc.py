"""Independent Monte Carlo reference for P_h and F_h (HANDOFF 22.17); no merging anywhere.

One path: draw an initial atom by its weight; at every round draw one accepted outcome with
probability p_y / sum_y p_y and multiply the path weight by the round acceptance sum_y p_y.
Then E[W_h] = P_h and E[W_h <GHZ|rho_h|GHZ>] = N_h exactly (same-history branching is linear
in the branch measure). F_h = N_h / P_h with a delta-method standard error.
"""
import hr_runtime  # noqa: F401
import numpy as np
import torch
import ghz_factors as gf
from ghz_physics import ghz_diagonal
from g1_core import check_deadline

KEYS = ('w', 'w2', 'n', 'n2', 'wn')


def summarize(S, total):
    P = S['w']/total
    var_w = torch.clamp(S['w2']/total-P*P, min=0.)
    Nn = S['n']/total
    var_n = torch.clamp(S['n2']/total-Nn*Nn, min=0.)
    cov = S['wn']/total-P*Nn
    F = torch.where(P > 0, Nn/torch.clamp(P, min=1e-300), torch.zeros_like(P))
    var_F = torch.clamp(var_n-2*F*cov+F*F*var_w, min=0.)/torch.clamp(P*P, min=1e-300)/total
    se_P = (var_w/total).sqrt()
    return dict(P=P, N=Nn, F=F, se_P=se_P, se_P_relative=se_P/torch.clamp(P, min=1e-300), se_F=var_F.sqrt(),
                ess=(S['w']**2/torch.clamp(S['w2'], min=1e-300)))


@torch.no_grad()
def reference(prep, word, letters, aux, c, mc, device, hard, seed):
    states, weights, factors = gf.prepare(prep, c, device)
    B, H = aux['ghz'], len(word)
    g = torch.Generator(device=device).manual_seed(int(seed))
    S = {k: torch.zeros(H, dtype=torch.float64, device=device) for k in KEYS}
    total, target = 0, mc['initial_paths']
    while True:
        while total < target:
            check_deadline(hard)
            m = min(mc['chunk'], target-total)
            f = factors[torch.multinomial(weights, m, replacement=True, generator=g)]
            W = torch.ones(m, dtype=torch.float64, device=device)
            rows = torch.arange(m, device=device)
            for h, a in enumerate(word):
                cf, p = gf.children(f, letters[a], c['chunk'])
                acc = p.sum(1)
                W = W*acc
                draw = torch.where((acc > 0)[:, None], p, torch.ones_like(p))
                f = cf[rows, torch.multinomial(draw, 1, generator=g).squeeze(1)]
                WF = W*ghz_diagonal(gf.density(f), B)[:, 0]
                S['w'][h] += W.sum(); S['w2'][h] += (W*W).sum()
                S['n'][h] += WF.sum(); S['n2'][h] += (WF*WF).sum(); S['wn'][h] += (W*WF).sum()
            total += m
        st = summarize(S, total)
        resolved = bool(st['se_P_relative'].max() <= mc['target_P_relative'] and st['se_F'].max() <= mc['target_F_absolute'])
        if resolved or target >= mc['max_paths']:
            break
        target = min(2*target, mc['max_paths'])
    out = {k: v.cpu().numpy() for k, v in st.items()}
    return out, dict(paths=total, physics_calls=total*H, resolved=resolved,
                     max_se_P_relative=float(out['se_P_relative'].max()), max_se_F=float(out['se_F'].max()),
                     min_ess=float(out['ess'].min()))

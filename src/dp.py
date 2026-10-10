"""Exact dynamic programming for in-match tactical policy.

State: (minute index k, goal difference g from our perspective).
Action a: a style choice with per-minute scoring rates (lf[a], la[a]) that may depend on (k, g).
Within a minute: P(we score)=lf, P(they score)=la, P(both) ignored (O(dt^2)); rates are small (≈0.015/min).

value(k, g) = max_a  lf*V(k+1, g+1) + la*V(k+1, g-1) + (1-lf-la)*V(k+1, g)
Terminal utility U(g): default league points (3/1/0); any utility (e.g. title-race weighting) can be passed.
"""
import numpy as np

GMAX = 8


def solve(rates_for, rates_against, utility=None, fixed_policy=None):
    """rates_*: arrays of shape (K, G, A) with per-minute rates for each minute, goal-diff index and action.
    Goal-diff index i corresponds to g = i - GMAX.
    fixed_policy: optional int array (K, G) to evaluate instead of optimise.
    Returns V (K+1, G), policy (K, G)."""
    K, G, A = rates_for.shape
    g = np.arange(G) - GMAX
    if utility is None:
        utility = np.where(g > 0, 3.0, np.where(g == 0, 1.0, 0.0))
    V = np.zeros((K + 1, G))
    V[K] = utility
    pol = np.zeros((K, G), dtype=int)
    for k in range(K - 1, -1, -1):
        nxt = V[k + 1]
        up = np.append(nxt[1:], nxt[-1])     # g+1 (clamped)
        dn = np.insert(nxt[:-1], 0, nxt[0])   # g-1 (clamped)
        lf = rates_for[k]; la = rates_against[k]
        q = lf * up[:, None] + la * dn[:, None] + (1 - lf - la) * nxt[:, None]
        if fixed_policy is None:
            pol[k] = q.argmax(1)
        else:
            pol[k] = fixed_policy[k]
        V[k] = q[np.arange(G), pol[k]]
    return V, pol


def outcome_dist(rates_for, rates_against, policy, start_g=0):
    """Forward pass: distribution over final goal difference under a policy."""
    K, G, A = rates_for.shape
    p = np.zeros(G); p[start_g + GMAX] = 1.0
    for k in range(K):
        a = policy[k]
        lf = rates_for[k, np.arange(G), a]; la = rates_against[k, np.arange(G), a]
        stay = p * (1 - lf - la)
        new = stay.copy()
        new[1:] += (p * lf)[:-1]; new[-1] += (p * lf)[-1]
        new[:-1] += (p * la)[1:]; new[0] += (p * la)[0]
        p = new
    return p

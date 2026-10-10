"""P2: value of the DP-optimal in-match policy for a heavy underdog (pwin ~ 0.15).

Two frontiers:
 (A) registered: lagged-style GLM (P1 'full' lagged model). Actions = own style knobs in {-1,0,+1} SD (27 combos,
     within observed support). The opponent's style is held at typical (z=0).
 (B) post-hoc: coaching-decision modes {defensive, neutral, attacking} with multiplicative effects from the event study.
Baseline rates per (minute, goal diff) come from the strength+state GLM.

Gain is reported three ways:
  naive      optimise and evaluate on the same fit (the optimiser's curse inflates this)
  literal    pre-registered criterion: refit on bootstrap resamples, re-optimise, share with gain > 0
  crossfit   optimise on resample b, evaluate under an independent resample b' (the honest number)
"""
import itertools, json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from frontier import *
import dp

K = 95  # minutes of play incl. stoppage
G = 2 * dp.GMAX + 1
RNG = np.random.default_rng(20261011)
ACTIONS = list(itertools.product([-1, 0, 1], repeat=3))  # def_height, press_rate, long_share


def underdog_rates(lam_u=0.85, lam_f=2.05):
    """pwin ~0.15 for the underdog with these goal expectations (checked below)."""
    return lam_u, lam_f


def state_frame(lam_own, lam_opp, gd_sign=1):
    k = np.arange(K); g = np.arange(G) - dp.GMAX
    kk, gg = np.meshgrid(k, g, indexing="ij")
    q = pd.DataFrame(dict(gd_c=np.clip(gd_sign * gg.ravel(), -2, 2), men_c=0.0, tmin=(kk.ravel() + 0.5) / 90.0,
                          minutes=1.0, lam_own=lam_own, log_ratio=np.log(lam_own / lam_opp)))
    return q


def rates_A(params, sc, lam_u, lam_f):
    """Per-minute rates (K,G,A) under the lagged-style GLM for every action."""
    qf = state_frame(lam_u, lam_f, 1); qa = state_frame(lam_f, lam_u, -1)
    RF = np.zeros((K, G, len(ACTIONS))); RA = np.zeros_like(RF)
    for ai, a in enumerate(ACTIONS):
        for q, R, own in ((qf, RF, True), (qa, RA, False)):
            q = q.copy()
            for c in STYLE:
                z_us = dict(zip(POLICY_KNOBS, a)).get(c, 0.0)
                us_val = sc.mu[c] + z_us * sc.sd[c]
                q[c + "_lag"] = us_val if own else sc.mu[c]
                q[c + "_lag_opp"] = sc.mu[c] if own else us_val
            X, off = design(q, sc, "full", lag=True)
            R[:, :, ai] = np.exp(X.values @ params + off.values).reshape(K, G)
    return RF, RA


def rates_base(params_b, sc, lam_u, lam_f):
    qf = state_frame(lam_u, lam_f, 1); qa = state_frame(lam_f, lam_u, -1)
    Xf, of = design(qf, sc, "base"); Xa, oa = design(qa, sc, "base")
    return np.exp(Xf.values @ params_b + of.values).reshape(K, G), np.exp(Xa.values @ params_b + oa.values).reshape(K, G)


def value(RF, RA, pol=None):
    V, p = dp.solve(RF, RA, fixed_policy=pol)
    return V[0, dp.GMAX], p


def main():
    s = load_segments()
    need = ["lam_own"] + [c + "_lag" for c in STYLE] + [c + "_lag_opp" for c in STYLE]
    s = s.dropna(subset=need)
    sc = StyleScaler().fit(s, STYLE)
    lam_u, lam_f = underdog_rates()
    out = {}
    # baseline calibration
    Xb, ob = design(s, sc, "base"); rb = fit_glm(s, Xb, ob)
    bf, ba = rates_base(rb.params, sc, lam_u, lam_f)
    V0, _ = dp.solve(bf[:, :, None], ba[:, :, None])
    dist = dp.outcome_dist(bf[:, :, None], ba[:, :, None], np.zeros((K, G), int))
    g = np.arange(G) - dp.GMAX
    out["baseline"] = dict(lam_u=lam_u, lam_f=lam_f, E_pts=float(V0[0, dp.GMAX]),
                           pW=float(dist[g > 0].sum()), pD=float(dist[g == 0].sum()), pL=float(dist[g < 0].sum()),
                           xg_u=float(bf[:, dp.GMAX].sum()), xg_f=float(ba[:, dp.GMAX].sum()))
    print("baseline", out["baseline"])

    # ---------- (A) registered lagged-style frontier ----------
    Xf, of = design(s, sc, "full", lag=True)
    neutral = ACTIONS.index((0, 0, 0))
    def fitA(d):
        X, o = design(d, sc, "full", lag=True); return fit_glm(d, X, o).params
    pA = fitA(s)
    RF, RA = rates_A(pA, sc, lam_u, lam_f)
    v_typ, _ = value(RF, RA, np.full((K, G), neutral))
    v_opt, pol = value(RF, RA)
    out["A_naive"] = dict(typical=float(v_typ), optimal=float(v_opt), gain=float(v_opt - v_typ),
                          policy_at_0_0_min1=ACTIONS[pol[0, dp.GMAX]], policy_at_0_0_min70=ACTIONS[pol[70, dp.GMAX]],
                          policy_trail1_min75=ACTIONS[pol[75, dp.GMAX - 1]], policy_lead1_min75=ACTIONS[pol[75, dp.GMAX + 1]])
    print("A naive", out["A_naive"])
    mids = s.match_id.unique(); grp = {m: ix for m, ix in s.groupby("match_id").indices.items()}
    def resample():
        pick = RNG.choice(mids, len(mids)); return s.iloc[np.concatenate([grp[m] for m in pick])]
    B = 60
    lit, cross = [], []
    fits = [fitA(resample()) for _ in range(2 * B)]
    for b in range(B):
        RFb, RAb = rates_A(fits[b], sc, lam_u, lam_f)
        vt, _ = value(RFb, RAb, np.full((K, G), neutral)); vo, polb = value(RFb, RAb)
        lit.append(vo - vt)
        RFe, RAe = rates_A(fits[B + b], sc, lam_u, lam_f)
        vt2, _ = value(RFe, RAe, np.full((K, G), neutral)); vo2, _ = value(RFe, RAe, polb)
        cross.append(vo2 - vt2)
    lit, cross = np.array(lit), np.array(cross)
    out["A_literal"] = dict(mean_gain=float(lit.mean()), share_gt0=float((lit > 0).mean()))
    out["A_crossfit"] = dict(mean_gain=float(cross.mean()), ci95=[float(np.percentile(cross, 2.5)), float(np.percentile(cross, 97.5))],
                             share_gt0=float((cross > 0).mean()))
    print("A literal", out["A_literal"], "A crossfit", out["A_crossfit"])

    # ---------- (B) post-hoc coaching-mode frontier ----------
    es = json.load(open(os.path.join(ROOT, "results", "event_study.json")))
    def modes(bf_att, ba_att, bf_dfn, ba_dfn):
        mf = np.array([np.exp(bf_dfn), 1.0, np.exp(bf_att)]); ma = np.array([np.exp(ba_dfn), 1.0, np.exp(ba_att)])
        return bf[:, :, None] * mf, ba[:, :, None] * ma
    pe = [es["f"]["att"][0], es["a"]["att"][0], es["f"]["dfn"][0], es["a"]["dfn"][0]]
    se = [es["f"]["att"][1], es["a"]["att"][1], es["f"]["dfn"][1], es["a"]["dfn"][1]]
    RF, RA = modes(*pe)
    vt, _ = value(RF, RA, np.full((K, G), 1)); vo, polB = value(RF, RA)
    out["B_naive"] = dict(typical=float(vt), optimal=float(vo), gain=float(vo - vt),
                          share_states_attacking=float((polB == 2).mean()), share_states_defensive=float((polB == 0).mean()))
    cross = []
    for b in range(400):
        d1 = RNG.normal(pe, se); d2 = RNG.normal(pe, se)
        _, pb = value(*modes(*d1))
        RFe, RAe = modes(*d2)
        cross.append(value(RFe, RAe, pb)[0] - value(RFe, RAe, np.full((K, G), 1))[0])
    cross = np.array(cross)
    out["B_crossfit"] = dict(mean_gain=float(cross.mean()), ci95=[float(np.percentile(cross, 2.5)), float(np.percentile(cross, 97.5))],
                             share_gt0=float((cross > 0).mean()))
    print("B naive", out["B_naive"], "B crossfit", out["B_crossfit"])

    # ---------- (C) what-if bound: how large would mode effects need to be? ----------
    # A hypothetical 'low-block' mode that cuts both teams' rates by x% (share unchanged), switched on at will
    bound = {}
    for cut in (0.05, 0.10, 0.20, 0.30):
        RF = np.stack([bf, bf * (1 - cut)], 2); RA = np.stack([ba, ba * (1 - cut)], 2)
        vt, _ = value(RF, RA, np.zeros((K, G), int)); vo, pc = value(RF, RA)
        bound[f"cut{int(cut*100)}"] = dict(gain=float(vo - vt), share_states_low=float((pc[:, dp.GMAX - 3:dp.GMAX + 4] == 1).mean()))
    out["C_intensity_bound"] = bound
    print("C", bound)
    json.dump(out, open(os.path.join(ROOT, "results", "p2_policy.json"), "w"), indent=1, default=lambda o: list(o) if isinstance(o, tuple) else float(o))


if __name__ == "__main__":
    main()

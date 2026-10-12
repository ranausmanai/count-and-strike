"""Round 7: escalation profile of trailing teams and the optimal 'all-in' timing (PREREGISTRATION_7.md)."""
import json, os, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
sys.path.insert(0, os.path.dirname(__file__))
import dp
from frontier import load_segments
ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")
BUCKETS = [(0, 30), (30, 45), (45, 60), (60, 75), (75, 200)]
BNAMES = ["0-30", "30-45", "45-60", "60-75", "75+"]
K = 95; G = 2 * dp.GMAX + 1


def prep_sb15():
    s = load_segments(min_minutes=0.0).dropna(subset=["lam_own"])
    s["s_for"] = s.lam_own; s["s_ag"] = s.lam_opp
    s["comp"] = s.league
    s["pwin_u"] = s.pwin
    return s


def prep_loo(s):
    """Strength = leave-this-match-out xG for/against per match within comp-season."""
    tm = s.groupby(["comp", "match_id", "team"])[["xg_for", "xg_against"]].sum().reset_index()
    g = tm.groupby(["comp", "team"]); n = g.match_id.transform("size")
    tm["att"] = (g.xg_for.transform("sum") - tm.xg_for) / (n - 1)
    tm["dfn"] = (g.xg_against.transform("sum") - tm.xg_against) / (n - 1)
    s = s.merge(tm[["comp", "match_id", "team", "att", "dfn"]], on=["comp", "match_id", "team"])
    o = tm[["comp", "match_id", "team", "att", "dfn"]].rename(columns={"team": "opp", "att": "o_att", "dfn": "o_dfn"})
    s = s.merge(o, on=["comp", "match_id", "opp"])
    # expected match xG for/against (multiplicative, league-normalised)
    lm = s.groupby("comp").xg_for.transform("sum") / s.groupby("comp").minutes.transform("sum") * 90
    s["s_for"] = s.att * s.o_dfn / lm; s["s_ag"] = s.o_att * s.dfn / lm
    s = s[(s.s_for > 0) & (s.s_ag > 0)]
    return s


def bucket(cm):
    return np.select([cm < 30, cm < 45, cm < 60, cm < 75], [0, 1, 2, 3], 4)


def multipliers(s, side):
    """Poisson GLM with buckets × state (gd −1, 0, +1; no red cards). Returns dict state->[m per bucket]."""
    d = s[(s.men_diff == 0) & (s.gd.between(-1, 1)) & (s.minutes > 0.5)].copy()
    d["b"] = bucket(d.clock_mid)
    y = d.xg_for if side == "for" else d.xg_against
    strength = d.s_for if side == "for" else d.s_ag
    X = pd.DataFrame(index=d.index)
    X["home"] = d.home if side == "for" else 1 - d.home
    for b in range(5):
        X[f"b{b}"] = (d.b == b).astype(float)
        X[f"tr{b}"] = ((d.b == b) & (d.gd == -1)).astype(float)
        X[f"ld{b}"] = ((d.b == b) & (d.gd == 1)).astype(float)
    m = sm.GLM(y.values, X.values, family=sm.families.Poisson(), offset=np.log(d.minutes / 90 * strength).values).fit(scale="X2")
    p = dict(zip(X.columns, m.params))
    return dict(trail=[float(np.exp(p[f"tr{b}"])) for b in range(5)], lead=[float(np.exp(p[f"ld{b}"])) for b in range(5)],
                base=[float(np.exp(p[f"b{b}"])) for b in range(5)])


def profile(s, B=200, seed=20261012):
    mf, ma = multipliers(s, "for"), multipliers(s, "against")
    rng = np.random.default_rng(seed); mids = s.match_id.unique(); ix = s.groupby("match_id").indices
    lr = []
    for _ in range(B):
        d = s.iloc[np.concatenate([ix[m] for m in rng.choice(mids, len(mids))])]
        f = multipliers(d, "for")
        lr.append(np.log(f["trail"][4] / f["trail"][2]))
    return mf, ma, dict(log_ratio=float(np.log(mf["trail"][4] / mf["trail"][2])), ci=[float(np.percentile(lr, 2.5)), float(np.percentile(lr, 97.5))])


def rates(lam_u, lam_f, mf, ma, allin_shrink=0.0):
    """Per-minute rates (K,G,2) for the underdog; action 0 normal, 1 all-in (only differs when trailing)."""
    RF = np.zeros((K, G, 2)); RA = np.zeros((K, G, 2))
    for k in range(K):
        b = bucket(np.array([k + 0.5]))[0]
        base_f = lam_u / 90 * mf["base"][b] / np.mean(mf["base"]); base_a = lam_f / 90 * ma["base"][b] / np.mean(ma["base"])
        for gi in range(G):
            g = gi - dp.GMAX
            if g < 0:
                nf, na = mf["trail"][b], ma["trail"][b]
                af = 1 + (mf["trail"][4] - 1) * (1 - allin_shrink); aa = ma["trail"][4]
                af = max(af, nf); aa = max(aa, na) if af > nf else na  # all-in is never less attacking than normal
            elif g > 0:
                nf, na = mf["lead"][b], ma["lead"][b]; af, aa = nf, na
            else:
                nf = na = af = aa = 1.0
            RF[k, gi] = [base_f * nf, base_f * af]; RA[k, gi] = [base_a * na, base_a * aa]
    return RF, RA


def analyse(mf, ma, lam_u, lam_f, fall_minutes, shrink=0.0):
    RF, RA = rates(lam_u, lam_f, mf, ma, shrink)
    V, pol = dp.solve(RF, RA)
    g1 = dp.GMAX - 1
    opt_switch = next((k for k in range(K) if pol[k, g1] == 1), None)
    # observed escalation: first bucket where m_for exceeds the midpoint of 45-60 and 75+
    mid = (mf["trail"][2] + mf["trail"][4]) / 2
    ob = next(b for b in range(5) if mf["trail"][b] >= mid and b >= 2)
    obs_switch = [0, 30, 45, 60, 75][ob]
    pol_obs = np.zeros((K, G), int)
    pol_obs[:, :dp.GMAX] = (np.arange(K) >= obs_switch)[:, None].astype(int)
    pol_mix = pol.copy(); pol_mix[:, :dp.GMAX] = pol_obs[:, :dp.GMAX]
    Vo, _ = dp.solve(RF, RA, fixed_policy=pol_mix)
    fm = np.clip(np.array(fall_minutes).astype(int), 0, K - 1)
    gain = float(np.mean(V[fm, g1] - Vo[fm, g1]))
    return dict(opt_switch_minute=opt_switch, observed_switch_minute=obs_switch, gain_per_trailing_situation=gain,
                policy_monotone=bool(all(np.diff(pol[:, g1]) >= 0)))


def fall_behind_minutes(s, pw_lo=0.15, pw_hi=0.35):
    """Minutes at which underdogs first go 1 down (from segments: first segment with gd == -1 after gd == 0)."""
    s = s.sort_values(["match_id", "team", "clock_mid"])
    out = []
    for (m, t), g in s.groupby(["match_id", "team"]):
        if not (pw_lo <= g.pwin_u.iloc[0] <= pw_hi):
            continue
        prev = g.gd.shift(1)
        hit = g[(g.gd == -1) & (prev == 0)]
        if len(hit):
            out.append(hit.clock_mid.iloc[0] - hit.minutes.iloc[0] / 2)
    return out


def main():
    pops = {}
    s15 = prep_sb15(); pops["SB_1516"] = s15
    for name, f in (("WY_1718", "r7_segments_wy.parquet"), ("SB2_unseen", "r7_segments_sb2.parquet")):
        s = prep_loo(pd.read_parquet(os.path.join(PROC, f)))
        # underdog definition without odds: ratio of expected xG
        r = s.groupby(["match_id", "team"]).apply(lambda g: g.s_for.iloc[0] / (g.s_for.iloc[0] + g.s_ag.iloc[0])).rename("share")
        s = s.merge(r.reset_index(), on=["match_id", "team"])
        s["pwin_u"] = np.interp(s.share, [0.2, 0.3, 0.4, 0.5, 0.6], [0.08, 0.17, 0.28, 0.38, 0.50])  # share→pwin map from the DP (λ total 2.7)
        pops[name] = s
    out = {}
    lam_u, lam_f = 1.05, 1.75  # pwin ≈ 0.25 underdog
    prof = {}
    for name, s in pops.items():
        mf, ma, h1 = profile(s)
        prof[name] = (mf, ma)
        fm = fall_behind_minutes(s)
        res = dict(m_for_trail=mf["trail"], m_against_trail=ma["trail"], m_for_lead=mf["lead"], m_against_lead=ma["lead"], H7_1=h1, n_fall=len(fm))
        for sh in (0.0, 0.25, 0.5):
            res[f"decision_shrink{int(sh*100)}"] = analyse(mf, ma, lam_u, lam_f, fm, sh)
        out[name] = res
        print(name, json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in res.items()}, default=str))
    # cross-population: optimise on A's multipliers, evaluate under B's
    cross = {}
    names = list(pops)
    for a in names:
        for b in names:
            if a == b:
                continue
            RFa, RAa = rates(lam_u, lam_f, *prof[a]); _, pa = dp.solve(RFa, RAa)
            RFb, RAb = rates(lam_u, lam_f, *prof[b])
            mf = prof[b][0]; mid = (mf["trail"][2] + mf["trail"][4]) / 2
            ob = next(x for x in range(5) if mf["trail"][x] >= mid and x >= 2); obs = [0, 30, 45, 60, 75][ob]
            po = pa.copy(); po[:, :dp.GMAX] = (np.arange(K) >= obs)[:, None].astype(int)
            Va, _ = dp.solve(RFb, RAb, fixed_policy=pa); Vo, _ = dp.solve(RFb, RAb, fixed_policy=po)
            fm = np.clip(np.array(fall_behind_minutes(pops[b])).astype(int), 0, K - 1)
            cross[f"{a}->{b}"] = float(np.mean(Va[fm, dp.GMAX - 1] - Vo[fm, dp.GMAX - 1]))
    out["cross_population_gain"] = cross
    print("cross", cross)
    json.dump(out, open(os.path.join(ROOT, "results", "r7_chase.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

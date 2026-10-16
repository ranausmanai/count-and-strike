"""Round 22B: underdog-specific rules (PREREGISTRATION_22.md). B1 underdog-only mining, B2 underdog x favourite interaction mining, B3 rule effects by tier."""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import PROC, ROOT, mine_family, describe_attack, describe_press
from r20_bridge import within
from r22_ratings import ratings
POPS = ("discovery", "confirm_club", "confirm_women")


def tiers():
    R = ratings().dropna(subset=["gap"])
    M = pd.read_parquet(os.path.join(PROC, "r15_attack.parquet"))[["match_id", "pop"]].drop_duplicates()
    R = R.merge(M, on="match_id")
    q = R.groupby("pop").gap.quantile([1 / 3, 2 / 3]).unstack()
    R = R.join(q, on="pop")
    R["tier"] = np.select([R.gap <= R[1 / 3], R.gap >= R[2 / 3]], ["under", "fav"], "mid")
    print(R.groupby(["pop", "tier"]).size().unstack(), "\n", q)
    return R[["match_id", "team", "gap", "tier"]]


def add_tier(D, T):
    return D.merge(T, on=["match_id", "team"], how="inner")


def stats(D):
    st = D.groupby(["cell", "action"]).net.agg(["mean", "var", "size"]).reset_index()
    return st.set_index(["cell", "action"])


def interaction_mine(D, fam):
    disc = D[D["pop"] == "discovery"]
    ref = disc.groupby(["cell", "action"]).size().reset_index(name="n").sort_values("n", ascending=False).drop_duplicates("cell").set_index("cell").action
    out = {}
    for p in POPS:
        d = D[D["pop"] == p]; su = stats(d[d.tier == "under"]); sf = stats(d[d.tier == "fav"]); rows = []
        for (c, a) in su.index.intersection(sf.index):
            if c not in ref.index or a == ref[c] or (c, ref[c]) not in su.index or (c, ref[c]) not in sf.index:
                continue
            u, ur, f, fr = su.loc[(c, a)], su.loc[(c, ref[c])], sf.loc[(c, a)], sf.loc[(c, ref[c])]
            n = min(u["size"], ur["size"], f["size"], fr["size"])
            du = u["mean"] - ur["mean"]; df = f["mean"] - fr["mean"]
            se = np.sqrt(u["var"] / u["size"] + ur["var"] / ur["size"] + f["var"] / f["size"] + fr["var"] / fr["size"])
            rows.append(dict(cell=c, action=a, ref=ref[c], n=int(n), du=du, df=df, inter=du - df, se=se))
        out[p] = pd.DataFrame(rows).set_index(["cell", "action"]) if rows else pd.DataFrame()
    S = out["discovery"]; S = S[S.n >= 30].copy()
    S["p"] = 2 * norm.sf(np.abs(S.inter / S.se)); S["q"] = multipletests(S.p, method="fdr_bh")[1] if len(S) else []
    hits = S[S.q < 0.05]; conf = []
    for (c, a), r in hits.iterrows():
        e = [out[p].loc[(c, a)] for p in ("confirm_club", "confirm_women") if (c, a) in out[p].index and out[p].loc[(c, a)].n >= 10]
        if len(e) < 2:
            continue
        w = np.array([1 / x.se ** 2 for x in e]); pooled = float((w * np.array([x.inter for x in e])).sum() / w.sum()); pse = float(1 / np.sqrt(w.sum()))
        conf.append(dict(family=fam, cell=c, action=a, ref=r.ref, disc_inter=float(r.inter), disc_du=float(r.du), disc_df=float(r.df), pooled=pooled, pooled_se=pse,
                         p1=float(norm.sf(np.sign(r.inter) * pooled / pse)), same=bool(all(np.sign(x.inter) == np.sign(r.inter) for x in e)),
                         conf_du=[float(x.du) for x in e], conf_df=[float(x.df) for x in e]))
    conf = sorted(conf, key=lambda d: d["p1"]); k = len(conf); flag = True
    for i, d in enumerate(conf):
        flag = flag and d["p1"] < 0.05 / (k - i); d["confirmed"] = bool(flag and d["same"])
    print(f"[B2 {fam}] interactions searched {len(S)} | BH5% {len(hits)} | testable {k} | confirmed {sum(d['confirmed'] for d in conf)}")
    return dict(searched=int(len(S)), bh=int(len(hits)), testable=k, rules=conf)


def by_tier(name, D, ma, mb):
    r = {}
    for t in ("under", "fav"):
        ests = [within(D[(D["pop"] == p) & (D.tier == t)], ma[(D["pop"] == p) & (D.tier == t)], mb[(D["pop"] == p) & (D.tier == t)]) for p in POPS]
        ests = [e for e in ests if np.isfinite(e[1]) and e[1] > 0]
        w = np.array([1 / e[1] ** 2 for e in ests]); m = float((w * np.array([e[0] for e in ests])).sum() / w.sum())
        r[t] = dict(est=m, se=float(1 / np.sqrt(w.sum())), per_pop=[(e[0], e[1]) for e in ests])
    r["diff_under_minus_fav"] = r["under"]["est"] - r["fav"]["est"]; r["diff_se"] = float(np.hypot(r["under"]["se"], r["fav"]["se"]))
    print(f"[B3] {name}: under {r['under']['est']:+.4f} ({r['under']['se']:.4f}) fav {r['fav']['est']:+.4f} ({r['fav']['se']:.4f})")
    return r


def main():
    T = tiers()
    A = add_tier(pd.read_parquet(os.path.join(PROC, "r15_attack.parquet")), T)
    P = add_tier(pd.read_parquet(os.path.join(PROC, "r15_press.parquet")), T)
    old = json.load(open(os.path.join(ROOT, "results", "r15_rules.json")))
    known = {(d["cell"], d["action"], d["ref"]) for d in old["all_attack"] + old["all_press"] if d.get("confirmed")}
    res = {}
    # B1
    b1 = {}
    for fam, D, desc in (("attack", A, describe_attack), ("press", P, describe_press)):
        conf, s = mine_family(D[D.tier == "under"], fam)
        new = [dict(d, situation=desc(d["cell"]), new=(d["cell"], d["action"], d["ref"]) not in known) for d in conf if d["confirmed"]]
        b1[fam] = dict(summary=s, confirmed=new)
        for d in new:
            print(f"  B1 {fam} {'NEW ' if d['new'] else 'known'} {d['situation']}: {d['action']} vs {d['ref']} pooled {d['pooled']:+.4f}")
    res["B1"] = b1
    # B2
    res["B2"] = {fam: interaction_mine(D, fam) for fam, D in (("attack", A), ("press", P))}
    for fam in res["B2"]:
        for d in res["B2"][fam]["rules"]:
            if d["confirmed"]:
                print("  B2 CONFIRMED", fam, d["cell"], d["action"], "vs", d["ref"], round(d["pooled"], 4))
    # B3 rule effects by tier
    b3 = {}
    s = A[(A.x >= 96) & ((A.y - 40).abs() <= 20) & A.action.isin(["shot", "lateral", "back_pass", "short_fwd_ground", "short_fwd_air"])].copy()
    b3["strike"] = by_tier("strike", s, s.action == "shot", s.action != "shot")
    r = A[A.action.isin(["carry_fwd", "dribble"]) & (A.x >= 80) & (A.nearest_opp > 7)].copy()
    b3["run"] = by_tier("run", r, r.action == "carry_fwd", r.action == "dribble")
    O = add_tier(pd.read_parquet(os.path.join(PROC, "r18a_passes.parquet")), T)
    b3["overload"] = by_tier("overload", O, O.target_ov > 0, O.target_ov < 0)
    pr = P[(P.x >= 96) & (P.our_goalside <= 5) & (P.already == 0)].copy()
    b3["press_exposed"] = by_tier("press_exposed", pr, pr.action == "press", pr.action == "hold")
    G = add_tier(pd.read_parquet(os.path.join(PROC, "r19_regain.parquet")), T)
    G["cell"] = G.zx.astype(str)
    b3["count"] = by_tier("count (≤3 back minus 8+ back)", G, G.opp_goalside <= 3, G.opp_goalside >= 8)
    L = add_tier(pd.read_parquet(os.path.join(PROC, "agent_secondball_lb.parquet")), T)
    L = L[L.vis == 1] if "vis" in L else L
    L["adv"] = (L.ours10 - L.theirs10).clip(-3, 3); L["cell"] = pd.cut(L.ex, [0, 60, 80, 100, 121], labels=False).astype(str)
    L["net"] = L.keep5.astype(float)
    b3["second_ball_keep5"] = by_tier("second ball keep5 (adv>0 minus adv<0)", L, L.adv > 0, L.adv < 0)
    res["B3"] = b3
    json.dump(res, open(os.path.join(ROOT, "results", "r22b_underdog.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

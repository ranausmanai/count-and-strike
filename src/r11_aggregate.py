"""Round 11 added analysis: does the mined rule set replicate in aggregate? (DEVIATIONS #4)"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(__file__))
from r11_miner import cell_stats
ROOT = os.path.join(os.path.dirname(__file__), ".."); PROC = os.path.join(ROOT, "data", "proc")
A = pd.read_parquet(os.path.join(PROC, "r11_actions.parquet"))
def rules(D):
    st = cell_stats(D)
    ref = st.sort_values("size", ascending=False).drop_duplicates("cell").set_index("cell")
    c = st.merge(ref[["action", "mean", "var", "size"]].rename(columns=lambda x: "ref_" + x), left_on="cell", right_index=True)
    c = c[(c.action != c.ref_action)].copy()
    c["diff"] = c["mean"] - c.ref_mean; c["se"] = np.sqrt(c["var"] / c["size"] + c.ref_var / c.ref_size)
    return c
disc = rules(A[A["pop"] == "discovery"]); disc = disc[(disc["size"] >= 30) & (disc.ref_size >= 30)]
disc["p"] = 2 * norm.sf(np.abs(disc["diff"] / disc.se))
out = {}
for popn in ("confirm_EPL", "confirm_unseen"):
    C = A[A["pop"] == popn]
    st = cell_stats(C).set_index(["cell", "action"])
    def conf_diff(df):
        v = []
        for r in df.itertuples():
            try:
                a = st.loc[(r.cell, r.action)]; b = st.loc[(r.cell, r.ref_action)]
                v.append((a["mean"] - b["mean"], a["size"], b["size"]))
            except KeyError:
                v.append((np.nan, 0, 0))
        return np.array(v, dtype=float)
    res = {}
    for label, sub in (("nominal_hits_p<.05", disc[disc.p < 0.05]), ("strong_hits_p<.001", disc[disc.p < 0.001]), ("all_rules", disc)):
        v = conf_diff(sub); ok = (v[:, 1] >= 10) & (v[:, 2] >= 10)
        d0, d1 = sub["diff"].values[ok], v[ok, 0]
        agree = np.sign(d0) == np.sign(d1)
        rng = np.random.default_rng(1); bs = [agree[rng.integers(0, len(agree), len(agree))].mean() for _ in range(2000)]
        res[label] = dict(n=int(ok.sum()), sign_agreement=float(agree.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                          corr=float(np.corrcoef(d0, d1)[0, 1]), shrinkage_slope=float(np.polyfit(d0, d1, 1)[0]))
    # combined policy: in each cell pick discovery-best action (among actions with n>=30 in discovery); value it on confirm data
    cells = disc.groupby("cell")
    gain = 0.0; n_dec = 0; changed = 0
    Cst = cell_stats(C)
    for cell, g in cells:
        best = g.loc[g["diff"].idxmax()]
        if best["diff"] <= 0 or best.p >= 0.05:
            continue  # keep default when discovery shows no clear improvement
        try:
            a = st.loc[(cell, best.action)]; b = st.loc[(cell, best.ref_action)]
        except KeyError:
            continue
        if a["size"] < 10:
            continue
        # if every ref-action use in this cell switched to the better action:
        gain += (a["mean"] - b["mean"]) * b["size"]; changed += b["size"]
    n_tm = C.groupby(["match_id", "team"]).ngroups
    res["combined_policy"] = dict(decisions_changed_per_team_match=float(changed / n_tm), net_xg_gain_per_team_match=float(gain / n_tm),
                                  net_xg_gain_per_season=float(gain / n_tm * 38))
    out[popn] = res
    print(popn, json.dumps(res, indent=1))
json.dump(out, open(os.path.join(ROOT, "results", "r11_aggregate.json"), "w"), indent=1)

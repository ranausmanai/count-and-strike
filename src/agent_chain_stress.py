"""Agent chain: stress battery (AIPW + 20-s pre-decision placebo, r22c) and exploratory power checks.
No pre-registered test confirmed, so every row here is EXPLORATORY (reported, not a success claim).
Appends results['stress'] and results['exploratory'] to results/agent_chain.json."""
import json, os, sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import ROOT
from r22c_stress_all import add_pre, shots, battery
from r22a_strike import verdict
from agent_chain_analyse import load, s_h1, s_h2, POPS


def fe_slope(d, y, x="tov"):
    d = d.dropna(subset=[y]).copy()
    d["y_dm"] = d[y] - d.groupby("cell")[y].transform("mean"); d["x_dm"] = d[x] - d.groupby("cell")[x].transform("mean")
    m = smf.ols("y_dm ~ x_dm - 1", d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.match_id)[0]})
    return float(m.params["x_dm"]), float(m.bse["x_dm"])


def main():
    out_f = os.path.join(ROOT, "results", "agent_chain.json"); res = json.load(open(out_f))
    L, G = load(); S = shots(); st = {}
    L["h"] = L.height.astype("category").cat.codes
    H1 = s_h1(L).copy(); H1["T"] = H1.O.astype(int); H1 = add_pre(H1, S)
    f1 = ["fx", "fy", "opp_goalside", "nearest_opp", "start_ov", "max_ov", "min_ov", "ex", "ey", "length", "h", "gk"]
    st["H1a_overload_net20F"] = battery("H1 net20F", H1, f1, y="net20F")
    st["H1b_overload_net45F"] = battery("H1 net45F", H1, f1, y="net45F")
    H2 = s_h2(L).copy()
    H2["T"] = (H2.ours10 >= 2).astype(int); H2 = add_pre(H2, S)
    f2 = ["theirs10", "ex", "ey", "x", "y", "length", "h", "gk", "pressure", "keeper"]
    st["H2c_ours10ge2_net45L"] = battery("H2 net45L", H2, f2, y="net45L")
    H2["neg_opp45L"] = -H2.opp45L
    st["H2a_ours10ge2_minus_opp45L"] = battery("H2 -opp45L", H2, f2, y="neg_opp45L")
    g = G[G.Tany.notna()].copy(); g["T"] = g.Tany.astype(int); g = add_pre(g, S)
    st["H3_any_long_vs_short_net45G"] = battery("GK long v short", g, ["score_diff", "minute", "press", "x", "y"], y="net45G")
    res["stress"] = st
    # exploratory: r18a-style dose-response with cell FE on the full H1 sample (pre-registered estimator trims to ~15%)
    ex = {}
    for y in ("net20F", "net45F"):
        r = {}
        for p in POPS:
            d = s_h1(L[L["pop"] == p]).copy(); d["tov"] = d.target_ov.clip(-4, 4); r[p] = fe_slope(d, y)
        ex[f"H1_dose_cellFE_{y}"] = verdict(r)
        r = {}
        for p in POPS:
            d = L[(L["pop"] == p) & (L.f_has_frame == 1) & (L.fx >= 40)].copy(); d["tov"] = d.target_ov.clip(-4, 4); r[p] = fe_slope(d, y)
        ex[f"H1_dose_cellFE_all_ov_{y}"] = verdict(r)
    ex["share_followups_into_overload"] = {p: float((s_h1(L[L["pop"] == p]).O == 1).mean()) for p in POPS}
    ex["share_followups_with_any_overload_lane"] = {p: float((L[(L["pop"] == p) & (L.f_has_frame == 1) & (L.fx >= 40)].max_ov >= 1).mean()) for p in POPS}
    res["exploratory"] = ex
    json.dump(res, open(out_f, "w"), indent=1, default=float)
    print(json.dumps(ex, indent=0, default=float))


if __name__ == "__main__":
    main()

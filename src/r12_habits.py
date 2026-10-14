"""Round 12: do mined decision rules work as team habits? (PREREGISTRATION_12.md)"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")
ACTIONS = ["carry_fwd", "short_fwd", "long_fwd", "cross", "shot", "dribble", "recycle"]
ZONES = {"all": (0, 6), "own_half": (0, 2), "middle": (2, 4), "final_third": (4, 6)}  # zx bands: 0:<40 1:40-60 2:60-80 3:80-96 4:96-108 5:108+


def demean2(y, x, a, b, w, it=30):
    y = y.copy(); x = x.copy()
    for _ in range(it):
        for g in (a, b):
            for v in (y, x):
                s = pd.Series(v * w).groupby(g).transform("sum").values / pd.Series(w).groupby(g).transform("sum").values
                v -= s
    return y, x


def habit_test(A, action, zone):
    lo, hi = ZONES[zone]
    d = A[(A.zx >= lo) & (A.zx < hi)].copy()
    d["tm"] = d.league + "|" + d.team
    order = d.drop_duplicates(["tm", "match_id"])[["tm", "match_id"]].sort_values(["tm", "match_id"])
    order["half"] = order.groupby("tm").cumcount() % 2
    d = d.merge(order, on=["tm", "match_id"])
    d["isA"] = (d.action == action).astype(float)
    g = d.groupby(["tm", "cell", "half"]).agg(n=("net", "size"), pA=("isA", "mean"), y=("net", "mean")).reset_index()
    h0 = g[g.half == 0].set_index(["tm", "cell"]); h1 = g[g.half == 1].set_index(["tm", "cell"])
    j = h0.join(h1, lsuffix="_0", rsuffix="_1", how="inner")
    j = j[(j.n_0 >= 10) & (j.n_1 >= 10)].reset_index()
    rows = pd.concat([pd.DataFrame(dict(tm=j.tm, cell=j.cell, p=j.pA_0, y=j.y_1, w=j.n_1)),
                      pd.DataFrame(dict(tm=j.tm, cell=j.cell, p=j.pA_1, y=j.y_0, w=j.n_0))], ignore_index=True)
    # z-score propensity within cell (between teams)
    sd = rows.groupby("cell").p.transform("std")
    rows["pz"] = (rows.p - rows.groupby("cell").p.transform("mean")) / sd.replace(0, np.nan)
    rows = rows.dropna(subset=["pz"])
    if len(rows) < 200 or rows.tm.nunique() < 10:
        return None
    yd, xd = demean2(rows.y.values.astype(float), rows.pz.values.astype(float), rows.tm.values, rows.cell.values, rows.w.values.astype(float))
    w = rows.w.values.astype(float)
    beta = (w * xd * yd).sum() / (w * xd * xd).sum()
    e = yd - beta * xd
    # cluster-robust SE by team
    sc = pd.Series(w * xd * e).groupby(rows.tm.values).sum().values
    se = np.sqrt((sc ** 2).sum()) / (w * xd * xd).sum()
    # between-team SD of raw propensity (for value calc) and decisions in zone per team-match
    return dict(beta=float(beta), se=float(se), n_rows=int(len(rows)), n_teams=int(rows.tm.nunique()),
                mean_sd_prop=float(sd.mean()), decisions_per_team_match=float(len(d) / d.groupby(["tm", "match_id"]).ngroups))


def main():
    A = pd.read_parquet(os.path.join(PROC, "r11_actions.parquet"))
    res = []
    for action in ACTIONS:
        for zone in ZONES:
            r = {}
            for popn in ("discovery", "confirm_EPL", "confirm_unseen"):
                r[popn] = habit_test(A[A["pop"] == popn], action, zone)
            res.append(dict(action=action, zone=zone, **{k: v for k, v in r.items()}))
    R = pd.DataFrame(res)
    R["d_beta"] = R.discovery.map(lambda v: v["beta"] if v else np.nan); R["d_se"] = R.discovery.map(lambda v: v["se"] if v else np.nan)
    R = R.dropna(subset=["d_beta"])
    R["d_p"] = 2 * norm.sf(np.abs(R.d_beta / R.d_se)); R["q"] = multipletests(R.d_p, method="fdr_bh")[1]
    for popn in ("confirm_EPL", "confirm_unseen"):
        R[popn[8:] + "_beta"] = R[popn].map(lambda v: v["beta"] if v else np.nan); R[popn[8:] + "_se"] = R[popn].map(lambda v: v["se"] if v else np.nan)
    pd.set_option("display.width", 250)
    print(R[["action", "zone", "d_beta", "d_se", "d_p", "q", "EPL_beta", "EPL_se", "unseen_beta", "unseen_se"]].round(5).to_string())
    S = R[R.q < 0.10].copy(); k = len(S)
    print("shortlist (BH 10%):", k)
    conf = []
    for _, r in S.iterrows():
        w1, w2 = 1 / r.EPL_se ** 2, 1 / r.unseen_se ** 2
        pooled = (w1 * r.EPL_beta + w2 * r.unseen_beta) / (w1 + w2); pse = 1 / np.sqrt(w1 + w2)
        sg = np.sign(r.d_beta); p1 = norm.sf(sg * pooled / pse)
        conf.append(dict(action=r.action, zone=r.zone, disc_beta=r.d_beta, pooled=pooled, pooled_se=pse, p_one=p1,
                         same_both=bool(np.sign(r.EPL_beta) == sg and np.sign(r.unseen_beta) == sg)))
    C = pd.DataFrame(conf)
    if k:
        C = C.sort_values("p_one").reset_index(drop=True); C["holm"] = [0.05 / (k - i) for i in range(k)]
        ok, flag = [], True
        for _, r in C.iterrows():
            flag = flag and (r.p_one < r.holm); ok.append(bool(flag and r.same_both))
        C["confirmed"] = ok
        print(C.round(5).to_string())
    json.dump(dict(all=R.drop(columns=["discovery", "confirm_EPL", "confirm_unseen"]).to_dict(orient="records"),
                   details=res, confirmation=C.to_dict(orient="records")), open(os.path.join(ROOT, "results", "r12_habits.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

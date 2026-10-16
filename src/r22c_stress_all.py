"""Round 22C (post-hoc, DEVIATIONS #7): AIPW + pre-decision placebo for every confirmed rule."""
import gzip, json, os, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import PROC, ROOT, RAW, tsec
from r22a_strike import aipw, verdict, POPS
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold


def shots_one(m):
    try:
        ev = json.load(gzip.open(os.path.join(RAW, f"{m['match_id']}_events.json.gz")))
    except Exception:
        return None
    return pd.DataFrame([(m["match_id"], e["period"], tsec(e), e["team"]["name"], e["shot"].get("statsbomb_xg", 0.0)) for e in ev
                         if e["period"] <= 2 and e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] != "Penalty"],
                        columns=["match_id", "period", "t", "team", "xg"])


def shots():
    f = os.path.join(PROC, "r22_shots.parquet")
    if not os.path.exists(f):
        ms = json.load(open(os.path.join(ROOT, "data", "raw", "sb360_matches.json")))
        with ProcessPoolExecutor(8) as ex:
            pd.concat([s for s in ex.map(shots_one, ms) if s is not None], ignore_index=True).to_parquet(f)
    return pd.read_parquet(f)


def add_pre(D, S):
    D = D.reset_index(drop=True); pre = np.zeros(len(D))
    Sg = {k: g.sort_values("t") for k, g in S.groupby(["match_id", "period"])}
    for (mid, per), idx in D.groupby(["match_id", "period"]).indices.items():
        s = Sg.get((mid, per))
        if s is None:
            continue
        st = s.t.values
        for team in np.unique(D.team.values[idx]):
            sel = idx[D.team.values[idx] == team]; mine = s.team.values == team
            cm = np.concatenate([[0], np.cumsum(np.where(mine, s.xg.values, 0))]); co = np.concatenate([[0], np.cumsum(np.where(~mine, s.xg.values, 0))])
            tt = D.t.values[sel]; lo = np.searchsorted(st, tt - 20, "left"); hi = np.searchsorted(st, tt - 0.01, "left")
            pre[sel] = (cm[hi] - cm[lo]) - (co[hi] - co[lo])
    D["pre_net"] = pre
    return D


def crossfit(D, feat, y):
    X = D[feat].astype(float).values; T = D["T"].values; Y = D[y].values
    e = np.zeros(len(D)); m1 = np.zeros(len(D)); m0 = np.zeros(len(D))
    for tr, te in GroupKFold(5).split(X, T, D.match_id):
        e[te] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15).fit(X[tr], T[tr]).predict_proba(X[te])[:, 1]
        for t, mm in ((1, m1), (0, m0)):
            s = tr[T[tr] == t]
            mm[te] = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15).fit(X[s], Y[s]).predict(X[te])
    return np.clip(e, 0.02, 0.98), m1, m0


def battery(name, D, feat, y="net"):
    eff, plc, cnt = {}, {}, {}
    for p in POPS:
        d = D[(D["pop"] == p) & D[y].notna()].reset_index(drop=True)
        eff[p] = aipw(d, *crossfit(d, feat, y), y=y); plc[p] = aipw(d, *crossfit(d, feat, "pre_net"), y="pre_net")
        cnt[p] = (int(d["T"].sum()), int((1 - d["T"]).sum()))
    v = verdict(eff); pl = verdict(plc); pl["fails"] = bool(pl["pooled"] > 0 and pl["pooled_p"] < .05)
    out = dict(effect=v, placebo=pl, n=cnt, survives=bool(v["passed"] and not pl["fails"]))
    print(f"{name:14s} AIPW disc {eff['discovery'][0]:+.4f}({eff['discovery'][1]:.4f}) pooled {v['pooled']:+.4f}({v['pooled_se']:.4f}) pass={v['passed']} | "
          f"placebo pooled {pl['pooled']:+.4f}({pl['pooled_se']:.4f}) fails={pl['fails']} | SURVIVES={out['survives']}", flush=True)
    return out


def main():
    S = shots(); res = {}
    A = pd.read_parquet(os.path.join(PROC, "r15_attack.parquet"))
    base = ["x", "y", "opp_goalside", "nearest_opp", "mates_ahead", "lane"]
    r = A[A.action.isin(["carry_fwd", "dribble"]) & (A.x >= 80)].copy(); r["T"] = (r.action == "carry_fwd").astype(int)
    res["run"] = battery("run", add_pre(r, S), base)
    O = pd.read_parquet(os.path.join(PROC, "r18a_passes.parquet")); O = O[O.target_ov != 0].copy(); O["T"] = (O.target_ov > 0).astype(int)
    res["overload"] = battery("overload", add_pre(O, S), ["x", "y", "opp_goalside", "nearest_opp", "start_ov", "max_ov", "min_ov"])
    G = pd.read_parquet(os.path.join(PROC, "r19_regain.parquet")); G = G[G.opp_goalside <= 3].copy(); G["T"] = G.go.astype(int)
    res["count_go_le3"] = battery("count go(≤3)", add_pre(G, S), ["x", "opp_goalside", "zy"])
    P = pd.read_parquet(os.path.join(PROC, "r15_press.parquet")); P = P[P.x >= 80].copy(); P["T"] = P.press.astype(int)
    res["press_exposed"] = battery("press", add_pre(P, S), ["x", "y", "support", "our_goalside", "already"])
    L = pd.read_parquet(os.path.join(PROC, "agent_secondball_lb.parquet")); L = L[(L.ours10 != L.theirs10)].copy()
    L["T"] = (L.ours10 > L.theirs10).astype(int); L["keep5"] = L.keep5.astype(float)
    for c in ("gk", "keeper", "pressure", "vis"):
        L[c] = L[c].astype(float)
    L["h"] = L.height.astype("category").cat.codes
    res["second_ball_keep5"] = battery("2nd ball keep5", add_pre(L, S), ["x", "y", "ex", "ey", "length", "h", "gk", "keeper", "pressure"], y="keep5")
    res["second_ball_net"] = battery("2nd ball net", add_pre(L, S), ["x", "y", "ex", "ey", "length", "h", "gk", "keeper", "pressure"])
    json.dump(res, open(os.path.join(ROOT, "results", "r22c_stress_all.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

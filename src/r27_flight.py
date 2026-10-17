"""Round 27 (PREREGISTRATION_27.md): fast vs slow long balls, retention."""
import gzip, json, os, sys
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import tsec, RAW, PROC, ROOT
from r20_bridge import within
from r22a_strike import aipw, verdict
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
POPS = ("discovery", "confirm_club", "confirm_women")


def durs(mid):
    try:
        ev = json.load(gzip.open(os.path.join(RAW, f"{mid}_events.json.gz")))
    except Exception:
        return []
    return [(mid, e["period"], round(tsec(e), 3), e.get("duration", np.nan)) for e in ev if e["type"]["name"] == "Pass"]


def crossfit(D, feat, y):
    X = D[feat].astype(float).values; T = D["T"].values; Y = D[y].values
    e = np.zeros(len(D)); m1 = np.zeros(len(D)); m0 = np.zeros(len(D))
    for tr, te in GroupKFold(5).split(X, T, D.match_id):
        e[te] = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15).fit(X[tr], T[tr]).predict_proba(X[te])[:, 1]
        for t, mm in ((1, m1), (0, m0)):
            s = tr[T[tr] == t]
            mm[te] = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15).fit(X[s], Y[s]).predict(X[te])
    return np.clip(e, 0.02, 0.98), m1, m0


def main():
    L = pd.read_parquet(os.path.join(PROC, "agent_chain_L.parquet"))
    with ProcessPoolExecutor(6) as ex:
        rows = [r for rs in ex.map(durs, L.match_id.unique()) for r in rs]
    Dd = pd.DataFrame(rows, columns=["match_id", "period", "t", "duration"])
    L["t"] = L.t.round(3)
    L = L.merge(Dd.drop_duplicates(["match_id", "period", "t"]), on=["match_id", "period", "t"], how="left")
    L = L[(L.duration > 0.2) & (L.duration <= 8) & L.keep5.notna()].copy()
    L["speed"] = L.length / L.duration
    L["tert"] = L.groupby("pop").speed.transform(lambda s: pd.qcut(s, 3, labels=False))
    L["bal"] = np.where(L.vis == 1, (L.ours10 - L.theirs10).clip(-1, 2), 9)
    L["lb"] = pd.cut(L.length, [34, 45, 60, 200], labels=False); L["zx"] = pd.cut(L.x, [-1, 40, 60, 121], labels=False)
    L["cell"] = L.lb.astype(str) + L.height.astype(str) + L.bal.astype(str) + L.zx.astype(str)
    L["net"] = L.keep5.astype(float)
    print(L.groupby("pop").size().to_dict(), "speed quantiles", L.speed.quantile([.33, .67]).round(1).to_dict())
    res = {}
    S = L[L.tert != 1]
    h1 = {p: within(S[S["pop"] == p], S[S["pop"] == p].tert == 2, S[S["pop"] == p].tert == 0)[:2] for p in POPS}
    res["H1_keep5_fast_vs_slow"] = verdict(h1)
    S = S.copy(); S["T"] = (S.tert == 2).astype(int); S["h"] = S.height.astype("category").cat.codes
    feat = ["length", "h", "x", "y", "ex", "ey", "vis", "ours10", "theirs10", "pressure", "gk", "minute", "score_diff"]
    h2 = {}
    for p in POPS:
        d = S[S["pop"] == p].reset_index(drop=True); d["keep5"] = d.keep5.astype(float)
        h2[p] = aipw(d, *crossfit(d, feat, "keep5"), y="keep5")
    res["H2_aipw"] = verdict(h2)
    r1 = L.groupby(["tert", "bal"]).keep5.agg(["mean", "size"]).round(3)
    res["R1_keep5_by_tercile_balance"] = {f"{int(a)}_{int(b)}": dict(mean=float(v["mean"]), n=int(v["size"])) for (a, b), v in r1.iterrows()}
    for y in ("net20L", "net45L"):
        S2 = L[L.tert != 1].copy(); S2["net"] = S2[y]
        res["R2_" + y] = {p: within(S2[S2["pop"] == p], S2[S2["pop"] == p].tert == 2, S2[S2["pop"] == p].tert == 0)[:2] for p in POPS}
    res["speed_terciles_by_pop"] = {p: L[L["pop"] == p].groupby("tert").speed.median().round(1).tolist() for p in POPS}
    for k, v in res.items():
        print(k, json.dumps(v, default=float)[:500])
    json.dump(res, open(os.path.join(ROOT, "results", "r27_flight.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

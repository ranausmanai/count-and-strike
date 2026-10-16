"""Round 20: does the simulator's tactic discovery hold in real 360 data? (PREREGISTRATION_20.md)"""
import gzip, json, os, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import tsec, windows, RAW, DISC, WOMEN, RESTART
ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")
POPS = ("discovery", "confirm_club", "confirm_women")


def full_width(f):
    va = f.get("visible_area") or []
    ys = va[1::2]
    return len(ys) > 0 and min(ys) <= 5 and max(ys) >= 75


def one(m):
    mid = m["match_id"]
    try:
        ev = [e for e in json.load(gzip.open(os.path.join(RAW, f"{mid}_events.json.gz"))) if e["period"] <= 2]
        fr = {f["event_uuid"]: f for f in json.load(gzip.open(os.path.join(RAW, f"{mid}_three-sixty.json.gz")))}
    except Exception:
        return None
    S = pd.DataFrame([(e["period"], tsec(e), e["team"]["name"], e["shot"].get("statsbomb_xg", 0.0)) for e in ev
                      if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] != "Penalty"], columns=["period", "t", "team", "xg"])
    teams = sorted({e["team"]["name"] for e in ev})
    att, dfn = [], []
    for e in ev:
        f = fr.get(e["id"]); loc = e.get("location")
        if f is None or not loc or not full_width(f):
            continue
        x, y = loc[:2]
        typ = e["type"]["name"]
        mates = [q["location"] for q in f["freeze_frame"] if q["teammate"] and not q.get("keeper")]
        opps = [q["location"] for q in f["freeze_frame"] if not q["teammate"] and not q.get("keeper")]
        if typ == "Pass" and e["team"]["name"] == e["possession_team"]["name"] and e["pass"].get("type", {}).get("name") not in RESTART and 40 <= x <= 96:
            ys = [q[1] for q in mates]
            if len(ys) >= 6:
                d_opp = [np.hypot(q[0] - x, q[1] - y) for q in opps]
                att.append(dict(match_id=mid, comp=m["_comp"], period=e["period"], t=tsec(e), team=e["team"]["name"], x=x, y=y,
                                width=max(ys) - min(ys), n_vis=len(ys), opp_goalside=sum(q[0] > x for q in opps), nearest_opp=min(d_opp) if d_opp else 30.0))
        elif typ == "Ball Receipt*" and "outcome" not in e.get("ball_receipt", {}) and x >= 60:
            # opponent (receiver) attacking in OUR half: x >= 60 in receiver frame
            recv = e["team"]["name"]; us = [t for t in teams if t != recv]
            ys = [q[1] for q in opps]
            if us and len(ys) >= 6:
                sup = [np.hypot(q[0] - x, q[1] - y) for q in mates]
                dfn.append(dict(match_id=mid, comp=m["_comp"], period=e["period"], t=tsec(e), team=us[0], x=x, y=y,
                                compact=max(ys) - min(ys), n_vis=len(ys), support=min(sup) if sup else 30.0))
    return windows(att, S), windows(dfn, S)


def within(d, ma, mb):
    a = d[ma].groupby("cell").net.agg(["mean", "var", "size"]); b = d[mb].groupby("cell").net.agg(["mean", "var", "size"])
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner"); j = j[(j.size_a >= 10) & (j.size_b >= 10)]
    w = np.minimum(j.size_a, j.size_b)
    diff = ((j.mean_a - j.mean_b) * w).sum() / w.sum(); var = ((w / w.sum()) ** 2 * (j.var_a / j.size_a + j.var_b / j.size_b)).sum()
    return float(diff), float(np.sqrt(var)), int(len(j)), int(ma.sum()), int(mb.sum())


def verdict(r, sign=1):
    (d0, s0, *_), c1, c2 = r["discovery"], r["confirm_club"], r["confirm_women"]
    w = np.array([1 / c1[1] ** 2, 1 / c2[1] ** 2]); pooled = (w[0] * c1[0] + w[1] * c2[0]) / w.sum(); pse = 1 / np.sqrt(w.sum())
    return dict(per_pop={p: dict(diff=v[0], se=v[1], cells=v[2], n_a=v[3], n_b=v[4]) for p, v in r.items()},
                disc_p_one=float(norm.sf(sign * d0 / s0)), pooled=float(pooled), pooled_p_one=float(norm.sf(sign * pooled / pse)),
                confirmed=bool(sign * d0 > 0 and norm.sf(sign * d0 / s0) < 0.05 and sign * pooled > 0 and norm.sf(sign * pooled / pse) < 0.05))


def main():
    cache_a, cache_d = os.path.join(PROC, "r20_att.parquet"), os.path.join(PROC, "r20_def.parquet")
    if not os.path.exists(cache_a):
        ms = json.load(open(os.path.join(ROOT, "data", "raw", "sb360_matches.json")))
        A, D = [], []
        with ProcessPoolExecutor(8) as ex:
            for r in ex.map(one, ms, chunksize=4):
                if r is not None:
                    if len(r[0]): A.append(r[0])
                    if len(r[1]): D.append(r[1])
        for name, L in (("att", A), ("def", D)):
            X = pd.concat(L, ignore_index=True)
            X["pop"] = np.select([X.comp.str.startswith(DISC), X.comp.str.startswith(WOMEN)], ["discovery", "confirm_women"], "confirm_club")
            X["zx"] = pd.cut(X.x, [-1, 60, 80, 96, 108, 121], labels=False); X["zy"] = np.select([(X.y - 40).abs() <= 13, X.y < 40], [1, 0], 2)
            X.to_parquet(os.path.join(PROC, f"r20_{name}.parquet"))
    A = pd.read_parquet(cache_a); D = pd.read_parquet(cache_d)
    A["cell"] = A.zx.astype(int).astype(str) + A.zy.astype(str) + np.select([A.opp_goalside <= 3, A.opp_goalside <= 6], ["0", "1"], "2") + np.select([A.nearest_opp < 3, A.nearest_opp <= 7], ["0", "1"], "2")
    D["cell"] = D.zx.astype(int).astype(str) + D.zy.astype(str) + np.select([D.support < 8, D.support <= 15], ["0", "1"], "2")
    print("full-width frames: att", A.groupby("pop").size().to_dict(), "def", D.groupby("pop").size().to_dict())
    print("attacking width quantiles:", A.width.quantile([.1, .25, .5, .75, .9]).round(1).to_dict(), "| defensive range:", D.compact.quantile([.1, .25, .5, .75, .9]).round(1).to_dict())
    res = {}
    res["H20_1_width_32_48_vs_wide"] = verdict({p: within(A[A["pop"] == p], A[A["pop"] == p].width.between(32, 48), A[A["pop"] == p].width > 48) for p in POPS})
    res["H20_2_compact_le32_vs_gt40"] = verdict({p: within(D[D["pop"] == p], D[D["pop"] == p].compact <= 32, D[D["pop"] == p].compact > 40) for p in POPS})
    P = pd.read_parquet(os.path.join(PROC, "r19_press.parquet"))
    P["cell"] = P.zx.astype(int).astype(str) + P.zy.astype(str) + P.touchline.astype(str) + P.pressers.clip(upper=2).astype(str) + np.select([P.support < 8, P.support <= 15], ["0", "1"], "2")
    pr = {p: within(P[P["pop"] == p], P[P["pop"] == p].action == "press", P[P["pop"] == p].action == "hold") for p in POPS}
    res["H20_3_press_minus_hold"] = dict(per_pop={p: dict(diff=v[0], se=v[1]) for p, v in pr.items()}, all_le_zero=bool(all(v[0] <= 0 for v in pr.values())))
    for k, v in res.items():
        print(k, json.dumps(v, default=float))
    json.dump(res, open(os.path.join(ROOT, "results", "r20_bridge.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

"""Round 19: sweet-spot search (PREREGISTRATION_19.md).
  build -> data/proc/r19_{tempo,press,regain}.parquet ;  search -> results/r19_sweetspot.json
"""
import gzip, json, os, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import tsec, windows, RAW, DISC, WOMEN, RESTART
ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")
POPS = ("discovery", "confirm_club", "confirm_women")


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
    press_t = {}
    for e in ev:
        if e["type"]["name"] == "Pressure":
            press_t.setdefault((e["period"], e["team"]["name"]), []).append(tsec(e))
    press_t = {k: np.sort(v) for k, v in press_t.items()}
    pattern = {}
    for e in ev:
        pattern.setdefault(e["possession"], (e["play_pattern"]["name"], e["possession_team"]["name"], tsec(e), e["period"]))
    last_receipt = {}
    tempo, prs, reg = [], [], []
    seen_poss = set()
    for i, e in enumerate(ev):
        typ = e["type"]["name"]; t = tsec(e)
        pid = e.get("player", {}).get("id")
        if typ == "Ball Receipt*" and pid is not None:
            last_receipt[pid] = (e["period"], t)
        f = fr.get(e["id"])
        loc = e.get("location")
        # ---- regain first action (C family) ----
        k = e["possession"]
        if k not in seen_poss and typ in ("Pass", "Carry") and e["team"]["name"] == e["possession_team"]["name"]:
            seen_poss.add(k)
            pat, pteam, p0, pper = pattern[k]; prev = pattern.get(k - 1)
            if pat == "Regular Play" and prev is not None and prev[1] != pteam and prev[3] == pper and f is not None and loc and loc[0] < 80 and t - p0 <= 3:
                end = e["pass"]["end_location"] if typ == "Pass" else e["carry"]["end_location"]
                opp = [q["location"] for q in f["freeze_frame"] if not q["teammate"] and not q.get("keeper")]
                reg.append(dict(match_id=mid, comp=m["_comp"], period=e["period"], t=t, team=e["team"]["name"], x=loc[0],
                                opp_goalside=sum(q[0] > loc[0] for q in opp), go=int(end[0] - loc[0] >= 10)))
        if f is None or not loc:
            continue
        x, y = loc[:2]
        # ---- tempo (T family) ----
        if typ == "Pass" and e["team"]["name"] == e["possession_team"]["name"] and e["pass"].get("type", {}).get("name") not in RESTART:
            lr = last_receipt.get(pid)
            if lr and lr[0] == e["period"] and 0 <= t - lr[1] <= 8:
                opp = [q["location"] for q in f["freeze_frame"] if not q["teammate"] and not q.get("keeper")]
                mates = [q["location"] for q in f["freeze_frame"] if q["teammate"] and not q.get("actor") and not q.get("keeper")]
                d_opp = [np.hypot(q[0] - x, q[1] - y) for q in opp]
                tempo.append(dict(match_id=mid, comp=m["_comp"], period=e["period"], t=t, team=e["team"]["name"], x=x, y=y,
                                  action="first_time" if t - lr[1] < 1.0 else "controlled",
                                  opp_goalside=sum(q[0] > x for q in opp), nearest_opp=min(d_opp) if d_opp else 30.0,
                                  mates_ahead=sum(q[0] > x for q in mates)))
        # ---- pressing traps (P family) ----
        if typ == "Ball Receipt*" and "outcome" not in e.get("ball_receipt", {}) and pattern[e["possession"]][0] in ("Regular Play", "From Counter"):
            recv = e["team"]["name"]; us = [tm for tm in teams if tm != recv]
            if not us:
                continue
            us = us[0]
            ours = [q["location"] for q in f["freeze_frame"] if not q["teammate"] and not q.get("keeper")]
            theirs = [q["location"] for q in f["freeze_frame"] if q["teammate"] and not q.get("actor") and not q.get("keeper")]
            pt = press_t.get((e["period"], us), np.array([]))
            prs.append(dict(match_id=mid, comp=m["_comp"], period=e["period"], t=t, team=us, x=x, y=y,
                            touchline=int(min(y, 80 - y) <= 10), pressers=sum(np.hypot(q[0] - x, q[1] - y) <= 10 for q in ours),
                            support=min([np.hypot(q[0] - x, q[1] - y) for q in theirs]) if theirs else 30.0,
                            action="press" if (len(pt) and ((pt >= t) & (pt <= t + 2)).any()) else "hold"))
    return windows(tempo, S), windows(prs, S), windows(reg, S)


def build():
    ms = json.load(open(os.path.join(ROOT, "data", "raw", "sb360_matches.json")))
    T, P, R = [], [], []
    with ProcessPoolExecutor(8) as ex:
        for r in ex.map(one, ms, chunksize=4):
            if r is not None:
                T.append(r[0]); P.append(r[1]); R.append(r[2])
    for name, L in (("tempo", T), ("press", P), ("regain", R)):
        D = pd.concat([x for x in L if len(x)], ignore_index=True)
        D["pop"] = np.select([D.comp.str.startswith(DISC), D.comp.str.startswith(WOMEN)], ["discovery", "confirm_women"], "confirm_club")
        D["zx"] = pd.cut(D.x, [-1, 40, 60, 80, 96, 108, 121], labels=False)
        D["zy"] = np.select([(D.y - 40).abs() <= 13, D.y < 40], [1, 0], 2) if "y" in D else 1
        D.to_parquet(os.path.join(PROC, f"r19_{name}.parquet"))
        print(name, D.groupby("pop").size().to_dict())


def family_search(D, name):
    def table(d):
        st = d.groupby(["cell", "action"]).net.agg(["mean", "var", "size"]).unstack("action")
        return st
    acts = sorted(D.action.unique()); a, b = acts[0], acts[1]  # compare a vs b (alphabetical: controlled vs first_time; hold vs press)
    res = []
    T = {p: table(D[D["pop"] == p]) for p in POPS}
    td = T["discovery"]
    for cell in td.index:
        r = td.loc[cell]
        if r[("size", a)] < 30 or r[("size", b)] < 30 or np.isnan(r[("size", a)]) or np.isnan(r[("size", b)]):
            continue
        diff = r[("mean", b)] - r[("mean", a)]; se = np.sqrt(r[("var", a)] / r[("size", a)] + r[("var", b)] / r[("size", b)])
        row = dict(cell=cell, diff=float(diff), se=float(se), p=float(2 * norm.sf(abs(diff / se))), n_a=int(r[("size", a)]), n_b=int(r[("size", b)]))
        for p in POPS[1:]:
            tc = T[p]
            if cell in tc.index and tc.loc[cell][("size", a)] >= 10 and tc.loc[cell][("size", b)] >= 10:
                c = tc.loc[cell]
                row[p] = (float(c[("mean", b)] - c[("mean", a)]), float(np.sqrt(c[("var", a)] / c[("size", a)] + c[("var", b)] / c[("size", b)])))
        res.append(row)
    R = pd.DataFrame(res)
    R["q"] = multipletests(R.p, method="fdr_bh")[1]
    S = R[(R.q < 0.05)].copy()
    S = S[S.confirm_club.notna() & S.confirm_women.notna()] if len(S) else S
    conf = []
    k = len(S)
    if k:
        S["pooled"] = S.apply(lambda r: (r.confirm_club[0] / r.confirm_club[1] ** 2 + r.confirm_women[0] / r.confirm_women[1] ** 2) / (1 / r.confirm_club[1] ** 2 + 1 / r.confirm_women[1] ** 2), axis=1)
        S["pse"] = S.apply(lambda r: 1 / np.sqrt(1 / r.confirm_club[1] ** 2 + 1 / r.confirm_women[1] ** 2), axis=1)
        S["p1"] = norm.sf(np.sign(S["diff"]) * S.pooled / S.pse)
        S["same"] = S.apply(lambda r: np.sign(r.confirm_club[0]) == np.sign(r["diff"]) and np.sign(r.confirm_women[0]) == np.sign(r["diff"]), axis=1)
        S = S.sort_values("p1").reset_index(drop=True); flag = True
        for i, r in S.iterrows():
            flag = flag and r.p1 < 0.05 / (k - i); conf.append(bool(flag and r.same))
        S["confirmed"] = conf
    print(f"[{name}] {b} − {a}: searched {len(R)}, BH5% {int((R.q < 0.05).sum())}, testable {k}, confirmed {int(sum(conf))}; nominal agreement:",
          _agree(R))
    return R, S, (a, b)


def _agree(R):
    out = {}
    for p in POPS[1:]:
        m = R[(R.p < 0.05) & R[p].notna()]
        if len(m):
            out[p] = round(float((np.sign(m["diff"]) == np.sign(m[p].map(lambda v: v[0]))).mean()), 2)
    return out


def search():
    out = {}
    # T family
    T = pd.read_parquet(os.path.join(PROC, "r19_tempo.parquet"))
    T["cell"] = (T.zx.astype(int).astype(str) + T.zy.astype(str) + np.select([T.opp_goalside <= 3, T.opp_goalside <= 6], ["0", "1"], "2")
                 + np.select([T.nearest_opp < 3, T.nearest_opp <= 7], ["0", "1"], "2") + np.select([T.mates_ahead <= 1, T.mates_ahead <= 3], ["0", "1"], "2"))
    print("first-time share:", T.groupby("pop").action.apply(lambda s: round((s == "first_time").mean(), 3)).to_dict())
    R, S, ab = family_search(T, "tempo")
    out["T"] = dict(compare=f"{ab[1]} − {ab[0]}", searched=len(R), confirmed=S[S.get("confirmed", False) == True].drop(columns=["confirm_club", "confirm_women"]).to_dict(orient="records") if len(S) else [])
    # P family
    P = pd.read_parquet(os.path.join(PROC, "r19_press.parquet"))
    P["cell"] = (P.zx.astype(int).astype(str) + P.zy.astype(str) + P.touchline.astype(str) + P.pressers.clip(upper=2).astype(str)
                 + np.select([P.support < 8, P.support <= 15], ["0", "1"], "2"))
    R, S, ab = family_search(P, "press")
    out["P"] = dict(compare=f"{ab[1]} − {ab[0]}", searched=len(R), confirmed=S[S.get("confirmed", False) == True].drop(columns=["confirm_club", "confirm_women"]).to_dict(orient="records") if len(S) else [])
    # C family: count threshold sweep
    G = pd.read_parquet(os.path.join(PROC, "r19_regain.parquet"))
    G["zone"] = pd.cut(G.x, [-1, 40, 60, 80], labels=["own third", "own half", "middle third"])
    curve = []
    for z in G.zone.cat.categories:
        for k in range(2, 10):
            row = dict(zone=z, opp_back=k)
            for p in POPS:
                d = G[(G["pop"] == p) & (G.zone == z) & (G.opp_goalside == k)]
                a, b = d[d.go == 1].net, d[d.go == 0].net
                if len(a) >= 15 and len(b) >= 15:
                    df = a.mean() - b.mean(); se = np.sqrt(a.var() / len(a) + b.var() / len(b))
                    row[p] = dict(adv=float(df), se=float(se), p_one=float(norm.sf(df / se)), n_go=len(a), n_keep=len(b))
            row["confirmed_go"] = all(p in row and row[p]["p_one"] < 0.05 for p in POPS)
            curve.append(row)
    out["C"] = curve
    print("[count sweep] confirmed 'go' cells:", [(r["zone"], r["opp_back"]) for r in curve if r["confirmed_go"]])
    for r in curve:
        print("  ", r["zone"], r["opp_back"], {p: round(r[p]["adv"], 4) if p in r else None for p in POPS})
    # O family: overload sweep
    O = pd.read_parquet(os.path.join(PROC, "r18a_passes.parquet"))
    osw = []
    for zx in range(5):
        for t in (1, 2, 3):
            row = dict(zone=int(zx), threshold=t)
            for p in POPS:
                d = O[(O["pop"] == p) & (O.zx == zx)]
                a, b = d[d.target_ov >= t].net, d[d.target_ov <= -t].net
                if len(a) >= 20 and len(b) >= 20:
                    df = a.mean() - b.mean(); se = np.sqrt(a.var() / len(a) + b.var() / len(b))
                    row[p] = dict(adv=float(df), se=float(se), p_one=float(norm.sf(df / se)), n=len(a))
            row["confirmed"] = all(p in row and row[p]["p_one"] < 0.05 for p in POPS)
            osw.append(row)
    out["O"] = osw
    print("[overload sweep] confirmed:", [(r["zone"], r["threshold"]) for r in osw if r["confirmed"]])
    json.dump(out, open(os.path.join(ROOT, "results", "r19_sweetspot.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    {"build": build, "search": search}[sys.argv[1]]()

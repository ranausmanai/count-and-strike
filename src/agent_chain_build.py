"""Agent chain: build long-ball -> follow-up-pass table and goal-kick table (PREREGISTRATION_AGENT_chain.md).
-> data/proc/agent_chain_L.parquet, data/proc/agent_chain_G.parquet
"""
import gzip, json, os, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import RAW, PROC, ROOT, DISC, WOMEN, RESTART, tsec
from agent_secondball_build import pip

ALLOWED = {None, "Goal Kick"}


def lane(y):
    return 0 if y < 26.7 else (2 if y > 53.3 else 1)


def win(t, per, team, S, H):
    own = np.zeros(len(t)); opp = np.zeros(len(t))
    for i in range(len(t)):
        s = S[S[:, 0] == per[i]]
        m = (s[:, 1] >= t[i] - 0.01) & (s[:, 1] <= t[i] + H)
        mine = s[:, 2] == team[i]
        own[i] = s[m & mine, 3].sum()
        opp[i] = s[m & ~mine & (s[:, 1] > t[i] + 0.01), 3].sum()
    return own, opp


def ov_feats(e, f):
    x, y = e["location"][:2]; ex, ey = e["pass"]["end_location"][:2]
    mates = [q["location"] for q in f["freeze_frame"] if q["teammate"] and not q.get("actor") and not q.get("keeper")]
    opps = [q["location"] for q in f["freeze_frame"] if not q["teammate"] and not q.get("keeper")]
    ov = [0, 0, 0]
    for q in mates:
        if q[0] > x - 5:
            ov[lane(q[1])] += 1
    for q in opps:
        if q[0] > x - 5:
            ov[lane(q[1])] -= 1
    d = [np.hypot(q[0] - x, q[1] - y) for q in opps]
    return dict(fx=x, fy=y, fex=ex, fey=ey, target_ov=ov[lane(ey)], start_ov=ov[lane(y)], max_ov=max(ov), min_ov=min(ov),
                opp_goalside=sum(q[0] > x for q in opps), nearest_opp=min(d) if d else 30.0, f_fwd=int(ex > x + 5),
                f_complete=int("outcome" not in e["pass"]))


def one(m):
    mid = m["match_id"]
    try:
        ev = [e for e in json.load(gzip.open(os.path.join(RAW, f"{mid}_events.json.gz"))) if e["period"] <= 2]
        fr = {f["event_uuid"]: f for f in json.load(gzip.open(os.path.join(RAW, f"{mid}_three-sixty.json.gz")))}
    except Exception:
        return None
    teams = {e["team"]["name"] for e in ev}
    S = np.array([(e["period"], tsec(e), e["team"]["name"], e["shot"].get("statsbomb_xg", 0.0)) for e in ev
                  if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] != "Penalty"], dtype=object).reshape(-1, 4)
    T = np.array([tsec(e) for e in ev]); P = np.array([e["period"] for e in ev])
    # running score (before event)
    score = {t: 0 for t in teams}; sd = []
    for e in ev:
        tm = e["team"]["name"]; other = [t for t in teams if t != tm]
        sd.append(score[tm] - (score[other[0]] if other else 0))
        if e["type"]["name"] == "Shot" and e["shot"].get("outcome", {}).get("name") == "Goal":
            score[tm] += 1
        if e["type"]["name"] == "Own Goal For":
            score[tm] += 1
    L, G = [], []
    for k, e in enumerate(ev):
        if e["type"]["name"] != "Pass" or "location" not in e:
            continue
        p = e["pass"]; ptype = p.get("type", {}).get("name"); f = fr.get(e["id"])
        if f is None:
            continue
        ex, ey = p["end_location"][:2]
        ours10 = theirs10 = 0
        for q in f["freeze_frame"]:
            if q.get("keeper") or q.get("actor"):
                continue
            if np.hypot(q["location"][0] - ex, q["location"][1] - ey) <= 10:
                if q["teammate"]: ours10 += 1
                else: theirs10 += 1
        vis = int(pip(ex, ey, f["visible_area"]))
        t0 = tsec(e); per = e["period"]; team = e["team"]["name"]
        base = dict(match_id=mid, comp=m["_comp"], period=per, t=t0, team=team, x=e["location"][0], y=e["location"][1], ex=ex, ey=ey,
                    length=p.get("length", 0), height=p.get("height", {}).get("name", "Ground Pass"), gk=int(ptype == "Goal Kick"),
                    keeper=int(e.get("position", {}).get("name") == "Goalkeeper"), pressure=int(bool(e.get("under_pressure"))),
                    vis=vis, ours10=ours10, theirs10=theirs10, score_diff=sd[k], minute=e["minute"],
                    press=sum(1 for q in f["freeze_frame"] if not q["teammate"] and not q.get("keeper") and q["location"][0] < 40))
        if ptype == "Goal Kick":
            G.append(base)
        if p.get("length", 0) < 35 or ptype not in ALLOWED:
            continue
        r = dict(base); r["retained"] = 0; r["f_has_frame"] = 0
        for j in range(k + 1, len(ev)):
            g = ev[j]
            if g["period"] != per or T[j] > t0 + 15 or g["possession"] != e["possession"]:
                break
            if g["type"]["name"] == "Pass" and g["team"]["name"] == team and "location" in g:
                if g["pass"].get("type", {}).get("name") in RESTART:
                    break
                r["retained"] = 1; r["ft"] = T[j]; r["fdt"] = T[j] - t0
                ff = fr.get(g["id"])
                if ff is not None:
                    r["f_has_frame"] = 1; r.update(ov_feats(g, ff))
                break
        jj = np.where((P == per) & (T >= t0 + 5) & (np.arange(len(ev)) > k))[0]
        r["keep5"] = float(ev[jj[0]]["possession_team"]["name"] == team) if len(jj) else np.nan
        L.append(r)
    L = pd.DataFrame(L); G = pd.DataFrame(G)
    for D, specs in ((L, [("45L", 45), ("20L", 20)]), (G, [("45G", 45)])):
        if D.empty:
            continue
        for nm, H in specs:
            o, q = win(D.t.values, D.period.values, D.team.values, S, H)
            D["own" + nm] = o; D["opp" + nm] = q; D["net" + nm] = o - q
    if not L.empty and "ft" in L:
        m_ = L.ft.notna().values; ft = L.ft.fillna(0).values
        for nm, H in (("20F", 20), ("45F", 45)):
            o, q = win(ft, L.period.values, L.team.values, S, H)
            L["net" + nm] = np.where(m_, o - q, np.nan)
    return L, G


def main():
    ms = json.load(open(os.path.join(ROOT, "data", "raw", "sb360_matches.json")))
    Ls, Gs = [], []
    with ProcessPoolExecutor(2) as ex:
        for r in ex.map(one, ms, chunksize=4):
            if r is not None:
                Ls.append(r[0]); Gs.append(r[1])
    for nm, parts in (("L", Ls), ("G", Gs)):
        D = pd.concat([p for p in parts if len(p)], ignore_index=True)
        D["pop"] = np.select([D.comp.str.startswith(DISC), D.comp.str.startswith(WOMEN)], ["discovery", "confirm_women"], "confirm_club")
        D.to_parquet(os.path.join(PROC, f"agent_chain_{nm}.parquet"))
        print(nm, D.groupby("pop").size().to_dict())
    L = pd.read_parquet(os.path.join(PROC, "agent_chain_L.parquet"))
    print(L.groupby("pop")[["vis", "retained", "f_has_frame"]].mean())


if __name__ == "__main__":
    main()

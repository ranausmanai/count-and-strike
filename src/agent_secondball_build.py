"""Agent secondball: build long-ball table with landing-zone balance from 360 frames.
-> data/proc/agent_secondball_lb.parquet
"""
import gzip, json, os, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import RAW, PROC, ROOT, DISC, WOMEN, tsec

ALLOWED = {None, "Goal Kick"}


def pip(x, y, poly):
    xs = poly[0::2]; ys = poly[1::2]; n = len(xs); inside = False; j = n - 1
    for i in range(n):
        if (ys[i] > y) != (ys[j] > y):
            xi = (xs[j] - xs[i]) * (y - ys[i]) / (ys[j] - ys[i]) + xs[i]
            if x < xi:
                inside = not inside
        j = i
    return inside


def win(a, S, H):
    own = np.zeros(len(a)); opp = np.zeros(len(a))
    for i, r in enumerate(a.itertuples()):
        s = S[(S.period == r.period) & (S.t >= r.t - 0.01) & (S.t <= r.t + H)]
        own[i] = s.xg[s.team == r.team].sum()
        s2 = s[(s.team != r.team) & (s.t > r.t + 0.01)]
        opp[i] = s2.xg.sum()
    return own - opp


def one(m):
    mid = m["match_id"]
    try:
        ev = [e for e in json.load(gzip.open(os.path.join(RAW, f"{mid}_events.json.gz"))) if e["period"] <= 2]
        fr = {f["event_uuid"]: f for f in json.load(gzip.open(os.path.join(RAW, f"{mid}_three-sixty.json.gz")))}
    except Exception:
        return None
    S = pd.DataFrame([(e["period"], tsec(e), e["team"]["name"], e["shot"].get("statsbomb_xg", 0.0)) for e in ev
                      if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] != "Penalty"], columns=["period", "t", "team", "xg"])
    T = np.array([tsec(e) for e in ev]); P = np.array([e["period"] for e in ev]); PT = [e["possession_team"]["name"] for e in ev]
    rows = []; n_all = 0
    for k, e in enumerate(ev):
        if e["type"]["name"] != "Pass":
            continue
        p = e["pass"]
        if p.get("length", 0) < 35 or p.get("type", {}).get("name") not in ALLOWED:
            continue
        f = fr.get(e["id"])
        if f is None:
            continue
        n_all += 1
        ex, ey = p["end_location"][:2]
        vis = pip(ex, ey, f["visible_area"])
        ours = theirs = ours10 = theirs10 = 0
        for q in f["freeze_frame"]:
            if q.get("keeper") or q.get("actor"):
                continue
            d = np.hypot(q["location"][0] - ex, q["location"][1] - ey)
            if d <= 15:
                if q["teammate"]: ours += 1
                else: theirs += 1
            if d <= 10:
                if q["teammate"]: ours10 += 1
                else: theirs10 += 1
        t0 = tsec(e); per = e["period"]
        j = np.where((P == per) & (T >= t0 + 5) & (np.arange(len(ev)) > k))[0]
        keep5 = float(PT[j[0]] == e["team"]["name"]) if len(j) else np.nan
        rows.append(dict(match_id=mid, comp=m["_comp"], period=per, t=t0, team=e["team"]["name"], x=e["location"][0], y=e["location"][1],
                         ex=ex, ey=ey, length=p["length"], height=p.get("height", {}).get("name", "Ground Pass"),
                         gk=int(p.get("type", {}).get("name") == "Goal Kick"), keeper=int(e.get("position", {}).get("name") == "Goalkeeper"),
                         pressure=int(bool(e.get("under_pressure"))), complete=int("outcome" not in p), vis=int(vis),
                         ours=ours, theirs=theirs, ours10=ours10, theirs10=theirs10, keep5=keep5))
    a = pd.DataFrame(rows)
    if a.empty:
        return a
    a["net"] = win(a, S, 20); a["net45"] = win(a, S, 45)
    return a


def main():
    ms = json.load(open(os.path.join(ROOT, "data", "raw", "sb360_matches.json")))
    out = []
    with ProcessPoolExecutor(2) as ex:
        for r in ex.map(one, ms, chunksize=4):
            if r is not None and len(r):
                out.append(r)
    D = pd.concat(out, ignore_index=True)
    D["pop"] = np.select([D.comp.str.startswith(DISC), D.comp.str.startswith(WOMEN)], ["discovery", "confirm_women"], "confirm_club")
    D.to_parquet(os.path.join(PROC, "agent_secondball_lb.parquet"))
    print(D.groupby("pop").agg(n=("vis", "size"), vis=("vis", "sum")))


if __name__ == "__main__":
    main()

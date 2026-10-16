"""Agent 'cross': build wide attacking-third decision table from 360 (PREREGISTRATION_AGENT_cross.md).
-> data/proc/agent_cross_actions.parquet
"""
import gzip, json, os, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from matplotlib.path import Path
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import RAW, DISC, WOMEN, RESTART, tsec

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.environ.get("AGENT_CROSS_OUT", os.path.join(ROOT, "data", "proc", "agent_cross_actions.parquet"))
SETPIECE = {"From Corner", "From Free Kick", "From Throw In", "From Kick Off", "From Goal Kick"}


def inbox(p):
    return p[0] >= 102 and 18 <= p[1] <= 62


def horizon(a, S, H):
    own = np.zeros(len(a)); opp = np.zeros(len(a))
    for per, idx in a.groupby("period").indices.items():
        s = S[S.period == per].sort_values("t")
        if s.empty:
            continue
        st = s.t.values; sx = s.xg.values; steam = s.team.values
        for team in np.unique(a.team.values[idx]):
            sel = idx[a.team.values[idx] == team]
            mine = steam == team
            cm = np.concatenate([[0], np.cumsum(np.where(mine, sx, 0))]); co = np.concatenate([[0], np.cumsum(np.where(~mine, sx, 0))])
            tt = a.t.values[sel]
            lo = np.searchsorted(st, tt - 0.01, "left"); hi = np.searchsorted(st, tt + H, "right"); lo2 = np.searchsorted(st, tt + 0.01, "right")
            own[sel] = cm[hi] - cm[lo]; opp[sel] = co[hi] - co[lo2]
    return own, opp


def one(m):
    mid = m["match_id"]
    try:
        ev = [e for e in json.load(gzip.open(os.path.join(RAW, f"{mid}_events.json.gz"))) if e["period"] <= 2]
        fr = {f["event_uuid"]: f for f in json.load(gzip.open(os.path.join(RAW, f"{mid}_three-sixty.json.gz")))}
    except Exception:
        return None
    S = pd.DataFrame([(e["period"], tsec(e), e["team"]["name"], e["shot"].get("statsbomb_xg", 0.0)) for e in ev
                      if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] != "Penalty"], columns=["period", "t", "team", "xg"])
    rows = []
    for e in ev:
        typ = e["type"]["name"]
        if typ not in ("Pass", "Carry", "Dribble") or "location" not in e or e["team"]["name"] != e["possession_team"]["name"]:
            continue
        if e["play_pattern"]["name"] in SETPIECE:
            continue
        x, y = e["location"][:2]
        if x < 96 or abs(y - 40) <= 18:
            continue
        f = fr.get(e["id"])
        if f is None:
            continue
        if typ == "Carry":
            ex, ey = e["carry"]["end_location"][:2]
            if abs(ex - x) + abs(ey - y) < 1:
                continue
            ch = "carry"
        elif typ == "Dribble":
            ch = "carry"
        else:
            p = e["pass"]
            if p.get("type", {}).get("name") in RESTART:
                continue
            ex, ey = p["end_location"][:2]
            h = p.get("height", {}).get("name", "Ground Pass")
            if p.get("cut_back"):
                ch = "cutback"
            elif p.get("cross") or inbox((ex, ey)):
                ch = "cross_high" if h == "High Pass" else "cross_low"
            elif ex - x <= -5:
                ch = "recycle"
            elif abs(y - 40) - abs(ey - 40) >= 5:
                ch = "inside"
            else:
                ch = "other_pass"
        ff = f["freeze_frame"]
        ours = sum(1 for q in ff if q["teammate"] and not q.get("actor") and not q.get("keeper") and inbox(q["location"]))
        theirs = sum(1 for q in ff if not q["teammate"] and not q.get("keeper") and inbox(q["location"]))
        d_opp = [np.hypot(q["location"][0] - x, q["location"][1] - y) for q in ff if not q["teammate"]]
        va = np.array(f.get("visible_area", []), dtype=float).reshape(-1, 2)
        vis = bool(len(va) >= 3 and Path(va).contains_point((108, 40)))
        rows.append(dict(match_id=mid, comp=m["_comp"], period=e["period"], t=tsec(e), team=e["team"]["name"], x=x, y=y,
                         choice=ch, ours=ours, theirs=theirs, nearest=min(d_opp) if d_opp else 30.0, vis=vis))
    a = pd.DataFrame(rows)
    if a.empty:
        return a
    for H in (20, 45):
        own, opp = horizon(a, S, H)
        a[f"net{H}"] = own - opp
    return a


def main():
    ms = json.load(open(os.path.join(ROOT, "data", "raw", "sb360_matches.json")))
    with ProcessPoolExecutor(2) as ex:
        L = [r for r in ex.map(one, ms, chunksize=8) if r is not None and len(r)]
    D = pd.concat(L, ignore_index=True)
    D["pop"] = np.select([D.comp.str.startswith(DISC), D.comp.str.startswith(WOMEN)], ["discovery", "confirm_women"], "confirm_club")
    D.to_parquet(OUT)
    print(D.groupby(["pop", "choice"]).size().unstack(0))
    print(D.groupby("pop").vis.mean())


if __name__ == "__main__":
    main()

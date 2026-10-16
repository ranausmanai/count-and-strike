"""agent corners360: per-corner 360 setup counts + outcomes -> data/proc/agent_corners360.parquet"""
import gzip, json, os, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from matplotlib.path import Path
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import RAW, DISC, WOMEN, tsec, ROOT, PROC

gx, gy = np.meshgrid(np.linspace(102.25, 119.75, 36), np.linspace(18.25, 61.75, 88))
GBOX = np.c_[gx.ravel(), gy.ravel()]
sx, sy = np.meshgrid(np.linspace(114.1, 119.9, 12), np.linspace(30.1, 49.9, 40))
GSIX = np.c_[sx.ravel(), sy.ravel()]


def inbox(p):
    return p[0] >= 102 and 18 <= p[1] <= 62


def insix(p):
    return p[0] >= 114 and 30 <= p[1] <= 50


def one(m):
    mid = m["match_id"]
    try:
        ev = [e for e in json.load(gzip.open(os.path.join(RAW, f"{mid}_events.json.gz"))) if e["period"] <= 2]
    except Exception:
        return None
    try:
        fr = {f["event_uuid"]: f for f in json.load(gzip.open(os.path.join(RAW, f"{mid}_three-sixty.json.gz")))}
    except Exception:
        fr = {}
    shots = [(e["period"], tsec(e), e["team"]["name"], e["shot"].get("statsbomb_xg", 0.0)) for e in ev
             if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] != "Penalty"]
    out = []
    for e in ev:
        if e["type"]["name"] != "Pass" or e["pass"].get("type", {}).get("name") != "Corner":
            continue
        t = tsec(e); per = e["period"]; team = e["team"]["name"]
        r = dict(match_id=mid, comp=m["_comp"], period=per, t=t, team=team, y0=e["location"][1],
                 end_in_box=inbox(e["pass"]["end_location"]), has_frame=False)
        for H, Ho in ((20, 30), (45, 60)):
            r[f"own{H}"] = sum(x for p, ts, tm, x in shots if p == per and tm == team and t - 0.01 <= ts <= t + H)
            r[f"opp{Ho}"] = sum(x for p, ts, tm, x in shots if p == per and tm != team and t + 0.01 < ts <= t + Ho)
        f = fr.get(e["id"])
        if f is not None:
            va = Path(np.array(f["visible_area"]).reshape(-1, 2))
            ff = f["freeze_frame"]
            ours = [p["location"] for p in ff if p["teammate"] and not p.get("actor") and not p.get("keeper")]
            opps = [p["location"] for p in ff if not p["teammate"] and not p.get("keeper")]
            r.update(has_frame=True, cov_box=float(va.contains_points(GBOX).mean()), cov_six=float(va.contains_points(GSIX).mean()),
                     our_six=sum(map(insix, ours)), our_box=sum(map(inbox, ours)),
                     our_edge=sum((not inbox(p)) and p[0] >= 90 for p in ours),
                     opp_box=sum(map(inbox, opps)), opp_six=sum(map(insix, opps)), n_ours_vis=len(ours), n_opps_vis=len(opps))
        out.append(r)
    return out


if __name__ == "__main__":
    ms = json.load(open(os.path.join(ROOT, "data", "raw", "sb360_matches.json")))
    rows = []
    with ProcessPoolExecutor(2) as ex:
        for r in ex.map(one, ms, chunksize=8):
            if r:
                rows += r
    D = pd.DataFrame(rows)
    D["pop"] = np.select([D.comp.str.startswith(DISC), D.comp.str.startswith(WOMEN)], ["discovery", "confirm_women"], "confirm_club")
    D.to_parquet(os.path.join(PROC, "agent_corners360.parquet"))
    print(D.groupby("pop").agg(n=("t", "size"), frame=("has_frame", "mean"), matches=("match_id", "nunique")))

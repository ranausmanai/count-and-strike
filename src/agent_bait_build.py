"""Agent bait: build pass-level decision table with bait / switch features (NO outcomes here).
Output: data/proc/agent_bait_passes.parquet
"""
import gzip, json, os, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from matplotlib.path import Path
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import tsec, RAW, DISC, WOMEN, RESTART, PROC, ROOT

WIN = 8.0        # bait look-back window (s)
DEAD = 5.0       # central dead zone: |y-40| <= DEAD is "no side"
ONBALL = {"Pass", "Carry", "Dribble"}


def lane(y):
    return 0 if y < 26.7 else (2 if y > 53.3 else 1)


def side(y):
    return -1 if abs(y - 40) <= DEAD else (0 if y < 40 else 1)


def frame_feats(f, x, y, sd):
    """Counts from a freeze frame, relative to ball (x,y) and ball side sd (0 left / 1 right)."""
    ff = f["freeze_frame"]
    mates = [q["location"] for q in ff if q["teammate"] and not q.get("actor") and not q.get("keeper")]
    opps = [q["location"] for q in ff if not q["teammate"] and not q.get("keeper")]
    ov = [0, 0, 0]
    for q in mates:
        if q[0] > x - 5:
            ov[lane(q[1])] += 1
    for q in opps:
        if q[0] > x - 5:
            ov[lane(q[1])] -= 1
    ball_half = (lambda yy: yy < 40) if sd == 0 else (lambda yy: yy > 40)
    opp_bh = sum(ball_half(q[1]) for q in opps)
    mate_bh = sum(ball_half(q[1]) for q in mates)
    return ov, opp_bh, mate_bh, opps, mates


def far_visibility(f, x, sd):
    va = np.array(f["visible_area"], dtype=float).reshape(-1, 2)
    if len(va) < 3:
        return 0.0
    ylo, yhi = (53.3, 80.0) if sd == 0 else (0.0, 26.7)
    gx = np.linspace(max(x - 5, 0) + 1, 119, 8); gy = np.linspace(ylo + 1, yhi - 1, 5)
    P = np.array([(a, b) for a in gx for b in gy])
    return float(Path(va).contains_points(P).mean())


def one(m):
    mid = m["match_id"]
    try:
        ev = [e for e in json.load(gzip.open(os.path.join(RAW, f"{mid}_events.json.gz"))) if e["period"] <= 2]
        fr = {f["event_uuid"]: f for f in json.load(gzip.open(os.path.join(RAW, f"{mid}_three-sixty.json.gz")))}
    except Exception:
        return None
    ev.sort(key=lambda e: e["index"])
    poss_start = {}
    for e in ev:
        poss_start.setdefault((e["period"], e["possession"]), tsec(e))
    hist = {}  # (period, possession) -> list of (t, x, y, opp_bh_by_side dict or None, frame)
    rows = []
    for e in ev:
        typ = e["type"]["name"]
        if typ not in ONBALL or "location" not in e or e["team"]["name"] != e["possession_team"]["name"]:
            continue
        key = (e["period"], e["possession"]); t = tsec(e); x, y = e["location"][:2]
        f = fr.get(e["id"])
        H = hist.setdefault(key, [])
        if typ == "Pass":
            p = e["pass"]; sd = side(y)
            if p.get("type", {}).get("name") not in RESTART and x >= 40 and f is not None and sd >= 0 and "end_location" in p:
                ex, ey = p["end_location"][:2]
                ov, opp_bh, mate_bh, opps, mates = frame_feats(f, x, y, sd)
                far_l = 2 if sd == 0 else 0
                prior = [h for h in H if t - WIN <= h[0] < t]
                same = [h for h in prior if side(h[2]) == sd]
                bait_loose = len(prior) >= 2 and len(same) == len(prior)
                start_opp_bh = start_far_ov = np.nan
                fh = [h for h in prior if h[3] is not None]
                if fh:
                    h0 = fh[0]
                    if side(h0[2]) == sd:
                        ov0, ob0, _, _, _ = frame_feats(h0[3], h0[1], h0[2], sd)
                        start_opp_bh = ob0; start_far_ov = ov0[far_l]
                lt = lane(ey); ls = lane(y); lat = abs(ey - y); end_other = (ey - 40) * (y - 40) < 0
                d_opp = [np.hypot(q[0] - x, q[1] - y) for q in opps]
                rows.append(dict(match_id=mid, comp=m["_comp"], period=e["period"], t=t, idx=e["index"], team=e["team"]["name"],
                                 possession=e["possession"], minute=e["minute"], x=x, y=y, ady=abs(y - 40), sd=sd, ex=ex, ey=ey,
                                 far_ov=ov[far_l], ball_ov=ov[ls], mid_ov=ov[1], opp_bh=opp_bh, mate_bh=mate_bh,
                                 n_opp_vis=len(opps), n_mate_vis=len(mates), far_vis=far_visibility(f, x, sd),
                                 n_prior=len(prior), n_same=len(same), bait_loose=bait_loose, start_opp_bh=start_opp_bh,
                                 start_far_ov=start_far_ov, prior_has_frame=bool(fh),
                                 switch=bool(abs(lt - ls) == 2 or (lat >= 30 and end_other)), stay=bool((ey - 40) * (y - 40) > 0),
                                 lat=lat, opp_goalside=sum(q[0] > x for q in opps), nearest_opp=min(d_opp) if d_opp else 30.0,
                                 complete="outcome" not in p, poss_age=t - poss_start[key]))
        H.append((t, x, y, f))
    if not rows:
        return None
    D = pd.DataFrame(rows)
    return D


def main():
    ms = json.load(open(os.path.join(ROOT, "data", "raw", "sb360_matches.json")))
    out = []
    with ProcessPoolExecutor(2) as ex:
        for r in ex.map(one, ms, chunksize=4):
            if r is not None:
                out.append(r)
    D = pd.concat(out, ignore_index=True)
    D["pop"] = np.select([D.comp.str.startswith(DISC), D.comp.str.startswith(WOMEN)], ["discovery", "confirm_women"], "confirm_club")
    D["zx"] = pd.cut(D.x, [39, 60, 80, 96, 108, 121], labels=False).astype(int)
    D["zy"] = np.select([(D.y - 40).abs() <= 13, D.y < 40], [1, 0], 2)
    D["cell"] = (D.zx.astype(str) + D.zy.astype(str) + np.select([D.opp_goalside <= 3, D.opp_goalside <= 6], ["0", "1"], "2")
                 + np.select([D.nearest_opp < 3, D.nearest_opp <= 7], ["0", "1"], "2"))
    D["draw"] = D.opp_bh - D.start_opp_bh
    D.to_parquet(os.path.join(PROC, "agent_bait_passes.parquet"))
    print(D.groupby("pop").size())


if __name__ == "__main__":
    main()

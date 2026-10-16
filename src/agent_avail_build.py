"""Agent avail: rebuild r18a overload passes with availability of the overload receiver (PREREGISTRATION_AGENT_avail.md)."""
import gzip, json, os, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import tsec, windows, RAW, DISC, WOMEN, RESTART
from r18a_overload import lane
from r20_bridge import full_width
ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")


def seg_dist(p, a, b):
    """distance from points p (n,2) to segment a->b."""
    ab = b - a; L2 = ab @ ab
    if L2 == 0:
        return np.hypot(*(p - a).T)
    t = np.clip(((p - a) @ ab) / L2, 0, 1)
    return np.hypot(*(p - (a + t[:, None] * ab)).T)


def inside(pt, poly, tol=1.0):
    n = len(poly)
    if n < 3:
        return False
    x, y = pt; c = False
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            c = not c
    if c:
        return True
    P = np.array(poly, float)
    for i in range(n):
        if seg_dist(np.array([pt], float), P[i], P[(i + 1) % n])[0] <= tol:
            return True
    return False


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
        if e["type"]["name"] != "Pass" or e["team"]["name"] != e["possession_team"]["name"]:
            continue
        p = e["pass"]
        if p.get("type", {}).get("name") in RESTART or "location" not in e:
            continue
        x, y = e["location"][:2]
        if x < 40:
            continue
        f = fr.get(e["id"])
        if f is None:
            continue
        ex, ey = p["end_location"][:2]
        ff = f["freeze_frame"]
        mates = [q["location"] for q in ff if q["teammate"] and not q.get("actor") and not q.get("keeper")]
        opps = [q["location"] for q in ff if not q["teammate"] and not q.get("keeper")]
        allopp = np.array([q["location"] for q in ff if not q["teammate"]], float).reshape(-1, 2)
        ov = [0, 0, 0]
        for q in mates:
            if q[0] > x - 5:
                ov[lane(q[1])] += 1
        for q in opps:
            if q[0] > x - 5:
                ov[lane(q[1])] -= 1
        if not (max(ov) >= 1 and min(ov) <= -1):
            continue
        va = f.get("visible_area") or []
        poly = list(zip(va[0::2], va[1::2]))
        ball = np.array([x, y], float)
        n_on_ov = n_on_cr = 0
        for q in mates:
            if q[0] <= x - 5:
                continue
            lv = ov[lane(q[1])]
            if lv == 0:
                continue
            qq = np.array(q[:2], float); d = np.hypot(*(qq - ball))
            if not (5 <= d <= 35):
                continue
            if len(allopp) and np.hypot(*(allopp - qq).T).min() <= 3:
                continue
            if len(allopp) and seg_dist(allopp, ball, qq).min() <= 2:
                continue
            if not inside(qq, poly):
                continue
            if lv >= 1:
                n_on_ov += 1
            else:
                n_on_cr += 1
        d_opp = [np.hypot(q[0] - x, q[1] - y) for q in opps]
        lt, ls = lane(ey), lane(y)
        rows.append(dict(match_id=mid, comp=m["_comp"], period=e["period"], t=tsec(e), team=e["team"]["name"], x=x, y=y,
                         target_ov=ov[lt], start_ov=ov[ls], max_ov=max(ov), min_ov=min(ov), lane_change=abs(lt - ls),
                         opp_goalside=sum(q[0] > x for q in opps), nearest_opp=min(d_opp) if d_opp else 30.0,
                         complete="outcome" not in p, n_on_ov=n_on_ov, n_on_cr=n_on_cr, full_width=full_width(f),
                         n_mates_vis=len(mates), n_opps_vis=len(opps)))
    A = windows(rows, S)
    if A.empty:
        return A
    # 45 s horizon (same definition as windows(), 45 s)
    own = np.zeros(len(A)); opp = np.zeros(len(A))
    for i, r in A.iterrows():
        s = S[(S.period == r.period) & (S.t >= r.t - 0.01) & (S.t <= r.t + 45)]
        own[i] = s[s.team == r.team].xg.sum(); opp[i] = s[(s.team != r.team) & (s.t > r.t + 0.01)].xg.sum()
    A["net45"] = own - opp
    return A


def main():
    ms = json.load(open(os.path.join(ROOT, "data", "raw", "sb360_matches.json")))
    out, tm = [], []
    with ProcessPoolExecutor(2) as ex:
        for m, r in zip(ms, ex.map(one, ms, chunksize=4)):
            if r is not None and len(r):
                out.append(r)
            if r is not None:
                tm.append(dict(match_id=m["match_id"], comp=m["_comp"]))
    D = pd.concat(out, ignore_index=True)
    D["pop"] = np.select([D.comp.str.startswith(DISC), D.comp.str.startswith(WOMEN)], ["discovery", "confirm_women"], "confirm_club")
    D["zx"] = pd.cut(D.x, [39, 60, 80, 96, 108, 121], labels=False)
    D["zy"] = np.select([(D.y - 40).abs() <= 13, D.y < 40], [1, 0], 2)
    D["cell"] = (D.zx.astype(int).astype(str) + D.zy.astype(str) + np.select([D.opp_goalside <= 3, D.opp_goalside <= 6], ["0", "1"], "2")
                 + np.select([D.nearest_opp < 3, D.nearest_opp <= 7], ["0", "1"], "2"))
    D.to_parquet(os.path.join(PROC, "agent_avail_passes.parquet"))
    M = pd.DataFrame(tm); M["pop"] = np.select([M.comp.str.startswith(DISC), M.comp.str.startswith(WOMEN)], ["discovery", "confirm_women"], "confirm_club")
    M.to_parquet(os.path.join(PROC, "agent_avail_matches.parquet"))
    print(len(D), D.groupby("pop").size().to_dict(), "matches with frames", len(M))


if __name__ == "__main__":
    main()

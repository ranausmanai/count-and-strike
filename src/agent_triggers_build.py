"""Agent 'triggers': build per-receipt table of opponent open-play receipts with pressing triggers.
-> data/proc/agent_triggers_receipts.parquet  (written via path with tag)"""
import gzip, json, os, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import tsec, RAW, DISC, WOMEN, RESTART, ROOT, cellify

OUT = os.path.join(ROOT, "data", "proc", "agent_triggers_receipts.parquet")
CM = ("Center Midfield", "Defensive Midfield")


def win(a, S, H):
    own = np.zeros(len(a)); opp = np.zeros(len(a))
    for per, idx in a.groupby("period").indices.items():
        s = S[S.period == per].sort_values("t")
        if s.empty:
            continue
        st = s.t.values; sx = s.xg.values; steam = s.team.values
        for team in np.unique(a.team.values[idx]):
            sel = idx[a.team.values[idx] == team]; mine = steam == team
            cm = np.concatenate([[0], np.cumsum(np.where(mine, sx, 0))]); co = np.concatenate([[0], np.cumsum(np.where(~mine, sx, 0))])
            tt = a.t.values[sel]
            lo = np.searchsorted(st, tt - 0.01, "left"); hi = np.searchsorted(st, tt + H, "right"); lo2 = np.searchsorted(st, tt + 0.01, "right")
            own[sel] = cm[hi] - cm[lo]; opp[sel] = co[hi] - co[lo2]
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
    press_t = {}
    for e in ev:
        if e["type"]["name"] == "Pressure":
            press_t.setdefault((e["period"], e["team"]["name"]), []).append(tsec(e))
    press_t = {k: np.sort(v) for k, v in press_t.items()}
    byid = {e["id"]: e for e in ev}
    teams = sorted({e["team"]["name"] for e in ev})
    pattern = {}
    for e in ev:
        pattern.setdefault(e["possession"], e["play_pattern"]["name"])
    rows = []
    last_pass = {}
    for i, e in enumerate(ev):
        typ = e["type"]["name"]
        if typ == "Pass":
            last_pass[e["team"]["name"]] = e
        if typ != "Ball Receipt*" or pattern.get(e["possession"]) not in ("Regular Play", "From Counter"):
            continue
        f = fr.get(e["id"])
        if f is None or "location" not in e:
            continue
        rt = e["team"]["name"]; us = [t for t in teams if t != rt]
        if not us:
            continue
        us = us[0]
        p = None
        for rid in e.get("related_events", []):
            r = byid.get(rid)
            if r is not None and r["type"]["name"] == "Pass":
                p = r; break
        if p is None:
            p = last_pass.get(rt)
            if p is None or p["period"] != e["period"] or tsec(e) - tsec(p) > 6:
                continue
        pp = p["pass"]
        if pp.get("type", {}).get("name") in RESTART:
            continue
        x, y = e["location"][:2]; t0 = tsec(e)
        sx, sy = p["location"][:2]; ex, ey = pp["end_location"][:2]
        ln = pp.get("length", np.hypot(ex - sx, ey - sy)); h = pp.get("height", {}).get("name", "Ground Pass")
        ours = [q["location"] for q in f["freeze_frame"] if not q["teammate"] and not q.get("keeper")]
        mates = [q["location"] for q in f["freeze_frame"] if q["teammate"] and not q.get("actor") and not q.get("keeper")]
        d_us = np.array([np.hypot(o[0] - x, o[1] - y) for o in ours]) if ours else np.array([99.0])
        d_m = [np.hypot(o[0] - x, o[1] - y) for o in mates]
        # poor touch: incomplete receipt, or receiver's next own event within 3 s is Miscontrol
        mis = False
        pid = e.get("player", {}).get("id")
        for nx in ev[i + 1:i + 15]:
            if tsec(nx) - t0 > 3 or nx["period"] != e["period"]:
                break
            if nx.get("player", {}).get("id") == pid and nx["type"]["name"] not in ("Ball Receipt*",):
                mis = nx["type"]["name"] == "Miscontrol"; break
        inc = "outcome" in e.get("ball_receipt", {})
        pos = e.get("position", {}).get("name", "")
        pt = press_t.get((e["period"], us), np.array([]))
        rows.append(dict(match_id=mid, comp=m["_comp"], period=e["period"], t=t0, x=x, y=y, team=us,
                         press=bool(len(pt) and ((pt >= t0) & (pt <= t0 + 2)).any()),
                         support=min(d_m) if d_m else 30.0, n_us10=int((d_us <= 10).sum()), n_us8=int((d_us <= 8).sum()),
                         our_goalside=sum(o[0] > x for o in ours), already=int(d_us.min() < 3), incomplete=inc,
                         T_backward=ex < sx - 3, T_touchline=(y <= 8) or (y >= 72), T_poor_touch=inc or mis,
                         T_long_air=(ln > 30) and (h == "High Pass"),
                         T_cm_crowded=any(c in pos for c in CM) and int((d_us <= 8).sum()) >= 2))
    if not rows:
        return None
    a = pd.DataFrame(rows)
    a["net"] = win(a, S, 20); a["net45"] = win(a, S, 45)
    return a


def main():
    ms = json.load(open(os.path.join(ROOT, "data", "raw", "sb360_matches.json")))
    L = []
    with ProcessPoolExecutor(2) as ex:
        for r in ex.map(one, ms, chunksize=4):
            if r is not None:
                L.append(r)
    D = cellify(pd.concat(L, ignore_index=True))
    D["pop"] = np.select([D.comp.str.startswith(DISC), D.comp.str.startswith(WOMEN)], ["discovery", "confirm_women"], "confirm_club")
    D["cell"] = (D.zx.astype(int).astype(str) + D.zy.astype(str) + np.select([D.support < 8, D.support <= 15], ["0", "1"], "2")
                 + np.clip(D.n_us10, 0, 2).astype(str))
    D.to_parquet(OUT)
    T = [c for c in D.columns if c.startswith("T_")]
    print(D.groupby("pop").size()); print(D.groupby("pop")[T + ["press"]].mean().round(3))


if __name__ == "__main__":
    main()

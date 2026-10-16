"""Agent 'combos': build decision points P and continuation types (PREREGISTRATION_AGENT_combos.md).
-> data/proc/agent_combos_P.parquet
"""
import gzip, json, os, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import tsec, RAW, PROC, ROOT, RESTART, DISC, WOMEN

IGNORE = {"Pressure", "Ball Receipt*", "Dribbled Past", "Duel"}
HOR = (20, 45)


def lane(y):
    return 0 if y < 80 / 3 else (1 if y <= 160 / 3 else 2)


def phase(ev, j, pid, team):
    """Walk the on-ball phase of player pid starting after list position j.
    Returns carry_gain, dribble, release (event or None), position of release."""
    gain, drib = 0.0, False
    n = len(ev)
    k = j + 1
    while k < n:
        e = ev[k]; typ = e["type"]["name"]
        if e["period"] != ev[j]["period"]:
            break
        if typ in IGNORE:
            k += 1; continue
        if e.get("player", {}).get("id") == pid and e["team"]["name"] == team:
            if typ == "Carry":
                gain += e["carry"]["end_location"][0] - e["location"][0]; k += 1; continue
            if typ == "Dribble":
                drib = True; k += 1; continue
            if typ in ("Pass", "Shot"):
                return gain, drib, e, k
        break
    return gain, drib, None, k


def find_pos(ev, k, pass_e):
    for kk in range(k + 1, min(k + 10, len(ev))):
        e = ev[kk]
        if e["type"]["name"] == "Ball Receipt*" and pass_e["id"] in e.get("related_events", []):
            return kk
    return None


def windows_h(a, S, H):
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
    return own, opp


def one(m):
    mid = m["match_id"]
    try:
        ev = [e for e in json.load(gzip.open(os.path.join(RAW, f"{mid}_events.json.gz"))) if e["period"] <= 2]
        fr = {f["event_uuid"]: f for f in json.load(gzip.open(os.path.join(RAW, f"{mid}_three-sixty.json.gz")))}
    except Exception:
        return None
    ev.sort(key=lambda e: e["index"])
    byid = {e["id"]: e for e in ev}
    S = pd.DataFrame([(e["period"], tsec(e), e["team"]["name"], e["shot"].get("statsbomb_xg", 0.0)) for e in ev
                      if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] != "Penalty"], columns=["period", "t", "team", "xg"])
    rows = []
    for j, e in enumerate(ev):
        if e["type"]["name"] != "Ball Receipt*" or "outcome" in e.get("ball_receipt", {}):
            continue
        team = e["team"]["name"]
        if team != e["possession_team"]["name"] or "location" not in e:
            continue
        x, y = e["location"][:2]
        if x < 40:
            continue
        f = fr.get(e["id"])
        if f is None:
            continue
        pe = next((byid[r] for r in e.get("related_events", []) if r in byid and byid[r]["type"]["name"] == "Pass"), None)
        if pe is None or pe["pass"].get("type", {}).get("name") in RESTART:
            continue
        A = pe["player"]["id"]; ax = pe["location"][0]; B = e["player"]["id"]; t0 = tsec(e)
        opp = [p["location"] for p in f["freeze_frame"] if not p["teammate"] and not p.get("keeper")]
        mates = [p["location"] for p in f["freeze_frame"] if p["teammate"] and not p.get("actor") and not p.get("keeper")]
        d_opp = [np.hypot(o[0] - x, o[1] - y) for o in opp]
        gain, drib, rel, k = phase(ev, j, B, team)
        a1 = "none"; a1_complete = False; cat = "other"
        if rel is not None:
            a1 = rel["type"]["name"]
            a1_complete = a1 == "Pass" and "outcome" not in rel["pass"]
        if rel is not None and a1 == "Shot":
            cat = "shot"
        elif drib or gain >= 10:
            cat = "solo_carry"
        elif a1_complete:
            p = rel["pass"]; sx, sy = rel["location"][:2]; ex, ey = p["end_location"][:2]
            rcpt = p.get("recipient", {}).get("id"); dt1 = tsec(rel) - t0
            kr = find_pos(ev, k, rel)
            tr = tsec(ev[kr]) if kr is not None else tsec(rel) + rel.get("duration", 0.0)
            # receiver phase (action 2), only if receipt found
            g2, d2, rel2 = 0.0, False, None
            if kr is not None:
                g2, d2, rel2, _ = phase(ev, kr, rcpt, team)
            r2_dt = (tsec(rel2) - tr) if rel2 is not None else 99
            r2_pass_dx = (rel2["pass"]["end_location"][0] - rel2["location"][0]) if (rel2 is not None and rel2["type"]["name"] == "Pass") else -99
            if rcpt == A and dt1 <= 4 and ex >= ax + 5:
                cat = "one_two"
            elif rcpt != A and dt1 <= 4 and ex >= ax + 10:
                cat = "third_man"
            elif ex < sx - 2 and p.get("length", 99) <= 15 and r2_pass_dx >= 15 and r2_dt <= 4:
                cat = "layoff_fwd"
            elif ((lane(ey) != lane(sy) and abs(ey - sy) >= 15 and abs(ex - sx) < 10) or p.get("switch")) and \
                    (d2 or g2 >= 10 or (r2_pass_dx >= 10 and r2_dt <= 4)):
                cat = "switch_attack"
        rows.append(dict(match_id=mid, comp=m["_comp"], period=e["period"], t=t0, team=team, x=x, y=y,
                         opp_goalside=sum(o[0] > x for o in opp), nearest_opp=min(d_opp) if d_opp else 30.0,
                         mates_ahead=sum(mm[0] > x for mm in mates), cat=cat, a1=a1, a1_complete=a1_complete,
                         pattern=e["play_pattern"]["name"]))
    a = pd.DataFrame(rows)
    if a.empty:
        return None
    for H in HOR:
        own, opp_ = windows_h(a, S, H)
        a[f"own{H}"] = own; a[f"opp{H}"] = opp_; a[f"net{H}"] = own - opp_
    own3, _ = windows_h(a, S, 3)
    a["shot3"] = own3 > 0
    # own-team shot within 3 s: own3 counts xG>0; also catch xG==0 shots
    return a


def build():
    ms = json.load(open(os.path.join(ROOT, "data", "raw", "sb360_matches.json")))
    out = []
    with ProcessPoolExecutor(2) as ex:
        for r in ex.map(one, ms, chunksize=8):
            if r is not None:
                out.append(r)
    D = pd.concat(out, ignore_index=True)
    D["zx"] = pd.cut(D.x, [-1, 40, 60, 80, 96, 108, 121], labels=False)
    D["zy"] = np.select([(D.y - 40).abs() <= 13, D.y < 40], [1, 0], 2)
    D["pop"] = np.select([D.comp.str.startswith(DISC), D.comp.str.startswith(WOMEN)], ["discovery", "confirm_women"], "confirm_club")
    D["cell"] = (D.zx.astype(int).astype(str) + D.zy.astype(str) + np.select([D.opp_goalside <= 3, D.opp_goalside <= 6], ["0", "1"], "2")
                 + np.select([D.nearest_opp < 3, D.nearest_opp <= 7], ["0", "1"], "2") + np.select([D.mates_ahead <= 1, D.mates_ahead <= 3], ["0", "1"], "2"))
    D.to_parquet(os.path.join(PROC, "agent_combos_P.parquet"))
    print(len(D), D.match_id.nunique())
    print(D.groupby(["pop", "cat"]).size().unstack(0))


if __name__ == "__main__":
    build()

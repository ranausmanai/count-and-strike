"""Round 22 build (PREREGISTRATION_22.md): rich near-box decision features + per-match team xG for ratings."""
import gzip, json, os, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from r4_structure import in_tri
from r15_360 import tsec, RAW, PROC, ROOT, RESTART


def pre_window(rows, S):
    a = pd.DataFrame(rows)
    own = np.zeros(len(a)); opp = np.zeros(len(a))
    for per, idx in a.groupby("period").indices.items():
        s = S[S.period == per].sort_values("t")
        if s.empty:
            continue
        st = s.t.values
        for team in np.unique(a.team.values[idx]):
            sel = idx[a.team.values[idx] == team]; mine = s.team.values == team
            cm = np.concatenate([[0], np.cumsum(np.where(mine, s.xg.values, 0))]); co = np.concatenate([[0], np.cumsum(np.where(~mine, s.xg.values, 0))])
            tt = a.t.values[sel]
            lo = np.searchsorted(st, tt - 20, "left"); hi = np.searchsorted(st, tt - 0.01, "left")
            own[sel] = cm[hi] - cm[lo]; opp[sel] = co[hi] - co[lo]
    a["pre_net"] = own - opp
    return a


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
    xgt = S.groupby("team").xg.sum().to_dict()
    match = [dict(match_id=mid, comp=m["_comp"], team=t, opp=[u for u in teams if u != t][0], xg_for=xgt.get(t, 0.0),
                  xg_against=sum(v for k, v in xgt.items() if k != t)) for t in teams]
    score = {t: 0 for t in teams}; poss_start = {}; last_h = {}; rows = []
    for e in ev:
        typ = e["type"]["name"]; tm = e["team"]["name"]
        poss_start.setdefault(e["possession"], tsec(e))
        if typ == "Pass":
            last_h[e["possession"]] = e["pass"].get("height", {}).get("name", "Ground Pass")
        f = fr.get(e["id"]); loc = e.get("location")
        if f is not None and loc and typ in ("Pass", "Carry", "Dribble", "Shot") and tm == e["possession_team"]["name"]:
            x, y = loc[:2]
            ok = x >= 96 and abs(y - 40) <= 20
            if typ == "Pass" and e["pass"].get("type", {}).get("name") in RESTART:
                ok = False
            if typ == "Shot" and e["shot"]["type"]["name"] != "Open Play":
                ok = False
            if ok:
                opp = [p["location"] for p in f["freeze_frame"] if not p["teammate"] and not p.get("keeper")]
                gk = [p["location"] for p in f["freeze_frame"] if not p["teammate"] and p.get("keeper")]
                mates = [p["location"] for p in f["freeze_frame"] if p["teammate"] and not p.get("actor") and not p.get("keeper")]
                d = np.array([np.hypot(o[0] - x, o[1] - y) for o in opp]) if opp else np.array([30.0])
                a1 = np.arctan2(36 - y, 120 - x); a2 = np.arctan2(44 - y, 120 - x)
                r = dict(match_id=mid, idx=e["index"], period=e["period"], t=tsec(e), team=tm, x=x, y=y,
                         dist=float(np.hypot(120 - x, 40 - y)), angle=float(abs(a2 - a1)),
                         lane=sum(in_tri(o, (x, y), (120, 36), (120, 44)) for o in opp), opp2=int((d < 2).sum()), opp5=int((d < 5).sum()),
                         nearest=float(d.min()), mates_ahead=sum(q[0] > x for q in mates), under_pressure=int(bool(e.get("under_pressure"))),
                         air=int(last_h.get(e["possession"], "Ground Pass") != "Ground Pass"), poss_dur=tsec(e) - poss_start[e["possession"]],
                         minute=e["minute"], score_diff=score[tm] - sum(v for k, v in score.items() if k != tm), gk_vis=int(len(gk) > 0))
                if gk:
                    gx, gy = gk[0][:2]
                    # lateral offset of keeper from the ball -> goal-centre line
                    vx, vy = 120 - x, 40 - y; n = np.hypot(vx, vy) or 1.0
                    r.update(gk_line=float(np.hypot(120 - gx, 40 - gy)), gk_off=float(abs((gx - x) * vy - (gy - y) * vx) / n),
                             gk_ball=float(np.hypot(gx - x, gy - y)))
                rows.append(r)
        if typ == "Shot" and e["shot"]["outcome"]["name"] == "Goal":
            score[tm] = score.get(tm, 0) + 1
        elif typ == "Own Goal Against":
            o = [u for u in teams if u != tm]
            if o:
                score[o[0]] += 1
    R = pre_window(rows, S) if rows else pd.DataFrame()
    return R, pd.DataFrame(match)


def main():
    ms = json.load(open(os.path.join(ROOT, "data", "raw", "sb360_matches.json")))
    R, M = [], []
    with ProcessPoolExecutor(8) as ex:
        for r in ex.map(one, ms, chunksize=4):
            if r is not None:
                R.append(r[0]); M.append(r[1])
    R = pd.concat(R, ignore_index=True); M = pd.concat(M, ignore_index=True)
    R.to_parquet(os.path.join(PROC, "r22_box.parquet")); M.to_parquet(os.path.join(PROC, "r22_match_xg.parquet"))
    print(R.shape, M.shape); print(R.describe().T[["mean", "min", "max"]])


if __name__ == "__main__":
    main()

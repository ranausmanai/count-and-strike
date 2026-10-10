"""Turn StatsBomb event JSON into (a) a compact event table and (b) team-segment rows.

A segment is a stretch of play with constant game state: split at 15-minute marks
inside each half and at every goal or red card. For each (segment, team) we record
exposure (minutes), game state, xG for/against, and style descriptors:

  def_height  mean x (0..120, own goal = 0) of the team's defensive actions
  press_rate  pressures applied per opponent pass
  ppda        opponent passes in their own 60% / our def. actions in that zone
  long_share  share of the team's open-play passes >= 32 yards
  poss_share  team passes / all passes in the segment
  fwd_speed   forward progress (yards) per second of own possession (directness/tempo)
"""
import gzip, json, os, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw", "sb")
PROC = os.path.join(ROOT, "data", "proc")

DEF_TYPES = {"Pressure", "Ball Recovery", "Interception", "Block", "Clearance", "Foul Committed"}


def tsec(e):
    h, m, s = e["timestamp"].split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def compact_events(ev, mid):
    rows = []
    for e in ev:
        p = e["period"]
        if p > 4:
            continue
        typ = e["type"]["name"]
        loc = e.get("location") or [np.nan, np.nan]
        r = dict(match_id=mid, idx=e["index"], period=p, t=tsec(e) / 60.0, minute=e["minute"],
                 type=typ, team=e["team"]["name"], poss=e["possession"],
                 poss_team=e["possession_team"]["name"], pattern=e["play_pattern"]["name"],
                 x=loc[0], y=loc[1], end_x=np.nan, end_y=np.nan, length=np.nan, xg=np.nan,
                 outcome=None, sub=None, dur=e.get("duration") or 0.0,
                 under_pressure=bool(e.get("under_pressure", False)),
                 counter=bool(e.get(typ.lower(), {}).get("counterpress", False)) if isinstance(e.get(typ.lower()), dict) else False)
        if typ == "Pass":
            ps = e["pass"]
            r["end_x"], r["end_y"] = ps["end_location"][:2]
            r["length"] = ps.get("length", np.nan)
            r["outcome"] = ps.get("outcome", {}).get("name", "Complete")
            r["sub"] = ps.get("type", {}).get("name", "Open")
        elif typ == "Shot":
            sh = e["shot"]
            r["xg"] = sh.get("statsbomb_xg", np.nan)
            r["outcome"] = sh["outcome"]["name"]
            r["sub"] = sh["type"]["name"]
            r["end_x"], r["end_y"] = sh["end_location"][:2]
        elif typ == "Carry":
            r["end_x"], r["end_y"] = e["carry"]["end_location"][:2]
        elif typ == "Duel":
            r["sub"] = e["duel"].get("type", {}).get("name")
            r["outcome"] = e["duel"].get("outcome", {}).get("name")
        elif typ in ("Foul Committed", "Bad Behaviour"):
            key = "foul_committed" if typ == "Foul Committed" else "bad_behaviour"
            card = e.get(key, {}).get("card", {}).get("name")
            r["outcome"] = card
        elif typ == "Ball Recovery":
            r["outcome"] = "Fail" if e.get("ball_recovery", {}).get("recovery_failure") else "OK"
        elif typ == "Interception":
            r["outcome"] = e["interception"]["outcome"]["name"]
        elif typ == "Half End":
            pass
        rows.append(r)
    return pd.DataFrame(rows)


def process(m):
    mid = m["match_id"]
    ev = json.load(gzip.open(os.path.join(RAW, f"{mid}.json.gz")))
    df = compact_events(ev, mid)
    # event files sometimes use a different team name than the match list; resolve by id
    idname = {e["team"]["id"]: e["team"]["name"] for e in ev if e["type"]["name"] == "Starting XI"}
    home = idname.get(m["home_team"]["home_team_id"], m["home_team"]["home_team_name"])
    away = idname.get(m["away_team"]["away_team_id"], m["away_team"]["away_team_name"])
    teams = (home, away)

    # continuous clock per period: minutes since period start (+ offset 0/45/90/105)
    off = {1: 0.0, 2: 45.0, 3: 90.0, 4: 105.0}
    df["clock"] = df["t"] + df["period"].map(off)
    df["ptime"] = df["t"]  # minutes since the period started
    pend = df[df.type == "Half End"].groupby("period")["t"].max().to_dict()
    for p in df.period.unique():
        pend.setdefault(p, df.loc[df.period == p, "t"].max())

    # goals and red cards (team that benefits / team penalised)
    events_change = []  # (period, t, kind, team)
    for r in df.itertuples():
        if r.type == "Shot" and r.outcome == "Goal":
            events_change.append((r.period, r.t, "goal", r.team))
        elif r.type == "Own Goal For":
            events_change.append((r.period, r.t, "goal", r.team))
        elif r.type in ("Foul Committed", "Bad Behaviour") and r.outcome in ("Red Card", "Second Yellow"):
            events_change.append((r.period, r.t, "red", r.team))
    events_change.sort()

    seg_rows = []
    score = {home: 0, away: 0}
    reds = {home: 0, away: 0}
    seg_id = 0
    for p in sorted(df.period.unique()):
        plen_nominal = 45.0 if p <= 2 else 15.0
        cuts = [c for c in (15.0, 30.0) if c < plen_nominal] + [pend[p]]
        cuts = sorted(set([c for c in cuts if c <= pend[p]]))
        bounds = [0.0]
        chg = [c for c in events_change if c[0] == p]
        pts = sorted(set(cuts + [c[1] for c in chg]))
        bounds += [x for x in pts if x > 0]
        evp = df[df.period == p]
        for a, b in zip(bounds[:-1], bounds[1:]):
            lo = (evp.t > a) if a > 0 else (evp.t >= a)
            seg = evp[lo & (evp.t <= b)] if b < bounds[-1] else evp[lo]
            dur = b - a
            if dur > 0.05:
                for i, tm in enumerate(teams):
                    op = teams[1 - i]
                    seg_rows.append(_seg_features(seg, tm, op, i == 0, dur, p, a, b, score, reds, mid, seg_id))
                seg_id += 1
            # apply state changes occurring at time b
            for c in chg:
                if abs(c[1] - b) < 1e-9:
                    if c[2] == "goal":
                        score[c[3]] += 1
                    else:
                        reds[c[3]] += 1
    seg = pd.DataFrame(seg_rows)
    seg["league"] = m["_league"]
    seg["date"] = m["match_date"]
    seg["home_team"], seg["away_team"] = m["home_team"]["home_team_name"], m["away_team"]["away_team_name"]
    canon = {home: m["home_team"]["home_team_name"], away: m["away_team"]["away_team_name"]}
    seg["team"] = seg["team"].map(canon); seg["opp"] = seg["opp"].map(canon)
    df["team"] = df["team"].map(canon).fillna(df["team"]); df["poss_team"] = df["poss_team"].map(canon).fillna(df["poss_team"])
    seg["home_score"], seg["away_score"] = m["home_score"], m["away_score"]
    df["league"] = m["_league"]
    return df, seg


def _seg_features(seg, tm, op, is_home, dur, p, a, b, score, reds, mid, seg_id):
    own = seg[seg.team == tm]
    opp = seg[seg.team == op]
    shots_own = own[own.type == "Shot"]
    shots_opp = opp[opp.type == "Shot"]
    np_own = shots_own[shots_own["sub"] != "Penalty"]
    np_opp = shots_opp[shots_opp["sub"] != "Penalty"]
    defs = own[own.type.isin(DEF_TYPES) | ((own.type == "Duel") & (own["sub"] == "Tackle"))]
    passes_own = own[own.type == "Pass"]
    passes_opp = opp[opp.type == "Pass"]
    op_open = passes_own[passes_own["sub"].isin(["Open", "Recovery", "Interception"]) | passes_own["sub"].isna()]
    # PPDA zone: opponent passes in their own 60% (x < 72 in their frame); our def actions at x > 48 in our frame
    opp_build = (passes_opp.x < 72).sum()
    our_high_def = (defs.x > 48).sum()
    # forward progress per possession second: pass/carry progress in own frame
    prog = 0.0
    for d in (passes_own[passes_own.outcome == "Complete"], own[own.type == "Carry"]):
        prog += np.clip(d.end_x - d.x, 0, None).sum()
    poss_time = own[own.poss_team == tm].dur.sum() + 1e-9
    gd = score[tm] - score[op]
    return dict(match_id=mid, seg=seg_id, team=tm, opp=op, home=int(is_home), period=p,
                t0=a, t1=b, minutes=dur, clock_mid=(a + b) / 2 + (0 if p == 1 else 45 if p == 2 else 90),
                gd=gd, goals_for=score[tm], goals_against=score[op], men_diff=reds[op] - reds[tm],
                xg_for=shots_own.xg.sum(), xg_against=shots_opp.xg.sum(),
                npxg_for=np_own.xg.sum(), npxg_against=np_opp.xg.sum(),
                shots_for=len(shots_own), shots_against=len(shots_opp),
                goals_in_seg_for=int((shots_own.outcome == "Goal").sum() + (own.type == "Own Goal For").sum()),
                goals_in_seg_against=int((shots_opp.outcome == "Goal").sum() + (opp.type == "Own Goal For").sum()),
                n_def=len(defs), def_height_sum=defs.x.sum(),
                n_press=int((own.type == "Pressure").sum()), n_pass=len(passes_own), n_pass_opp=len(passes_opp),
                n_open_pass=len(op_open), n_long=int((op_open.length >= 32).sum()),
                opp_build=int(opp_build), our_high_def=int(our_high_def),
                prog=prog, poss_time=poss_time)


def main():
    os.makedirs(PROC, exist_ok=True)
    ms = json.load(open(os.path.join(ROOT, "data", "raw", "sb_matches.json")))
    ms = [m for m in ms if os.path.exists(os.path.join(RAW, f"{m['match_id']}.json.gz"))]
    print("matches", len(ms), flush=True)
    evs, segs = [], []
    with ProcessPoolExecutor(8) as ex:
        for i, (d, s) in enumerate(ex.map(process, ms, chunksize=8)):
            evs.append(d)
            segs.append(s)
            if i % 200 == 0:
                print(i, flush=True)
    ev = pd.concat(evs, ignore_index=True)
    ev.to_parquet(os.path.join(PROC, "events.parquet"))
    sg = pd.concat(segs, ignore_index=True)
    sg.to_parquet(os.path.join(PROC, "segments.parquet"))
    print(ev.shape, sg.shape)


if __name__ == "__main__":
    main()

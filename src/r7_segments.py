"""Segment tables (constant game state, ≤15-min) for the unseen StatsBomb leagues and Wyscout 2017/18, matching build_segments.py
in the columns round 7 needs: match_id, team, opp, home, period, minutes, clock_mid, gd, men_diff, xg_for, xg_against, comp."""
import json, os, pickle, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import build_segments as bs
ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")
COLS = ["match_id", "team", "opp", "home", "period", "minutes", "clock_mid", "gd", "men_diff", "xg_for", "xg_against"]


def sb2_one(m):
    bs.RAW = os.path.join(ROOT, "data", "raw", "sb2")
    m = dict(m); m["_league"] = m["_comp"]
    _, seg = bs.process(m)
    seg["comp"] = m["_comp"]
    return seg[COLS + ["comp"]]


def sb2():
    ms = json.load(open(os.path.join(ROOT, "data", "raw", "sb2_matches.json")))
    with ProcessPoolExecutor(8) as ex:
        segs = list(ex.map(sb2_one, ms, chunksize=8))
    return pd.concat(segs, ignore_index=True)


def wy():
    from xg_model import feats
    XG = pickle.load(open(os.path.join(PROC, "xg_model.pkl"), "rb"))
    WY = os.path.join(ROOT, "data", "raw", "wy")
    out = []
    for country, lg in {"England": "EPL", "Spain": "LaLiga", "Germany": "Bundesliga", "Italy": "SerieA", "France": "Ligue1"}.items():
        ms = {m["wyId"]: m for m in json.load(open(os.path.join(WY, f"matches_{country}.json")))}
        ev = json.load(open(os.path.join(WY, f"events_{country}.json")))
        by = {}
        for e in ev:
            if e["matchPeriod"] in ("1H", "2H"):
                by.setdefault(e["matchId"], []).append(e)
        # batch xG for all shots
        shots = [e for e in ev if e["matchPeriod"] in ("1H", "2H") and (e["eventName"] == "Shot" or (e["eventName"] == "Free Kick" and e["subEventName"] in ("Free kick shot", "Penalty")))]
        df = pd.DataFrame(dict(x=[s["positions"][0]["x"] / 100 * 105 for s in shots], y=[s["positions"][0]["y"] / 100 * 68 for s in shots],
                               head=[any(t["id"] == 403 for t in s["tags"]) for s in shots], fk=[s["subEventName"] == "Free kick shot" for s in shots]))
        p = XG.predict_proba(feats(df))[:, 1]
        xgmap = {s["id"]: (0.76 if s["subEventName"] == "Penalty" else float(pp)) for s, pp in zip(shots, p)}
        for mid, evs in by.items():
            m = ms[mid]
            sides = {v["side"]: int(k) for k, v in m["teamsData"].items()}
            home, away = sides["home"], sides["away"]
            evs.sort(key=lambda e: (e["matchPeriod"], e["eventSec"]))
            changes = []
            for e in evs:
                tags = {t["id"] for t in e["tags"]}
                t = e["eventSec"] / 60
                if e["id"] in xgmap and 101 in tags:
                    changes.append((e["matchPeriod"], t, "goal", e["teamId"]))
                if 102 in tags:  # own goal: credit the other team
                    changes.append((e["matchPeriod"], t, "goal", away if e["teamId"] == home else home))
                if e["eventName"] == "Foul" and (1701 in tags or 1703 in tags):
                    changes.append((e["matchPeriod"], t, "red", e["teamId"]))
            score = {home: 0, away: 0}; reds = {home: 0, away: 0}
            for per in ("1H", "2H"):
                pe = [e for e in evs if e["matchPeriod"] == per]
                if not pe:
                    continue
                end = pe[-1]["eventSec"] / 60
                chg = sorted([c for c in changes if c[0] == per], key=lambda c: c[1])
                bounds = sorted(set([0.0] + [b for b in (15.0, 30.0) if b < end] + [c[1] for c in chg if 0 < c[1] < end] + [end]))
                for a, b in zip(bounds[:-1], bounds[1:]):
                    sh = [e for e in pe if e["id"] in xgmap and a < e["eventSec"] / 60 <= b] if a > 0 else [e for e in pe if e["id"] in xgmap and e["eventSec"] / 60 <= b]
                    if b - a > 0.05:
                        for tm, op, h in ((home, away, 1), (away, home, 0)):
                            out.append(dict(match_id=mid, team=tm, opp=op, home=h, period=1 if per == "1H" else 2, minutes=b - a,
                                            clock_mid=(a + b) / 2 + (0 if per == "1H" else 45), gd=score[tm] - score[op], men_diff=reds[op] - reds[tm],
                                            xg_for=sum(xgmap[e["id"]] for e in sh if e["teamId"] == tm),
                                            xg_against=sum(xgmap[e["id"]] for e in sh if e["teamId"] == op), comp=lg))
                    for c in chg:
                        if abs(c[1] - b) < 1e-9:
                            if c[2] == "goal":
                                score[c[3]] += 1
                            else:
                                reds[c[3]] += 1
    return pd.DataFrame(out)


if __name__ == "__main__":
    a = sb2(); a.to_parquet(os.path.join(PROC, "r7_segments_sb2.parquet")); print("sb2", a.shape, a.groupby("comp").match_id.nunique().to_dict())
    b = wy(); b.to_parquet(os.path.join(PROC, "r7_segments_wy.parquet")); print("wy", b.shape, b.groupby("comp").match_id.nunique().to_dict())

"""Agent keeper: build keeper open-play release table (PREREGISTRATION_AGENT_keeper.md).
-> data/proc/agent_keeper_releases.parquet"""
import gzip, json, os, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")
DISC = ("FIFA World Cup 2022", "UEFA Euro 2020", "UEFA Euro 2024", "African Cup of Nations 2023")
WOMEN = ("UEFA Women's Euro 2022", "UEFA Women's Euro 2025", "Women's World Cup 2023")
GAIN = {"Collected": "claim", "Keeper Sweeper": "sweeper", "Shot Saved": "save", "Shot Saved Off Target": "save",
        "Shot Saved to Post": "save", "Smother": "save", "Save": "save"}
BP = {"Keeper Arm": "hand", "Drop Kick": "dropkick", "Left Foot": "foot", "Right Foot": "foot"}
NEUTRAL = {"Camera On", "Camera off", "Camera Off"}


def tsec(e):
    h, m, s = e["timestamp"].split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def one(args):
    path, frpath, mid, pop, sub = args
    try:
        ev = [e for e in json.load(gzip.open(path)) if e["period"] <= 2]
    except Exception:
        return []
    fr = {}
    if frpath and os.path.exists(frpath):
        try:
            fr = {f["event_uuid"]: f for f in json.load(gzip.open(frpath))}
        except Exception:
            fr = {}
    shots = {}
    for e in ev:
        if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] != "Penalty":
            shots.setdefault(e["period"], []).append((tsec(e), e["shot"].get("statsbomb_xg", 0.0) or 0.0, e["team"]["name"]))
    shots = {k: (np.array([s[0] for s in v]), np.array([s[1] for s in v]), np.array([s[2] for s in v])) for k, v in shots.items()}
    teams = []
    for e in ev:
        if e["team"]["name"] not in teams:
            teams.append(e["team"]["name"])
    score = {t: 0 for t in teams}
    out = []
    for i, e in enumerate(ev):
        typ = e["type"]["name"]
        if (typ == "Shot" and e["shot"]["outcome"]["name"] == "Goal") or typ == "Own Goal For":
            score[e["team"]["name"]] = score.get(e["team"]["name"], 0) + 1
        if typ != "Goal Keeper" or e["goalkeeper"]["type"]["name"] not in GAIN or "player" not in e:
            continue
        kid = e["player"]["id"]; team = e["team"]["name"]; per = e["period"]; t0 = tsec(e)
        opp_team = [t for t in teams if t != team]
        gd = score.get(team, 0) - sum(score.get(t, 0) for t in opp_team)
        rel = None; pressed = False; frame = fr.get(e["id"])
        for f in ev[i + 1:i + 26]:
            ft = f["type"]["name"]
            if f["period"] != per:
                break
            if ft in NEUTRAL:
                continue
            same_k = f.get("player", {}).get("id") == kid
            if ft == "Pressure" and f["team"]["name"] != team:
                pressed = True; continue
            if same_k and ft in ("Carry", "Ball Receipt*", "Goal Keeper"):
                if frame is None:
                    frame = fr.get(f["id"])
                continue
            if same_k and ft == "Pass" and e["period"] == per:
                pt = f["pass"].get("type", {}).get("name")
                if pt in (None, "Recovery"):
                    rel = f
                    if frame is None:
                        frame = fr.get(f["id"])
            break
        if rel is None:
            continue
        bp = BP.get(rel["pass"].get("body_part", {}).get("name"))
        if bp is None:
            continue
        t1 = tsec(rel); delay = t1 - t0
        if delay < 0 or delay > 60:
            continue
        row = dict(match_id=mid, pop=pop, sub=sub, team=team, period=per, t0=t0, t1=t1, delay=delay, rtype=bp,
                   gtype=GAIN[e["goalkeeper"]["type"]["name"]], gd=gd, length=rel["pass"].get("length", np.nan),
                   kx=e["location"][0] if "location" in e else np.nan, pressed=pressed,
                   complete="outcome" not in rel["pass"])
        s = shots.get(per)
        for w in (30, 60):
            if s is None:
                row[f"own{w}"] = 0.0; row[f"opp{w}"] = 0.0
            else:
                m = (s[0] >= t1 - 0.01) & (s[0] <= t1 + w)
                row[f"own{w}"] = float(s[1][m & (s[2] == team)].sum()); row[f"opp{w}"] = float(s[1][m & (s[2] != team)].sum())
        if fr:
            if frame is not None:
                ff = frame["freeze_frame"]
                opp = [p["location"] for p in ff if not p["teammate"] and not p.get("keeper")]
                row["n_opp_half"] = sum(1 for o in opp if o[0] < 60); row["n_vis"] = len(ff); row["has_frame"] = True
            else:
                row["n_opp_half"] = np.nan; row["n_vis"] = np.nan; row["has_frame"] = False
        out.append(row)
    return out


def build():
    args = []
    for mfile, raw, pop, fmt in (("sb_matches.json", "sb", "SB_1516", "{}.json.gz"), ("sb2_matches.json", "sb2", "SB_women_ISL", "{}.json.gz"),
                                 ("sb360_matches.json", "sb360", "SB_360comps", "{}_events.json.gz")):
        for m in json.load(open(os.path.join(ROOT, "data", "raw", mfile))):
            mid = m["match_id"]
            frp = os.path.join(ROOT, "data", "raw", "sb360", f"{mid}_three-sixty.json.gz") if raw == "sb360" else None
            sub = m.get("_comp") or m.get("_league")
            if raw == "sb360":
                sub = "DISC" if sub in DISC else ("WOMEN" if sub in WOMEN else "CLUB")
            args.append((os.path.join(ROOT, "data", "raw", raw, fmt.format(mid)), frp, mid, pop, sub))
    rows = []
    with ProcessPoolExecutor(2) as ex:
        for r in ex.map(one, args, chunksize=8):
            rows += r
    d = pd.DataFrame(rows)
    d.to_parquet(os.path.join(PROC, "agent_keeper_releases.parquet"))
    print(len(d), d.groupby("pop").size().to_dict())


if __name__ == "__main__":
    build()

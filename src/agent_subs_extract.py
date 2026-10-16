"""agent subs: extract per-match shots / goals / subs / red cards for sb, sb2, sb360 (StatsBomb format) and Wyscout.
Writes data/proc/agent_subs_{shots,goals,subs,reds,matches}.parquet."""
import gzip, json, os, pickle, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "proc")
OUT = os.environ.get("SUBS_OUT", "/private/tmp/claude-501/-Users-usman-Documents-research-football/3d8cdebd-b9be-4e7c-b5f3-c4c5a15f5f08/scratchpad")


def pclass(name):
    if name is None:
        return None
    if name == "Goalkeeper":
        return "GK"
    if "Back" in name:
        return "DEF"
    if "Forward" in name or "Striker" in name or "Wing" in name:
        return "FWD"
    return "MID"


def clock(e):
    return e["minute"] + e["second"] / 60.0


def sb_match(args):
    pop, path, mid, group, home = args
    ev = json.load(gzip.open(path))
    shots, goals, subs, reds = [], [], [], []
    ppos = {}   # player id -> Counter of positions after (for replacement class)
    teams = []
    for e in ev:
        tn = e.get("team", {}).get("name")
        if tn and tn not in teams:
            teams.append(tn)
        if e["period"] > 2:
            continue
        t = e["type"]["name"]; c = clock(e)
        if "player" in e and "position" in e:
            ppos.setdefault(e["player"]["id"], Counter())[e["position"]["name"]] += 1
        if t == "Shot":
            s = e["shot"]
            shots.append((mid, e["period"], c, tn, float(s.get("statsbomb_xg") or 0.0)))
            if s["outcome"]["name"] == "Goal":
                goals.append((mid, e["period"], c, tn))
        elif t == "Own Goal For":
            goals.append((mid, e["period"], c, tn))
        elif t == "Substitution":
            subs.append(dict(match_id=mid, period=e["period"], clock=c, team=tn, off_pos=e.get("position", {}).get("name"),
                             on_id=e["substitution"]["replacement"]["id"], reason=e["substitution"].get("outcome", {}).get("name")))
        card = None
        for k in ("bad_behaviour", "foul_committed"):
            if k in e and "card" in e[k]:
                card = e[k]["card"]["name"]
        if card in ("Red Card", "Second Yellow"):
            reds.append((mid, e["period"], c, tn))
    for s in subs:
        cnt = ppos.get(s.pop("on_id"))
        s["on_pos"] = cnt.most_common(1)[0][0] if cnt else None
        s["off_cls"] = pclass(s["off_pos"]); s["on_cls"] = pclass(s["on_pos"]) or s["off_cls"]
    return dict(pop=pop, match_id=mid, group=group, home=home, teams=teams[:2], shots=shots, goals=goals, subs=subs, reds=reds)


def sb_jobs():
    jobs = []
    for m in json.load(open(os.path.join(RAW, "sb_matches.json"))):
        jobs.append(("sb", os.path.join(RAW, "sb", f"{m['match_id']}.json.gz"), m["match_id"], m["_league"], m["home_team"]["home_team_name"]))
    for m in json.load(open(os.path.join(RAW, "sb2_matches.json"))):
        p = os.path.join(RAW, "sb2", f"{m['match_id']}.json.gz")
        if os.path.exists(p):
            jobs.append(("sb2", p, m["match_id"], m["_comp"], m["home_team"]["home_team_name"]))
    for m in json.load(open(os.path.join(RAW, "sb360_matches.json"))):
        p = os.path.join(RAW, "sb360", f"{m['match_id']}_events.json.gz")
        if os.path.exists(p):
            g = m["competition"]["competition_name"] + " " + m["season"]["season_name"]
            jobs.append(("sb360", p, m["match_id"], g, m["home_team"]["home_team_name"]))
    return jobs


def wyscout():
    from xg_model import feats
    XG = pickle.load(open(os.path.join(PROC, "xg_model.pkl"), "rb"))
    out = []
    for f in sorted(os.listdir(os.path.join(RAW, "wy"))):
        if not f.startswith("matches_"):
            continue
        country = f[8:-5]
        ms = json.load(open(os.path.join(RAW, "wy", f)))
        ev = json.load(open(os.path.join(RAW, "wy", f"events_{country}.json")))
        evm = {}
        for e in ev:
            if e["matchPeriod"] in ("1H", "2H"):
                evm.setdefault(e["matchId"], []).append(e)
        del ev
        for m in ms:
            mid = m["wyId"]
            lab = m["label"].split(",")[0]; h, a = [x.strip() for x in lab.split(" - ")]
            sides = {v["side"]: int(k) for k, v in m["teamsData"].items()}
            name = {sides["home"]: h, sides["away"]: a}
            shots, goals, subs, reds, rows = [], [], [], [], []
            for e in evm.get(mid, []):
                per = 1 if e["matchPeriod"] == "1H" else 2
                c = e["eventSec"] / 60.0 + (45.0 if per == 2 else 0.0)
                tags = {t["id"] for t in e["tags"]}
                tn = name.get(e["teamId"])
                if tn is None:
                    continue
                is_shot = e["eventName"] == "Shot" or (e["eventName"] == "Free Kick" and e["subEventName"] in ("Free kick shot", "Penalty"))
                if is_shot and e["positions"]:
                    pen = e["subEventName"] == "Penalty"
                    rows.append((per, c, tn, e["positions"][0]["x"] / 100 * 105, e["positions"][0]["y"] / 100 * 68, 403 in tags,
                                 e["subEventName"] == "Free kick shot", pen))
                    if 101 in tags:
                        goals.append((mid, per, c, tn))
                if 102 in tags:   # own goal: credited to the other team
                    other = [v for k, v in name.items() if k != e["teamId"]][0]
                    goals.append((mid, per, c, other))
                if 1701 in tags or 1703 in tags:
                    reds.append((mid, per, c, tn))
            if rows:
                d = pd.DataFrame(rows, columns=["period", "clock", "team", "x", "y", "head", "fk", "pen"])
                xg = np.where(d.pen, 0.76, XG.predict_proba(feats(d))[:, 1])
                shots = [(mid, int(p), float(c), t, float(x)) for p, c, t, x in zip(d.period, d.clock, d.team, xg)]
            for tid, td in m["teamsData"].items():
                for s in (td.get("formation") or {}).get("substitutions") or []:
                    if not isinstance(s, dict) or s.get("minute") is None:
                        continue
                    mn = float(s["minute"])
                    subs.append(dict(match_id=mid, period=1 if mn < 45 else 2, clock=mn + 0.5, team=name[int(tid)],
                                     off_pos=None, on_pos=None, reason=None, off_cls=None, on_cls=None))
            # de-duplicate red-card tags that appear on several events
            reds = list({(r[0], r[3]): r for r in sorted(reds, key=lambda r: -r[2])}.values())
            out.append(dict(pop="wy", match_id=mid, group=country, home=h, teams=[h, a], shots=shots, goals=goals, subs=subs, reds=reds))
        print("wy", country, len(ms), flush=True)
    return out


if __name__ == "__main__":
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(sb_match, sb_jobs(), chunksize=8))
    print("sb-format matches", len(res), flush=True)
    res += wyscout()
    M, S, G, U, R = [], [], [], [], []
    for r in res:
        if len(r["teams"]) < 2:
            continue
        M.append(dict(pop=r["pop"], match_id=r["match_id"], group=r["group"], home_team=r["home"], t1=r["teams"][0], t2=r["teams"][1]))
        S += [(r["pop"],) + x for x in r["shots"]]; G += [(r["pop"],) + x for x in r["goals"]]
        R += [(r["pop"],) + x for x in r["reds"]]; U += [dict(pop=r["pop"], **x) for x in r["subs"]]
    cols = ["pop", "match_id", "period", "clock", "team"]
    pd.DataFrame(M).to_parquet(os.path.join(OUT, "agent_subs_matches.parquet"))
    pd.DataFrame(S, columns=cols + ["xg"]).to_parquet(os.path.join(OUT, "agent_subs_shots.parquet"))
    pd.DataFrame(G, columns=cols).to_parquet(os.path.join(OUT, "agent_subs_goals.parquet"))
    pd.DataFrame(R, columns=cols).to_parquet(os.path.join(OUT, "agent_subs_reds.parquet"))
    pd.DataFrame(U).to_parquet(os.path.join(OUT, "agent_subs_subs.parquet"))
    print(pd.DataFrame(M).groupby("pop").size())

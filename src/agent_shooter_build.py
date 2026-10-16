"""Agent shooter: build tables.
  - data/proc/agent_shooter_ev.parquet   : per sb360 event (match_id, idx) -> player_id, position, net60
  - data/proc/agent_shooter_shots.parquet: all non-penalty shots (player_id, date, xg, goal) from sb, sb2, sb360
"""
import gzip, json, os, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import tsec

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "proc")
H = 60.0


def shots_of(ev, mid, date, src):
    out = []
    for e in ev:
        if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] != "Penalty" and "player" in e:
            out.append((src, mid, date, e["player"]["id"], e["shot"].get("statsbomb_xg", 0.0) or 0.0,
                        int(e["shot"]["outcome"]["name"] == "Goal")))
    return out


def one360(m):
    mid = m["match_id"]
    try:
        ev = [e for e in json.load(gzip.open(os.path.join(RAW, "sb360", f"{mid}_events.json.gz"))) if e["period"] <= 2]
    except Exception:
        return None
    S = [(e["period"], tsec(e), e["team"]["name"], e["shot"].get("statsbomb_xg", 0.0)) for e in ev
         if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] != "Penalty"]
    rows = []
    for e in ev:
        if e["type"]["name"] in ("Pass", "Carry", "Dribble", "Shot") and "player" in e:
            rows.append((mid, e["index"], e["period"], tsec(e), e["team"]["name"], e["player"]["id"],
                         e.get("position", {}).get("name", "")))
    A = pd.DataFrame(rows, columns=["match_id", "idx", "period", "t", "team", "player_id", "position"])
    S = pd.DataFrame(S, columns=["period", "t", "team", "xg"])
    own = np.zeros(len(A)); opp = np.zeros(len(A))
    sp, st, steam, sx = S.period.values, S.t.values, S.team.values, S.xg.values.astype(float)
    for i, (per, t, team) in enumerate(zip(A.period.values, A.t.values, A.team.values)):
        w = (sp == per) & (st >= t - 0.01) & (st <= t + H)
        mine = steam == team
        own[i] = sx[w & mine].sum()
        opp[i] = sx[w & ~mine & (st > t + 0.01)].sum()
    A["net60"] = own - opp
    return A[["match_id", "idx", "player_id", "position", "net60"]], shots_of(ev, mid, m["match_date"], "sb360")


def one_hist(args):
    path, mid, date, src = args
    try:
        ev = json.load(gzip.open(path))
    except Exception:
        return []
    return shots_of(ev, mid, date, src)


def main():
    ms = json.load(open(os.path.join(RAW, "sb360_matches.json")))
    jobs = [(os.path.join(RAW, "sb", f"{m['match_id']}.json.gz"), m["match_id"], m["match_date"], "sb")
            for m in json.load(open(os.path.join(RAW, "sb_matches.json")))]
    jobs += [(os.path.join(RAW, "sb2", f"{m['match_id']}.json.gz"), m["match_id"], m["match_date"], "sb2")
             for m in json.load(open(os.path.join(RAW, "sb2_matches.json")))]
    As, shots = [], []
    with ProcessPoolExecutor(2) as ex:
        for r in ex.map(one360, ms, chunksize=4):
            if r is not None:
                As.append(r[0]); shots += r[1]
        print("360 done", flush=True)
        for r in ex.map(one_hist, jobs, chunksize=8):
            shots += r
    pd.concat(As, ignore_index=True).to_parquet(os.path.join(PROC, "agent_shooter_ev.parquet"))
    Sh = pd.DataFrame(shots, columns=["src", "match_id", "date", "player_id", "xg", "goal"])
    Sh.to_parquet(os.path.join(PROC, "agent_shooter_shots.parquet"))
    print(Sh.groupby("src").size())


if __name__ == "__main__":
    main()

"""Extract in-match coaching decisions (substitutions and formation shifts) and score how attacking each is.

Attacking index by position (StatsBomb position names): GK 0, centre-backs 1, full-backs 1.3, wing-backs 1.6,
defensive mids 2, centre mids 2.3, wide/attacking mids 2.6, wingers 2.8, forwards 3.
For every team and every moment it changes something we compute two quantities:
  formation load: sum of the attacking indices of the 10 outfield *assigned* positions
  personnel load: sum of each on-pitch outfield player's *natural* index (his modal position over the season)
A decision = all of a team's changes within 2 minutes of each other. Its delta is the change in the mean of
the two loads. Output: data/proc/decisions.parquet
"""
import gzip, json, os, collections
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw", "sb")

IDX = {"Goalkeeper": 0.0, "Right Center Back": 1.0, "Left Center Back": 1.0, "Center Back": 1.0,
       "Right Back": 1.3, "Left Back": 1.3, "Right Wing Back": 1.6, "Left Wing Back": 1.6,
       "Center Defensive Midfield": 2.0, "Right Defensive Midfield": 2.0, "Left Defensive Midfield": 2.0,
       "Center Midfield": 2.3, "Right Center Midfield": 2.3, "Left Center Midfield": 2.3,
       "Right Midfield": 2.6, "Left Midfield": 2.6, "Center Attacking Midfield": 2.6,
       "Right Attacking Midfield": 2.6, "Left Attacking Midfield": 2.6,
       "Right Wing": 2.8, "Left Wing": 2.8, "Center Forward": 3.0, "Right Center Forward": 3.0,
       "Left Center Forward": 3.0, "Secondary Striker": 2.9}


def clock(e):
    h, m, s = e["timestamp"].split(":")
    t = int(h) * 60 + int(m) + float(s) / 60
    return t + {1: 0, 2: 45, 3: 90, 4: 105}.get(e["period"], 120)


def player_positions(mid):
    ev = json.load(gzip.open(os.path.join(RAW, f"{mid}.json.gz")))
    c = collections.Counter()
    for e in ev:
        if "player" in e and "position" in e:
            c[(e["player"]["id"], e["position"]["name"])] += 1
    return mid, c


_NAT = {}


def _init(nat):
    _NAT.update(nat)


def changes_for_match(mid):
    ev = json.load(gzip.open(os.path.join(RAW, f"{mid}.json.gz")))
    natural = _NAT
    out = []
    teams = {}
    for e in ev:
        if e["type"]["name"] == "Starting XI":
            lu = {p["player"]["id"]: p["position"]["name"] for p in e["tactics"]["lineup"]}
            teams[e["team"]["name"]] = dict(lineup=lu)
    def loads(lu):
        outf = [(pid, pos) for pid, pos in lu.items() if pos != "Goalkeeper"]
        f = sum(IDX.get(pos, 2.0) for _, pos in outf)
        p = sum(natural.get(pid, IDX.get(pos, 2.0)) for pid, pos in outf)
        return f, p, len(outf)
    for tm, st in teams.items():
        st["load"] = loads(st["lineup"])
    for e in ev:
        typ = e["type"]["name"]
        if e["period"] > 2 or typ not in ("Substitution", "Tactical Shift", "Bad Behaviour", "Foul Committed", "Player Off"):
            continue
        tm = e["team"]["name"]
        if tm not in teams:
            continue
        st = teams[tm]
        before = st["load"]
        kind = None
        if typ == "Substitution":
            off = e["player"]["id"]; on = e["substitution"]["replacement"]["id"]
            pos = st["lineup"].pop(off, e.get("position", {}).get("name", "Center Midfield"))
            st["lineup"][on] = pos
            kind = "sub"
        elif typ == "Tactical Shift":
            st["lineup"] = {p["player"]["id"]: p["position"]["name"] for p in e["tactics"]["lineup"]}
            kind = "shift"
        else:
            key = "foul_committed" if typ == "Foul Committed" else "bad_behaviour"
            card = e.get(key, {}).get("card", {}).get("name") if typ != "Player Off" else None
            if card in ("Red Card", "Second Yellow") and "player" in e:
                st["lineup"].pop(e["player"]["id"], None)
                kind = "red"
            else:
                continue
        st["load"] = loads(st["lineup"])
        after = st["load"]
        out.append(dict(match_id=mid, team=tm, clock=clock(e), period=e["period"], kind=kind,
                        d_form=after[0] - before[0], d_pers=after[1] - before[1], n_out=after[2]))
    return out


def main():
    ms = json.load(open(os.path.join(ROOT, "data", "raw", "sb_matches.json")))
    ids = [m["match_id"] for m in ms]
    # pass 1: natural position per player = attacking index of modal position over the season
    tot = collections.Counter()
    with ProcessPoolExecutor(8) as ex:
        for mid, c in ex.map(player_positions, ids, chunksize=8):
            tot.update(c)
    best = {}
    for (pid, pos), n in tot.items():
        if pid not in best or n > best[pid][1]:
            best[pid] = (pos, n)
    natural = {pid: IDX.get(pos, 2.0) for pid, (pos, n) in best.items()}
    rows = []
    with ProcessPoolExecutor(8, initializer=_init, initargs=(natural,)) as ex:
        for r in ex.map(changes_for_match, ids, chunksize=8):
            rows += r
    ch = pd.DataFrame(rows)
    # canonical team names via the match list
    canon = {}
    for m in ms:
        canon[(m["match_id"], 0)] = m["home_team"]["home_team_name"]; canon[(m["match_id"], 1)] = m["away_team"]["away_team_name"]
    ch.to_parquet(os.path.join(ROOT, "data", "proc", "changes_raw.parquet"))
    print(ch.kind.value_counts(), ch.describe())


if __name__ == "__main__":
    main()

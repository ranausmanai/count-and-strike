"""Possession table: one row per (match, possession) with origin, regain location, speed and xG.

channel:
  penalty | corner | free_kick | throw_in | counter (SB 'From Counter' or open-play regain → shot within 15 s)
  | high_regain (open-play possession won at x>=80, i.e. in the final third, not counter) | build_up (other open play)
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")


def build_possessions(ev):
    ev = ev[ev.period <= 2].sort_values(["match_id", "idx"])
    own = ev[ev.team == ev.poss_team]
    first = own.groupby(["match_id", "poss"]).first()
    last = ev.groupby(["match_id", "poss"]).t.max()
    shots = ev[ev.type == "Shot"]
    sh = shots.groupby(["match_id", "poss"]).agg(xg=("xg", "sum"), n_shots=("xg", "size"),
                                                 t_shot=("t", "min"), pen=("sub", lambda s: (s == "Penalty").any()),
                                                 goals=("outcome", lambda s: (s == "Goal").sum()))
    passes = own[own.type == "Pass"].groupby(["match_id", "poss"]).size().rename("n_pass")
    p = first[["poss_team", "pattern", "type", "x", "y", "t", "period", "league"]].rename(
        columns={"type": "start_type", "x": "start_x", "y": "start_y", "t": "t0"})
    p = p.join(last.rename("t_end")).join(sh).join(passes)
    p[["xg", "n_shots", "goals", "n_pass"]] = p[["xg", "n_shots", "goals", "n_pass"]].fillna(0)
    p["pen"] = p.pen.astype(object).fillna(False).astype(bool)
    p["secs_to_shot"] = (p.t_shot - p.t0) * 60
    pat = p.pattern
    ch = np.full(len(p), "build_up", dtype=object)
    ch[(p.start_x >= 80).values & (pat == "Regular Play").values] = "high_regain"
    fast = (p.secs_to_shot <= 15).values & pat.isin(["Regular Play", "From Counter"]).values & (p.start_x < 80).values
    ch[fast | (pat == "From Counter").values] = "counter"
    # restarts count as set pieces only when they are attacking restarts and any shot comes within 20 s;
    # longer restart-initiated possessions are open play (classified by the rules above)
    quick = (p.secs_to_shot.isna() | (p.secs_to_shot <= 20)).values
    ch[(pat == "From Throw In").values & (p.start_x >= 80).values & quick] = "throw_in"
    ch[(pat == "From Free Kick").values & (p.start_x >= 60).values & quick] = "free_kick"
    ch[(pat == "From Corner").values & quick] = "corner"
    ch[p.pen.values] = "penalty"
    p["channel"] = ch
    return p.reset_index()


def main():
    ev = pd.read_parquet(os.path.join(PROC, "events.parquet"),
                         columns=["match_id", "idx", "period", "t", "type", "team", "poss", "poss_team", "pattern",
                                  "x", "y", "xg", "outcome", "sub", "league"])
    p = build_possessions(ev)
    p.to_parquet(os.path.join(PROC, "possessions.parquet"))
    print(p.channel.value_counts())
    print(p.groupby("channel")[["xg", "goals", "n_shots"]].sum())


if __name__ == "__main__":
    main()

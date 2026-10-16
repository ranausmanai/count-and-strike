"""Agent 'runners': extra outcome windows (opp30, own45, opp45) and visible team-mate count at regain starts."""
import gzip, json, os, sys
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import RAW, PROC, tsec

R = None


def one(mid):
    pe, pf = os.path.join(RAW, f"{mid}_events.json.gz"), os.path.join(RAW, f"{mid}_three-sixty.json.gz")
    ev = [e for e in json.load(gzip.open(pe)) if e["period"] <= 2]
    fr = {f["event_uuid"]: f for f in json.load(gzip.open(pf))}
    S = pd.DataFrame([(e["period"], tsec(e), e["team"]["name"], e["shot"].get("statsbomb_xg", 0.0)) for e in ev
                      if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] != "Penalty"], columns=["period", "t", "team", "xg"])
    rows = []
    for e in ev:
        if e["type"]["name"] not in ("Pass", "Carry", "Dribble", "Shot") or "location" not in e or e["id"] not in fr:
            continue
        f = fr[e["id"]]
        nm = sum(1 for p in f["freeze_frame"] if p["teammate"] and not p.get("actor") and not p.get("keeper"))
        rows.append((mid, e["period"], tsec(e), e["team"]["name"], e["location"][0], e["location"][1], nm))
    a = pd.DataFrame(rows, columns=["match_id", "period", "t", "team", "x", "y", "n_vis_mates"]).drop_duplicates(["match_id", "period", "t", "team", "x", "y"])
    return a, S.assign(match_id=mid)


if __name__ == "__main__":
    A = pd.read_parquet(os.path.join(PROC, "r13_actions.parquet"))
    R = A[A.regain_start & (A.x < 80)].copy()
    mids = sorted(R.match_id.unique())
    feats, shots = [], []
    with ProcessPoolExecutor(2) as ex:
        for a, s in ex.map(one, mids, chunksize=4):
            feats.append(a); shots.append(s)
    F = pd.concat(feats); S = pd.concat(shots)
    R = R.merge(F, on=["match_id", "period", "t", "team", "x", "y"], how="left")
    print("n_vis_mates missing:", R.n_vis_mates.isna().sum(), "of", len(R))
    for h in (30, 45):
        own = np.zeros(len(R)); opp = np.zeros(len(R))
        Sg = {k: g for k, g in S.groupby(["match_id", "period"])}
        for i, r in enumerate(R[["match_id", "period", "t", "team"]].itertuples(index=False)):
            s = Sg.get((r.match_id, r.period))
            if s is None:
                continue
            w = (s.t.values >= r.t - 0.01) & (s.t.values <= r.t + h); mine = s.team.values == r.team
            own[i] = s.xg.values[w & mine].sum(); opp[i] = s.xg.values[w & ~mine & (s.t.values > r.t)].sum()
        R[f"own{h}"] = own; R[f"opp{h}"] = opp
    R.to_parquet(os.path.join(PROC, "agent_runners_regains.parquet"))
    print(R.groupby("pop").size())

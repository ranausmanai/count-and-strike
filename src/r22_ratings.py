"""Round 22 strength gap per (match, team): club matches from closing odds (expected-goal difference), tournaments from leave-one-match-out xG difference."""
import difflib, json, os
import numpy as np
import pandas as pd
from r15_360 import PROC, ROOT
LEAGUES = {"La Liga": "La Liga", "1. Bundesliga": "Bundesliga", "Ligue 1": "Ligue 1"}


def market_gap():
    ms = json.load(open(os.path.join(ROOT, "data", "raw", "sb360_matches.json")))
    mk = pd.read_parquet(os.path.join(PROC, "market.parquet")); mk["date"] = pd.to_datetime(mk.date).dt.normalize()
    out = {}
    for m in ms:
        if not any(m["_comp"].startswith(k) for k in LEAGUES):
            continue
        d = pd.Timestamp(m["match_date"]); c = mk[mk.date.between(d - pd.Timedelta(days=1), d + pd.Timedelta(days=1))]
        h, a = m["home_team"]["home_team_name"], m["away_team"]["away_team_name"]
        if c.empty:
            continue
        sc = c.apply(lambda r: difflib.SequenceMatcher(None, h, r.home).ratio() + difflib.SequenceMatcher(None, a, r.away).ratio(), axis=1)
        if sc.max() < 1.0:
            continue
        r = c.loc[sc.idxmax()]
        out[(m["match_id"], "home")] = r.lam_h - r.lam_a
    return out, {m["match_id"]: m["home_team"]["home_team_name"] for m in ms}


def ratings():
    M = pd.read_parquet(os.path.join(PROC, "r22_match_xg.parquet"))
    M["d"] = M.xg_for - M.xg_against
    g = M.groupby(["comp", "team"]).d.agg(["sum", "size"])
    M = M.join(g, on=["comp", "team"])
    M["rating"] = np.where(M["size"] >= 3, (M["sum"] - M.d) / (M["size"] - 1), np.nan)
    o = M[["match_id", "team", "rating"]].rename(columns={"team": "opp", "rating": "opp_rating"})
    M = M.merge(o, on=["match_id", "opp"])
    M["gap"] = M.rating - M.opp_rating; M["gap_src"] = np.where(M.gap.notna(), "xg_loo", None)
    mg, home = market_gap()
    hg = M.match_id.map({k[0]: v for k, v in mg.items()})
    hn = M.match_id.map(home)
    sim = lambda a, b: difflib.SequenceMatcher(None, str(a), str(b)).ratio()
    is_home = np.array([sim(t, h) > sim(o, h) for t, o, h in zip(M.team, M.opp, hn)])
    mgap = np.where(is_home, hg, -hg)
    has = hg.notna().values
    M.loc[has, "gap"] = mgap[has]; M.loc[has, "gap_src"] = "market"
    return M[["match_id", "team", "rating", "opp_rating", "gap", "gap_src"]]


if __name__ == "__main__":
    r = ratings(); print(r.groupby("gap_src", dropna=False).gap.describe())

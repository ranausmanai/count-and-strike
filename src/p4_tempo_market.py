"""P4 (pre-registered): does an underdog's habitual match tempo predict its points residual vs the closing market?

Tempo = rolling mean of total shots (both teams) in the team's previous 10 league matches,
z-scored within league-season. Residual = actual points - market expected points (de-vigged closing odds).
Underdog = side with the lower market win probability. Seasons 2005/06-2025/26.
Cluster-robust SEs by team-season.
"""
import os, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = os.path.join(os.path.dirname(__file__), "..")
mk = pd.read_parquet(os.path.join(ROOT, "data", "proc", "market.parquet"))
mk = mk[(mk.season >= 2005)].dropna(subset=["hs", "as_", "date"]).sort_values("date").reset_index(drop=True)
mk["mid"] = np.arange(len(mk))
mk["tot_shots"] = mk.hs + mk.as_

# team-match long table
long = pd.concat([
    pd.DataFrame(dict(mid=mk.mid, date=mk.date, league=mk.league, season=mk.season, team=mk.home, opp=mk.away, home=1,
                      tot=mk.tot_shots, sf=mk.hs, sa=mk.as_, pwin=mk.pH, ploss=mk.pA, xpts=mk.xpts_h, pts=mk.pts_h)),
    pd.DataFrame(dict(mid=mk.mid, date=mk.date, league=mk.league, season=mk.season, team=mk.away, opp=mk.home, home=0,
                      tot=mk.tot_shots, sf=mk.as_, sa=mk.hs, pwin=mk.pA, ploss=mk.pH, xpts=mk.xpts_a, pts=mk.pts_a))])
long = long.sort_values(["league", "team", "date"]).reset_index(drop=True)
g = long.groupby(["league", "team"])
long["tempo_raw"] = g["tot"].transform(lambda s: s.shift(1).rolling(10, min_periods=10).mean())
long["share_raw"] = g["sf"].transform(lambda s: s.shift(1).rolling(10, min_periods=10).sum()) / \
    (g["tot"].transform(lambda s: s.shift(1).rolling(10, min_periods=10).sum()))
ls = long.groupby(["league", "season"])["tot"]
long["tempo"] = (long.tempo_raw - ls.transform("mean")) / ls.transform("std").clip(lower=1e-6) * np.sqrt(10)  # z of a 10-match mean
# simpler and more interpretable: shots per match deviation from league-season mean
long["tempo_dev"] = long.tempo_raw - ls.transform("mean")
opp = long[["mid", "team", "tempo_dev", "share_raw"]].rename(columns={"team": "opp", "tempo_dev": "opp_tempo_dev", "share_raw": "opp_share"})
long = long.merge(opp, on=["mid", "opp"], how="left")
long["resid"] = long.pts - long.xpts
long["gap"] = long.pwin - long.ploss
long["ts"] = long.team + "_" + long.season.astype(str)

und = long[(long.pwin < long.ploss)].dropna(subset=["tempo_dev", "opp_tempo_dev"]).copy()
print("underdog team-matches:", len(und), "seasons", und.season.min(), und.season.max())

res = {}
for name, d in [("all_underdogs", und), ("heavy_underdogs_p<0.25", und[und.pwin < 0.25])]:
    m = smf.ols("resid ~ tempo_dev + opp_tempo_dev + gap + home + C(league)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.ts)[0]})
    b, se = m.params["tempo_dev"], m.bse["tempo_dev"]
    res[name] = dict(n=int(len(d)), coef_per_shot=b, se=se, ci=[b - 1.96 * se, b + 1.96 * se], p=m.pvalues["tempo_dev"],
                     coef_opp=m.params["opp_tempo_dev"], p_opp=m.pvalues["opp_tempo_dev"],
                     mean_resid=d.resid.mean())
    print(name, json.dumps(res[name], indent=1, default=float))

# also: favourites, as a contrast (exploratory)
fav = long[(long.pwin > long.ploss)].dropna(subset=["tempo_dev", "opp_tempo_dev"])
m = smf.ols("resid ~ tempo_dev + opp_tempo_dev + gap + home + C(league)", data=fav).fit(
    cov_type="cluster", cov_kwds={"groups": pd.factorize(fav.ts)[0]})
res["favourites_exploratory"] = dict(n=int(len(fav)), coef_per_shot=m.params["tempo_dev"], se=m.bse["tempo_dev"], p=m.pvalues["tempo_dev"])
print("fav", res["favourites_exploratory"])

# realised (not habitual) tempo, descriptive: does the underdog do better vs market in low-shot games? (endogenous; exploratory)
und["real_tot_dev"] = und.tot - und.groupby(["league", "season"])["tot"].transform("mean")
m = smf.ols("resid ~ real_tot_dev + gap + home + C(league)", data=und).fit()
res["realised_tempo_descriptive"] = dict(coef_per_shot=m.params["real_tot_dev"], se=m.bse["real_tot_dev"])
print("realised", res["realised_tempo_descriptive"])

# by-season stability
bys = []
for s, d in und.groupby("season"):
    mm = smf.ols("resid ~ tempo_dev + opp_tempo_dev + gap + home", data=d).fit()
    bys.append((int(s), mm.params["tempo_dev"], mm.bse["tempo_dev"]))
res["by_season"] = bys
json.dump(res, open(os.path.join(ROOT, "results", "p4_tempo_market.json"), "w"), indent=1, default=float)
long.to_parquet(os.path.join(ROOT, "data", "proc", "team_matches_fd.parquet"))

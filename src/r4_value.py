"""Exploratory: goals/points value of the 'cone' defensive-structure trait (team level, SB 2015/16)."""
import os, json, sys, numpy as np, pandas as pd, statsmodels.formula.api as smf
sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.join(os.path.dirname(__file__), ".."); PROC = os.path.join(ROOT, "data", "proc")
d = pd.read_parquet(os.path.join(PROC, "r4_shots.parquet")).dropna(subset=["xg"])
tm = pd.read_parquet(os.path.join(PROC, "team_match_channels.parquet"))
gl = pd.read_parquet(os.path.join(PROC, "segments.parquet"), columns=["match_id", "team", "home", "home_score", "away_score"]).drop_duplicates(["match_id", "team"])
gl["ga"] = np.where(gl.home == 1, gl.away_score, gl.home_score)
tm = tm.merge(gl[["match_id", "team", "ga"]], on=["match_id", "team"])
T = tm.groupby(["league", "team"]).agg(rating=("lam_own", lambda s: 0), xga=("xg_total_ag", "mean"), ga=("ga", "mean"), n=("match_id", "size")).reset_index()
T["rating"] = tm.groupby(["league", "team"]).apply(lambda g: np.mean(np.log(g.lam_own / g.lam_opp))).values
T["exp_ga_mkt"] = tm.groupby(["league", "team"]).lam_opp.mean().values
c = d.groupby("opp").agg(cone=("cone_def", "mean"), press=("press_3y", "mean"), xgps=("xg", "mean"), shots=("xg", "size"))
T = T.merge(c, left_on="team", right_index=True)
T["shots_pm"] = T.shots / T.n
T["over"] = T.ga - T.xga
T["over_dm"] = T.over - T.groupby("league").over.transform("mean")
T["ga_vs_mkt"] = T.ga - T.exp_ga_mkt
out = {}
for y in ("xga", "over_dm", "ga", "ga_vs_mkt"):
    m = smf.ols(f"{y} ~ cone + press + rating + C(league)", data=T).fit(cov_type="HC1")
    out[y] = dict(cone=float(m.params["cone"]), se=float(m.bse["cone"]), p=float(m.pvalues["cone"]),
                  press=float(m.params["press"]), p_press=float(m.pvalues["press"]))
    print(y, out[y])
out["cone_sd_between_teams"] = float(T.cone.std()); out["cone_true_sd"] = float(T.cone.std() * np.sqrt(0.75))
print("cone sd", T.cone.std(), "mean", T.cone.mean())
json.dump(out, open(os.path.join(ROOT, "results", "r4_value.json"), "w"), indent=1)

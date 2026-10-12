"""Round 4 follow-ups (exploratory): is H4-3 robust (league-demeaned, wealth), and is it goalkeeping or structure?
Decomposition per defending team (SB 2015/16):
  miss_effect = expected on-target − actual on-target conceded   (defence makes shooters miss/blocked)
  gk_effect   = expected goals from on-target shots − actual goals (keeper)
Expectations from logistic models on shot location features + SB xG, fit league-wide (no team info)."""
import json, os, sys, gzip
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(__file__))
from r4_structure import split_half_team
ROOT = os.path.join(os.path.dirname(__file__), ".."); PROC = os.path.join(ROOT, "data", "proc")
d = pd.read_parquet(os.path.join(PROC, "r4_shots.parquet")).dropna(subset=["xg"])
# on-target flag requires outcome; re-derive from events table
ev = pd.read_parquet(os.path.join(PROC, "events.parquet"), columns=["match_id", "type", "x", "y", "outcome", "sub", "period"])
ev = ev[(ev.type == "Shot") & (ev["sub"] != "Penalty") & (ev.period <= 4)]
ev["on_target"] = ev.outcome.isin(["Goal", "Saved", "Saved To Post", "Saved Off Target"]).astype(int)
d = d.merge(ev[["match_id", "x", "y", "on_target"]].drop_duplicates(["match_id", "x", "y"]), on=["match_id", "x", "y"], how="left").dropna(subset=["on_target"])
d["lx"] = np.log(d.xg.clip(1e-4, 1 - 1e-4) / (1 - d.xg.clip(1e-4, 1 - 1e-4)))
d["dist"] = np.hypot(120 - d.x, 40 - d.y)
F = ["lx", "dist"]
m_ot = LogisticRegression(max_iter=1000).fit(d[F], d.on_target)
d["e_ot"] = m_ot.predict_proba(d[F])[:, 1]
ot = d[d.on_target == 1]
m_g = LogisticRegression(max_iter=1000).fit(ot[F], ot.goal)
d["e_g_given_ot"] = m_g.predict_proba(d[F])[:, 1]
d["miss_eff"] = d.e_ot - d.on_target
d["gk_eff"] = np.where(d.on_target == 1, d.e_g_given_ot - d.goal, 0.0)
d["over"] = d.goal - d.xg
for c in ("over", "miss_eff", "gk_eff"):
    d[c + "_dm"] = d[c] - d.groupby("league")[c].transform("mean")
tm = pd.read_parquet(os.path.join(PROC, "team_match_channels.parquet"))
rating = tm.groupby("team").apply(lambda g: np.mean(np.log(g.lam_own / g.lam_opp))).rename("rating")
out = {}
for c in ("over_dm", "miss_eff_dm", "gk_eff_dm", "cone_def", "press_3y"):
    rel, r = split_half_team(d, "opp", lambda g, c=c: g[c].sum() / g.match_id.nunique())
    r = r.merge(rating, left_on="team", right_index=True)
    out[c] = dict(reliability=rel, wealth_r=float(r[["full", "rating"]].corr().iloc[0, 1]), sd_per_match=float(r.full.std()))
# relation of structure to the outcome components at team level
agg = d.groupby("opp").agg(cone=("cone_def", "mean"), press=("press_3y", "mean"), miss=("miss_eff_dm", "mean"), gk=("gk_eff_dm", "mean"),
                           xgps=("xg", "mean"), over=("over_dm", "mean"))
out["team_corr"] = agg.corr().round(3).to_dict()
for k, v in out.items():
    print(k, v if k != "team_corr" else "")
print(agg.corr().round(2))
json.dump(out, open(os.path.join(ROOT, "results", "r4_followup.json"), "w"), indent=1, default=float)

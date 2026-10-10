"""P3 (pre-registered): in held-out EPL 2015/16, is an underdog's habitual style closer to the P2 prescription
associated with a positive points residual vs the closing market?
Habitual style = mean of the team's match-level z-styles over its previous matches this season (min 5).
Prescription = registered frontier (A) optimal action at 0-0 for a heavy underdog (read from results/p2_policy.json).
Closeness = projection of habitual z-style onto the unit prescription vector."""
import json, os, sys
import numpy as np, pandas as pd, statsmodels.formula.api as smf
sys.path.insert(0, os.path.dirname(__file__))
from frontier import load_segments, StyleScaler, POLICY_KNOBS, ROOT
s = load_segments(min_minutes=0.0).dropna(subset=["lam_own"])
sc = StyleScaler().fit(s.dropna(subset=POLICY_KNOBS), POLICY_KNOBS)
z = sc.z(s); s[[k + "_z" for k in POLICY_KNOBS]] = z.values
s["w"] = s.minutes
agg = s.dropna(subset=[k + "_z" for k in POLICY_KNOBS]).groupby(["match_id", "team"]).apply(
    lambda d: pd.Series({k: np.average(d[k + "_z"], weights=d.w) for k in POLICY_KNOBS})).reset_index()
info = s.drop_duplicates(["match_id", "team"])[["match_id", "team", "league", "date", "pwin", "ploss", "home", "home_score", "away_score"]]
m = info.merge(agg, on=["match_id", "team"]).sort_values("date")
gf = np.where(m.home == 1, m.home_score, m.away_score); ga = np.where(m.home == 1, m.away_score, m.home_score)
m["pts"] = np.where(gf > ga, 3, np.where(gf == ga, 1, 0)); m["xpts"] = 3 * m.pwin + (1 - m.pwin - m.ploss); m["resid"] = m.pts - m.xpts
for k in POLICY_KNOBS:
    m[k + "_hab"] = m.groupby("team")[k].transform(lambda x: x.shift(1).expanding(min_periods=5).mean())
p2 = json.load(open(os.path.join(ROOT, "results", "p2_policy.json")))
v = np.array(p2["A_naive"]["policy_at_0_0_min1"], float); v = v / np.linalg.norm(v)
m["closeness"] = m[[k + "_hab" for k in POLICY_KNOBS]].values @ v
m["gap"] = m.pwin - m.ploss
out = {"prescription": v.tolist()}
for name, d in [("EPL_registered", m[(m.league == "EPL") & (m.pwin < m.ploss)]),
                ("other_leagues_exploratory", m[(m.league != "EPL") & (m.pwin < m.ploss)])]:
    d = d.dropna(subset=["closeness"])
    r = smf.ols("resid ~ closeness + gap + home", data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.team)[0]})
    out[name] = dict(n=len(d), coef=r.params["closeness"], se=r.bse["closeness"], p=r.pvalues["closeness"],
                     ci=[r.params["closeness"] - 1.96 * r.bse["closeness"], r.params["closeness"] + 1.96 * r.bse["closeness"]])
    print(name, out[name])
json.dump(out, open(os.path.join(ROOT, "results", "p3_style_market.json"), "w"), indent=1, default=float)

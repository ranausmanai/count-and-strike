"""Robustness for the tempo habit: market-rating control (SB 2015/16), possession-share control, and points vs market."""
import os, sys, json, numpy as np, pandas as pd, statsmodels.formula.api as smf
sys.path.insert(0, os.path.dirname(__file__))
from r14b_habits import halves
from frontier import load_segments
ROOT = os.path.join(os.path.dirname(__file__), ".."); PROC = os.path.join(ROOT, "data", "proc")
T = pd.read_parquet(os.path.join(PROC, "r14b_team_match.parquet"))
T = T[T.comp.isin(["EPL", "LaLiga", "SerieA", "Ligue1"])].copy()
s = load_segments(min_minutes=0.0).drop_duplicates(["match_id", "team"])
s["rating"] = np.log(s.lam_own / s.lam_opp)
gf = np.where(s.home == 1, s.home_score, s.away_score); ga = np.where(s.home == 1, s.away_score, s.home_score)
s["pts_resid"] = np.where(gf > ga, 3, np.where(gf == ga, 1, 0)) - (3 * s.pwin + (1 - s.pwin - s.ploss))
seg = load_segments(min_minutes=0.0)
ps = seg.groupby(["match_id", "team"]).apply(lambda g: np.average(g.poss_share.fillna(0.5), weights=g.minutes)).rename("poss_share").reset_index()
T = T.merge(s[["match_id", "team", "rating", "pts_resid"]], on=["match_id", "team"]).merge(ps, on=["match_id", "team"])
X = halves(T)
out = {}
cl = lambda d: dict(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.comp + d.team)[0]})
for name, f in [("base", "net_e ~ tempo_z + xgf_o + xga_o + C(comp)"),
                ("+market_rating(both halves)", "net_e ~ tempo_z + xgf_o + xga_o + rating_o + rating_e + C(comp)"),
                ("+rating +possession", "net_e ~ tempo_z + xgf_o + xga_o + rating_o + rating_e + poss_share_o + C(comp)"),
                ("points_vs_market", "pts_resid_e ~ tempo_z + rating_o + C(comp)")]:
    m = smf.ols(f, data=X).fit(**cl(X))
    out[name] = dict(coef=float(m.params["tempo_z"]), se=float(m.bse["tempo_z"]), p=float(m.pvalues["tempo_z"]))
    print(f"{name:32s} {out[name]}")
out["corr_tempo_rating"] = float(X[["tempo_o", "rating_o"]].corr().iloc[0, 1]); out["corr_tempo_poss"] = float(X[["tempo_o", "poss_share_o"]].corr().iloc[0, 1])
print({k: v for k, v in out.items() if k.startswith("corr")})
json.dump(out, open(os.path.join(ROOT, "results", "r14b_tempo_robust.json"), "w"), indent=1)

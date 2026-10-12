"""Exploratory: does a deep defensive block 'bait' the opponent into the high-value regain window?
Per team (SB 2015/16): mean defensive-action height, share of regains made after the opponent reached within 30 yds of our goal
(opp_prev_max_x < 30 in our frame), counter xG per match from regains, controlling for market rating."""
import os, json, sys, numpy as np, pandas as pd, statsmodels.formula.api as smf
sys.path.insert(0, os.path.dirname(__file__))
from frontier import load_segments
ROOT = os.path.join(os.path.dirname(__file__), ".."); PROC = os.path.join(ROOT, "data", "proc")
r = pd.read_parquet(os.path.join(PROC, "r3_regains_sb.parquet"))
s = load_segments(min_minutes=0.0)
h = s.dropna(subset=["def_height"]).groupby("team").apply(lambda g: np.average(g.def_height, weights=g.minutes)).rename("def_height")
rat = s.drop_duplicates(["match_id", "team"]).groupby("team").apply(lambda g: np.mean(np.log(g.lam_own / g.lam_opp))).rename("rating")
nm = s.drop_duplicates(["match_id", "team"]).groupby("team").size().rename("n")
r["deep_window"] = (r.opp_prev_max_x < 30).astype(float)
T = r.groupby("poss_team").agg(win_share=("deep_window", "mean"), regains=("xg20", "size"), xg_regain=("xg20", "sum"),
                               xg_window=("xg20", lambda x: x[r.loc[x.index, "deep_window"] == 1].sum()))
T = T.join(h).join(rat).join(nm).dropna()
T["xg_regain_pm"] = T.xg_regain / T.n; T["xg_window_pm"] = T.xg_window / T.n
out = {}
for y in ("win_share", "xg_window_pm", "xg_regain_pm"):
    m = smf.ols(f"{y} ~ def_height + rating", data=T).fit(cov_type="HC1")
    out[y] = dict(coef_def_height=float(m.params["def_height"]), se=float(m.bse["def_height"]), p=float(m.pvalues["def_height"]),
                  coef_rating=float(m.params["rating"]), mean=float(T[y].mean()))
    print(y, out[y])
out["def_height_sd"] = float(T.def_height.std())
w = r[r.deep_window == 1]; nw = r[r.deep_window == 0]
out["xg_per_regain_window"] = float(w.xg20.mean()); out["xg_per_regain_other"] = float(nw.xg20.mean()); out["share_regains_window"] = float(r.deep_window.mean())
out["share_xg_in_window"] = float(w.xg20.sum() / r.xg20.sum())
print({k: v for k, v in out.items() if not isinstance(v, dict)})
json.dump(out, open(os.path.join(ROOT, "results", "r6_bait.json"), "w"), indent=1)

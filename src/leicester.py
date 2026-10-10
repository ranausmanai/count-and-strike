"""Exploratory: Leicester 2015/16 (the one genuine underdog champion in the event data) vs the EPL field."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from frontier import load_segments, POLICY_KNOBS, StyleScaler, ROOT, PROC
from odds import _probs
tm = pd.read_parquet(os.path.join(PROC, "team_match_channels.parquet"))
e = tm[tm.league == "EPL"].copy()
seg = load_segments(min_minutes=0.0)
s = seg[seg.league == "EPL"].dropna(subset=POLICY_KNOBS + ["poss_share"])
sty = s.groupby("team").apply(lambda d: pd.Series({k: np.average(d[k], weights=d.minutes) for k in POLICY_KNOBS + ["poss_share"]}))
# results & expected points
info = seg[seg.league == "EPL"].drop_duplicates(["match_id", "team"])
gf = np.where(info.home == 1, info.home_score, info.away_score); ga = np.where(info.home == 1, info.away_score, info.home_score)
info = info.assign(pts=np.where(gf > ga, 3, np.where(gf == ga, 1, 0)), xpts_mkt=3 * info.pwin + (1 - info.pwin - info.ploss))
e = e.merge(info[["match_id", "team", "pts", "xpts_mkt"]], on=["match_id", "team"])
def xpts_xg(r):
    h, d, a, _ = _probs(max(r.xg_total, 0.01), max(r.xg_total_ag, 0.01)); return 3 * h + d
e["xpts_xg"] = e.apply(xpts_xg, axis=1)
T = e.groupby("team").agg(pts=("pts", "sum"), xpts_mkt=("xpts_mkt", "sum"), xpts_xg=("xpts_xg", "sum"),
                          xg=("xg_total", "sum"), xga=("xg_total_ag", "sum"), sp=("set_piece", "sum"), sp_ag=("set_piece_ag", "sum"),
                          ctr=("counter", "sum"), ctr_ag=("counter_ag", "sum"), pen=("penalty", "sum"),
                          pre_season_pwin=("pwin", lambda x: x.iloc[:5].mean()))
T["tempo_xg"] = (T.xg + T.xga) / 38; T["sp_share"] = T.sp / T.xg; T["ctr_share"] = T.ctr / T.xg
T = T.join(sty)
for c in ["tempo_xg", "sp_share", "ctr_share", "def_height", "press_rate", "long_share", "poss_share"]:
    T[c + "_rank"] = T[c].rank(ascending=False).astype(int)
T = T.sort_values("pts", ascending=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(T.round(3).to_string())
L = T.loc["Leicester City"].to_dict()
json.dump({"leicester": L, "table": T.reset_index().to_dict(orient="records")}, open(os.path.join(ROOT, "results", "leicester.json"), "w"), indent=1, default=float)

"""Robustness for H18-1/H18-3: also match on pass length band and lane change; separate completion vs value."""
import json, os, sys, numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.join(os.path.dirname(__file__), ".."); PROC = os.path.join(ROOT, "data", "proc")
D = pd.read_parquet(os.path.join(PROC, "r18a_passes.parquet"))
# recover pass length from geometry is not stored; approximate with lane change + zone; add length via end not stored -> use lane_change and complete
D["cell2"] = D.cell + D.lane_change.astype(str)
out = {}
for p in ("discovery", "confirm_club", "confirm_women"):
    d = D[D["pop"] == p].copy(); d["tov"] = d.target_ov.clip(-4, 4)
    for c in ("net", "tov", "own20", "opp20"):
        d[c + "_dm"] = d[c] - d.groupby("cell2")[c].transform("mean")
    r = {}
    for y in ("net", "own20", "opp20"):
        m = smf.ols(f"{y}_dm ~ tov_dm - 1", data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.match_id)[0]})
        r[y] = dict(per_player=float(m.params["tov_dm"]), se=float(m.bse["tov_dm"]), p=float(m.pvalues["tov_dm"]))
    # completion rate by overload (descriptive)
    r["completion_by_target_ov"] = d.groupby(d.tov.clip(-2, 2)).complete.mean().round(3).to_dict()
    out[p] = r; print(p, json.dumps(r, default=float))
json.dump(out, open(os.path.join(ROOT, "results", "r18a_robust.json"), "w"), indent=1, default=float)

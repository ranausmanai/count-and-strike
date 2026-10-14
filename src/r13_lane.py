"""Robustness for confirmed shot-vs-recycle rules: hold the shooting lane (defenders inside the shooter–posts triangle) fixed."""
import os, json, numpy as np, pandas as pd
from scipy.stats import norm
ROOT = os.path.join(os.path.dirname(__file__), ".."); PROC = os.path.join(ROOT, "data", "proc")
A = pd.read_parquet(os.path.join(PROC, "r13_actions.parquet"))
R = json.load(open(os.path.join(ROOT, "results", "r13_miner360.json")))["H13_2"]
rules = [r for r in R if r["confirmed"] and "shot" in (r["action"], r["ref"])]
A["lane_b"] = A.lane.clip(upper=3)
out = []
for r in rules:
    shot_is_a = r["action"] == "shot"; other = r["ref"] if shot_is_a else r["action"]
    res = {"cell": r["cell"], "rule": f"shot vs {other}", "orig_pooled_shot_minus_other": r["pooled"] if shot_is_a else -r["pooled"]}
    for popn in ("discovery", "confirm_club", "confirm_women"):
        d = A[(A["pop"] == popn) & (A.cell_space == r["cell"]) & A.action.isin(["shot", other])]
        g = d.groupby(["lane_b", "action"]).net.agg(["mean", "var", "size"]).unstack("action")
        diffs, ws = [], []
        for lb, row in g.iterrows():
            if row["size"].get("shot", 0) >= 5 and row["size"].get(other, 0) >= 5:
                se2 = row["var"]["shot"] / row["size"]["shot"] + row["var"][other] / row["size"][other]
                diffs.append(row["mean"]["shot"] - row["mean"][other]); ws.append(1 / se2)
        if ws:
            ws = np.array(ws); est = float((ws * np.array(diffs)).sum() / ws.sum()); se = float(1 / np.sqrt(ws.sum()))
            res[popn] = dict(lane_adjusted=est, se=se, p_one=float(norm.sf(est / se)),
                             mean_lane_shot=float(d[d.action == "shot"].lane.mean()), mean_lane_other=float(d[d.action == other].lane.mean()))
    out.append(res); print(json.dumps(res))
json.dump(out, open(os.path.join(ROOT, "results", "r13_lane.json"), "w"), indent=1)

"""Set-piece playbook value under the lower (Wyscout 2017/18) persistence estimates."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from season_sim import simulate, ROOT
sb = json.load(open(os.path.join(ROOT, "results", "channels.json")))["cheap_skill_map"]
wy = json.load(open(os.path.join(ROOT, "results", "wyscout_replication.json")))["cheap_skill_map"]
ss = json.load(open(os.path.join(ROOT, "results", "season_sim.json")))["setpiece_deltas"]
# scale SB-derived deltas by sqrt(rel_wy / rel_sb), component-wise (corners and throws share the 'for' delta ~ 85/15 by volume)
z = 1.2816
sd = lambda k: sb[k]["spread"] * np.sqrt(max(wy[k]["reliability"], 0))
f = z * (sd("corner_xg_per_corner") * ss["corners_pm"] + sd("throw_xg_per_att_throw") * ss["throws_pm"])
a = z * sd("corner_xg_conceded_per_match")
out = {"sp_for_low": f, "sp_against_low": a}
out["setpiece_p90_lowrel"] = simulate(sp_for=f, sp_against=a)
out["setpiece+tempo_lowrel"] = simulate(sp_for=f, sp_against=a, c=0.09)
print(out)
json.dump(out, open(os.path.join(ROOT, "results", "season_sensitivity.json"), "w"), indent=1, default=float)

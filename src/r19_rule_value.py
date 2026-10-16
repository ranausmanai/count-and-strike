"""Round 19 (simulation): what each Count & Strike rule is worth on its own, per club (50% adoption).
Per-match xG values at 50% adoption come from confirmed estimates; Count's in-game uplift is an assumption (labelled)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from season_sim import simulate, ROOT
sens = json.load(open(os.path.join(ROOT, "results", "season_sensitivity.json")))
pb = json.load(open(os.path.join(ROOT, "results", "r15_rules.json")))["confirmed_playbook"]
per_match = lambda fam: sum(d["season_xg_50pct"] for d in pb if fam(d)) / 38
RULES = {
    "Strike through": dict(sp_for=per_match(lambda d: d["better"] == "shot")),
    "Run into space": dict(sp_for=per_match(lambda d: d["better"] == "carry_fwd")),
    "Press the exposed": dict(sp_against=per_match(lambda d: d["family"] == "press")),
    "Play into the overload": dict(sp_for=0.5 * 90 * 0.0017),
    "Count (assumed +0.01 xG/match at 50%)": dict(sp_for=0.01),
    "Dead balls (p90 set pieces)": dict(sp_for=sens["sp_for_low"], sp_against=sens["sp_against_low"]),
    "Match management (model)": dict(c=0.09),
}
allr = dict(sp_for=sum(v.get("sp_for", 0) for v in RULES.values()), sp_against=sum(v.get("sp_against", 0) for v in RULES.values()), c=0.09)
clubs = ["Man City", "Chelsea", "Newcastle", "Brighton", "Brentford", "Leeds", "Sunderland", "Wolves"]
out = {}
for c in clubs:
    base = simulate(n_sims=40_000, team=c)
    row = {"baseline_pts": base["E_pts"]}
    for name, kw in list(RULES.items()) + [("ALL RULES", allr)]:
        r = simulate(n_sims=40_000, team=c, **kw)
        row[name] = dict(pts=r["E_pts"] - base["E_pts"], title=r["p_title"] - base["p_title"], top4=r["p_top4"] - base["p_top4"], releg=r["p_releg"] - base["p_releg"])
    out[c] = row
    print(c, {k: (round(v["pts"], 2) if isinstance(v, dict) else round(v, 1)) for k, v in row.items()})
json.dump(dict(rules={k: v for k, v in RULES.items()}, results=out), open(os.path.join(ROOT, "results", "r19_rule_value.json"), "w"), indent=1, default=float)

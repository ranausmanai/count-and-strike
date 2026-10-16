"""Season value of the full eight-rule system (50% adoption), 2025/26 Premier League, 40k seasons per club.
Rule values from r19_rule_value.py; second ball and third-man are not added (no xG translation estimated) — conservative."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from season_sim import simulate, ROOT
J = json.load(open(os.path.join(ROOT, "results", "r19_rule_value.json")))
R = J["rules"]; allr = dict(sp_for=sum(v.get("sp_for", 0) for v in R.values()), sp_against=sum(v.get("sp_against", 0) for v in R.values()), c=0.09)
out = {}
for c in ["Man City", "Chelsea", "Newcastle", "Brighton", "Brentford", "Leeds", "Sunderland", "Wolves"]:
    b = simulate(n_sims=40_000, team=c); p = simulate(n_sims=40_000, team=c, **allr)
    out[c] = dict(base=b, playbook=p); print(c, round(b["E_pts"], 1), "->", round(p["E_pts"], 1), {k: (round(b[k] * 100, 1), round(p[k] * 100, 1)) for k in ("p_title", "p_top4", "p_releg")})
json.dump(out, open(os.path.join(ROOT, "results", "clubs_playbook_v2.json"), "w"), indent=1)

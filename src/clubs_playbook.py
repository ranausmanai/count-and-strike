"""The general playbook applied to clubs across the 2025/26 Premier League table (simulated seasons).
Levers: set pieces to 90th pct (low-persistence estimate), tempo/match-management policy (±9%), and the
'break after they commit' rule, assumed to lift window-regain xG by 25% (+0.02 xG/match; an assumption, not measured)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from season_sim import simulate, ROOT
sens = json.load(open(os.path.join(ROOT, "results", "season_sensitivity.json")))
spf, spa = sens["sp_for_low"], sens["sp_against_low"]
BREAK = 0.02
clubs = ["Man City", "Chelsea", "Newcastle", "Brighton", "Brentford", "Everton", "Leeds", "Sunderland", "Wolves"]
out = {}
for c in clubs:
    b = simulate(n_sims=100_000, team=c)
    p = simulate(n_sims=100_000, team=c, sp_for=spf + BREAK, sp_against=spa, c=0.09)
    out[c] = dict(base=b, playbook=p)
    print(f"{c:12s} pts {b['E_pts']:.1f} -> {p['E_pts']:.1f} (+{p['E_pts']-b['E_pts']:.1f}) | title {b['p_title']*100:.2f}% -> {p['p_title']*100:.2f}% | "
          f"top4 {b['p_top4']*100:.1f}% -> {p['p_top4']*100:.1f}% | relegation {b['p_releg']*100:.1f}% -> {p['p_releg']*100:.1f}%")
json.dump(out, open(os.path.join(ROOT, "results", "clubs_playbook.json"), "w"), indent=1)

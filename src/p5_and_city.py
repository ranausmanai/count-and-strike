"""P5 at contender strength + single-match case: Brentford vs Man City."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from season_sim import *
out = {}
# P5: risk dial for a contender-strength version (att boost 0.4 ≈ 3.6% title) and Brentford, c = 0.09, common seed
for boost in (0.0, 0.4, 0.6):
    for k in (0, 1, 3, 8):
        r = simulate(n_sims=300_000, c=0.09, kappa=k, att_boost=boost, seed=11)
        out[f"boost{boost}_kappa{k}"] = r
        print(boost, k, {a: round(b, 4) for a, b in r.items()})
# Brentford vs Man City, home and away
teams, mu, home, att, dfn, actual = ratings()
sp = json.load(open(os.path.join(ROOT, "results", "season_sim.json")))["setpiece_deltas"]
g = np.arange(G) - dp.GMAX
def summ(p):
    return dict(W=float(p[g > 0].sum()), D=float(p[g == 0].sum()), L=float(p[g < 0].sum()), E=float(3 * p[g > 0].sum() + p[g == 0].sum()))
for venue, ih in (("home", 1), ("away", 0)):
    lf = rate(mu, home, att, dfn, TEAM, "Man City", ih); la = rate(mu, home, att, dfn, "Man City", TEAM, 1 - ih)
    res = {"lam_brentford": lf, "lam_city": la,
           "baseline": summ(match_dist(lf, la)),
           "always_low_tempo": summ(match_dist(lf, la, 0.09, policy=0)),
           "dp_tempo": summ(match_dist(lf, la, 0.09)),
           "setpiece": summ(match_dist(lf + sp["sp_for"], max(la - sp["sp_against"], .05))),
           "setpiece+dp_tempo": summ(match_dist(lf + sp["sp_for"], max(la - sp["sp_against"], .05), 0.09))}
    # the DP policy itself: mode by goal diff at minutes 1, 45, 70, 85
    m = np.array([0.91, 1.0, 1.09]); RF = np.broadcast_to(lf / K * m, (K, G, 3)).copy(); RA = np.broadcast_to(la / K * m, (K, G, 3)).copy()
    _, pol = dp.solve(RF, RA)
    names = ["low", "normal", "high"]
    res["policy"] = {f"min{t}": {int(gg): names[pol[t, gg + dp.GMAX]] for gg in (-2, -1, 0, 1, 2)} for t in (1, 45, 70, 85)}
    out["city_" + venue] = res
    print(venue, json.dumps(res, indent=0, default=float))
json.dump(out, open(os.path.join(ROOT, "results", "p5_and_city.json"), "w"), indent=1, default=float)

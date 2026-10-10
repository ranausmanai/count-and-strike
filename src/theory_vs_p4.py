"""Compare the empirical P4 tempo coefficient with the Poisson 'variance' prediction, by underdog strength bin.
Theory: scale both teams' goal rates by (1+δ) with share fixed; dE[pts_u]/dδ from the bivariate Poisson (DC rho=-0.06).
One extra habitual shot per match ≈ δ = 1 / (league-season mean total shots)."""
import os, sys, json
import numpy as np, pandas as pd, statsmodels.formula.api as smf
sys.path.insert(0, os.path.dirname(__file__))
from odds import _probs
ROOT = os.path.join(os.path.dirname(__file__), "..")
long = pd.read_parquet(os.path.join(ROOT, "data", "proc", "team_matches_fd.parquet"))
mk = pd.read_parquet(os.path.join(ROOT, "data", "proc", "market.parquet"))[["home", "away", "date", "lam_h", "lam_a"]]
und = long[(long.pwin < long.ploss)].dropna(subset=["tempo_dev", "opp_tempo_dev"]).copy()
und["tot_mean"] = long.groupby(["league", "season"])["tot"].transform("mean").loc[und.index]
# underdog / favourite rates from the market
und = und.merge(mk.rename(columns={"home": "h", "away": "a"}), left_on=["date"], right_on=["date"], how="left")
und = und[((und.home == 1) & (und.h == und.team) & (und.a == und.opp)) | ((und.home == 0) & (und.a == und.team) & (und.h == und.opp))]
und["lu"] = np.where(und.home == 1, und.lam_h, und.lam_a); und["lf"] = np.where(und.home == 1, und.lam_a, und.lam_h)
def epts(lu, lf):
    h, d, a, _ = _probs(lu, lf); return 3 * h + d
eps = 0.02
und["theory_per_shot"] = [(epts(lu * (1 + eps), lf * (1 + eps)) - epts(lu * (1 - eps), lf * (1 - eps))) / (2 * eps) / tm
                          for lu, lf, tm in zip(und.lu, und.lf, und.tot_mean)]
out = {}
bins = [(0, 0.2), (0.2, 0.3), (0.3, 0.45), (0, 0.45)]
for lo, hi in bins:
    d = und[(und.pwin >= lo) & (und.pwin < hi)]
    r = smf.ols("resid ~ tempo_dev + opp_tempo_dev + gap + home + C(league)", data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.ts)[0]})
    out[f"{lo}-{hi}"] = dict(n=len(d), empirical=r.params["tempo_dev"], se=r.bse["tempo_dev"], theory=d.theory_per_shot.mean())
    print(f"pwin {lo}-{hi}", {k: round(v, 4) if isinstance(v, float) else v for k, v in out[f'{lo}-{hi}'].items()})
json.dump(out, open(os.path.join(ROOT, "results", "theory_vs_p4.json"), "w"), indent=1, default=float)

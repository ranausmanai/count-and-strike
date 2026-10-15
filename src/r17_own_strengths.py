"""Exploratory (post-hoc) follow-up to round 17: does leaning into your OWN strong routes pay?
Index = Σ_k (match share_k − usual share_k) × log(own prior attacking strength_k). Same specification as H17-3."""
import json, os, sys, numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(__file__))
from r17_routes import profiles, ROUTES, PROC, ROOT
T = pd.read_parquet(os.path.join(PROC, "r17_team_routes.parquet"))
out = {}
for pop, d0 in T.groupby("pop"):
    d = profiles(d0.copy()); d = d[(d.n_prior >= 8) & d.D_opp.notna() & (d.xg_tot > 0)].copy()
    sh = d[ROUTES].div(d.xg_tot, axis=0).values; ush = d[["ush_" + k for k in ROUTES]].values
    d["own"] = np.nansum((sh - ush) * np.log(d[["att_" + k for k in ROUTES]].values), axis=1)
    d["opp"] = np.nansum((sh - ush) * np.log(d[["odef_" + k for k in ROUTES]].values), axis=1)
    d["exp_xg"] = d[["m_" + k for k in ROUTES]].sum(1) * d.A * d.D_opp; d["y"] = d.xg_tot - d.exp_xg
    for c in ("own", "opp"):
        d[c + "z"] = d[c] / d[c].std()
    d["lA"], d["lD"] = np.log(d.A), np.log(d.D_opp)
    m = smf.ols("y ~ ownz + oppz + lA + lD + home + C(comp)", data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.match_id)[0]})
    out[pop] = {c: dict(coef=float(m.params[c]), se=float(m.bse[c]), p_two=float(m.pvalues[c])) for c in ("ownz", "oppz")}
    print(pop, {k: {a: round(b, 4) for a, b in v.items()} for k, v in out[pop].items()})
json.dump(out, open(os.path.join(ROOT, "results", "r17_own_strengths.json"), "w"), indent=1)

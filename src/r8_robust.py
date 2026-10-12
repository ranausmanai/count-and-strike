"""Round 8 robustness (registered caveat: selection). Same location, within-team, net of counter risk."""
import os, json, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
ROOT = os.path.join(os.path.dirname(__file__), ".."); PROC = os.path.join(ROOT, "data", "proc")
A = pd.read_parquet(os.path.join(PROC, "r8_deadballs.parquet"))
A["net"] = A.xg20 - A.xga30
A["tm"] = A.comp + "|" + A.team.astype(str)
out = {}
def fit(d, y, f, fam="pois"):
    fam_ = sm.families.Poisson() if fam == "pois" else sm.families.Gaussian()
    m0 = smf.glm(f"{y} ~ {f}", data=d, family=fam_).fit()
    dd = d.loc[m0.model.data.row_labels]
    m = smf.glm(f"{y} ~ {f}", data=dd, family=fam_).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.match_id)[0]})
    return dict(coef=float(m.params.iloc[1]), se=float(m.bse.iloc[1]), n=len(d))
for popn in ("discovery", "confirm_SB_EPL", "confirm_WY", "confirm_SB2"):
    P = A[A["pop"] == popn]
    t = P[P.throw_opt.notna() & (P.x >= 100)].copy(); t["lb"] = (t.throw_opt == "long_box").astype(float)
    t["xb"] = pd.cut(t.x, [100, 105, 110, 115, 121]).astype(str); t["yb"] = (np.abs(t.y - 40) > 30).astype(int)
    keep = t.groupby("tm").lb.transform(lambda s: 0 < s.mean() < 1)  # teams that use both
    r = {"share_long": float(t.lb.mean()),
         "xg_same_loc": fit(t, "xg20", "lb + C(xb)"),
         "xg_same_loc_team_fe": fit(t[keep], "xg20", "lb + C(xb) + C(tm)"),
         "net_same_loc_team_fe_ols": fit(t[keep], "net", "lb + C(xb) + C(tm)", "ols"),
         "risk_xga30_same_loc": fit(t, "xga30", "lb + C(xb)")}
    c = P[(P.kind == "corner") & P.zone.isin(["penalty_area", "near"])].copy(); c["pa"] = (c.zone == "penalty_area").astype(float); c["side"] = (c.y < 40).astype(int)
    keepc = c.groupby("tm").pa.transform(lambda s: 0 < s.mean() < 1)
    r.update({"corner_share_pa_of_pa_near": float(c.pa.mean()),
              "corner_xg_team_fe": fit(c[keepc], "xg20", "pa + side + C(tm)"),
              "corner_net_team_fe_ols": fit(c[keepc], "net", "pa + side + C(tm)", "ols")})
    out[popn] = r
    print(popn, json.dumps({k: (v if not isinstance(v, dict) else {a: round(b, 4) for a, b in v.items()}) for k, v in r.items()}))
json.dump(out, open(os.path.join(ROOT, "results", "r8_robust.json"), "w"), indent=1)

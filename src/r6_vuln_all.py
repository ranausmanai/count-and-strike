"""Vulnerability-window effect with identical specification in each population (for the paper figure)."""
import os, json, numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
ROOT = os.path.join(os.path.dirname(__file__), ".."); PROC = os.path.join(ROOT, "data", "proc")
sb = pd.read_parquet(os.path.join(PROC, "r3_regains_sb.parquet")); wy = pd.read_parquet(os.path.join(PROC, "r3_regains_wy.parquet"))
def prep(r):
    r = r.dropna(subset=["opp_prev_max_x", "rx", "xg20"]).copy()
    r["rx"] = r.rx / 120; r["om"] = r.opp_prev_max_x / 120; r["gpos"] = (r.gd > 0).astype(float); r["gneg"] = (r.gd < 0).astype(float); r["clk"] = r.clock / 90
    return r
f = "xg20 ~ rx + I(rx**2) + I(rx**3) + cent + cent:rx + gpos + gneg + clk + om + om:rx + C(league)"
out = {}
for name, d in (("StatsBomb 15/16 (4 leagues, discovery)", prep(sb[sb.league != "EPL"])), ("StatsBomb 15/16 EPL", prep(sb[sb.league == "EPL"])),
                ("Wyscout 17/18 (5 leagues)", prep(wy))):
    m = smf.glm(f, data=d, family=sm.families.Poisson()).fit(scale="X2", cov_type="cluster", cov_kwds={"groups": pd.factorize(d.match_id)[0]})
    names = list(m.params.index); L = np.zeros(len(names)); L[names.index("om")] = 1; L[names.index("om:rx")] = 36 / 120
    est = L @ m.params; se = np.sqrt(L @ m.cov_params().values @ L); k = (15 - 65) / 120
    out[name] = dict(n=len(d), mult=float(np.exp(est * k)), lo=float(np.exp((est + 1.96 * se) * k)), hi=float(np.exp((est - 1.96 * se) * k)))
r6 = json.load(open(os.path.join(ROOT, "results", "r6_confirm.json")))["H6_3"]
k = (15 - 65) / 120
out["WSL/NWSL/ISL (confirmation)"] = dict(n=r6["n_regains"], mult=float(np.exp(r6["marginal_effect_at_rx36"] * k)),
                                           lo=float(np.exp((r6["marginal_effect_at_rx36"] + 1.96 * r6["se"]) * k)), hi=float(np.exp((r6["marginal_effect_at_rx36"] - 1.96 * r6["se"]) * k)))
for kk, v in out.items(): print(kk, {a: round(b, 3) for a, b in v.items()})
json.dump(out, open(os.path.join(ROOT, "results", "r6_vuln_all.json"), "w"), indent=1)

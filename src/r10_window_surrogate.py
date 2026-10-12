"""Is the 'break after they commit' effect visible in the fast surrogate (final-third entry ≤10 s)? SB 2015/16 + SB unseen."""
import os, json, sys, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
sys.path.insert(0, os.path.dirname(__file__))
from possessions import build_possessions
ROOT = os.path.join(os.path.dirname(__file__), ".."); PROC = os.path.join(ROOT, "data", "proc")
def regains(ev):
    ev = ev[ev.period <= 2]
    p = build_possessions(ev[["match_id", "idx", "period", "t", "type", "team", "poss", "poss_team", "pattern", "x", "y", "xg", "outcome", "sub", "league"]])
    own = ev[ev.team == ev.poss_team]
    mx = own.groupby(["match_id", "poss"]).x.max().rename("max_x").reset_index()
    q = p.merge(mx, on=["match_id", "poss"]).sort_values(["match_id", "poss"]).reset_index(drop=True)
    prev = q.groupby("match_id").shift(1)
    q["om"] = (120 - prev.max_x) / 120; q["ok"] = (q.pattern == "Regular Play") & prev.poss_team.notna() & (prev.poss_team != q.poss_team) & (prev.period == q.period)
    r = q[q.ok & (q.start_x < 80)].copy()
    o2 = own.merge(r[["match_id", "poss", "t0"]], on=["match_id", "poss"]); o2["dt"] = (o2.t - o2.t0) * 60
    ft = o2[(o2.x >= 80) & (o2.dt <= 10)].groupby(["match_id", "poss"]).size().rename("n")
    r = r.merge(ft.reset_index(), on=["match_id", "poss"], how="left"); r["ft10"] = r.n.notna().astype(float)
    r["rx"] = r.start_x / 120; r["cent"] = 1 - (r.start_y - 40).abs() / 40; r["clk"] = (r.t0 + np.where(r.period == 2, 45, 0)) / 90
    return r.dropna(subset=["om"])
out = {}
cols = ["match_id", "idx", "period", "t", "type", "team", "poss", "poss_team", "pattern", "x", "y", "xg", "outcome", "sub", "league"]
for name, f in (("SB_1516", "events.parquet"), ("SB_unseen", "sb2_events.parquet")):
    r = regains(pd.read_parquet(os.path.join(PROC, f), columns=cols))
    m = smf.glm("ft10 ~ rx + I(rx**2) + I(rx**3) + cent + cent:rx + clk + om + om:rx + C(league)", data=r, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(r.match_id)[0]})
    nm = list(m.params.index); L = np.zeros(len(nm)); L[nm.index("om")] = 1; L[nm.index("om:rx")] = 36 / 120
    est = L @ m.params; se = np.sqrt(L @ m.cov_params().values @ L); k = (15 - 65) / 120
    out[name] = dict(n=len(r), ft10_rate=float(r.ft10.mean()), odds_ratio_deep_vs_shallow=float(np.exp(est * k)),
                     lo=float(np.exp((est + 1.96 * se) * k)), hi=float(np.exp((est - 1.96 * se) * k)))
    print(name, out[name])
json.dump(out, open(os.path.join(ROOT, "results", "r10_window_surrogate.json"), "w"), indent=1)

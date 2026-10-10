"""Exploratory: does habitual set-piece xG (for minus against, previous matches, min 8) predict points residual vs market?"""
import os, json, numpy as np, pandas as pd, statsmodels.formula.api as smf
ROOT = os.path.join(os.path.dirname(__file__), "..")
mk = pd.read_parquet(os.path.join(ROOT, "data", "proc", "market.parquet"))
out = {}
for name, f, season in (("statsbomb_1516", "team_match_channels.parquet", 2015), ("wyscout_1718", "wy_team_match_channels.parquet", 2017)):
    t = pd.read_parquet(os.path.join(ROOT, "data", "proc", f)).copy()
    t["date"] = pd.to_datetime(t.date); t = t.sort_values("date")
    t["sp_net"] = t.set_piece - t.set_piece_ag; t["op_net"] = t.open_play - t.open_play_ag
    for c in ("sp_net", "op_net"):
        t[c + "_hab"] = t.groupby("team")[c].transform(lambda x: x.shift(1).expanding(min_periods=8).mean())
    # results from matching market rows
    if "ploss" not in t:
        t["ploss"] = np.nan
    m = mk[mk.season == season]
    t = t.merge(m[["lam_h", "lam_a", "pH", "pD", "pA", "pts_h", "pts_a"]], left_on=["lam_own" if False else "lam_own"], right_on=["lam_h"], how="left") if False else t
    # recover points from the market table via (lam_own, lam_opp) pairs
    key_h = m.set_index(["lam_h", "lam_a"]); 
    rows = []
    for r in t.itertuples():
        k = (r.lam_own, r.lam_opp) if r.home == 1 else (r.lam_opp, r.lam_own)
        try:
            row = key_h.loc[k]; row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
            pts = row.pts_h if r.home == 1 else row.pts_a; xp = row.xpts_h if r.home == 1 else row.xpts_a
            pw, pl = (row.pH, row.pA) if r.home == 1 else (row.pA, row.pH)
        except KeyError:
            pts = xp = pw = pl = np.nan
        rows.append((pts, xp, pw, pl))
    t[["pts", "xpts", "pw", "pl"]] = rows
    t["resid"] = t.pts - t.xpts; t["gap"] = t.pw - t.pl
    d = t.dropna(subset=["sp_net_hab", "resid"])
    r = smf.ols("resid ~ sp_net_hab + op_net_hab + gap + home", data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.team)[0]})
    out[name] = dict(n=len(d), coef_sp=r.params["sp_net_hab"], se_sp=r.bse["sp_net_hab"], p_sp=r.pvalues["sp_net_hab"],
                     coef_op=r.params["op_net_hab"], se_op=r.bse["op_net_hab"], sd_sp_hab=d.sp_net_hab.std())
    print(name, {k: round(v, 4) for k, v in out[name].items()})
json.dump(out, open(os.path.join(ROOT, "results", "setpiece_market.json"), "w"), indent=1, default=float)

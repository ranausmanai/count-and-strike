"""Round 3-B: opponent weak-spot matchups (PREREGISTRATION_3.md, H3b-1 and H3b-2).

Profiles are built from PRIOR matches only (expanding, shrunk toward 1 with k=5 pseudo-matches):
  att_ratio[c]  = attacker's observed channel-c xG / baseline-expected, prior matches
  def_ratio[c]  = defender's observed channel-c xG conceded / baseline-expected, prior matches
Baseline expectation per channel = Poisson GLM on log market strengths (fit on the training leagues).
M0: baseline + overall attack/defence form (all channels pooled)
M1: M0 + channel-specific profile deviations (the weak-spot terms)
Primary test fold: EPL (train on the other leagues of the same dataset).
"""
import json, os, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
sys.path.insert(0, os.path.dirname(__file__))
from frontier import poisson_dev
ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")
CHS = ["build_up", "counter", "high_regain", "corner", "free_kick", "throw_in"]
K = 5.0


def long_table(t):
    rows = []
    for c in CHS:
        d = t[["match_id", "team", "opp", "home", "date", "league", "lam_own", "lam_opp", c]].rename(columns={c: "y"})
        d["ch"] = c
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def add_baseline(L, train_mask):
    L["mu0"] = np.nan
    for c in CHS:
        m = (L.ch == c)
        tr = L[m & train_mask]
        X = sm.add_constant(np.column_stack([np.log(tr.lam_own), np.log(tr.lam_opp), tr.home]))
        f = sm.GLM(tr.y.values, X, family=sm.families.Poisson()).fit()
        Xa = sm.add_constant(np.column_stack([np.log(L[m].lam_own), np.log(L[m].lam_opp), L[m].home]))
        L.loc[m, "mu0"] = Xa @ f.params
    L["mu0"] = np.exp(L.mu0)
    return L


def prior_ratio(L, who, key):
    """Expanding prior-match ratio of observed/expected for `who` ('team' attacking or 'opp' as defender), by channel or pooled."""
    L = L.sort_values("date")
    grp = [who] + (["ch"] if key == "ch" else [])
    g = L.groupby(grp)
    if key == "ch":
        so = g.y.transform(lambda s: s.shift(1).expanding().sum()).fillna(0)
        se = g.mu0.transform(lambda s: s.shift(1).expanding().sum()).fillna(0)
        n = g.y.transform(lambda s: s.shift(1).expanding().count()).fillna(0)
    else:
        # pooled over channels, per match: aggregate first
        agg = L.groupby([who, "match_id", "date"])[["y", "mu0"]].sum().reset_index().sort_values("date")
        gg = agg.groupby(who)
        agg["so"] = gg.y.transform(lambda s: s.shift(1).expanding().sum()).fillna(0)
        agg["se"] = gg.mu0.transform(lambda s: s.shift(1).expanding().sum()).fillna(0)
        agg["n"] = gg.y.transform(lambda s: s.shift(1).expanding().count()).fillna(0)
        m = L[[who, "match_id"]].merge(agg[[who, "match_id", "so", "se", "n"]], on=[who, "match_id"], how="left")
        so, se, n = m.so.values, m.se.values, m.n.values
        so = pd.Series(so, index=L.index); se = pd.Series(se, index=L.index); n = pd.Series(n, index=L.index)
    mean_e = (se / n.replace(0, np.nan)).fillna(L.mu0)
    ratio = (so + K * mean_e) / (se + K * mean_e)
    return ratio, n


def run(t, name):
    t = t.dropna(subset=["lam_own"]).copy()
    t["date"] = pd.to_datetime(t.date)
    L = long_table(t)
    train = L.league != "EPL"
    L = add_baseline(L, train)
    L = L.sort_values("date").reset_index(drop=True)
    L["att_all"], n1 = prior_ratio(L, "team", "all")
    L["def_all"], _ = prior_ratio(L, "opp", "all")
    L["att_ch"], n2 = prior_ratio(L, "team", "ch")
    L["def_ch"], _ = prior_ratio(L, "opp", "ch")
    L["n_prior"] = n2
    L = L[L.n_prior >= 5]
    L["x_att_all"] = np.log(L.att_all); L["x_def_all"] = np.log(L.def_all)
    L["x_att_dev"] = np.log(L.att_ch / L.att_all); L["x_def_dev"] = np.log(L.def_ch / L.def_all)
    off = np.log(L.mu0)
    out = {"n_rows": int(len(L))}
    fits = {}
    for mname, cols in (("M0", ["x_att_all", "x_def_all"]), ("M1", ["x_att_all", "x_def_all", "x_att_dev", "x_def_dev"])):
        tr = L[L.league != "EPL"]
        X = sm.add_constant(tr[cols].values)
        fits[mname] = (sm.GLM(tr.y.values, X, family=sm.families.Poisson(), offset=np.log(tr.mu0).values).fit(scale="X2"), cols)
        out[mname + "_coefs"] = dict(zip(["const"] + cols, [[float(p), float(s)] for p, s in zip(fits[mname][0].params, fits[mname][0].bse)]))
    te = L[L.league == "EPL"]
    devs = {}
    for mname, (f, cols) in fits.items():
        mu = np.exp(sm.add_constant(te[cols].values, has_constant="add") @ f.params + np.log(te.mu0).values)
        devs[mname] = poisson_dev(te.y, mu)
    per = pd.DataFrame({"m": te.match_id.values, "d": devs["M0"] - devs["M1"]}).groupby("m").d.sum()
    rng = np.random.default_rng(20261012)
    bs = [per.values[rng.integers(0, len(per), len(per))].sum() for _ in range(2000)]
    out["EPL_gain_M1_vs_M0"] = float(per.sum()); out["ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    out["rel_gain"] = float(per.sum() / devs["M0"].sum())
    # H3b-1: split-half reliability of strength-adjusted concession shares
    t2 = L.drop_duplicates(["match_id", "opp", "ch"])
    rel = {}
    for c in CHS:
        rows = []
        for (lg, tm), d in t2[t2.ch == c].sort_values("date").groupby(["league", "opp"]):
            allm = t2[(t2.opp == tm) & (t2.league == lg)].sort_values("date")
            mids = allm.match_id.unique()
            if len(mids) < 20:
                continue
            a, b = set(mids[0::2]), set(mids[1::2])
            def share(ms):
                x = allm[allm.match_id.isin(ms)]
                return (x[x.ch == c].y.sum() / x[x.ch == c].mu0.sum()) / (x.y.sum() / x.mu0.sum())
            rows.append((share(a), share(b)))
        r = np.corrcoef(np.array(rows).T)[0, 1]
        rel[c] = float(2 * r / (1 + r))
    out["concession_profile_reliability"] = rel
    print(name, json.dumps({k: v for k, v in out.items()}, indent=0, default=float))
    return out


if __name__ == "__main__":
    res = {}
    res["SB_1516"] = run(pd.read_parquet(os.path.join(PROC, "team_match_channels.parquet")), "SB")
    res["WY_1718"] = run(pd.read_parquet(os.path.join(PROC, "wy_team_match_channels.parquet")), "WY")
    json.dump(res, open(os.path.join(ROOT, "results", "r3b_weakspots.json"), "w"), indent=1, default=float)

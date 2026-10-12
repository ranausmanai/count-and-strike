"""Round 5: confirm the cone-density candidate (PREREGISTRATION_5.md)."""
import gzip, json, os, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(__file__))
from r4_structure import in_tri
ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")


def shots(path):
    mid = int(os.path.basename(path).split(".")[0])
    ev = json.load(gzip.open(path))
    out = []
    for e in ev:
        if e["type"]["name"] != "Shot" or e["period"] > 4:
            continue
        s = e["shot"]
        if s["type"]["name"] == "Penalty" or "freeze_frame" not in s:
            continue
        x, y = e["location"][:2]
        de = [f["location"] for f in s["freeze_frame"] if not f["teammate"] and f["position"]["name"] != "Goalkeeper"]
        cone = sum(in_tri(l, (x, y), (120, 36), (120, 44)) for l in de)
        out.append(dict(match_id=mid, team_id=e["team"]["id"], x=x, y=y, xg=s.get("statsbomb_xg"), goal=int(s["outcome"]["name"] == "Goal"),
                        head=int(s["body_part"]["name"] == "Head"), cone_def=cone))
    return out


def load(raw_dir, matches, comp_key):
    paths = [os.path.join(raw_dir, f"{m['match_id']}.json.gz") for m in matches]
    rows = []
    with ProcessPoolExecutor(8) as ex:
        for r in ex.map(shots, paths, chunksize=8):
            rows += r
    d = pd.DataFrame(rows)
    mi = {m["match_id"]: m for m in matches}
    d["comp"] = [mi[m][comp_key] for m in d.match_id]
    d["date"] = [mi[m]["match_date"] for m in d.match_id]
    d["shoot_team"] = [mi[m]["home_team"]["home_team_name"] if t == mi[m]["home_team"]["home_team_id"] else mi[m]["away_team"]["away_team_name"]
                       for m, t in zip(d.match_id, d.team_id)]
    d["def_team"] = [mi[m]["away_team"]["away_team_name"] if t == mi[m]["home_team"]["home_team_id"] else mi[m]["home_team"]["home_team_name"]
                     for m, t in zip(d.match_id, d.team_id)]
    return d


def add_excess(d):
    d["dist"] = np.hypot(120 - d.x, 40 - d.y); d["ang"] = np.abs(np.arctan2(d.y - 40, 120 - d.x))
    d["cone_excess"] = np.nan
    for c, g in d.groupby("comp"):
        X = sm.add_constant(np.column_stack([g.dist, g.dist ** 2, g.ang, g["head"]]))
        f = sm.GLM(g.cone_def.values, X, family=sm.families.Poisson()).fit()
        d.loc[g.index, "cone_excess"] = g.cone_def - f.predict(X)
    return d


def team_matches(matches, comp_key):
    rows = []
    for m in matches:
        for side, o in (("home", "away"), ("away", "home")):
            rows.append(dict(match_id=m["match_id"], comp=m[comp_key], date=m["match_date"], team=m[f"{side}_team"][f"{side}_team_name"],
                             ga=m[f"{o}_score"]))
    return pd.DataFrame(rows)


def halves(tm):
    tm = tm.sort_values("date").copy()
    tm["k"] = tm.groupby(["comp", "team"]).cumcount(); tm["n"] = tm.groupby(["comp", "team"]).match_id.transform("size")
    tm["half"] = np.where(tm.k < tm.n // 2, 1, 2)
    return tm


def main():
    out = {}
    # ---- SB 2015/16 (H5-1) ----
    ms1 = json.load(open(os.path.join(ROOT, "data", "raw", "sb_matches.json")))
    ms1 = [m for m in ms1 if m["_league"] != "Bundesliga"]
    d1 = add_excess(load(os.path.join(ROOT, "data", "raw", "sb"), ms1, "_league"))
    tm1 = halves(team_matches(ms1, "_league"))
    ch = pd.read_parquet(os.path.join(PROC, "team_match_channels.parquet"))[["match_id", "team", "lam_own", "lam_opp", "xg_total", "xg_total_ag"]]
    tm1 = tm1.merge(ch, on=["match_id", "team"], how="left")
    sh = d1.merge(tm1[["match_id", "team", "half"]].rename(columns={"team": "def_team"}), on=["match_id", "def_team"])
    trait = sh[sh.half == 1].groupby(["comp", "def_team"]).cone_excess.mean().rename("trait")
    a = tm1[tm1.half == 1].groupby(["comp", "team"]).apply(lambda g: np.mean(np.log(g.lam_own / g.lam_opp))).rename("rating")
    b = tm1[tm1.half == 2].groupby(["comp", "team"]).apply(lambda g: np.mean(g.ga - g.lam_opp)).rename("ga_vs_mkt")
    b2 = tm1[tm1.half == 2].groupby(["comp", "team"]).apply(lambda g: np.mean(g.ga - g.xg_total_ag)).rename("ga_vs_xg")
    T = pd.concat([trait.rename_axis(["comp", "team"]), a, b, b2], axis=1).dropna().reset_index()
    m = smf.ols("ga_vs_mkt ~ trait + rating + C(comp)", data=T).fit(cov_type="HC1")
    m2 = smf.ols("ga_vs_xg ~ trait + rating + C(comp)", data=T).fit(cov_type="HC1")
    out["H5_1"] = dict(n_teams=len(T), coef=float(m.params["trait"]), se=float(m.bse["trait"]), p_one=float(norm.cdf(m.params["trait"] / m.bse["trait"])),
                       secondary_ga_vs_xg=dict(coef=float(m2.params["trait"]), se=float(m2.bse["trait"]), p_one=float(norm.cdf(m2.params["trait"] / m2.bse["trait"]))),
                       trait_sd=float(T.trait.std()))
    print("H5-1", out["H5_1"])
    # ---- unseen populations (H5-2, H5-3) ----
    ms2 = json.load(open(os.path.join(ROOT, "data", "raw", "sb2_matches.json")))
    d2 = add_excess(load(os.path.join(ROOT, "data", "raw", "sb2"), ms2, "_comp"))
    tm2 = halves(team_matches(ms2, "_comp"))
    xga = d2.groupby(["match_id", "def_team"]).xg.sum().rename("xga").reset_index().rename(columns={"def_team": "team"})
    xgf = d2.groupby(["match_id", "shoot_team"]).xg.sum().rename("xgf").reset_index().rename(columns={"shoot_team": "team"})
    tm2 = tm2.merge(xga, on=["match_id", "team"], how="left").merge(xgf, on=["match_id", "team"], how="left").fillna({"xga": 0, "xgf": 0})
    sh2 = d2.merge(tm2[["match_id", "team", "half"]].rename(columns={"team": "def_team"}), on=["match_id", "def_team"])
    trait2 = sh2[sh2.half == 1].groupby(["comp", "def_team"]).cone_excess.mean().rename("trait").rename_axis(["comp", "team"])
    q = tm2[tm2.half == 1].groupby(["comp", "team"]).apply(lambda g: np.mean(g.xgf - g.xga)).rename("xgd1")
    o = tm2[tm2.half == 2].groupby(["comp", "team"]).apply(lambda g: np.mean(g.ga - g.xga)).rename("ga_vs_xg")
    T2 = pd.concat([trait2, q, o], axis=1).dropna().reset_index()
    m = smf.ols("ga_vs_xg ~ trait + xgd1 + C(comp)", data=T2).fit(cov_type="HC1")
    out["H5_2"] = dict(n_teams=len(T2), coef=float(m.params["trait"]), se=float(m.bse["trait"]), p_one=float(norm.cdf(m.params["trait"] / m.bse["trait"])),
                       by_comp={c: float(smf.ols("ga_vs_xg ~ trait + xgd1", data=g).fit().params["trait"]) for c, g in T2.groupby("comp")})
    print("H5-2", out["H5_2"])
    s = d2.dropna(subset=["xg"]).copy()
    s["lx"] = np.log(s.xg.clip(1e-4, 1 - 1e-4) / (1 - s.xg.clip(1e-4, 1 - 1e-4)))
    g = smf.glm("goal ~ lx + cone_excess", data=s, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s.match_id)[0]})
    out["H5_3"] = dict(n_shots=len(s), coef=float(g.params["cone_excess"]), se=float(g.bse["cone_excess"]), p_one=float(norm.cdf(g.params["cone_excess"] / g.bse["cone_excess"])))
    # same shot-level model on SB 2015/16 for reference
    s1 = d1.dropna(subset=["xg"]).copy(); s1["lx"] = np.log(s1.xg.clip(1e-4, 1 - 1e-4) / (1 - s1.xg.clip(1e-4, 1 - 1e-4)))
    g1 = smf.glm("goal ~ lx + cone_excess", data=s1, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s1.match_id)[0]})
    out["shot_level_SB1516_reference"] = dict(coef=float(g1.params["cone_excess"]), se=float(g1.bse["cone_excess"]))
    print("H5-3", out["H5_3"], "ref", out["shot_level_SB1516_reference"])
    json.dump(out, open(os.path.join(ROOT, "results", "r5_confirm.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

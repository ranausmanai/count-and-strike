"""Round 18B: formation matchups (PREREGISTRATION_18.md)."""
import gzip, json, os, sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import norm
ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")


def one(args):
    path, mid, pop, comp, date, home, away, hid, aid = args
    try:
        ev = json.load(gzip.open(path))
    except Exception:
        return None
    form = {}; xg = {hid: 0.0, aid: 0.0}; goals = {hid: 0, aid: 0}
    for e in ev:
        t = e["type"]["name"]
        if t == "Starting XI":
            form[e["team"]["id"]] = str(e["tactics"]["formation"])
        elif t == "Shot" and e["period"] <= 4 and e["shot"]["type"]["name"] != "Penalty":
            xg[e["team"]["id"]] = xg.get(e["team"]["id"], 0) + e["shot"].get("statsbomb_xg", 0.0)
        if (t == "Shot" and e["period"] <= 4 and e["shot"]["outcome"]["name"] == "Goal") or t == "Own Goal For":
            goals[e["team"]["id"]] = goals.get(e["team"]["id"], 0) + 1
    if hid not in form or aid not in form:
        return None
    return [dict(pop=pop, comp=comp, match_id=mid, date=date, team=home, opp=away, home=1, f=form[hid], fo=form[aid], xgf=xg[hid], xga=xg[aid], gf=goals[hid], ga=goals[aid]),
            dict(pop=pop, comp=comp, match_id=mid, date=date, team=away, opp=home, home=0, f=form[aid], fo=form[hid], xgf=xg[aid], xga=xg[hid], gf=goals[aid], ga=goals[hid])]


def build():
    args = []
    for mfile, raw, pop, key, fmt in (("sb_matches.json", "sb", "SB_1516", "_league", "{}.json.gz"), ("sb2_matches.json", "sb2", "SB_women_ISL", "_comp", "{}.json.gz"),
                                      ("sb360_matches.json", "sb360", "SB_360comps", "_comp", "{}_events.json.gz")):
        for m in json.load(open(os.path.join(ROOT, "data", "raw", mfile))):
            args.append((os.path.join(ROOT, "data", "raw", raw, fmt.format(m["match_id"])), m["match_id"], pop, m[key], m["match_date"],
                         m["home_team"]["home_team_name"], m["away_team"]["away_team_name"], m["home_team"]["home_team_id"], m["away_team"]["away_team_id"]))
    rows = []
    with ProcessPoolExecutor(8) as ex:
        for r in ex.map(one, args, chunksize=8):
            if r:
                rows += r
    T = pd.DataFrame(rows).drop_duplicates(["match_id", "team"])
    T.to_parquet(os.path.join(PROC, "r18b_formations.parquet"))
    return T


def families(T):
    T["back3"] = T.f.str[0].isin(["3", "5"]).astype(int); T["oback3"] = T.fo.str[0].isin(["3", "5"]).astype(int)
    st = lambda s: np.where(s.str[-1] == "1", "1", np.where(s.str[-1] == "2", "2", "3"))
    T["str"] = st(T.f); T["ostr"] = st(T.fo)
    # strength: leave-this-match-out xG for/against within comp
    g = T.groupby(["comp", "team"]); n = g.match_id.transform("size")
    T["att"] = (g.xgf.transform("sum") - T.xgf) / (n - 1).replace(0, np.nan); T["dfn"] = (g.xga.transform("sum") - T.xga) / (n - 1).replace(0, np.nan)
    o = T[["match_id", "team", "att", "dfn"]].rename(columns={"team": "opp", "att": "oatt", "dfn": "odfn"})
    T = T.merge(o, on=["match_id", "opp"])
    T["exp_xgd"] = (T.att + T.odfn) / 2 - (T.oatt + T.dfn) / 2
    T["y"] = (T.xgf - T.xga) - T.exp_xgd
    return T.dropna(subset=["y"])


def main():
    T = build() if not os.path.exists(os.path.join(PROC, "r18b_formations.parquet")) else pd.read_parquet(os.path.join(PROC, "r18b_formations.parquet"))
    T = families(T)
    print(T.groupby("pop").size().to_dict(), "| back-3 share:", T.groupby("pop").back3.mean().round(3).to_dict())
    print(T.groupby("pop").f.value_counts(normalize=True).groupby(level=0).head(5).round(3))
    res = {}
    base = "y ~ back3 + C(str) + oback3 + C(ostr) + home"
    full = base + " + back3:C(ostr) + C(str):oback3"
    from frontier import poisson_dev  # noqa (not used; OLS here)
    for pop, d in T.groupby("pop"):
        comps = d.comp.unique()
        gains = []
        if len(comps) >= 3:
            for c in comps:
                tr, te = d[d.comp != c], d[d.comp == c]
                try:
                    mb = smf.ols(base, data=tr).fit(); mf = smf.ols(full, data=tr).fit()
                    eb = (te.y - mb.predict(te)) ** 2; ef = (te.y - mf.predict(te)) ** 2
                    gains.append(pd.DataFrame(dict(m=te.match_id, g=eb - ef)))
                except Exception:
                    pass
        else:  # few competitions: 5-fold by match
            mids = d.match_id.unique(); rng = np.random.default_rng(1); fold = dict(zip(mids, rng.integers(0, 5, len(mids))))
            d = d.assign(fold=d.match_id.map(fold))
            for k in range(5):
                tr, te = d[d.fold != k], d[d.fold == k]
                mb = smf.ols(base, data=tr).fit(); mf = smf.ols(full, data=tr).fit()
                gains.append(pd.DataFrame(dict(m=te.match_id, g=(te.y - mb.predict(te)) ** 2 - (te.y - mf.predict(te)) ** 2)))
        G = pd.concat(gains).groupby("m").g.sum(); rng = np.random.default_rng(20261015)
        bs = [G.values[rng.integers(0, len(G), len(G))].sum() for _ in range(2000)]
        # H18-5: back three vs two strikers, relative to back four vs two strikers
        s = d[d.ostr == "2"]
        m5 = smf.ols("y ~ back3 + home", data=s).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s.match_id)[0]})
        res[pop] = dict(n=int(len(d)), H18_4_oos_gain=float(G.sum()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                        H18_5_back3_vs_2strikers=dict(coef=float(m5.params["back3"]), se=float(m5.bse["back3"]), p_one=float(norm.sf(m5.params["back3"] / m5.bse["back3"])),
                                                      n=int(len(s)), n_back3=int(s.back3.sum())))
        print(pop, json.dumps(res[pop], default=float))
    json.dump(res, open(os.path.join(ROOT, "results", "r18b_formations.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

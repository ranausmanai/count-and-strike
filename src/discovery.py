"""Round 2 discovery engine (PREREGISTRATION_2.md).

  python3 src/discovery.py search     -> discovery seasons only (2005-2016); writes results/r2_shortlist.json
  python3 src/discovery.py confirm    -> reads the committed shortlist, evaluates on the locked holdout (2017-2025)
  python3 src/discovery.py ml         -> LightGBM arm (fit on discovery, evaluated once on holdout)
"""
import glob, json, os, sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = os.path.join(os.path.dirname(__file__), "..")
FD = os.path.join(ROOT, "data", "raw", "fd")
LEAGUE = {"E0": "EPL", "SP1": "LaLiga", "D1": "Bundesliga", "I1": "SerieA", "F1": "Ligue1"}
TRAITS = ["shots_for", "shots_against", "sot_rate_for", "sot_rate_against", "fouls_committed", "fouls_won",
          "corners_for", "corners_against", "yellows", "gd_resid_form"]
DISC = (2005, 2016); HOLD = (2017, 2025)


def stats_table():
    fr = []
    for f in glob.glob(os.path.join(FD, "*.csv")):
        code, s = os.path.basename(f)[:-4].split("_")
        season = 2000 + int(s[:2]) if int(s[:2]) < 50 else 1900 + int(s[:2])
        if season < 2004:
            continue
        d = pd.read_csv(f, encoding="latin-1", on_bad_lines="skip").dropna(subset=["HomeTeam", "FTHG"])
        cols = ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY"]
        if not all(c in d for c in cols):
            continue
        d = d[["Date", "HomeTeam", "AwayTeam"] + cols].copy()
        d["date"] = pd.to_datetime(d.Date, dayfirst=True, errors="coerce")
        d["league"] = LEAGUE[code]; d["season"] = season
        fr.append(d.rename(columns={"HomeTeam": "home", "AwayTeam": "away"}).drop(columns="Date"))
    return pd.concat(fr, ignore_index=True)


def build():
    mk = pd.read_parquet(os.path.join(ROOT, "data", "proc", "market.parquet"))
    st = stats_table()
    mk = mk.merge(st, on=["league", "season", "home", "away", "date"], how="inner")
    for c in ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY"]:
        mk[c] = pd.to_numeric(mk[c], errors="coerce")
    mk = mk.dropna(subset=["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "lam_h"]).sort_values("date").reset_index(drop=True)
    mk["mid"] = np.arange(len(mk))
    def side(h):
        p, o = ("H", "A") if h else ("A", "H")
        return pd.DataFrame(dict(mid=mk.mid, date=mk.date, league=mk.league, season=mk.season,
                                 team=mk.home if h else mk.away, opp=mk.away if h else mk.home, home=int(h),
                                 gf=mk.hg if h else mk.ag, ga=mk.ag if h else mk.hg,
                                 lam_f=mk.lam_h if h else mk.lam_a, lam_a=mk.lam_a if h else mk.lam_h,
                                 pwin=mk.pH if h else mk.pA, ploss=mk.pA if h else mk.pH,
                                 xpts=mk.xpts_h if h else mk.xpts_a, pts=mk.pts_h if h else mk.pts_a,
                                 shots_for=mk[p + "S"], shots_against=mk[o + "S"], sot_for=mk[p + "ST"], sot_against=mk[o + "ST"],
                                 fouls_committed=mk[p + "F"], fouls_won=mk[o + "F"], corners_for=mk[p + "C"], corners_against=mk[o + "C"],
                                 yellows=mk[p + "Y"]))
    L = pd.concat([side(True), side(False)], ignore_index=True).sort_values(["league", "team", "date"])
    L["gd_resid"] = (L.gf - L.ga) - (L.lam_f - L.lam_a)
    g = L.groupby(["league", "team"])
    roll = lambda c: g[c].transform(lambda s: s.shift(1).rolling(10, min_periods=10).sum())
    H = pd.DataFrame(index=L.index)
    for c in ["shots_for", "shots_against", "fouls_committed", "fouls_won", "corners_for", "corners_against", "yellows", "gd_resid"]:
        H[c] = roll(c) / 10
    H["sot_rate_for"] = roll("sot_for") / roll("shots_for")
    H["sot_rate_against"] = roll("sot_against") / roll("shots_against")
    H = H.rename(columns={"gd_resid": "gd_resid_form"})
    for c in TRAITS:
        L[c] = H[c] - H[c].groupby([L.league, L.season]).transform("mean")
    U = L[L.pwin < L.ploss].copy()
    F = L[L.pwin > L.ploss][["mid"] + TRAITS].rename(columns={c: "F_" + c for c in TRAITS})
    U = U.rename(columns={c: "U_" + c for c in TRAITS}).merge(F, on="mid")
    U["y_gd"] = (U.gf - U.ga) - (U.lam_f - U.lam_a)
    U["y_pts"] = U.pts - U.xpts
    U["gap"] = U.pwin - U.ploss
    U["ts"] = U.team + "_" + U.season.astype(str)
    cols = ["U_" + c for c in TRAITS] + ["F_" + c for c in TRAITS]
    U = U.dropna(subset=cols)
    # standardise traits on discovery-set SDs (fixed constants, applied to holdout too)
    disc = U[(U.season >= DISC[0]) & (U.season <= DISC[1])]
    for c in cols:
        U[c] = U[c] / disc[c].std()
    return U


def tests(d, y="y_gd"):
    rows = []
    base = "gap + home"
    for side in ("U", "F"):
        for c in TRAITS:
            f = f"{y} ~ {side}_{c} + {base}"
            rows.append((f"{side}:{c}", f, f"{side}_{c}"))
    for a in TRAITS:
        for b in TRAITS:
            f = f"{y} ~ U_{a} * F_{b} + {base}"
            rows.append((f"U:{a} x F:{b}", f, f"U_{a}:F_{b}"))
    out = []
    for name, f, term in rows:
        m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.ts)[0]})
        out.append(dict(test=name, formula=f, term=term, coef=float(m.params[term]), se=float(m.bse[term]), p=float(m.pvalues[term])))
    return pd.DataFrame(out)


def main(mode):
    U = build()
    if mode == "search":
        d = U[(U.season >= DISC[0]) & (U.season <= DISC[1])]
        print("discovery underdog-matches:", len(d))
        r = tests(d)
        r["q_bh"] = multipletests(r.p, method="fdr_bh")[1]
        r = r.sort_values("p")
        print(r.head(20).to_string())
        passed = r[r.q_bh < 0.10]
        short = passed if len(passed) else r.head(5)
        label = "BH_FDR10" if len(passed) else "weak_candidates_top5"
        # positive control
        pc = smf.ols("y_gd ~ I(U_shots_for + U_shots_against) + gap + home", data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.ts)[0]})
        out = dict(n=len(d), label=label, shortlist=short.to_dict(orient="records"), all=r.to_dict(orient="records"),
                   positive_control=dict(coef=float(pc.params.iloc[1]), p=float(pc.pvalues.iloc[1])))
        print("positive control (U tempo):", out["positive_control"], "| shortlist:", label, list(short.test))
        json.dump(out, open(os.path.join(ROOT, "results", "r2_shortlist.json"), "w"), indent=1)
    elif mode == "confirm":
        sl = json.load(open(os.path.join(ROOT, "results", "r2_shortlist.json")))
        d = U[(U.season >= HOLD[0]) & (U.season <= HOLD[1])]
        k = len(sl["shortlist"]); res = []
        for s in sl["shortlist"]:
            for y in ("y_gd", "y_pts"):
                f = s["formula"].replace("y_gd ~", f"{y} ~")
                m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.ts)[0]})
                b, se = float(m.params[s["term"]]), float(m.bse[s["term"]])
                from scipy.stats import norm
                p1 = float(norm.sf(b / se)) if s["coef"] > 0 else float(norm.cdf(b / se))
                res.append(dict(test=s["test"], outcome=y, disc_coef=s["coef"], hold_coef=b, hold_se=se, p_one_sided=p1))
        res = pd.DataFrame(res)
        prim = res[res.outcome == "y_gd"].sort_values("p_one_sided").reset_index(drop=True)
        prim["holm_threshold"] = [0.05 / (k - i) for i in range(len(prim))]
        conf = []
        for _, r in prim.iterrows():
            if r.p_one_sided < r.holm_threshold:
                conf.append(r.test)
            else:
                break
        prim["confirmed"] = prim.test.isin(conf)
        print("holdout underdog-matches:", len(d)); print(prim.to_string()); print(res[res.outcome == "y_pts"].to_string())
        json.dump(dict(n=len(d), primary=prim.to_dict(orient="records"), all=res.to_dict(orient="records")),
                  open(os.path.join(ROOT, "results", "r2_confirm.json"), "w"), indent=1)
    elif mode == "ml":
        import lightgbm as lgb
        cols = ["U_" + c for c in TRAITS] + ["F_" + c for c in TRAITS] + ["gap", "home"]
        tr = U[(U.season >= DISC[0]) & (U.season <= 2014)]; va = U[(U.season >= 2015) & (U.season <= DISC[1])]
        te = U[(U.season >= HOLD[0]) & (U.season <= HOLD[1])]
        m = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.01, num_leaves=8, min_child_samples=200, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1)
        m.fit(tr[cols], tr.y_gd, eval_set=[(va[cols], va.y_gd)], callbacks=[lgb.early_stopping(100, verbose=False)])
        p = m.predict(te[cols])
        e_model = (te.y_gd - p) ** 2; e_mkt = te.y_gd ** 2
        diff = (e_mkt - e_model).values
        rng = np.random.default_rng(20261012)
        bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        out = dict(best_iter=int(m.best_iteration_), mse_market=float(e_mkt.mean()), mse_model=float(e_model.mean()),
                   gain=float(diff.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                   importance=dict(zip(cols, map(int, m.feature_importances_))))
        print(out)
        json.dump(out, open(os.path.join(ROOT, "results", "r2_ml.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])

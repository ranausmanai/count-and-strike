"""Round 9: referee strictness as a natural experiment (PREREGISTRATION_9.md).  usage: r9_referees.py discovery|confirm"""
import glob, json, os, sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(__file__))
from discovery import build as build_u
ROOT = os.path.join(os.path.dirname(__file__), "..")
FD = os.path.join(ROOT, "data", "raw", "fd")


def referee_table():
    fr = []
    for f in sorted(glob.glob(os.path.join(FD, "E0_*.csv"))):
        s = os.path.basename(f)[3:7]; season = 2000 + int(s[:2]) if int(s[:2]) < 50 else 1900 + int(s[:2])
        d = pd.read_csv(f, encoding="latin-1", on_bad_lines="skip").dropna(subset=["HomeTeam"])
        if "Referee" not in d:
            continue
        d["date"] = pd.to_datetime(d.Date, dayfirst=True, errors="coerce"); d["season"] = season
        for c in ["HY", "AY", "HR", "AR", "HF", "AF"]:
            d[c] = pd.to_numeric(d.get(c), errors="coerce")
        d["cards"] = d.HY + d.AY + 2 * (d.HR + d.AR); d["fouls"] = d.HF + d.AF
        fr.append(d[["date", "season", "HomeTeam", "AwayTeam", "Referee", "cards", "fouls"]])
    r = pd.concat(fr).dropna(subset=["Referee", "cards"]).sort_values("date")
    r["Referee"] = r.Referee.str.strip()
    g = r.groupby("Referee")
    for c in ("cards", "fouls"):
        r[c + "_prev"] = g[c].transform(lambda s: s.shift(1).expanding(min_periods=20).mean())
        r[c + "_strict"] = (r[c + "_prev"] - r.groupby("season")[c + "_prev"].transform("mean")) / r.groupby("season")[c + "_prev"].transform("std")
    r["ref_season"] = r.Referee + "_" + r.season.astype(str)
    return r.rename(columns={"HomeTeam": "home_t", "AwayTeam": "away_t"})


def main(mode):
    U = build_u()
    U = U[U.league == "EPL"].copy()
    U["home_t"] = np.where(U.home == 1, U.team, U.opp); U["away_t"] = np.where(U.home == 1, U.opp, U.team)
    U["date"] = pd.to_datetime(U.date)
    R = referee_table()
    D = U.merge(R[["date", "home_t", "away_t", "cards_strict", "fouls_strict", "ref_season", "Referee"]], on=["date", "home_t", "away_t"], how="inner")
    D = D.dropna(subset=["cards_strict"])
    D["y_tot"] = (D.gf + D.ga) - (D.lam_f + D.lam_a)
    span = (2005, 2016) if mode == "discovery" else (2017, 2025)
    d = D[(D.season >= span[0]) & (D.season <= span[1])]
    cl = dict(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.ref_season)[0]})
    out = {"n": len(d), "n_referees": int(d.Referee.nunique())}
    m1 = smf.ols("y_gd ~ cards_strict + gap + home", data=d).fit(**cl)
    m2 = smf.ols("y_gd ~ cards_strict * U_fouls_committed + gap + home", data=d).fit(**cl)
    m3 = smf.ols("y_tot ~ cards_strict + gap", data=d).fit(**cl)
    m1f = smf.ols("y_gd ~ fouls_strict + gap + home", data=d.dropna(subset=["fouls_strict"])).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.dropna(subset=["fouls_strict"]).ref_season)[0]})
    def rec(m, term, direction):
        b, se = float(m.params[term]), float(m.bse[term])
        p = float(norm.cdf(b / se)) if direction == "neg" else float(2 * norm.sf(abs(b / se)))
        return dict(coef=b, se=se, p=p)
    out["H9_1"] = rec(m1, "cards_strict", "neg"); out["H9_2"] = rec(m2, "cards_strict:U_fouls_committed", "neg")
    out["H9_3"] = rec(m3, "cards_strict", "two"); out["H9_1_fouls_version"] = rec(m1f, "fouls_strict", "neg")
    # assignment check: is strictness related to match importance (gap) or teams?
    out["assignment_corr_gap"] = float(d[["cards_strict", "gap"]].corr().iloc[0, 1])
    print(mode, json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(ROOT, "results", f"r9_{mode}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])

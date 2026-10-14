"""Round 14B: team habits that create valuable moments (PREREGISTRATION_14.md)."""
import json, os, sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, os.path.dirname(__file__))
from possessions import build_possessions
ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")
COLS = ["match_id", "idx", "period", "t", "type", "team", "poss", "poss_team", "pattern", "x", "y", "end_x", "end_y", "length", "xg", "outcome", "sub", "under_pressure", "league"]
PAIRS = [("counterpress", "high_regains"), ("press_height", "high_regains"), ("directness", "fast_ft"), ("cross_rate", "box_entries"), ("tempo", "shots")]


def team_match(ev):
    ev = ev[ev.period <= 2].sort_values(["match_id", "idx"]).reset_index(drop=True)
    p = build_possessions(ev[COLS])
    p = p.sort_values(["match_id", "poss"]).reset_index(drop=True)
    prev = p.groupby("match_id").shift(1)
    p["regain"] = (p.pattern == "Regular Play") & prev.poss_team.notna() & (prev.poss_team != p.poss_team) & (prev.period == p.period)
    p["loser"] = np.where(p.regain, prev.poss_team, None)
    own = ev[ev.team == ev.poss_team]
    # first own pass/carry of each possession
    fa = own[own.type.isin(["Pass", "Carry"])].groupby(["match_id", "poss"]).first()
    p = p.join((fa.end_x - fa.x).rename("first_gain"), on=["match_id", "poss"])
    reach = own.assign(dt=0).merge(p[["match_id", "poss", "t0", "start_x"]], on=["match_id", "poss"])
    reach["dt"] = (reach.t - reach.t0) * 60
    ft = reach[(reach.x >= 80) & (reach.dt <= 10)].groupby(["match_id", "poss"]).size().rename("ft")
    box = reach[(reach.x >= 102) & reach.y.between(18, 62)].groupby(["match_id", "poss"]).size().rename("bx")
    f3 = reach[reach.x >= 80].groupby(["match_id", "poss"]).size().rename("f3")
    p = p.join(ft, on=["match_id", "poss"]).join(box, on=["match_id", "poss"]).join(f3, on=["match_id", "poss"])
    # counter-press: pressures by the losing team within 5 s of the opponent's regain
    pr = ev[ev.type == "Pressure"][["match_id", "period", "t", "team", "x"]]
    rg = p[p.regain][["match_id", "period", "t0", "loser"]]
    cp = rg.merge(pr, left_on=["match_id", "period", "loser"], right_on=["match_id", "period", "team"])
    cp = cp[(cp.t >= cp.t0) & ((cp.t - cp.t0) * 60 <= 5)]
    cpn = cp.groupby(["match_id", "loser"]).size()
    losses = rg.groupby(["match_id", "loser"]).size()
    crosses = own[(own.type == "Pass") & ((own.y - 40).abs() > 22) & (own.x >= 96) & (own.end_x >= 102) & own.end_y.between(18, 62) & ~own["sub"].isin(["Corner", "Free Kick", "Throw-in"])]
    shots = ev[(ev.type == "Shot") & (ev["sub"] != "Penalty")]
    rows = []
    teams = ev.groupby("match_id").team.unique()
    P = dict(tuple(p.groupby("match_id"))); PR = dict(tuple(pr.groupby("match_id"))); OWN = dict(tuple(own[["match_id", "team", "type"]].groupby("match_id")))
    SH = dict(tuple(shots[["match_id", "team", "xg"]].groupby("match_id"))); CR = crosses.groupby(["match_id", "team"]).size()
    LG = ev.drop_duplicates("match_id").set_index("match_id").league
    empty_pr = pr.iloc[:0]; empty_sh = shots[["match_id", "team", "xg"]].iloc[:0]
    for m, ts in teams.items():
        for t in ts:
            o = [x for x in ts if x != t]
            if not o:
                continue
            o = o[0]
            pm = P[m]; prm = PR.get(m, empty_pr); ownm = OWN[m]; shm = SH.get(m, empty_sh)
            mine = pm[pm.poss_team == t]
            reg = mine[mine.regain]
            prs = prm[prm.team == t]
            own_t = ownm[ownm.team == t]
            poss_min = ((mine.t_end - mine.t0).clip(lower=0)).sum()
            rows.append(dict(match_id=m, team=t, comp=LG[m],
                             counterpress=cpn.get((m, t), 0) / max(losses.get((m, t), 0), 1),
                             press_height=prs.x.mean() if len(prs) else np.nan,
                             directness=(reg.first_gain >= 10).mean() if len(reg) else np.nan,
                             cross_rate=CR.get((m, t), 0) / max(mine.f3.notna().sum(), 1),
                             tempo=(own_t.type == "Pass").sum() / max(poss_min, 1e-6),
                             high_regains=int((reg.start_x >= 80).sum()), fast_ft=int((reg.ft.notna() & (reg.start_x < 80)).sum()),
                             box_entries=int(mine.bx.notna().sum()), shots=int((shm.team == t).sum()),
                             xgf=float(shm.xg[shm.team == t].sum()), xga=float(shm.xg[shm.team == o].sum())))
    return pd.DataFrame(rows)


def halves(T):
    T = T.sort_values(["comp", "team", "match_id"]).copy()
    T["h"] = T.groupby(["comp", "team"]).cumcount() % 2
    H = T.groupby(["comp", "team", "h"]).mean(numeric_only=True).reset_index()
    a = H[H.h == 0].set_index(["comp", "team"]); b = H[H.h == 1].set_index(["comp", "team"])
    X = []
    for src, dst in ((a, b), (b, a)):
        j = src.join(dst, lsuffix="_o", rsuffix="_e", how="inner").reset_index()
        X.append(j)
    X = pd.concat(X, ignore_index=True)
    X["net_e"] = X.xgf_e - X.xga_e
    for hb, _ in PAIRS:
        X[hb + "_z"] = (X[hb + "_o"] - X.groupby("comp")[hb + "_o"].transform("mean")) / X.groupby("comp")[hb + "_o"].transform("std")
    return X


def fit(X, habit, y):
    d = X.dropna(subset=[habit + "_z", y])
    m = smf.ols(f"{y} ~ {habit}_z + xgf_o + xga_o + C(comp)", data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.comp + d.team)[0]})
    return float(m.params[habit + "_z"]), float(m.bse[habit + "_z"]), len(d) // 2


def main():
    cache = os.path.join(PROC, "r14b_team_match.parquet")
    if not os.path.exists(cache):
        e1 = pd.read_parquet(os.path.join(PROC, "events.parquet"), columns=COLS)
        e2 = pd.read_parquet(os.path.join(PROC, "sb2_events.parquet"), columns=COLS)
        T = pd.concat([team_match(e1), team_match(e2)], ignore_index=True)
        T.to_parquet(cache)
    T = pd.read_parquet(cache)
    T["pop"] = np.where(T.comp == "EPL", "confirm", np.where(T.comp.isin(["LaLiga", "SerieA", "Ligue1", "Bundesliga"]), "discovery", "confirm"))
    T = T[T.comp != "Bundesliga"]  # only 34 matches (one team) in 2015/16
    out = []
    Xd, Xc = halves(T[T["pop"] == "discovery"]), halves(T[T["pop"] == "confirm"])
    for hb, mo in PAIRS:
        for y in ("net_e", mo + "_e"):
            b, se, n = fit(Xd, hb, y); bc, sec, nc = fit(Xc, hb, y)
            out.append(dict(habit=hb, outcome=y, disc=b, disc_se=se, n_teams=n, conf=bc, conf_se=sec, n_conf=nc))
    R = pd.DataFrame(out)
    R["p"] = 2 * norm.sf(np.abs(R.disc / R.disc_se)); R["q"] = multipletests(R.p, method="fdr_bh")[1]
    R["conf_p_one"] = norm.sf(np.sign(R.disc) * R.conf / R.conf_se)
    k = int((R.q < 0.10).sum())
    S = R[R.q < 0.10].sort_values("conf_p_one").reset_index(drop=True)
    flag = True; conf = []
    for i, r in S.iterrows():
        flag = flag and r.conf_p_one < 0.05 / (k - i) and np.sign(r.conf) == np.sign(r.disc); conf.append(bool(flag))
    S["confirmed"] = conf
    pd.set_option("display.width", 250)
    print(R.round(4).to_string()); print("shortlist", k); print(S.round(4).to_string())
    json.dump(dict(all=R.to_dict(orient="records"), shortlist=S.to_dict(orient="records")), open(os.path.join(ROOT, "results", "r14b_habits.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

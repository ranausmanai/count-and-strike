"""Round 22A: Strike-through stress tests (PREREGISTRATION_22.md)."""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
import statsmodels.formula.api as smf
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import PROC, ROOT
from r22_ratings import ratings
POPS = ("discovery", "confirm_club", "confirm_women")
RECYCLE = {"lateral", "back_pass", "short_fwd_ground", "short_fwd_air"}
FEAT = ["x", "y", "dist", "angle", "lane", "opp2", "opp5", "nearest", "opp_goalside", "mates_ahead", "under_pressure", "air", "poss_dur",
        "minute", "score_diff", "gk_vis", "gk_line", "gk_off", "gk_ball"]


def load():
    B = pd.read_parquet(os.path.join(PROC, "r22_box.parquet"))
    A = pd.read_parquet(os.path.join(PROC, "r15_attack.parquet"))[["match_id", "idx", "action", "net", "pop", "opp_goalside"]]
    E = pd.read_parquet(os.path.join(PROC, "agent_shooter_ev.parquet"))[["match_id", "idx", "net60"]]
    D = B.merge(A, on=["match_id", "idx"]).merge(E, on=["match_id", "idx"], how="left")
    D = D[D.action.isin(RECYCLE | {"shot"})].copy()
    bad = (D.gk_line > 40) | D.gk_line.isna()
    D.loc[bad, "gk_vis"] = 0
    for c in ("gk_line", "gk_off", "gk_ball"):
        D.loc[bad, c] = np.nan
    D["T"] = (D.action == "shot").astype(int)
    D = D.merge(ratings(), on=["match_id", "team"], how="left")
    return D.reset_index(drop=True)


def cells(D, keeper=False):
    base = (pd.Series(np.where(D.x >= 108, "b", "e"), index=D.index) + np.select([D.opp_goalside <= 3, D.opp_goalside <= 6], ["0", "1"], "2")
            + np.select([D.nearest < 3, D.nearest <= 7], ["0", "1"], "2") + np.where(D.lane >= 2, "2", D.lane.astype(str)))
    if keeper:
        base = base + np.where(D.gk_line > 4, "o", "l") + np.where(D.gk_off > 2, "m", "a")
    return base


def within(D, cell, y="net"):
    st = D.assign(cell=cell).groupby(["cell", "T"])[y].agg(["mean", "var", "size"]).unstack("T").dropna()
    st = st[(st[("size", 1)] >= 5) & (st[("size", 0)] >= 5)]
    w = np.minimum(st[("size", 1)], st[("size", 0)])
    diff = ((st[("mean", 1)] - st[("mean", 0)]) * w).sum() / w.sum()
    var = ((w / w.sum()) ** 2 * (st[("var", 1)] / st[("size", 1)] + st[("var", 0)] / st[("size", 0)])).sum()
    return float(diff), float(np.sqrt(var))


def crossfit(D, y="net", seed=0):
    X = D[FEAT].values; T = D["T"].values; Y = D[y].values
    e = np.zeros(len(D)); m1 = np.zeros(len(D)); m0 = np.zeros(len(D))
    for tr, te in GroupKFold(5).split(X, T, D.match_id):
        c = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=seed).fit(X[tr], T[tr])
        e[te] = c.predict_proba(X[te])[:, 1]
        for t, mm in ((1, m1), (0, m0)):
            s = tr[T[tr] == t]
            mm[te] = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=seed).fit(X[s], Y[s]).predict(X[te])
    return np.clip(e, 0.02, 0.98), m1, m0


def aipw(D, e, m1, m0, y="net", target="ate"):
    T = D["T"].values; Y = D[y].values; n = len(D)
    if target == "ate":
        phi = m1 - m0 + T * (Y - m1) / e - (1 - T) * (Y - m0) / (1 - e); th = phi.mean(); psi = phi - th
    else:  # atu: effect among recyclers
        p0 = (1 - T).mean()
        a = (1 - T) * (m1 - Y) + T * (1 - e) / e * (Y - m1)
        th = a.mean() / p0; psi = (a - (1 - T) * th) / p0
    cl = pd.Series(psi).groupby(D.match_id.values).sum()
    return float(th), float(np.sqrt((cl ** 2).sum()) / n)


def verdict(r):
    d0, s0 = r["discovery"]; c = [r["confirm_club"], r["confirm_women"]]
    w = np.array([1 / s ** 2 for _, s in c]); pooled = float((w * np.array([m for m, _ in c])).sum() / w.sum()); pse = float(1 / np.sqrt(w.sum()))
    return dict(per_pop={k: dict(est=v[0], se=v[1]) for k, v in r.items()}, disc_p=float(norm.sf(d0 / s0)), pooled=pooled, pooled_se=pse,
                pooled_p=float(norm.sf(pooled / pse)),
                passed=bool(d0 > 0 and norm.sf(d0 / s0) < .05 and all(m > 0 for m, _ in c) and norm.sf(pooled / pse) < .05))


def main():
    D = load()
    print(D.groupby(["pop", "T"]).size().unstack(), "\nkeeper visible:", D.gk_vis.mean().round(3))
    res = {}; S1, S2, S2ate, S3, S4, S60, raw = {}, {}, {}, {}, {}, {}, {}
    D["e"] = np.nan
    for p in POPS:
        d = D[D["pop"] == p].reset_index(drop=True)
        raw[p] = (float(d[d["T"] == 1].net.mean() - d[d["T"] == 0].net.mean()), within(d, cells(d)))
        k = d[d.gk_vis == 1].reset_index(drop=True); S1[p] = within(k, cells(k, keeper=True))
        e, m1, m0 = crossfit(d); D.loc[D["pop"] == p, "e"] = e
        S2[p] = aipw(d, e, m1, m0, target="atu"); S2ate[p] = aipw(d, e, m1, m0)
        lo = e <= np.quantile(e, 0.4); dl = d[lo].reset_index(drop=True)
        S3[p] = aipw(dl, e[lo], m1[lo], m0[lo]) + (float(dl["T"].mean()), int(dl["T"].sum()))
        e4, a1, a0 = crossfit(d, y="pre_net"); S4[p] = aipw(d, e4, a1, a0)
        d6 = d[d.net60.notna()].reset_index(drop=True); e6, b1, b0 = crossfit(d6, y="net60"); S60[p] = aipw(d6, e6, b1, b0, y="net60", target="atu")
        print(p, "raw", raw[p], "S1", S1[p], "S2 atu", S2[p], "ate", S2ate[p], "S3", S3[p], "S4", S4[p], "60s atu", S60[p], flush=True)
    res["S1_keeper_cells"] = verdict(S1); res["S2_aipw_atu"] = verdict(S2); res["S2_aipw_ate"] = verdict(S2ate)
    res["S3_low_propensity"] = verdict({k: v[:2] for k, v in S3.items()}); res["S3_low_propensity"]["shot_share_in_stratum"] = {k: v[2:] for k, v in S3.items()}
    s4 = verdict(S4); s4["fails"] = bool(s4["pooled"] > 0 and s4["pooled_p"] < .05); res["S4_placebo_pre20"] = s4
    res["robust_60s_atu"] = verdict(S60)
    res["raw_and_cell"] = {k: dict(raw=v[0], cell=v[1]) for k, v in raw.items()}
    # S5 Oster delta (report only)
    osr = {}
    for p in POPS:
        d = D[D["pop"] == p].copy(); d["cell"] = cells(d); d = d.fillna({c: -1 for c in ("gk_line", "gk_off", "gk_ball")})
        r0 = smf.ols("net ~ T", d).fit(); f = "net ~ T + C(cell) + " + " + ".join(FEAT); r1 = smf.ols(f, d).fit()
        b0, b1, R0, R1 = r0.params["T"], r1.params["T"], r0.rsquared, r1.rsquared; Rm = min(1.0, 1.3 * R1)
        osr[p] = dict(beta_raw=float(b0), beta_ctrl=float(b1), R2_raw=float(R0), R2_ctrl=float(R1),
                      delta_for_zero=float(b1 * (R1 - R0) / ((b0 - b1) * (Rm - R1))) if b0 != b1 else None)
    res["S5_oster"] = osr
    # S6 team ITT (report only)
    D["dev"] = D["T"] - D.e
    tm = D.groupby(["pop", "match_id", "team"]).agg(net=("net", "mean"), n=("net", "size"), dev=("dev", "sum"), gap=("gap", "first")).reset_index()
    D["comp"] = D.match_id.map(pd.read_parquet(os.path.join(PROC, "r22_match_xg.parquet")).drop_duplicates("match_id").set_index("match_id").comp)
    tm["comp"] = tm.match_id.map(D.drop_duplicates("match_id").set_index("match_id").comp)
    g = D.groupby(["comp", "team"]).agg(sd=("dev", "sum"), sn=("dev", "size")).reset_index()
    tm = tm.merge(g, on=["comp", "team"]); tm["n_all"] = tm.sn
    tm["habit"] = (tm.sd - tm.dev) / (tm.sn - tm.n)
    tm = tm[(tm.sn - tm.n) >= 15].dropna(subset=["gap", "habit"])
    it = {}
    for p in POPS:
        x = tm[tm["pop"] == p]; r = smf.wls("net ~ habit + gap", x, weights=x.n).fit(cov_type="cluster", cov_kwds={"groups": x.team.astype("category").cat.codes})
        it[p] = dict(slope=float(r.params["habit"]), se=float(r.bse["habit"]), n_team_matches=int(len(x)), habit_sd=float(x.habit.std()))
    res["S6_team_itt"] = it
    res["survives_stress"] = bool(res["S1_keeper_cells"]["passed"] and res["S2_aipw_atu"]["passed"] and res["S3_low_propensity"]["passed"] and not s4["fails"])
    for k, v in res.items():
        print(k, json.dumps(v, default=float)[:600])
    json.dump(res, open(os.path.join(ROOT, "results", "r22a_strike.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

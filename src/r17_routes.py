"""Round 17: opponent weaknesses × own strengths (PREREGISTRATION_17.md).
  python3 src/r17_routes.py build   -> data/proc/r17_team_routes.parquet
  python3 src/r17_routes.py test    -> results/r17_*.json
"""
import gzip, json, os, pickle, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")
ROUTES = ["corner", "free_kick", "throw_in", "counter", "high_regain", "cross_A", "cross_B", "cutback", "through_ball",
          "pass_central", "pass_wide", "aerial_pass", "solo_box", "long_shot"]


# ---------------- StatsBomb ----------------
def sb_match(args):
    path, m, chan = args
    try:
        ev = json.load(gzip.open(path))
    except Exception:
        return []
    byid = {e["id"]: e for e in ev}
    names = {m["home_team"]["home_team_id"]: m["home_team"]["home_team_name"], m["away_team"]["away_team_id"]: m["away_team"]["away_team_name"]}
    out = []
    for e in ev:
        if e["type"]["name"] != "Shot" or e["period"] > 4 or e["shot"]["type"]["name"] == "Penalty":
            continue
        s = e["shot"]; x, y = e["location"][:2]
        ch = chan.get(e["possession"], "build_up")
        if ch in ("corner", "free_kick", "throw_in", "counter", "high_regain"):
            r = ch
        else:
            kp = byid.get(s.get("key_pass_id"))
            if kp is not None and kp["type"]["name"] == "Pass":
                p = kp["pass"]; py = kp["location"][1]
                if p.get("cross"):
                    r = "cross_A" if py < 40 else "cross_B"
                elif p.get("cut_back"):
                    r = "cutback"
                elif p.get("through_ball") or p.get("technique", {}).get("name") == "Through Ball":
                    r = "through_ball"
                elif p.get("height", {}).get("name") == "High Pass":
                    r = "aerial_pass"
                else:
                    r = "pass_central" if abs(py - 40) <= 18 else "pass_wide"
            else:
                r = "solo_box" if (x >= 102 and 18 <= y <= 62) else "long_shot"
        out.append(dict(match_id=m["match_id"], team=names.get(e["team"]["id"], e["team"]["name"]), route=r, xg=s.get("statsbomb_xg", 0.0)))
    return out


def sb_population(matches, raw_dir, poss, comp_key):
    chan = {mid: dict(zip(g.poss, g.channel)) for mid, g in poss.groupby("match_id")}
    args = [(os.path.join(raw_dir, f"{m['match_id']}.json.gz"), m, chan.get(m["match_id"], {})) for m in matches]
    rows = []
    with ProcessPoolExecutor(8) as ex:
        for r in ex.map(sb_match, args, chunksize=8):
            rows += r
    S = pd.DataFrame(rows)
    info = [dict(match_id=m["match_id"], date=m["match_date"], comp=m[comp_key], home=m["home_team"]["home_team_name"], away=m["away_team"]["away_team_name"]) for m in matches]
    return S, pd.DataFrame(info)


# ---------------- Wyscout ----------------
def wy_population():
    from xg_model import feats
    XG = pickle.load(open(os.path.join(PROC, "xg_model.pkl"), "rb"))
    WY = os.path.join(ROOT, "data", "raw", "wy")
    CONTROL = {"Pass", "Free Kick", "Shot", "Others on the ball"}
    shots, info = [], []
    for country, lg in {"England": "EPL", "Spain": "LaLiga", "Germany": "Bundesliga", "Italy": "SerieA", "France": "Ligue1"}.items():
        ms = json.load(open(os.path.join(WY, f"matches_{country}.json")))
        names = {}
        for m in ms:
            lab = m["label"].split(",")[0]; h, a = [x.strip() for x in lab.split(" - ")]
            sides = {v["side"]: int(k) for k, v in m["teamsData"].items()}
            names[(m["wyId"], sides["home"])] = h; names[(m["wyId"], sides["away"])] = a
            info.append(dict(match_id=m["wyId"], date=m["dateutc"][:10], comp="WY_" + lg, home=h, away=a))
        ev = json.load(open(os.path.join(WY, f"events_{country}.json")))
        ev.sort(key=lambda e: (e["matchId"], e["matchPeriod"], e["eventSec"]))
        cur = None; prev_m = None; start = None; counter = False; last_pass = None
        pend = []
        for e in ev:
            if e["matchPeriod"] not in ("1H", "2H"):
                continue
            mid = e["matchId"]
            if mid != prev_m:
                cur = None; prev_m = mid
            name, sub, tm = e["eventName"], e["subEventName"], e["teamId"]
            pos = e["positions"]; x = pos[0]["x"] if pos else np.nan; y = pos[0]["y"] if pos else np.nan
            tags = {t["id"] for t in e["tags"]}
            if name in CONTROL and (tm != cur or name == "Free Kick"):
                cur = tm; start = dict(name=name, sub=sub, x=x, t=e["eventSec"], per=e["matchPeriod"]); counter = False; last_pass = None
            if tm == cur and 1901 in tags:
                counter = True
            if tm == cur and name == "Pass":
                last_pass = dict(sub=sub, x=x, y=y, ex=pos[1]["x"] if len(pos) > 1 else x, t=e["eventSec"])
            is_shot = name == "Shot" or (name == "Free Kick" and sub == "Free kick shot")
            if not is_shot or start is None:
                continue
            secs = e["eventSec"] - start["t"] if start["per"] == e["matchPeriod"] else 999
            st = start
            if st["name"] == "Free Kick" and st["sub"] == "Corner" and secs <= 20:
                r = "corner"
            elif st["name"] == "Free Kick" and st["sub"] == "Throw in" and st["x"] >= 66.7 and secs <= 20:
                r = "throw_in"
            elif st["name"] == "Free Kick" and st["sub"] in ("Free Kick", "Free kick cross", "Free kick shot") and st["x"] >= 50 and secs <= 20:
                r = "free_kick"
            elif counter or 1901 in tags or (secs <= 15 and st["name"] != "Free Kick" and st["x"] < 66.7):
                r = "counter"
            elif st["name"] != "Free Kick" and st["x"] >= 66.7:
                r = "high_regain"
            elif last_pass is not None and e["eventSec"] - last_pass["t"] <= 10:
                lp = last_pass
                if lp["sub"] == "Cross":
                    r = "cross_A" if lp["y"] < 50 else "cross_B"
                elif lp["sub"] == "Smart pass":
                    r = "through_ball"
                elif lp["sub"] in ("High pass", "Head pass", "Launch"):
                    r = "aerial_pass"
                elif lp["x"] >= 85 and abs(lp["y"] - 50) > 20 and lp["ex"] < lp["x"]:
                    r = "cutback"
                else:
                    r = "pass_central" if abs(lp["y"] - 50) <= 22.5 else "pass_wide"
            else:
                r = "solo_box" if (x >= 84.3 and 19 <= y <= 81) else "long_shot"
            pend.append((mid, names[(mid, tm)], r, x, y, 403 in tags, sub == "Free kick shot"))
        D = pd.DataFrame(pend, columns=["match_id", "team", "route", "x", "y", "head", "fk"])
        D["xg"] = XG.predict_proba(feats(pd.DataFrame(dict(x=D.x / 100 * 105, y=D.y / 100 * 68, head=D["head"], fk=D.fk))))[:, 1]
        shots.append(D[["match_id", "team", "route", "xg"]])
    return pd.concat(shots, ignore_index=True), pd.DataFrame(info)


def team_routes(S, I, pop):
    rows = []
    agg = S.groupby(["match_id", "team", "route"]).xg.sum()
    for r in I.itertuples():
        for t, o in ((r.home, r.away), (r.away, r.home)):
            d = dict(pop=pop, comp=r.comp, match_id=r.match_id, date=r.date, team=t, opp=o, home=int(t == r.home))
            for k in ROUTES:
                d[k] = float(agg.get((r.match_id, t, k), 0.0)); d[k + "_ag"] = float(agg.get((r.match_id, o, k), 0.0))
            rows.append(d)
    return pd.DataFrame(rows)


def build():
    from possessions import build_possessions
    m1 = json.load(open(os.path.join(ROOT, "data", "raw", "sb_matches.json")))
    m1 = [m for m in m1 if m["_league"] != "Bundesliga"]
    p1 = pd.read_parquet(os.path.join(PROC, "possessions.parquet"))
    S1, I1 = sb_population(m1, os.path.join(ROOT, "data", "raw", "sb"), p1, "_league")
    m2 = json.load(open(os.path.join(ROOT, "data", "raw", "sb2_matches.json")))
    cols = ["match_id", "idx", "period", "t", "type", "team", "poss", "poss_team", "pattern", "x", "y", "xg", "outcome", "sub", "league"]
    p2 = build_possessions(pd.read_parquet(os.path.join(PROC, "sb2_events.parquet"), columns=cols))
    S2, I2 = sb_population(m2, os.path.join(ROOT, "data", "raw", "sb2"), p2, "_comp")
    S3, I3 = wy_population()
    T = pd.concat([team_routes(S1, I1, "SB_1516"), team_routes(S2, I2, "SB_women_ISL"), team_routes(S3, I3, "WY_1718")], ignore_index=True)
    T.to_parquet(os.path.join(PROC, "r17_team_routes.parquet"))
    for name, S in (("SB_1516", S1), ("SB_women_ISL", S2), ("WY_1718", S3)):
        print(name, (S.groupby("route").xg.sum() / S.xg.sum()).round(3).to_dict())


if __name__ == "__main__":
    build() if sys.argv[1] == "build" else None


# ---------------- tests ----------------
K = 8.0


def profiles(T):
    """Prior-match (shrunk) attack/defence route profiles and overall levels for each team-match row."""
    T = T.sort_values(["comp", "team", "date", "match_id"]).reset_index(drop=True)
    T["xg_tot"] = T[ROUTES].sum(1); T["xga_tot"] = T[[k + "_ag" for k in ROUTES]].sum(1)
    m = T.groupby("comp")[ROUTES + ["xg_tot"]].transform("mean")
    g = T.groupby(["comp", "team"])
    n = g.cumcount().astype(float)
    def prior(col):
        return g[col].transform(lambda s: s.shift(1).fillna(0).cumsum())
    T["n_prior"] = n
    T["A"] = (prior("xg_tot") + K * m.xg_tot) / (n * m.xg_tot + K * m.xg_tot)
    T["D"] = (prior("xga_tot") + K * m.xg_tot) / (n * m.xg_tot + K * m.xg_tot)
    for k in ROUTES:
        T["m_" + k] = m[k]
        T["att_" + k] = (prior(k) + K * m[k] * T.A) / (n * m[k] * T.A + K * m[k] * T.A)
        T["def_" + k] = (prior(k + "_ag") + K * m[k] * T.D) / (n * m[k] * T.D + K * m[k] * T.D)
        T["ush_" + k] = (prior(k) + K * m[k]) / (prior("xg_tot") + K * m.xg_tot)  # usual attacking share (prior)
    o = T[["match_id", "team", "D", "A"] + ["def_" + k for k in ROUTES]].rename(
        columns={"team": "opp", "D": "D_opp", "A": "A_opp", **{"def_" + k: "odef_" + k for k in ROUTES}})
    return T.merge(o, on=["match_id", "opp"], how="left")


def h1(T):
    out = {}
    for side, cols, tot in (("defence", [k + "_ag" for k in ROUTES], "xga_tot"), ("attack", ROUTES, "xg_tot")):
        rel = {}
        for k, c in zip(ROUTES, cols):
            rows = []
            for (cp, tm), d in T.groupby(["comp", "team"]):
                if len(d) < 20:
                    continue
                d = d.sort_values("date"); a, b = d.iloc[0::2], d.iloc[1::2]
                rows.append((a[c].sum() / max(a[tot].sum(), 1e-9), b[c].sum() / max(b[tot].sum(), 1e-9), cp))
            R = pd.DataFrame(rows, columns=["a", "b", "comp"])
            R[["a", "b"]] = R[["a", "b"]] - R.groupby("comp")[["a", "b"]].transform("mean")
            r = R[["a", "b"]].corr().iloc[0, 1]
            rel[k] = float(2 * r / (1 + r)) if r > -1 else None
        out[side] = rel
    return out


def h2(T):
    import statsmodels.api as sm
    from frontier import poisson_dev
    rows = []
    for k in ROUTES:
        d = T[(T.n_prior >= 8) & T.D_opp.notna()][["comp", "match_id", k, "m_" + k, "A", "D_opp", "att_" + k, "odef_" + k]].copy()
        d.columns = ["comp", "match_id", "y", "m", "A", "Do", "att", "dfn"]; d["route"] = k
        rows.append(d)
    L = pd.concat(rows, ignore_index=True)
    L["off"] = np.log(L.m * L.A * L.Do)
    L["la"] = np.log(L.att); L["ld"] = np.log(L.dfn); L["lx"] = L.la * L.ld
    gains, coefs = [], []
    for c in L.comp.unique():
        tr, te = L[L.comp != c], L[L.comp == c]
        fit = {}
        for name, cols in (("base", ["la"]), ("full", ["la", "ld", "lx"])):
            X = sm.add_constant(tr[cols].values)
            fit[name] = sm.GLM(tr.y.values, X, family=sm.families.Poisson(), offset=tr.off.values).fit()
        coefs.append(dict(comp=c, **{f"full_{n}": float(v) for n, v in zip(["const", "la", "ld", "lx"], fit["full"].params)}))
        mu_b = np.exp(sm.add_constant(te[["la"]].values, has_constant="add") @ fit["base"].params + te.off.values)
        mu_f = np.exp(sm.add_constant(te[["la", "ld", "lx"]].values, has_constant="add") @ fit["full"].params + te.off.values)
        gains.append(pd.DataFrame(dict(match_id=te.match_id.values, g=poisson_dev(te.y, mu_b) - poisson_dev(te.y, mu_f))))
    G = pd.concat(gains).groupby("match_id").g.sum()
    rng = np.random.default_rng(20261015)
    bs = [G.values[rng.integers(0, len(G), len(G))].sum() for _ in range(2000)]
    allfit = sm.GLM(L.y.values, sm.add_constant(L[["la", "ld", "lx"]].values), family=sm.families.Poisson(), offset=L.off.values).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(L.match_id)[0]})
    return dict(n_rows=int(len(L)), oos_gain=float(G.sum()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                rel_gain=float(G.sum() / poisson_dev(L.y, np.exp(L.off + np.log(L.att))).sum()),
                coef_defence=[float(allfit.params[2]), float(allfit.bse[2])], coef_matchup=[float(allfit.params[3]), float(allfit.bse[3])],
                coef_attack=[float(allfit.params[1]), float(allfit.bse[1])], per_fold=coefs)


def h3(T):
    import statsmodels.formula.api as smf
    from scipy.stats import norm
    d = T[(T.n_prior >= 8) & T.D_opp.notna()].copy()
    sh = d[ROUTES].div(d.xg_tot.replace(0, np.nan), axis=0)
    ush = d[["ush_" + k for k in ROUTES]].values
    w = np.log(d[["odef_" + k for k in ROUTES]].values)
    d["exploit"] = np.nansum((sh.values - ush) * w, axis=1)
    d = d[d.xg_tot > 0]
    d["exp_xg"] = d[["m_" + k for k in ROUTES]].sum(1) * d.A * d.D_opp
    d["y"] = d.xg_tot - d.exp_xg; d["xgd"] = d.xg_tot - d.xga_tot
    d["lA"], d["lD"], d["lAo"], d["lDs"] = np.log(d.A), np.log(d.D_opp), np.log(d.A_opp), np.log(d.D)
    d["ez"] = d.exploit / d.exploit.std()
    out = {}
    for y in ("y", "xgd"):
        m = smf.ols(f"{y} ~ ez + lA + lD + lAo + lDs + home + C(comp)", data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.match_id)[0]})
        out[y] = dict(coef_per_sd=float(m.params["ez"]), se=float(m.bse["ez"]), p_one=float(norm.sf(m.params["ez"] / m.bse["ez"])))
    out["n"] = int(len(d))
    return out


def test():
    T = pd.read_parquet(os.path.join(PROC, "r17_team_routes.parquet"))
    res = {}
    for pop, d in T.groupby("pop"):
        P = profiles(d.copy())
        r = {"H17_1": h1(P), "H17_2": h2(P), "H17_3": h3(P)}
        res[pop] = r
        d1 = r["H17_1"]
        print(f"\n=== {pop} ===")
        print("H17-1 defence reliability:", {k: round(v, 2) for k, v in d1["defence"].items()})
        print("      attack  reliability:", {k: round(v, 2) for k, v in d1["attack"].items()})
        print("H17-2:", {k: v for k, v in r["H17_2"].items() if k != "per_fold"})
        print("H17-3:", r["H17_3"])
    json.dump(res, open(os.path.join(ROOT, "results", "r17_gameplan.json"), "w"), indent=1, default=float)


if __name__ == "__main__" and sys.argv[1] == "test":
    test()

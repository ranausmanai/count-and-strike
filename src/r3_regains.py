"""Round 3-A: build regain tables (StatsBomb 2015/16, Wyscout 2017/18) and test vulnerability-window hypotheses.

  python3 src/r3_regains.py build
  python3 src/r3_regains.py test
"""
import json, os, pickle, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")
WIN = 20.0  # seconds


def sb_regains():
    from frontier import load_segments
    ev = pd.read_parquet(os.path.join(PROC, "events.parquet"), columns=["match_id", "period", "t", "type", "team", "poss", "poss_team", "x", "y", "xg", "sub", "outcome"])
    ev = ev[ev.period <= 2]
    p = pd.read_parquet(os.path.join(PROC, "possessions.parquet"))
    own = ev[ev.team == ev.poss_team]
    mx = own.groupby(["match_id", "poss"]).x.max().rename("max_x")
    p = p.merge(mx.reset_index(), on=["match_id", "poss"], how="left").sort_values(["match_id", "poss"]).reset_index(drop=True)
    # xG within WIN seconds of possession start
    sh = ev[(ev.type == "Shot") & (ev["sub"] != "Penalty")].merge(p[["match_id", "poss", "t0"]], on=["match_id", "poss"])
    sh = sh[(sh.t - sh.t0) * 60 <= WIN]
    p = p.merge(sh.groupby(["match_id", "poss"]).xg.sum().rename("xg20").reset_index(), on=["match_id", "poss"], how="left")
    p["xg20"] = p.xg20.fillna(0)
    prev = p.groupby("match_id").shift(1)
    p["opp_prev_team"] = prev.poss_team; p["opp_prev_passes"] = prev.n_pass; p["opp_prev_max_x"] = 120 - prev.max_x
    p["opp_prev_secs"] = (prev.t_end - prev.t0) * 60; p["prev_period"] = prev.period
    r = p[(p.pattern == "Regular Play") & (p.opp_prev_team.notna()) & (p.opp_prev_team != p.poss_team) & (p.prev_period == p.period)].copy()
    r["regain_type"] = r.start_type.map(lambda s: {"Interception": "interception", "Ball Recovery": "recovery", "Duel": "tackle",
                                                   "Block": "block", "Clearance": "clearance"}.get(s, "other"))
    r["rx"] = r.start_x; r["cent"] = 1 - (r.start_y - 40).abs() / 40
    # game state & strength
    seg = load_segments(min_minutes=0.0)
    info = seg.drop_duplicates(["match_id", "team"])[["match_id", "team", "lam_own", "lam_opp", "league"]]
    r = r.drop(columns=["league"]).merge(info.rename(columns={"team": "poss_team"}), on=["match_id", "poss_team"], how="left")
    gl = ev[((ev.type == "Shot") & (ev.outcome == "Goal")) | (ev.type == "Own Goal For")][["match_id", "period", "t", "team"]]
    gl["clock"] = gl.t + np.where(gl.period == 2, 45, 0)
    r["clock"] = r.t0 + np.where(r.period == 2, 45, 0)
    gd = []
    gmap = {m: g for m, g in gl.groupby("match_id")}
    for m, tm, c in zip(r.match_id, r.poss_team, r.clock):
        g = gmap.get(m)
        if g is None:
            gd.append(0); continue
        g = g[g.clock < c]; gd.append(int((g.team == tm).sum() - (g.team != tm).sum()))
    r["gd"] = gd
    r["src"] = "SB"
    return r[["match_id", "league", "poss_team", "rx", "cent", "regain_type", "opp_prev_passes", "opp_prev_max_x", "opp_prev_secs",
              "gd", "clock", "lam_own", "lam_opp", "xg20", "src"]]


def wy_regains():
    from xg_model import feats
    XG = pickle.load(open(os.path.join(PROC, "xg_model.pkl"), "rb"))
    WY = os.path.join(ROOT, "data", "raw", "wy")
    LEAGUES = {"England": "EPL", "Spain": "LaLiga", "Germany": "Bundesliga", "Italy": "SerieA", "France": "Ligue1"}
    tm_tab = pd.read_parquet(os.path.join(PROC, "wy_team_match_channels.parquet"))
    out = []
    CONTROL = {"Pass", "Free Kick", "Shot", "Others on the ball"}
    for country, lg in LEAGUES.items():
        ms = json.load(open(os.path.join(WY, f"matches_{country}.json")))
        names = {}
        for m in ms:
            lab = m["label"].split(",")[0]; h, a = [x.strip() for x in lab.split(" - ")]
            sides = {v["side"]: int(k) for k, v in m["teamsData"].items()}
            names[(m["wyId"], sides["home"])] = h; names[(m["wyId"], sides["away"])] = a
        ev = json.load(open(os.path.join(WY, f"events_{country}.json")))
        ev.sort(key=lambda e: (e["matchId"], e["matchPeriod"], e["eventSec"]))
        poss = []; cur = None; prev_m = None; score = {}
        for e in ev:
            if e["matchPeriod"] not in ("1H", "2H"):
                continue
            mid = e["matchId"]
            if mid != prev_m:
                cur = None; prev_m = mid
            name, sub, tmid = e["eventName"], e["subEventName"], e["teamId"]
            pos = e["positions"]; x = pos[0]["x"] if pos else np.nan; y = pos[0]["y"] if pos else np.nan
            tags = {t["id"] for t in e["tags"]}
            new = name in CONTROL and (cur is None or tmid != cur["team"] or name == "Free Kick" or (cur["period"] != e["matchPeriod"]))
            if new:
                cur = dict(match_id=mid, team=tmid, period=e["matchPeriod"], t0=e["eventSec"], t_end=e["eventSec"], start_name=name,
                           start_sub=sub, x0=x, y0=y, n_pass=0, max_x=x, xg20=0.0,
                           gd=score.get((mid, tmid), 0) - sum(v for (mm, tt), v in score.items() if mm == mid and tt != tmid))
                poss.append(cur)
            if cur is not None and tmid == cur["team"]:
                cur["t_end"] = e["eventSec"]
                if name == "Pass":
                    cur["n_pass"] += 1
                if not np.isnan(x):
                    cur["max_x"] = np.nanmax([cur["max_x"], x])
                if name == "Shot" or (name == "Free Kick" and sub == "Free kick shot"):
                    if e["eventSec"] - cur["t0"] <= WIN:
                        d = pd.DataFrame(dict(x=[x / 100 * 105], y=[y / 100 * 68], head=[403 in tags], fk=[sub == "Free kick shot"]))
                        cur["xg20"] += float(XG.predict_proba(feats(d))[0, 1])
                    if 101 in tags:
                        score[(mid, tmid)] = score.get((mid, tmid), 0) + 1
        P = pd.DataFrame(poss)
        prev = P.groupby("match_id").shift(1)
        P["opp_prev_team"] = prev.team; P["opp_prev_passes"] = prev.n_pass; P["opp_prev_max_x"] = (100 - prev.max_x) * 1.2
        P["opp_prev_secs"] = prev.t_end - prev.t0; P["prev_period"] = prev.period
        r = P[(P.start_name != "Free Kick") & P.opp_prev_team.notna() & (P.opp_prev_team != P.team) & (P.prev_period == P.period)].copy()
        r["regain_type"] = r.start_name.map(lambda s: {"Pass": "other", "Others on the ball": "recovery", "Shot": "other"}.get(s, "other"))
        r["rx"] = r.x0 * 1.2; r["cent"] = 1 - (r.y0 - 50).abs() / 50
        r["clock"] = r.t0 / 60 + np.where(r.period == "2H", 45, 0)
        r["poss_team"] = [names[(m, t)] for m, t in zip(r.match_id, r.team)]
        r["league"] = lg
        out.append(r)
    r = pd.concat(out, ignore_index=True)
    info = tm_tab[["match_id", "team", "lam_own", "lam_opp"]].rename(columns={"team": "poss_team"})
    r = r.merge(info, on=["match_id", "poss_team"], how="left")
    r["src"] = "WY"
    return r[["match_id", "league", "poss_team", "rx", "cent", "regain_type", "opp_prev_passes", "opp_prev_max_x", "opp_prev_secs",
              "gd", "clock", "lam_own", "lam_opp", "xg20", "src"]]


def design(d, kind):
    X = pd.DataFrame(index=d.index)
    X["const"] = 1.0
    rx = d.rx / 120
    for k in range(1, 4):
        X[f"rx{k}"] = rx ** k
    X["cent"] = d.cent; X["cent_rx"] = d.cent * rx
    if kind in ("ctx", "full"):
        X["gd_pos"] = (d.gd > 0).astype(float); X["gd_neg"] = (d.gd < 0).astype(float)
        X["clock"] = d.clock / 90; X["ratio"] = np.log(d.lam_own / d.lam_opp)
    if kind == "full":
        X["opp_passes_log"] = np.log1p(d.opp_prev_passes.clip(0, 60))
        X["opp_maxx"] = d.opp_prev_max_x / 120
        X["opp_maxx_rx"] = (d.opp_prev_max_x / 120) * rx
        X["opp_secs_log"] = np.log1p(d.opp_prev_secs.clip(0, 300))
    return X


def test():
    sb = pd.read_parquet(os.path.join(PROC, "r3_regains_sb.parquet")).dropna(subset=["lam_own", "opp_prev_max_x", "rx"])
    wy = pd.read_parquet(os.path.join(PROC, "r3_regains_wy.parquet")).dropna(subset=["lam_own", "opp_prev_max_x", "rx"])
    tr = sb[sb.league != "EPL"]; t1 = sb[sb.league == "EPL"]
    print("train", len(tr), "test1", len(t1), "test2", len(wy), "| mean xg20", tr.xg20.mean(), t1.xg20.mean(), wy.xg20.mean())
    res = {}
    fits = {}
    for kind in ("loc", "ctx", "full"):
        X = design(tr, kind)
        fits[kind] = sm.GLM(tr.xg20.values, X.values, family=sm.families.Poisson()).fit(scale="X2")
    print("full model coefs:"); [print(f"  {n:16s} {p:+.3f} ({s:.3f})") for n, p, s in zip(design(tr, 'full').columns, fits['full'].params, fits['full'].bse)]
    rng = np.random.default_rng(20261012)
    from frontier import poisson_dev
    for tname, te in (("SB_EPL", t1), ("WY_1718", wy)):
        mu = {k: np.exp(design(te, k).values @ fits[k].params) for k in fits}
        # WY xG scale differs: recalibrate intercept only (scale factor) on the test set — harmless for ranking, fair for deviance
        if tname == "WY_1718":
            for k in mu:
                mu[k] = mu[k] * te.xg20.sum() / mu[k].sum()
        dev = {k: poisson_dev(te.xg20, mu[k]) for k in mu}
        per = pd.DataFrame({"m": te.match_id.values, "d_full_vs_ctx": dev["ctx"] - dev["full"], "d_full_vs_loc": dev["loc"] - dev["full"]}).groupby("m").sum()
        bs = np.array([per.values[rng.integers(0, len(per), len(per))].sum(0) for _ in range(2000)])
        # concentration: share of xg in top 20% predicted
        q = np.quantile(mu["full"], 0.8); top = mu["full"] >= q
        q2 = np.quantile(mu["loc"], 0.8); top2 = mu["loc"] >= q2
        res[tname] = dict(n=len(te), gain_vs_ctx=float(per.d_full_vs_ctx.sum()), ci_vs_ctx=[float(np.percentile(bs[:, 0], 2.5)), float(np.percentile(bs[:, 0], 97.5))],
                          gain_vs_loc=float(per.d_full_vs_loc.sum()), ci_vs_loc=[float(np.percentile(bs[:, 1], 2.5)), float(np.percentile(bs[:, 1], 97.5))],
                          rel_gain_vs_loc=float(per.d_full_vs_loc.sum() / dev["loc"].sum()),
                          top20_share_full=float(te.xg20[top].sum() / te.xg20.sum()), top20_share_loc=float(te.xg20[top2].sum() / te.xg20.sum()))
        print(tname, res[tname])
    # direction check on each test set: refit full model there
    for tname, te in (("SB_EPL", t1), ("WY_1718", wy)):
        m = sm.GLM(te.xg20.values, design(te, "full").values, family=sm.families.Poisson()).fit(scale="X2")
        cols = list(design(te, "full").columns)
        res[tname]["refit"] = {c: [float(m.params[i]), float(m.bse[i])] for i, c in enumerate(cols)}
    res["train_coefs"] = {c: [float(p), float(s)] for c, p, s in zip(design(tr, "full").columns, fits["full"].params, fits["full"].bse)}
    json.dump(res, open(os.path.join(ROOT, "results", "r3a_regains.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        sb_regains().to_parquet(os.path.join(PROC, "r3_regains_sb.parquet")); print("sb done")
        wy_regains().to_parquet(os.path.join(PROC, "r3_regains_wy.parquet")); print("wy done")
    else:
        test()

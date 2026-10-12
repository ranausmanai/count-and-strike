"""Round 6: confirm set-piece flatness and the 'break after they commit' regain effect on unseen leagues (PREREGISTRATION_6.md)."""
import gzip, json, os, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(__file__))
from build_segments import compact_events
from possessions import build_possessions
ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW2 = os.path.join(ROOT, "data", "raw", "sb2")
OPEN = ["build_up", "counter", "high_regain"]; SP = ["corner", "free_kick", "throw_in"]


def one(m):
    ev = json.load(gzip.open(os.path.join(RAW2, f"{m['match_id']}.json.gz")))
    idname = {e["team"]["id"]: e["team"]["name"] for e in ev if e["type"]["name"] == "Starting XI"}
    d = compact_events(ev, m["match_id"])
    d["league"] = m["_comp"]
    home = idname.get(m["home_team"]["home_team_id"]); away = idname.get(m["away_team"]["away_team_id"])
    return d, dict(match_id=m["match_id"], comp=m["_comp"], date=m["match_date"], home=home, away=away,
                   hs=m["home_score"], as_=m["away_score"])


def build():
    ms = json.load(open(os.path.join(ROOT, "data", "raw", "sb2_matches.json")))
    evs, info = [], []
    with ProcessPoolExecutor(8) as ex:
        for d, i in ex.map(one, ms, chunksize=8):
            evs.append(d); info.append(i)
    ev = pd.concat(evs, ignore_index=True); info = pd.DataFrame(info)
    ev.to_parquet(os.path.join(ROOT, "data", "proc", "sb2_events.parquet")); info.to_parquet(os.path.join(ROOT, "data", "proc", "sb2_info.parquet"))
    return ev, info


def team_match(p, info):
    x = p.pivot_table(index=["match_id", "poss_team"], columns="channel", values="xg", aggfunc="sum", fill_value=0).reset_index().rename(columns={"poss_team": "team"})
    rows = []
    for r in info.itertuples():
        rows += [dict(match_id=r.match_id, comp=r.comp, team=r.home, opp=r.away, home=1), dict(match_id=r.match_id, comp=r.comp, team=r.away, opp=r.home, home=0)]
    t = pd.DataFrame(rows).merge(x, on=["match_id", "team"], how="left").fillna(0)
    for c in OPEN + SP + ["penalty"]:
        if c not in t:
            t[c] = 0.0
    t["open_play"] = t[OPEN].sum(1); t["set_piece"] = t[SP].sum(1); t["tot"] = t[OPEN + SP + ["penalty"]].sum(1)
    opp = t[["match_id", "team", "open_play", "set_piece", "tot"]].rename(columns={"team": "opp", "open_play": "open_play_ag", "set_piece": "set_piece_ag", "tot": "tot_ag"})
    t = t.merge(opp, on=["match_id", "opp"])
    # leave-this-match-out strength
    g = t.groupby(["comp", "team"])
    n = g.tot.transform("size")
    t["att_loo"] = (g.tot.transform("sum") - t.tot) / (n - 1)
    t["def_loo"] = (g.tot_ag.transform("sum") - t.tot_ag) / (n - 1)
    o = t[["match_id", "team", "att_loo", "def_loo"]].rename(columns={"team": "opp", "att_loo": "opp_att_loo", "def_loo": "opp_def_loo"})
    return t.merge(o, on=["match_id", "opp"])


def elasticity(t, col):
    m = smf.glm(f"{col} ~ np.log(att_loo) + np.log(opp_def_loo) + home + C(comp)", data=t, family=sm.families.Poisson()).fit(scale="X2")
    return m.params["np.log(att_loo)"]


def main():
    ev, info = build()
    p = build_possessions(ev[["match_id", "idx", "period", "t", "type", "team", "poss", "poss_team", "pattern", "x", "y", "xg", "outcome", "sub", "league"]])
    t = team_match(p, info)
    t = t[(t.att_loo > 0) & (t.opp_def_loo > 0)]
    out = {}
    e_op, e_sp = elasticity(t, "open_play"), elasticity(t, "set_piece")
    rng = np.random.default_rng(20261012)
    teams = t[["comp", "team"]].drop_duplicates().values.tolist()
    by = {tuple(k): ix for k, ix in t.groupby(["comp", "team"]).indices.items()}
    bs = []
    for _ in range(500):
        pick = [teams[i] for i in rng.integers(0, len(teams), len(teams))]
        d = t.iloc[np.concatenate([by[tuple(k)] for k in pick])]
        bs.append(elasticity(d, "open_play") - elasticity(d, "set_piece"))
    out["H6_1"] = dict(e_open=float(e_op), e_setpiece=float(e_sp), diff=float(e_op - e_sp), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                       by_comp={c: dict(open=float(elasticity(g, "open_play")), sp=float(elasticity(g, "set_piece"))) for c, g in t.groupby("comp")})
    print("H6-1", out["H6_1"])
    # H6-2
    t["xgd_loo"] = t.att_loo - t.def_loo; t["opp_xgd_loo"] = t.opp_att_loo - t.opp_def_loo
    mm = t[t.xgd_loo <= t.opp_xgd_loo - 0.5]
    def shares(d):
        return d.set_piece.sum() / (d.set_piece.sum() + d.set_piece_ag.sum()), d.open_play.sum() / (d.open_play.sum() + d.open_play_ag.sum())
    sp, op = shares(mm)
    mids = mm.match_id.unique(); bym = mm.groupby("match_id").indices
    bs2 = []
    for _ in range(2000):
        d = mm.iloc[np.concatenate([bym[m] for m in rng.choice(mids, len(mids))])]
        a, b = shares(d); bs2.append(a - b)
    out["H6_2"] = dict(n=len(mm), share_setpiece=float(sp), share_open=float(op), diff=float(sp - op), ci=[float(np.percentile(bs2, 2.5)), float(np.percentile(bs2, 97.5))])
    print("H6-2", out["H6_2"])
    # H6-3: regains
    e = ev[ev.period <= 2]
    own = e[e.team == e.poss_team]
    mx = own.groupby(["match_id", "poss"]).x.max().rename("max_x").reset_index()
    q = p.merge(mx, on=["match_id", "poss"], how="left").sort_values(["match_id", "poss"]).reset_index(drop=True)
    sh = e[(e.type == "Shot") & (e["sub"] != "Penalty")].merge(q[["match_id", "poss", "t0"]], on=["match_id", "poss"])
    sh = sh[(sh.t - sh.t0) * 60 <= 20]
    q = q.merge(sh.groupby(["match_id", "poss"]).xg.sum().rename("xg20").reset_index(), on=["match_id", "poss"], how="left")
    q["xg20"] = q.xg20.fillna(0)
    prev = q.groupby("match_id").shift(1)
    q["opp_prev_team"] = prev.poss_team; q["opp_prev_max_x"] = 120 - prev.max_x; q["prev_period"] = prev.period
    r = q[(q.pattern == "Regular Play") & q.opp_prev_team.notna() & (q.opp_prev_team != q.poss_team) & (q.prev_period == q.period)].dropna(subset=["opp_prev_max_x", "start_x"]).copy()
    gl = e[((e.type == "Shot") & (e.outcome == "Goal")) | (e.type == "Own Goal For")][["match_id", "period", "t", "team"]]
    gl["clock"] = gl.t + np.where(gl.period == 2, 45, 0); r["clock"] = r.t0 + np.where(r.period == 2, 45, 0)
    gmap = {m: g for m, g in gl.groupby("match_id")}; gd = []
    for m, tm, c in zip(r.match_id, r.poss_team, r.clock):
        g = gmap.get(m)
        if g is None:
            gd.append(0); continue
        g = g[g.clock < c]; gd.append(int((g.team == tm).sum() - (g.team != tm).sum()))
    r["gd"] = gd
    r["rx"] = r.start_x / 120; r["cent"] = 1 - (r.start_y - 40).abs() / 40; r["om"] = r.opp_prev_max_x / 120
    r["gpos"] = (r.gd > 0).astype(float); r["gneg"] = (r.gd < 0).astype(float); r["clk"] = r.clock / 90
    f = "xg20 ~ rx + I(rx**2) + I(rx**3) + cent + cent:rx + gpos + gneg + clk + om + om:rx + C(league)"
    m = smf.glm(f, data=r, family=sm.families.Poisson()).fit(scale="X2", cov_type="cluster", cov_kwds={"groups": pd.factorize(r.match_id)[0]})
    x0 = 36 / 120
    L = np.zeros(len(m.params)); names = list(m.params.index)
    L[names.index("om")] = 1; L[names.index("om:rx")] = x0
    est = float(L @ m.params); se = float(np.sqrt(L @ m.cov_params().values @ L))
    out["H6_3"] = dict(n_regains=len(r), marginal_effect_at_rx36=est, se=se, p_one=float(norm.cdf(est / se)),
                       multiplier_deep_vs_shallow=float(np.exp(est * (15 / 120 - 65 / 120))))
    print("H6-3", out["H6_3"])
    json.dump(out, open(os.path.join(ROOT, "results", "r6_confirm.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

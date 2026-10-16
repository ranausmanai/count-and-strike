"""agent subs: H1 (early sub when trailing by 1), H2 (early sub when level), H3 (attacking vs like-for-like subs when trailing).
See PREREGISTRATION_AGENT_subs.md. Inputs from agent_subs_extract.py (SUBS_OUT dir)."""
import json, os, sys, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")
OUT = os.environ.get("SUBS_OUT", "/private/tmp/claude-501/-Users-usman-Documents-research-football/3d8cdebd-b9be-4e7c-b5f3-c4c5a15f5f08/scratchpad")
POPS = ["sb", "sb2", "sb360", "wy"]
CONF = ["sb2", "sb360", "wy"]

rd = lambda n: pd.read_parquet(os.path.join(OUT, f"agent_subs_{n}.parquet"))
M, S, G, U, R = rd("matches"), rd("shots"), rd("goals"), rd("subs"), rd("reds")
for d in (S, G, U, R):
    d["match_id"] = d.match_id.astype(np.int64)
M["match_id"] = M.match_id.astype(np.int64)


def before(d, T):  # strictly before match clock T (2nd half clocks start at 45)
    return (d.period == 1) | (d.clock < T)


# ---------- team-match table with strength ----------
def team_table():
    rows = []
    for r in M.itertuples():
        for t, o in ((r.t1, r.t2), (r.t2, r.t1)):
            rows.append(dict(pop=r.pop, match_id=r.match_id, group=r.group, team=t, opp=o, home=int(t == r.home_team)))
    T = pd.DataFrame(rows)
    xs = S.groupby(["match_id", "team"]).xg.sum()
    T["xgf"] = [xs.get((m, t), 0.0) for m, t in zip(T.match_id, T.team)]
    T["xga"] = [xs.get((m, o), 0.0) for m, o in zip(T.match_id, T.opp)]
    T["xgd"] = T.xgf - T.xga
    g = T.groupby(["pop", "group", "team"]).xgd
    tot, cnt = g.transform("sum"), g.transform("count")
    T["loo"] = ((tot - T.xgd) / (cnt - 1).replace(0, np.nan)).fillna(0.0)
    opp = T[["match_id", "team", "loo"]].rename(columns={"team": "opp", "loo": "loo_opp"})
    T = T.merge(opp, on=["match_id", "opp"], how="left")
    T["str"] = T.loo - T.loo_opp
    # market strength for 2015/16
    seg = pd.read_parquet(os.path.join(PROC, "segments.parquet"), columns=["match_id", "home_team", "away_team"]).drop_duplicates("match_id")
    tmap = json.load(open(os.path.join(PROC, "team_map.json")))
    mk = pd.read_parquet(os.path.join(PROC, "market.parquet")); mk = mk[mk.season == 2015]
    seg["home_fd"] = seg.home_team.map(tmap); seg["away_fd"] = seg.away_team.map(tmap)
    seg = seg.merge(mk[["home", "away", "pH", "pA"]].rename(columns={"home": "home_fd", "away": "away_fd"}), on=["home_fd", "away_fd"], how="left")
    seg["match_id"] = seg.match_id.astype(np.int64)
    T = T.merge(seg[["match_id", "pH", "pA"]], on="match_id", how="left")
    pw = np.where(T.home == 1, T.pH, T.pA); pl = np.where(T.home == 1, T.pA, T.pH)
    mkt = np.log(pw / pl)
    sbm = (T["pop"] == "sb") & np.isfinite(mkt)
    T["str_src"] = np.where(sbm, "market", "loo")
    T.loc[sbm, "str"] = mkt[sbm]
    return T


TT = team_table()
print("market coverage sb:", (TT[TT["pop"] == "sb"].str_src == "market").mean())
GB = {m: g for m, g in G.groupby("match_id")}
SB = {m: g for m, g in S.groupby("match_id")}
UB = {m: g.sort_values(["period", "clock"]) for m, g in U.groupby("match_id")}
RB = {m: g for m, g in R.groupby("match_id")}
E0 = G.iloc[:0]; ES = S.iloc[:0]; EU = U.iloc[:0]; ER = R.iloc[:0]


def timing_table(T0, T1, late, pre_name):
    """Population for H1 (gd=-1) / H2 (gd=0) at T0 and T1; returns rows with treatment and outcomes."""
    rows = []
    for r in TT.itertuples():
        g = GB.get(r.match_id, E0); s = SB.get(r.match_id, ES); u = UB.get(r.match_id, EU); rr = RB.get(r.match_id, ER)
        own = g.team == r.team
        gd0 = int((before(g, T0) & own).sum() - (before(g, T0) & ~own).sum())
        gd1 = int((before(g, T1) & own).sum() - (before(g, T1) & ~own).sum())
        target = -1 if pre_name == "trail" else 0
        if gd0 != target or gd1 != target:
            continue
        if before(rr, T1).any():
            continue
        us = u[u.team == r.team]
        if before(us, T0).any():
            continue
        first = us.clock.iloc[0] if len(us) else np.inf
        early = T0 <= first < T1
        lategrp = first >= late
        sown = s.team == r.team
        pre = before(s, T0)
        rec = (s.period == 2) & (s.clock < T0)
        post = ~before(s, T1)
        ga = g[~before(g, T1)].sort_values(["period", "clock"])
        run = np.cumsum(np.where(ga.team == r.team, 1, -1)) + gd1 if len(ga) else np.array([gd1])
        rows.append(dict(pop=r.pop, match_id=r.match_id, team=r.team, group=r.group, home=r.home, str=r.str,
                         first=first, early=int(early), late=int(lategrp), nosub_by_T1=int(first >= T1),
                         mom=s.xg[pre & sown].sum() - s.xg[pre & ~sown].sum(),
                         mom_rec=s.xg[rec & sown].sum() - s.xg[rec & ~sown].sum(),
                         y=s.xg[post & sown].sum() - s.xg[post & ~sown].sum(),
                         y_own=s.xg[post & sown].sum(),
                         equal=int((run >= 0).any()) if len(ga) else int(gd1 >= 0),
                         win=int(run[-1] > 0) if len(ga) else int(gd1 > 0)))
    return pd.DataFrame(rows)


def ols(d, ycol, tcol, cov=("str", "mom", "mom_rec", "home")):
    X = sm.add_constant(d[[tcol] + list(cov)].astype(float))
    m = sm.OLS(d[ycol].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.match_id)[0]})
    return float(m.params[tcol]), float(m.bse[tcol])


def logit_ame(d, ycol, tcol, cov=("str", "mom", "mom_rec", "home")):
    X = sm.add_constant(d[[tcol] + list(cov)].astype(float))
    try:
        m = sm.Logit(d[ycol].astype(float), X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d.match_id)[0]})
        me = m.get_margeff(at="overall")
        i = list(X.columns).index(tcol) - 1
        return float(me.margeff[i]), float(me.margeff_se[i])
    except Exception:
        return ols(d, ycol, tcol, cov)


def strat(d, ycol, tcol):
    d = d.copy()
    d["c"] = pd.qcut(d.str.rank(method="first"), 3, labels=False).astype(str) + pd.qcut(d.mom.rank(method="first"), 3, labels=False).astype(str)
    est, var, w = [], [], []
    for _, c in d.groupby("c"):
        a, b = c[c[tcol] == 1][ycol], c[c[tcol] == 0][ycol]
        if len(a) >= 2 and len(b) >= 2:
            n = 1 / (1 / len(a) + 1 / len(b))
            est.append(a.mean() - b.mean()); var.append(a.var() / len(a) + b.var() / len(b)); w.append(n)
    w = np.array(w) / sum(w)
    return float(np.dot(w, est)), float(np.sqrt(np.dot(w ** 2, var)))


def run_timing(name, T0, T1, late, state, binary):
    D = timing_table(T0, T1, late, state)
    res = {}
    for p in POPS:
        d = D[D["pop"] == p]
        main = d[(d.early == 1) | (d.late == 1)]
        land = d[d["first"] >= T0]   # all with no sub before T0 (already enforced): early vs no sub by T1
        land = land[(land.early == 1) | (land.nosub_by_T1 == 1)]
        r = dict(n_early=int(main.early.sum()), n_late=int((main.late == 1).sum()), n_landmark_ctrl=int((land.early == 0).sum()),
                 raw_diff=float(main[main.early == 1].y.mean() - main[main.early == 0].y.mean()),
                 pre_mom_early=float(main[main.early == 1].mom.mean()), pre_mom_late=float(main[main.early == 0].mom.mean()),
                 pre_str_early=float(main[main.early == 1].str.mean()), pre_str_late=float(main[main.early == 0].str.mean()),
                 rate_binary_early=float(main[main.early == 1][binary].mean()), rate_binary_late=float(main[main.early == 0][binary].mean()))
        r["netxg"] = ols(main, "y", "early")
        r["ownxg"] = ols(main, "y_own", "early")
        r[binary] = logit_ame(main, binary, "early")
        r["R1_landmark_netxg"] = ols(land, "y", "early")
        r["R1_landmark_" + binary] = logit_ame(land, binary, "early")
        r["R2_strat_netxg"] = strat(main, "y", "early")
        res[p] = r
    return res, D


def subs_clusters():
    rows = []
    for (m, t), u in U[U["pop"] != "wy"].groupby(["match_id", "team"]):
        u = u.sort_values(["period", "clock"])
        u = u[u.period == 2]
        if not len(u):
            continue
        k = (u.clock.diff().fillna(99) > 2.0).cumsum()
        for _, c in u.groupby(k):
            pairs = list(zip(c.off_cls, c.on_cls))
            rank = {"DEF": 0, "MID": 1, "FWD": 2, "GK": -1}
            att = any(on == "FWD" and off in ("DEF", "MID") for off, on in pairs) and not any(
                rank.get(on, 1) < rank.get(off, 1) for off, on in pairs if "GK" not in (on, off))
            like = all(off == on for off, on in pairs)
            rows.append(dict(pop=c["pop"].iloc[0], match_id=m, team=t, clock=c.clock.min(), clock_end=c.clock.max(),
                             kind="att" if att else ("like" if like else "other"), n=len(c)))
    return pd.DataFrame(rows)


def run_h3():
    C = subs_clusters()
    info = TT.set_index(["match_id", "team"])
    rate = {p: S[S["pop"] == p].xg.sum() / (2 * 90 * (M["pop"] == p).sum()) for p in POPS}
    rows = []
    for r in C.itertuples():
        g = GB.get(r.match_id, E0); s = SB.get(r.match_id, ES); rr = RB.get(r.match_id, ER)
        own = g.team == r.team
        gd = int((before(g, r.clock) & own).sum() - (before(g, r.clock) & ~own).sum())
        if gd >= 0 or r.clock > 80 or r.kind == "other":
            continue
        r2 = rr[rr.period == 2]
        if ((r2.clock - r.clock).abs() <= 3).any():
            continue
        g2 = g[g.period == 2]
        end = max(90.0, s[s.period == 2].clock.max() if (s.period == 2).any() else 90.0)
        prev_g = g2[g2.clock < r.clock].clock.max() if (g2.clock < r.clock).any() else 45.0
        next_g = g2[g2.clock > r.clock_end].clock.min() if (g2.clock > r.clock_end).any() else 1e9
        next_r = r2[r2.clock > r.clock_end].clock.min() if (r2.clock > r.clock_end).any() else 1e9
        pre0, pre1 = max(r.clock - 15, 45.0, prev_g), r.clock
        post0, post1 = r.clock_end, min(r.clock_end + 15, next_g, next_r, end)
        if post1 - post0 < 3 or pre1 - pre0 < 3:
            continue
        s2 = s[s.period == 2]; so = s2.team == r.team
        x = lambda a, b, o: s2.xg[(s2.clock > a) & (s2.clock <= b) & (so == o)].sum()
        u = UB.get(r.match_id, EU)
        opp_sub = int(((u.team != r.team) & (u.period == 2) & (u.clock >= post0 - 0.5) & (u.clock <= post1)).any())
        i = info.loc[(r.match_id, r.team)]
        k = rate[r.pop]
        rows.append(dict(pop=r.pop, match_id=r.match_id, team=r.team, att=int(r.kind == "att"), clock=r.clock, gd=gd,
                         str=float(i["str"]), home=int(i["home"]), pre_min=pre1 - pre0, post_min=post1 - post0,
                         xf_pre=x(pre0, pre1, True), xa_pre=x(pre0, pre1, False), xf_post=x(post0, post1, True), xa_post=x(post0, post1, False),
                         lpre_f=np.log((x(pre0, pre1, True) + k * 3) / (pre1 - pre0 + 3)), lpre_a=np.log((x(pre0, pre1, False) + k * 3) / (pre1 - pre0 + 3)),
                         opp_sub=opp_sub))
    D = pd.DataFrame(rows)
    res = {}
    for p in ["sb", "sb2", "sb360"]:
        d = D[D["pop"] == p].copy()
        d["trail2"] = (d.gd <= -2).astype(float); d["tmin"] = d.clock / 90
        cov = ["att", "lpre_f", "lpre_a", "str", "tmin", "trail2", "home", "opp_sub"]
        X = sm.add_constant(d[cov].astype(float))
        gr = pd.factorize(d.match_id)[0]
        out = dict(n_att=int(d.att.sum()), n_like=int((d.att == 0).sum()),
                   raw_rate_att=float(d[d.att == 1].xf_post.sum() / d[d.att == 1].post_min.sum() * 15),
                   raw_rate_like=float(d[d.att == 0].xf_post.sum() / d[d.att == 0].post_min.sum() * 15),
                   pre_rate_att=float(d[d.att == 1].xf_pre.sum() / d[d.att == 1].pre_min.sum() * 15),
                   pre_rate_like=float(d[d.att == 0].xf_pre.sum() / d[d.att == 0].pre_min.sum() * 15))
        for nm, y in (("own", d.xf_post), ("opp", d.xa_post)):
            m = sm.GLM(y.values, X.values, family=sm.families.Poisson(), offset=np.log(d.post_min.values)).fit(
                scale="X2", cov_type="cluster", cov_kwds={"groups": gr})
            out[nm + "_logRR"] = (float(m.params[1]), float(m.bse[1]))
        yn = (d.xf_post - d.xa_post) / d.post_min * 15
        m = sm.WLS(yn.values, X.values, weights=d.post_min.values).fit(cov_type="cluster", cov_kwds={"groups": gr})
        out["net_per15"] = (float(m.params[1]), float(m.bse[1]))
        # robustness: 30-min-capped post window not computed; finer control: restrict to trailing by exactly 1
        d1 = d[d.gd == -1]; X1 = sm.add_constant(d1[cov].drop(columns=["trail2"]).astype(float))
        m = sm.GLM(d1.xf_post.values, X1.values, family=sm.families.Poisson(), offset=np.log(d1.post_min.values)).fit(
            scale="X2", cov_type="cluster", cov_kwds={"groups": pd.factorize(d1.match_id)[0]})
        out["R_trail1_own_logRR"] = (float(m.params[1]), float(m.bse[1]))
        res[p] = out
    return res, D


def pool(res, key, pops):
    est = np.array([res[p][key][0] for p in pops]); se = np.array([res[p][key][1] for p in pops])
    w = 1 / se ** 2
    b = float((w * est).sum() / w.sum()); s = float(np.sqrt(1 / w.sum()))
    return dict(est=b, se=s, p_one_sided=float(1 - norm.cdf(b / s)), same_sign_pos=int((est > 0).sum()), n_pops=len(pops))


if __name__ == "__main__":
    out = {}
    h1, D1 = run_timing("H1", 55, 62, 70, "trail", "equal")
    h2, D2 = run_timing("H2", 60, 67, 75, "level", "win")
    h3, D3 = run_h3()
    for nm, h, keys, pops in (("H1", h1, ["netxg", "equal", "ownxg", "R1_landmark_netxg", "R1_landmark_equal", "R2_strat_netxg"], CONF),
                              ("H2", h2, ["netxg", "win", "ownxg", "R1_landmark_netxg", "R1_landmark_win", "R2_strat_netxg"], CONF),
                              ("H3", h3, ["own_logRR", "opp_logRR", "net_per15", "R_trail1_own_logRR"], ["sb2", "sb360"])):
        k0 = keys[0]
        disc = h["sb"][k0]
        pd_ = float(1 - norm.cdf(disc[0] / disc[1]))
        out[nm] = dict(per_pop=h, discovery_p_one_sided=pd_, pooled_confirmation={k: pool(h, k, pops) for k in keys})
        print("\n==", nm)
        for p in h:
            print(p, {k: (np.round(v, 3) if isinstance(v, tuple) else (round(v, 3) if isinstance(v, float) else v)) for k, v in h[p].items()})
        print("disc p", round(pd_, 3))
        for k, v in out[nm]["pooled_confirmation"].items():
            print("  pooled", k, {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()})
    ps = sorted([(out[h]["discovery_p_one_sided"], h) for h in ("H1", "H2", "H3")])
    holm = {}
    for i, (p, h) in enumerate(ps):
        holm[h] = min(1.0, max(p * (3 - i), holm.get(ps[i - 1][1], 0) if i else 0))
    out["holm_discovery"] = holm
    for h in ("H1", "H2", "H3"):
        pc = out[h]["pooled_confirmation"]
        k0 = list(pc)[0]
        need = 2 if h != "H3" else 2
        out[h]["verdict_primary"] = ("CONFIRMED" if out[h]["discovery_p_one_sided"] < 0.05 and pc[k0]["same_sign_pos"] >= need and pc[k0]["p_one_sided"] < 0.05
                                     else "NOT CONFIRMED")
    print(json.dumps({h: out[h]["verdict_primary"] for h in ("H1", "H2", "H3")}), holm)
    json.dump(out, open(os.path.join(ROOT, "results", "agent_subs.json"), "w"), indent=1)

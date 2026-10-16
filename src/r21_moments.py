"""Round 21: quick restarts, vulnerability windows, goal kicks vs press (PREREGISTRATION_21.md).
  build -> data/proc/r21_*.parquet ; test -> results/r21_moments.json"""
import gzip, json, os, pickle, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")


def tsec(e):
    h, m, s = e["timestamp"].split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def net_window(S, per, t, team, w):
    s = S.get(per)
    if s is None:
        return 0.0, 0.0
    m = (s[:, 0] >= t - 0.01) & (s[:, 0] <= t + w)
    own = s[m & (s[:, 2] == team), 1].sum(); opp = s[m & (s[:, 2] != team), 1].sum()
    return float(own), float(opp)


def sb_one(args):
    path, mid, pop, home, away = args
    try:
        ev = [e for e in json.load(gzip.open(path)) if e["period"] <= 2]
    except Exception:
        return None
    names = {}
    for e in ev:
        if e["type"]["name"] == "Starting XI":
            names[e["team"]["name"]] = len(names)
    S = {}
    for e in ev:
        if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] != "Penalty":
            S.setdefault(e["period"], []).append((tsec(e), e["shot"].get("statsbomb_xg", 0.0), names.get(e["team"]["name"], -1)))
    S = {k: np.array(v) for k, v in S.items()}
    restarts, gks, goals = [], [], []
    press = [(e["period"], tsec(e), e["team"]["name"]) for e in ev if e["type"]["name"] == "Pressure"]
    passes = [(e["period"], tsec(e), e["team"]["name"]) for e in ev if e["type"]["name"] == "Pass"]
    pend = {}
    for e in ev:
        pend[e["period"]] = max(pend.get(e["period"], 0), tsec(e))
    for i, e in enumerate(ev):
        typ = e["type"]["name"]; t = tsec(e); team = e["team"]["name"]
        if (typ == "Shot" and e["shot"]["outcome"]["name"] == "Goal") or typ == "Own Goal For":
            goals.append(dict(match_id=mid, period=e["period"], t=t, team=team))
        if typ != "Pass":
            continue
        pt = e["pass"].get("type", {}).get("name")
        if pt in ("Free Kick", "Throw-in") and e["location"][0] >= 60 and i > 0:
            prev = ev[i - 1]
            if prev["period"] == e["period"]:
                delay = t - (tsec(prev) + (prev.get("duration") or 0.0))
                own, opp = net_window(S, e["period"], t, names.get(team, -1), 20)
                restarts.append(dict(match_id=mid, pop=pop, team=team, kind=pt, x=e["location"][0], y=e["location"][1], delay=delay, own=own, opp=opp))
        elif pt == "Goal Kick":
            per = e["period"]; opp_team = [n for n in names if n != team]
            if not opp_team:
                continue
            o = opp_team[0]
            np_ = sum(1 for p in press if p[0] == per and p[2] == o and t - 900 <= p[1] < t)
            nps = sum(1 for p in passes if p[0] == per and p[2] == team and t - 900 <= p[1] < t)
            own, opp = net_window(S, per, t, names.get(team, -1), 30)
            gks.append(dict(match_id=mid, pop=pop, team=team, length=e["pass"].get("length", 0.0), opp_press=np_ / max(nps, 1), own=own, opp=opp))
    tn = list(names.keys()) + [home, away]
    info = dict(match_id=mid, pop=pop, home=tn[0], away=tn[1] if len(names) > 1 else away, p1=pend.get(1, 2700), p2=pend.get(2, 2700))
    return restarts, gks, goals, info


def wy_all():
    WY = os.path.join(ROOT, "data", "raw", "wy")
    from xg_model import feats
    XG = pickle.load(open(os.path.join(PROC, "xg_model.pkl"), "rb"))
    R, G, I = [], [], []
    for country in ("England", "Spain", "Germany", "Italy", "France"):
        ms = {m["wyId"]: m for m in json.load(open(os.path.join(WY, f"matches_{country}.json")))}
        ev = [e for e in json.load(open(os.path.join(WY, f"events_{country}.json"))) if e["matchPeriod"] in ("1H", "2H")]
        sh = [e for e in ev if e["eventName"] == "Shot" or (e["eventName"] == "Free Kick" and e["subEventName"] == "Free kick shot")]
        xg = XG.predict_proba(feats(pd.DataFrame(dict(x=[s["positions"][0]["x"] * 1.05 for s in sh], y=[s["positions"][0]["y"] * 0.68 for s in sh],
                                                      head=[any(t["id"] == 403 for t in s["tags"]) for s in sh], fk=[s["subEventName"] == "Free kick shot" for s in sh]))))[:, 1]
        S = {}
        for s, p in zip(sh, xg):
            S.setdefault((s["matchId"], s["matchPeriod"]), []).append((s["eventSec"], p, s["teamId"]))
        S = {k: np.array(v) for k, v in S.items()}
        by = {}
        for e in ev:
            by.setdefault(e["matchId"], []).append(e)
        for mid, es in by.items():
            es.sort(key=lambda e: (e["matchPeriod"], e["eventSec"]))
            m = ms[mid]; sides = {v["side"]: int(k) for k, v in m["teamsData"].items()}
            I.append(dict(match_id=mid, pop="WY_1718", home=sides["home"], away=sides["away"],
                          p1=max(e["eventSec"] for e in es if e["matchPeriod"] == "1H"), p2=max([e["eventSec"] for e in es if e["matchPeriod"] == "2H"] or [2700])))
            for i, e in enumerate(es):
                tags = {t["id"] for t in e["tags"]}
                per = 1 if e["matchPeriod"] == "1H" else 2
                if (e["eventName"] == "Shot" or e["subEventName"] == "Free kick shot") and 101 in tags:
                    G.append(dict(match_id=mid, period=per, t=e["eventSec"], team=e["teamId"]))
                if 102 in tags:
                    G.append(dict(match_id=mid, period=per, t=e["eventSec"], team=sides["away"] if e["teamId"] == sides["home"] else sides["home"]))
                if e["eventName"] == "Free Kick" and e["subEventName"] in ("Free Kick", "Free kick cross", "Throw in") and e["positions"][0]["x"] * 1.2 >= 60 and i > 0:
                    prev = es[i - 1]
                    if prev["matchPeriod"] == e["matchPeriod"]:
                        s = S.get((mid, e["matchPeriod"]))
                        own = opp = 0.0
                        if s is not None:
                            w = (s[:, 0] >= e["eventSec"] - 0.01) & (s[:, 0] <= e["eventSec"] + 20)
                            own = float(s[w & (s[:, 2] == e["teamId"]), 1].sum()); opp = float(s[w & (s[:, 2] != e["teamId"]), 1].sum())
                        R.append(dict(match_id=mid, pop="WY_1718", team=e["teamId"], kind="Throw-in" if e["subEventName"] == "Throw in" else "Free Kick",
                                      x=e["positions"][0]["x"] * 1.2, y=e["positions"][0]["y"] * 0.8, delay=e["eventSec"] - prev["eventSec"], own=own, opp=opp))
    return R, G, I


def build():
    args = []
    for mfile, raw, pop, fmt in (("sb_matches.json", "sb", "SB_1516", "{}.json.gz"), ("sb2_matches.json", "sb2", "SB_women_ISL", "{}.json.gz"),
                                 ("sb360_matches.json", "sb360", "SB_360comps", "{}_events.json.gz")):
        for m in json.load(open(os.path.join(ROOT, "data", "raw", mfile))):
            args.append((os.path.join(ROOT, "data", "raw", raw, fmt.format(m["match_id"])), m["match_id"], pop, m["home_team"]["home_team_name"], m["away_team"]["away_team_name"]))
    R, GK, G, I = [], [], [], []
    with ProcessPoolExecutor(8) as ex:
        for r in ex.map(sb_one, args, chunksize=8):
            if r:
                R += r[0]; GK += r[1]; G += [dict(g, pop=r[3]["pop"]) for g in r[2]]; I.append(r[3])
    wr, wg, wi = wy_all()
    R += wr; G += [dict(g, pop="WY_1718") for g in wg]; I += wi
    for n, L in (("restarts", R), ("goalkicks", GK), ("goals", G), ("info", I)):
        D = pd.DataFrame(L)
        for c in ("team", "home", "away"):
            if c in D:
                D[c] = D[c].astype(str)
        D.to_parquet(os.path.join(PROC, f"r21_{n}.parquet"))
        print(n, len(D))


POPS = ["SB_1516", "SB_women_ISL", "SB_360comps", "WY_1718"]


def confirm(per_pop, sign):
    d0 = per_pop["SB_1516"]
    others = [per_pop[p] for p in POPS[1:] if p in per_pop and per_pop[p] is not None]
    w = np.array([1 / o[1] ** 2 for o in others]); pooled = float((w * np.array([o[0] for o in others])).sum() / w.sum()); pse = float(1 / np.sqrt(w.sum()))
    same = sum(np.sign(o[0]) == sign for o in others)
    return dict(per_pop={p: dict(est=v[0], se=v[1]) for p, v in per_pop.items() if v}, disc_sign_ok=bool(np.sign(d0[0]) == sign),
                pooled_others=pooled, pooled_p_one=float(norm.sf(sign * pooled / pse)), n_same_sign_others=int(same),
                confirmed=bool(np.sign(d0[0]) == sign and same >= 3 and norm.sf(sign * pooled / pse) < 0.05))


def test():
    out = {}
    # 21A quick restarts
    R = pd.read_parquet(os.path.join(PROC, "r21_restarts.parquet"))
    R["net"] = R.own - R.opp
    R["cell"] = R.kind + "_" + pd.cut(R.x, [59, 80, 96, 121], labels=["60-80", "80-96", "96+"]).astype(str) + "_" + np.where((R.y - 40).abs() <= 20, "c", "w")
    print("restart delay quantiles by pop:", R.groupby("pop").delay.quantile([.25, .5, .75]).unstack().round(1).to_dict(orient="index"))
    pp = {}
    for p in POPS:
        d = R[R["pop"] == p]
        a = d[d.delay < 6].groupby("cell").net.agg(["mean", "var", "size"]); b = d[d.delay > 15].groupby("cell").net.agg(["mean", "var", "size"])
        j = a.join(b, lsuffix="_q", rsuffix="_s", how="inner"); j = j[(j.size_q >= 15) & (j.size_s >= 15)]
        w = np.minimum(j.size_q, j.size_s)
        est = ((j.mean_q - j.mean_s) * w).sum() / w.sum(); se = np.sqrt(((w / w.sum()) ** 2 * (j.var_q / j.size_q + j.var_s / j.size_s)).sum())
        pp[p] = (float(est), float(se)) if len(j) else None
        print(f"  {p}: quick {int((d.delay < 6).sum())}, slow {int((d.delay > 15).sum())}, quick−slow {est:+.4f} ± {se:.4f}")
    out["H21_1_quick_restarts"] = confirm(pp, +1)
    # 21B vulnerability windows
    G = pd.read_parquet(os.path.join(PROC, "r21_goals.parquet")); I = pd.read_parquet(os.path.join(PROC, "r21_info.parquet"))
    res_c, res_s = {}, {}
    for p in POPS:
        Ip = I[I["pop"] == p]; Gp = G[G["pop"] == p]
        O_after_scoring = E_after_scoring = 0.0; O_after_conc = E_after_conc = 0.0
        boots = []
        rows = []
        for r in Ip.itertuples():
            g = Gp[Gp.match_id == r.match_id]
            dur = r.p1 + r.p2
            for team in (str(r.home), str(r.away)):
                conc = g[g.team != team]; scored = g[g.team == team]
                rate_c = len(conc) / dur; rate_s = len(scored) / dur
                oc = ec = os_ = es = 0.0
                for gg in scored.itertuples():  # window after our goal: do we concede?
                    end = min(gg.t + 300, r.p1 if gg.period == 1 else r.p2)
                    wlen = max(end - gg.t, 0)
                    oc += ((conc.period == gg.period) & (conc.t > gg.t) & (conc.t <= end)).sum(); ec += rate_c * wlen
                for gg in conc.itertuples():  # window after conceding: do we score?
                    end = min(gg.t + 300, r.p1 if gg.period == 1 else r.p2)
                    wlen = max(end - gg.t, 0)
                    os_ += ((scored.period == gg.period) & (scored.t > gg.t) & (scored.t <= end)).sum(); es += rate_s * wlen
                rows.append((r.match_id, oc, ec, os_, es))
        A = pd.DataFrame(rows, columns=["m", "oc", "ec", "os", "es"]).groupby("m").sum()
        rng = np.random.default_rng(21)
        def ratio(col_o, col_e, idx=None):
            X = A if idx is None else A.iloc[idx]
            return X[col_o].sum() / max(X[col_e].sum(), 1e-9)
        bc = [ratio("oc", "ec", rng.integers(0, len(A), len(A))) for _ in range(1000)]
        bs = [ratio("os", "es", rng.integers(0, len(A), len(A))) for _ in range(1000)]
        rc, rs = ratio("oc", "ec"), ratio("os", "es")
        res_c[p] = (float(np.log(rc)), float(np.std(np.log(bc)))); res_s[p] = (float(np.log(rs)), float(np.std(np.log(bs))))
        print(f"  {p}: concede after scoring O/E {rc:.2f} [{np.percentile(bc,2.5):.2f},{np.percentile(bc,97.5):.2f}] (O={A.oc.sum():.0f}); score after conceding O/E {rs:.2f} [{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}] (O={A.os.sum():.0f})")
    out["H21_2_concede_after_scoring"] = confirm(res_c, +1)
    out["H21_3_score_after_conceding"] = confirm(res_s, +1)
    # 21C goal kicks vs press (SB only)
    K = pd.read_parquet(os.path.join(PROC, "r21_goalkicks.parquet")); K["net"] = K.own - K.opp; K["short"] = (K.length < 32).astype(int)
    gk = {}
    for p in POPS[:3]:
        d = K[K["pop"] == p].copy(); d["tert"] = pd.qcut(d.opp_press.rank(method="first"), 3, labels=["low", "mid", "high"])
        def sl(x):
            a, b = x[x.short == 1].net, x[x.short == 0].net
            return a.mean() - b.mean(), np.sqrt(a.var() / len(a) + b.var() / len(b)), len(a), len(b)
        hi, lo = sl(d[d.tert == "high"]), sl(d[d.tert == "low"])
        gk[p] = dict(high=hi, low=lo, diff=float(hi[0] - lo[0]), se=float(np.sqrt(hi[1] ** 2 + lo[1] ** 2)), short_share=float(d.short.mean()))
        print(f"  {p}: short−long vs high press {hi[0]:+.4f} (n {hi[2]}/{hi[3]}), vs low press {lo[0]:+.4f}; difference {hi[0]-lo[0]:+.4f} ± {gk[p]['se']:.4f}")
    out["H21_4_goalkicks"] = dict(per_pop=gk, all_negative=bool(all(v["diff"] < 0 for v in gk.values())))
    for k, v in out.items():
        print(k, "confirmed" if v.get("confirmed", v.get("all_negative")) else "NOT confirmed")
    json.dump(out, open(os.path.join(ROOT, "results", "r21_moments.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    {"build": build, "test": test}[sys.argv[1]]()

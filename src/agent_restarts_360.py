"""Agent 'restarts': quick-restart mechanism test on StatsBomb 360 (PREREGISTRATION_AGENT_restarts.md).
  python src/agent_restarts_360.py build  -> data/proc/agent_restarts_360.parquet
  python src/agent_restarts_360.py test   -> results/agent_restarts.json"""
import gzip, json, os, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import RAW, DISC, WOMEN, tsec
ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")
OUT = os.path.join(PROC, "agent_restarts_360.parquet")


def pop_of(comp):
    return "disc" if comp.startswith(DISC) else ("women" if comp.startswith(WOMEN) else "club")


def one(m):
    mid = m["match_id"]
    try:
        ev = [e for e in json.load(gzip.open(os.path.join(RAW, f"{mid}_events.json.gz"))) if e["period"] <= 2]
        fr = {f["event_uuid"]: f for f in json.load(gzip.open(os.path.join(RAW, f"{mid}_three-sixty.json.gz")))}
    except Exception:
        return []
    S = {}
    for e in ev:
        if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] != "Penalty":
            S.setdefault(e["period"], []).append((tsec(e), e["shot"].get("statsbomb_xg", 0.0), e["team"]["name"]))
    rows = []
    for i, e in enumerate(ev):
        if e["type"]["name"] != "Pass" or i == 0:
            continue
        kind = e["pass"].get("type", {}).get("name")
        if kind not in ("Free Kick", "Throw-in") or e["location"][0] < 30:
            continue
        prev = ev[i - 1]
        if prev["period"] != e["period"]:
            continue
        f = fr.get(e["id"])
        if f is None:
            continue
        t = tsec(e); x, y = e["location"][:2]; team = e["team"]["name"]
        ff = [p for p in f["freeze_frame"] if not p.get("keeper") and not p.get("actor")]
        opp = np.array([p["location"] for p in ff if not p["teammate"]]).reshape(-1, 2)
        mat = np.array([p["location"] for p in ff if p["teammate"]]).reshape(-1, 2)
        r = dict(match_id=mid, comp=m["_comp"], pop=pop_of(m["_comp"]), period=e["period"], t=t, team=team,
                 kind="FK" if kind == "Free Kick" else "TI", x=x, y=y, prev_type=prev["type"]["name"],
                 delay=t - (tsec(prev) + (prev.get("duration") or 0.0)),
                 opp_gs=int((opp[:, 0] > x).sum()), opp_vis=len(opp), opp10=int((np.hypot(opp[:, 0] - x, opp[:, 1] - y) <= 10).sum()),
                 mates_ahead=int((mat[:, 0] > x).sum()), mates_vis=len(mat), n_vis=len(f["freeze_frame"]))
        for w in (20, 45):
            s = S.get(e["period"], [])
            r[f"own{w}"] = sum(v for tt, v, tm in s if t - 0.01 <= tt <= t + w and tm == team)
            r[f"opp{w}"] = sum(v for tt, v, tm in s if t - 0.01 <= tt <= t + w and tm != team)
        rows.append(r)
    return rows


def build():
    M = json.load(open(os.path.join(ROOT, "data", "raw", "sb360_matches.json")))
    R = []
    with ProcessPoolExecutor(2) as ex:
        for r in ex.map(one, M, chunksize=8):
            R += r
    D = pd.DataFrame(R); D.to_parquet(OUT); print(len(D), D.groupby("pop").size().to_dict())


# ---------------------------------------------------------------- estimation
def prep(D):
    D = D[(D.delay >= 0) & (D.delay <= 90)].copy()
    D["q"] = np.where(D.delay < 6, 1, np.where(D.delay > 15, 0, -1))
    D = D[D.q >= 0].copy()
    D["zone"] = pd.cut(D.x, [29.99, 60, 80, 96, 121], labels=["30-60", "60-80", "80-96", "96+"], right=False).astype(str)
    D["base"] = D.kind + "_" + D.zone + "_" + np.where((D.y - 40).abs() <= 20, "c", "w")
    D["gsband"] = pd.cut(D.opp_gs, [-1, 4, 6, 8, 99], labels=["0-4", "5-6", "7-8", "9+"]).astype(str)
    D["o10band"] = np.minimum(D.opp10, 2).astype(str)
    D["net20"] = D.own20 - D.opp20; D["net45"] = D.own45 - D.opp45
    D["gs_share"] = D.opp_gs / D.opp_vis.clip(lower=1)
    return D


def cell_diff(d, y, cell, minn=5):
    g = d.groupby([cell, "q"])[y].agg(["sum", "size"]).unstack("q")
    g = g.dropna()
    g = g[(g[("size", 1)] >= minn) & (g[("size", 0)] >= minn)]
    if g.empty:
        return np.nan
    w = np.minimum(g[("size", 1)], g[("size", 0)])
    diff = g[("sum", 1)] / g[("size", 1)] - g[("sum", 0)] / g[("size", 0)]
    return float((w * diff).sum() / w.sum())


def boot(d, fn, B=500, seed=7):
    est = fn(d)
    mids = d.match_id.unique(); rng = np.random.default_rng(seed)
    idx = d.groupby("match_id").indices
    bs = []
    for _ in range(B):
        pick = rng.choice(mids, len(mids))
        bd = d.iloc[np.concatenate([idx[m] for m in pick])]
        v = fn(bd)
        if np.isfinite(v):
            bs.append(v)
    return est, float(np.std(bs))


def tests():
    D = prep(pd.read_parquet(OUT))
    att = lambda d: d[d.x >= 60]; own = lambda d: d[d.x < 60]
    T = {
        "H1a_opp_goalside": (-1, lambda d: cell_diff(d, "opp_gs", "base")),
        "H1b_opp_within10": (-1, lambda d: cell_diff(d, "opp10", "base")),
        "H0_net_base": (+1, lambda d: cell_diff(d, "net20", "base")),
        "H2_net_within_gsband": (+1, lambda d: cell_diff(d.assign(c=d.base + "_" + d.gsband), "net20", "c")),
        "H3a_att_minus_own": (+1, lambda d: cell_diff(att(d), "net20", "base") - cell_diff(own(d), "net20", "base")),
        "H3b_TI_minus_FK_att": (+1, lambda d: cell_diff(att(d)[att(d).kind == "TI"], "net20", "base") - cell_diff(att(d)[att(d).kind == "FK"], "net20", "base")),
        "H4_net_ownhalf": (+1, lambda d: cell_diff(own(d), "net20", "base")),
    }
    desc = {
        "H1c_mates_ahead": lambda d: cell_diff(d, "mates_ahead", "base"),
        "opp_visible": lambda d: cell_diff(d, "opp_vis", "base"),
        "gs_share_of_visible": lambda d: cell_diff(d, "gs_share", "base"),
        "H0_net_att_half": lambda d: cell_diff(att(d), "net20", "base"),
        "H2_att_half": lambda d: cell_diff(att(d).assign(c=att(d).base + "_" + att(d).gsband), "net20", "c"),
    }
    out = dict(n={}, delay_quantiles={}, tests={}, descriptive={})
    raw = pd.read_parquet(OUT)
    for p in ("disc", "women", "club"):
        r = raw[raw["pop"] == p]
        out["delay_quantiles"][p] = dict(n_all=len(r), share_neg=float((r.delay < 0).mean()), share_gt90=float((r.delay > 90).mean()),
                                         q=r.delay.quantile([.05, .25, .5, .75, .95]).round(2).to_dict(),
                                         by_kind_median=r.groupby("kind").delay.median().round(1).to_dict())
        d = D[D["pop"] == p]
        out["n"][p] = dict(quick=int((d.q == 1).sum()), slow=int((d.q == 0).sum()), matches=int(d.match_id.nunique()),
                           quick_by_kind_zone=d[d.q == 1].groupby(["kind", "zone"]).size().rename(lambda t: str(t)).to_dict() if False else
                           {f"{k}_{z}": int(v) for (k, z), v in d.groupby(["kind", "zone", "q"]).size().unstack("q").fillna(0)[1].items()},
                           slow_by_kind_zone={f"{k}_{z}": int(v) for (k, z), v in d.groupby(["kind", "zone", "q"]).size().unstack("q").fillna(0)[0].items()})
    for name, (sign, fn) in T.items():
        res = {p: boot(D[D["pop"] == p], fn) for p in ("disc", "women", "club")}
        out["tests"][name] = dict(sign=sign, per_pop={p: dict(est=v[0], se=v[1]) for p, v in res.items()})
        print(name, {p: f"{v[0]:+.4f}±{v[1]:.4f}" for p, v in res.items()})
    for name, fn in desc.items():
        res = {p: boot(D[D["pop"] == p], fn, B=300) for p in ("disc", "women", "club")}
        out["descriptive"][name] = {p: dict(est=v[0], se=v[1]) for p, v in res.items()}
        print("desc", name, {p: f"{v[0]:+.4f}±{v[1]:.4f}" for p, v in res.items()})
    # discovery BH
    names = list(T)
    pd_ = [norm.sf(out["tests"][n]["sign"] * out["tests"][n]["per_pop"]["disc"]["est"] / out["tests"][n]["per_pop"]["disc"]["se"]) for n in names]
    rej = multipletests(pd_, 0.05, "fdr_bh")[0]
    passed = []
    for n, p0, r in zip(names, pd_, rej):
        t = out["tests"][n]; s = t["sign"]; pp = t["per_pop"]
        w = np.array([1 / pp[k]["se"] ** 2 for k in ("women", "club")]); e = np.array([pp[k]["est"] for k in ("women", "club")])
        pooled = float((w * e).sum() / w.sum()); pse = float(1 / np.sqrt(w.sum()))
        t.update(disc_p_one=float(p0), disc_bh=bool(r), both_conf_same_sign=bool(all(np.sign(e) == s)),
                 pooled_conf=pooled, pooled_se=pse, pooled_p_one=float(norm.sf(s * pooled / pse)))
        if r:
            passed.append(n)
    if passed:
        hp = multipletests([out["tests"][n]["pooled_p_one"] for n in passed], 0.05, "holm")[0]
        for n, h in zip(passed, hp):
            out["tests"][n]["holm_ok"] = bool(h)
    for n in names:
        t = out["tests"][n]
        t["confirmed"] = bool(t["disc_bh"] and t["both_conf_same_sign"] and t.get("holm_ok", False))
        print(n, "disc p", round(t["disc_p_one"], 4), "BH", t["disc_bh"], "pooled", f"{t['pooled_conf']:+.4f}±{t['pooled_se']:.4f}", "p", round(t["pooled_p_one"], 4), "CONFIRMED" if t["confirmed"] else "")
    # robustness for confirmed
    rob = {}
    for n in names:
        if not out["tests"][n]["confirmed"]:
            continue
        if n.startswith("H1a"):
            rob[n + "_finer_y_band"] = {p: boot(D[D["pop"] == p].assign(c=lambda d: d.base + "_" + pd.cut(d.y, [0, 18, 30, 50, 62, 81]).astype(str)),
                                                lambda d: cell_diff(d, "opp_gs", "c"), B=300) for p in ("disc", "women", "club")}
            rob[n + "_share_of_visible"] = out["descriptive"]["gs_share_of_visible"]
        else:
            att_ = lambda d: d[d.x >= 60]
            def contrast(y, cellf):
                def f(d):
                    a = att_(d).assign(c=cellf)
                    return cell_diff(a[a.kind == "TI"], y, "c") - cell_diff(a[a.kind == "FK"], y, "c")
                return f
            base_c = lambda d: d.base
            fine_c = lambda d: d.base + "_" + d.gsband + "_" + d.o10band
            if n.startswith("H3b"):
                rob[n + "_h45"] = {p: boot(D[D["pop"] == p], contrast("net45", base_c), B=300) for p in ("disc", "women", "club")}
                rob[n + "_within_gsband"] = {p: boot(D[D["pop"] == p], contrast("net20", lambda d: d.base + "_" + d.gsband), B=300) for p in ("disc", "women", "club")}
                rob[n + "_finer_gs_opp10"] = {p: boot(D[D["pop"] == p], contrast("net20", fine_c), B=300) for p in ("disc", "women", "club")}
                for k in ("TI", "FK"):
                    rob[f"component_{k}_att_quick_minus_slow"] = {p: boot(D[D["pop"] == p], lambda d, k=k: cell_diff(att_(d)[att_(d).kind == k], "net20", "base"), B=300) for p in ("disc", "women", "club")}
                    rob[f"component_{k}_att_h45"] = {p: boot(D[D["pop"] == p], lambda d, k=k: cell_diff(att_(d)[att_(d).kind == k], "net45", "base"), B=300) for p in ("disc", "women", "club")}
    out["robustness"] = {k: {p: dict(est=v[0], se=v[1]) for p, v in r.items()} if isinstance(next(iter(r.values())), tuple) else r for k, r in rob.items()}
    for k, r in out["robustness"].items():
        print("rob", k, r)
    # attenuation (H2 vs H0) on same data
    out["attenuation_H2_over_H0"] = {p: out["tests"]["H2_net_within_gsband"]["per_pop"][p]["est"] / out["tests"]["H0_net_base"]["per_pop"][p]["est"]
                                     for p in ("disc", "women", "club")}
    json.dump(out, open(os.path.join(ROOT, "results", "agent_restarts.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    {"build": build, "test": tests}[sys.argv[1]]()

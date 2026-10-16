"""Agent keeper: tests per PREREGISTRATION_AGENT_keeper.md -> results/agent_keeper.json"""
import json, os
import numpy as np
import pandas as pd
from scipy.stats import norm
ROOT = os.path.join(os.path.dirname(__file__), "..")
D = pd.read_parquet(os.path.join(ROOT, "data", "proc", "agent_keeper_releases.parquet"))
D["gs"] = np.sign(D.gd).astype(int).astype(str)
for w in (30, 60):
    D[f"net{w}"] = D[f"own{w}"] - D[f"opp{w}"]
D["speed"] = np.where(D.delay < 5, "q", np.where(D.delay > 10, "s", "m"))
D["kxb"] = pd.cut(D.kx, [-1, 6, 18, 200], labels=["a", "b", "c"]).astype(str)
D["cellH1"] = D.rtype + "|" + D.gs + "|" + D.gtype
D["cellH3"] = D.gs + "|" + D.gtype
D["cellR"] = D.cellH1 + "|" + D.pressed.astype(str) + "|" + D.kxb
D["short"] = np.where(D.length < 35, "A", "B")
D["H1grp"] = np.where(D.speed == "q", "A", np.where(D.speed == "s", "B", ""))
B = 500
rng = np.random.default_rng(7)


def est(d, cell, grp, y, w):
    g = pd.DataFrame({"c": d[cell].values, "g": d[grp].values, "yw": d[y].values * w, "w": w})
    g = g[g.g.isin(["A", "B"])]
    a = g.groupby(["c", "g"])[["yw", "w"]].sum().unstack("g")
    a = a.dropna()
    nA, nB = a[("w", "A")], a[("w", "B")]
    ok = (nA >= 5) & (nB >= 5)
    if not ok.any():
        return np.nan
    diff = (a[("yw", "A")] / nA - a[("yw", "B")] / nB)[ok]
    wt = np.minimum(nA, nB)[ok]
    return float((diff * wt).sum() / wt.sum())


def boot(d, fn):
    """fn(d, weights) -> estimate; match-cluster bootstrap via multinomial match weights."""
    d = d.reset_index(drop=True)
    pt = fn(d, np.ones(len(d)))
    codes, uniq = pd.factorize(d.match_id)
    reps = []
    for _ in range(B):
        cnt = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)))
        reps.append(fn(d, cnt[codes].astype(float)))
    reps = np.array(reps)
    return pt, float(np.nanstd(reps))


def n_cells(d, cell, grp):
    g = d[d[grp].isin(["A", "B"])].groupby([cell, grp]).size().unstack(grp).fillna(0)
    ok = (g.get("A", 0) >= 5) & (g.get("B", 0) >= 5)
    return dict(cells=int(ok.sum()), nA=int(g.loc[ok, "A"].sum()) if ok.any() else 0, nB=int(g.loc[ok, "B"].sum()) if ok.any() else 0)


POPS = ["SB_1516", "SB_women_ISL", "SB_360comps"]


def per_pop(sub, cell, grp, y):
    r = {}
    for p in POPS:
        d = sub[sub["pop"] == p]
        e, se = boot(d, lambda x, w: est(x, cell, grp, y, w))
        r[p] = dict(est=e, se=se, z=e / se, **n_cells(d, cell, grp))
    return r


def confirm(r, sign):
    o = [r[p] for p in POPS[1:]]
    w = np.array([1 / v["se"] ** 2 for v in o]); pooled = float((w * np.array([v["est"] for v in o])).sum() / w.sum()); pse = float(1 / np.sqrt(w.sum()))
    d0 = r["SB_1516"]
    return dict(disc_p_one=float(norm.sf(sign * d0["est"] / d0["se"])), disc_sign_ok=bool(np.sign(d0["est"]) == sign),
                n_same_sign_others=int(sum(np.sign(v["est"]) == sign for v in o)), pooled_others=pooled, pooled_se=pse,
                pooled_p_one=float(norm.sf(sign * pooled / pse)))


def main():
    out = {"n": D.groupby("pop").size().to_dict()}
    # descriptive frequencies
    out["speed_share"] = (D.groupby("pop").speed.value_counts(normalize=True).unstack().round(3)).to_dict(orient="index")
    out["speed_share_by_rtype"] = {f"{k[0]}|{k[1]}": v for k, v in D.groupby(["pop", "rtype"]).speed.value_counts(normalize=True).unstack().round(3).to_dict(orient="index").items()}
    out["rtype_share"] = D.groupby("pop").rtype.value_counts(normalize=True).unstack().round(3).to_dict(orient="index")
    out["median_delay"] = D.groupby("pop").delay.median().round(2).to_dict()
    tests = {}
    tests["H1a_own30"] = (per_pop(D, "cellH1", "H1grp", "own30"), 1)
    tests["H1b_net30"] = (per_pop(D, "cellH1", "H1grp", "net30"), 1)
    Q = D[D.speed == "q"]
    r3 = per_pop(Q, "cellH3", "short", "net30")
    s3 = int(np.sign(r3["SB_1516"]["est"]))
    tests["H3_short_minus_long_net30_quick"] = (r3, s3)
    res = {}
    for k, (r, s) in tests.items():
        res[k] = dict(per_pop=r, direction=s, **confirm(r, s))
    res["H3_short_minus_long_net30_quick"]["disc_two_sided_p"] = float(2 * norm.sf(abs(r3["SB_1516"]["z"])))
    # Holm over 3 event tests on pooled p
    ks = list(tests); ps = np.array([res[k]["pooled_p_one"] for k in ks]); order = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, min(1, (m - rank) * ps[i])); adj[i] = run
    for k, a in zip(ks, adj):
        r = res[k]; r["pooled_p_holm"] = float(a)
        ok = r["disc_sign_ok"] and r["n_same_sign_others"] == 2 and a < 0.05
        if k.startswith("H3"):
            ok = ok and res[k]["disc_two_sided_p"] < 0.05
        r["confirmed"] = bool(ok)
    # secondary H3 within release type
    res["H3_secondary_within_rtype"] = per_pop(Q.assign(c2=Q.cellH3 + "|" + Q.rtype), "c2", "short", "net30")
    # descriptive raw means
    res["raw_means"] = D[D.speed != "m"].groupby(["pop", "speed"])[["own30", "opp30", "net30"]].mean().round(4).reset_index().to_dict(orient="records")
    # H2 (360)
    S = D[(D["pop"] == "SB_360comps") & D.has_frame.fillna(False).astype(bool)].copy()
    S["hi"] = S.n_opp_half > 5  # discovery median = 5, fixed before outcomes
    def h2(x, w):
        hi = x.hi.values
        return est(x[hi], "cellH1", "H1grp", "net30", w[hi]) - est(x[~hi], "cellH1", "H1grp", "net30", w[~hi])
    h2r = {}
    for s in ("DISC", "WOMEN", "CLUB"):
        d = S[S["sub"] == s]
        e, se = boot(d, h2)
        eh, seh = boot(d[d.hi], lambda x, w: est(x, "cellH1", "H1grp", "net30", w))
        el, sel = boot(d[~d.hi], lambda x, w: est(x, "cellH1", "H1grp", "net30", w))
        h2r[s] = dict(est=e, se=se, quick_adv_high=eh, se_high=seh, quick_adv_low=el, se_low=sel, n=len(d), n_high=int(d.hi.sum()),
                      n_vis_mean=float(d.n_vis.mean()), n_vis_quick=float(d[d.speed == "q"].n_vis.mean()), n_vis_slow=float(d[d.speed == "s"].n_vis.mean()))
    o = [h2r["WOMEN"], h2r["CLUB"]]; w = np.array([1 / v["se"] ** 2 for v in o]); pooled = float((w * np.array([v["est"] for v in o])).sum() / w.sum()); pse = float(1 / np.sqrt(w.sum()))
    dp = float(norm.sf(h2r["DISC"]["est"] / h2r["DISC"]["se"]))
    res["H2_360"] = dict(per_sub=h2r, disc_p_one=dp, pooled_conf=pooled, pooled_se=pse, pooled_p_one=float(norm.sf(pooled / pse)),
                         confirmed=bool(dp < 0.05 and all(v["est"] > 0 for v in o) and norm.sf(pooled / pse) < 0.05))
    # robustness (reported regardless): 60 s horizon and finer cells, for H1
    res["robust_H1b_net60"] = per_pop(D, "cellH1", "H1grp", "net60")
    res["robust_H1b_finer"] = per_pop(D, "cellR", "H1grp", "net30")
    res["robust_H1a_own60"] = per_pop(D, "cellH1", "H1grp", "own60")
    out["tests"] = res
    json.dump(out, open(os.path.join(ROOT, "results", "agent_keeper.json"), "w"), indent=1, default=float)
    for k in ("H1a_own30", "H1b_net30", "H3_short_minus_long_net30_quick"):
        r = res[k]; print(k, {p: (round(v["est"], 4), round(v["se"], 4), v["nA"], v["nB"]) for p, v in r["per_pop"].items()},
                          "pooled", round(r["pooled_others"], 4), "p", round(r["pooled_p_one"], 4), "holm", round(r["pooled_p_holm"], 3), "disc_p", round(r["disc_p_one"], 4), "conf", r["confirmed"])
    print("H2", json.dumps(res["H2_360"], default=float, indent=0))
    for k in ("robust_H1b_net60", "robust_H1b_finer", "robust_H1a_own60", "H3_secondary_within_rtype"):
        print(k, {p: (round(v["est"], 4), round(v["se"], 4)) for p, v in res[k].items()})
    print(json.dumps({k: out[k] for k in ("speed_share", "median_delay", "rtype_share")}, indent=0))
    print(json.dumps(out["speed_share_by_rtype"]))
    print(res["raw_means"])


if __name__ == "__main__":
    main()

"""agent corners360: pre-registered H1-H3 (PREREGISTRATION_AGENT_corners360.md) -> results/agent_corners360.json"""
import json, os, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
ROOT = os.path.join(os.path.dirname(__file__), "..")
D0 = pd.read_parquet(os.path.join(ROOT, "data", "proc", "agent_corners360.parquet"))
POPS = ["discovery", "confirm_club", "confirm_women"]


def fe_ols(d, y, X, absorb):
    """within-estimator: demean y and X by `absorb` group, OLS, cluster by match."""
    d = d.copy()
    cols = [y] + X
    d[cols] = d[cols].astype(float) - d.groupby(absorb)[cols].transform("mean")
    res = sm.OLS(d[y], d[X]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.match_id)[0]})
    return res


def prep(D):
    D = D.copy()
    D["side"] = (D.y0 > 40).astype(int)
    D["tc"] = D.team + "|" + D.comp
    D["net"] = D.own20 - D.opp30
    D["net_long"] = D.own45 - D.opp60
    return D


def est(D, hyp, y=None):
    if hyp == "H1":
        r = fe_ols(D, y or "own20", ["our_box", "opp_box", "side"], "tc"); k = "our_box"; n = len(D)
    elif hyp == "H3":
        r = fe_ols(D, y or "opp30", ["our_box", "opp_box", "side"], "tc"); k = "our_box"; n = len(D)
    else:
        d = D[(D.our_six >= 3) | (D.our_six <= 1)].copy(); d["hi6"] = (d.our_six >= 3).astype(int)
        d["g"] = d.tc + "|" + d.our_box.astype(str)  # team FE x box-count strata (nested absorb approximated by interacted groups)
        dd = pd.get_dummies(d.our_box.clip(upper=9), prefix="b", drop_first=True).astype(float)
        d = pd.concat([d, dd], axis=1)
        r = fe_ols(d, y or "own20", ["hi6", "opp_box", "side"] + list(dd.columns), "tc"); k = "hi6"; n = len(d)
        return dict(est=float(r.params[k]), se=float(r.bse[k]), n=int(n), n_hi=int(d.hi6.sum()))
    return dict(est=float(r.params[k]), se=float(r.bse[k]), n=int(n))


def run(D, label):
    out = {}
    for h in ("H1", "H2", "H3"):
        out[h] = {p: est(D[D["pop"] == p], h) for p in POPS}
        e = out[h]["discovery"]; e["p1"] = float(norm.sf(e["est"] / e["se"]))
    q = multipletests([out[h]["discovery"]["p1"] for h in out], method="fdr_bh")[1]
    carried = []
    for h, qq in zip(out, q):
        out[h]["disc_q"] = float(qq)
        c = [out[h][p] for p in POPS[1:]]
        w = np.array([1 / x["se"] ** 2 for x in c]); pooled = float((w * np.array([x["est"] for x in c])).sum() / w.sum()); pse = float(1 / np.sqrt(w.sum()))
        out[h]["conf_pooled"] = dict(est=pooled, se=pse, p1=float(norm.sf(pooled / pse)), same_sign=bool(all(x["est"] > 0 for x in c)))
        if qq < 0.05:
            carried.append(h)
    k = len(carried); flag = True
    for i, h in enumerate(sorted(carried, key=lambda h: out[h]["conf_pooled"]["p1"])):
        flag = flag and out[h]["conf_pooled"]["p1"] < 0.05 / (k - i)
        out[h]["confirmed"] = bool(flag and out[h]["conf_pooled"]["same_sign"])
    for h in out:
        out[h].setdefault("confirmed", False)
    # pooled all three (descriptive)
    for h in out:
        out[h]["all"] = est(D, h)
    print(f"\n== {label} ==")
    for h in out:
        print(h, " | ".join(f"{p[:12]} {out[h][p]['est']:+.4f}±{out[h][p]['se']:.4f} n={out[h][p]['n']}" for p in POPS),
              f"| q={out[h]['disc_q']:.3f} conf={out[h]['conf_pooled']['est']:+.4f}±{out[h]['conf_pooled']['se']:.4f} -> {out[h]['confirmed']}",
              f"| all {out[h]['all']['est']:+.4f}±{out[h]['all']['se']:.4f}")
    return out


def sweet(D):
    """FE-adjusted mean net by our_box bin, per population and all."""
    res = {}
    for name, d in [("all", D)] + [(p, D[D["pop"] == p]) for p in POPS]:
        d = d.copy(); d["bin"] = d.our_box.clip(3, 7)
        dd = pd.get_dummies(d.bin, prefix="k").astype(float)
        d = pd.concat([d, dd], axis=1)
        X = list(dd.columns[1:]) + ["opp_box", "side"]
        row = {}
        for y in ("own20", "opp30", "net"):
            r = fe_ols(d, y, X, "tc")
            row[y] = {c: (float(r.params[c]), float(r.bse[c])) for c in dd.columns[1:]}
        row["n_by_bin"] = d.bin.value_counts().sort_index().to_dict()
        res[name] = row
    return res


if __name__ == "__main__":
    D = prep(D0)
    F = D[D.has_frame]
    cov = {p: dict(corners=int((D["pop"] == p).sum()), with_frame=int((F["pop"] == p).sum()),
                   cov85=int(((F["pop"] == p) & (F.cov_box >= .85) & (F.cov_six >= .999)).sum()),
                   cov95=int(((F["pop"] == p) & (F.cov_box >= .95) & (F.cov_six >= .999)).sum())) for p in POPS}
    print(cov)
    P = F[(F.cov_box >= .85) & (F.cov_six >= .999)]
    print(P.groupby("pop")[["our_box", "our_six", "our_edge", "opp_box", "opp_six", "own20", "opp30"]].mean().round(3))
    print(pd.crosstab(P.our_box, P["pop"]))
    res = dict(coverage=cov, primary=run(P, "primary (cov>=85%)"))
    res["rob_cov95"] = run(P[P.cov_box >= .95], "cov>=95%")
    Q = P.copy(); Q["own20"] = Q.own45; Q["opp30"] = Q.opp60
    res["rob_h45_60"] = run(Q, "horizon 45/60 s")
    res["rob_no_short"] = run(P[P.end_in_box], "excluding short corners")
    res["sweetspot"] = sweet(P)
    for k, v in res["sweetspot"].items():
        print(k, v["n_by_bin"], {y: {c: round(a, 4) for c, (a, b) in v[y].items()} for y in ("own20", "opp30", "net")})
    json.dump(res, open(os.path.join(ROOT, "results", "agent_corners360.json"), "w"), indent=1)

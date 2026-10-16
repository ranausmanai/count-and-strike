"""Agent secondball: H1/H2/H3 + robustness -> results/agent_secondball.json"""
import json, os, sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import PROC, ROOT

POPS = ["discovery", "confirm_club", "confirm_women"]
D = pd.read_parquet(os.path.join(PROC, "agent_secondball_lb.parquet"))
D = D[D.vis == 1].copy()
D["zx"] = pd.cut(D.ex, [-1, 40, 60, 80, 102, 121], labels=False).astype(int)
D["zy"] = np.select([(D.ey - 40).abs() <= 13, D.ey < 40], [1, 0], 2)
D["ez"] = D.zx.astype(str) + D.zy.astype(str)
D["bal"] = D.ours - D.theirs; D["bal10"] = D.ours10 - D.theirs10
D["crowd"] = np.select([D.ours + D.theirs <= 2, D.ours + D.theirs <= 5], ["a", "b"], "c")
CTRL = "C(ez) + C(height) + pressure + gk"


def reg(d, y, x="bal", extra=""):
    d = d.dropna(subset=[y])
    m = smf.ols(f"{y} ~ {x} + {CTRL}{extra}", d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.match_id)[0]})
    return dict(est=float(m.params[x]), se=float(m.bse[x]), n=int(len(d)))


def strat(d, y, cellcol="ez", bal="bal"):
    d = d[(d.height == "High Pass") & (d[bal] != 0)].dropna(subset=[y])
    num = 0.0; var = 0.0; W = 0.0; n = 0
    for c, g in d.groupby(cellcol):
        s = g[g[bal] > 0][y]; o = g[g[bal] < 0][y]
        if len(s) < 5 or len(o) < 5:
            continue
        w = len(s) * len(o) / (len(s) + len(o)); diff = s.mean() - o.mean(); se2 = s.var() / len(s) + o.var() / len(o)
        num += w * diff; var += w * w * se2; W += w; n += len(s) + len(o)
    return dict(est=num / W, se=float(np.sqrt(var) / W), n=int(n)) if W else dict(est=np.nan, se=np.nan, n=0)


TESTS = {"H1_keep5": lambda d: reg(d, "keep5"), "H2_net": lambda d: reg(d, "net"), "H3_high_net": lambda d: strat(d, "net")}
res = {"n_visible": D.groupby("pop").size().to_dict(), "usage": {}, "tests": {}, "descriptive": {}, "robust": {}}
raw = pd.read_parquet(os.path.join(PROC, "agent_secondball_lb.parquet"))
res["n_long_balls_with_frame"] = raw.groupby("pop").size().to_dict()
for p in POPS:
    d = D[D["pop"] == p]
    res["usage"][p] = dict(outnumbered=float((d.bal < 0).mean()), level=float((d.bal == 0).mean()), superior=float((d.bal > 0).mean()),
                           high_outnumbered=float((d[d.height == "High Pass"].bal < 0).mean()), mean_ours=float(d.ours.mean()), mean_theirs=float(d.theirs.mean()),
                           keep5_rate=float(d.keep5.mean()))
for k, f in TESTS.items():
    res["tests"][k] = {p: f(D[D["pop"] == p]) for p in POPS}
res["descriptive"]["H3_high_keep5"] = {p: strat(D[D["pop"] == p], "keep5") for p in POPS}
res["descriptive"]["H1_raw_keep5_by_bal"] = {p: D[D["pop"] == p].assign(b=D.bal.clip(-3, 3)).groupby("b").keep5.agg(["mean", "size"]).round(3).to_dict() for p in POPS}

# discovery BH
disc = res["tests"]
ps = [2 * norm.sf(abs(disc[k]["discovery"]["est"] / disc[k]["discovery"]["se"])) for k in TESTS]
q = multipletests(ps, method="fdr_bh")[1]
carried = []
for (k, pv, qv) in zip(TESTS, ps, q):
    t = disc[k]; t["disc_p"] = float(pv); t["disc_q"] = float(qv); t["disc_pass"] = bool(qv < 0.05 and t["discovery"]["est"] > 0)
    e = [t["confirm_club"], t["confirm_women"]]
    w = np.array([1 / x["se"] ** 2 for x in e]); pooled = float(sum(wi * x["est"] for wi, x in zip(w, e)) / w.sum()); pse = float(1 / np.sqrt(w.sum()))
    t["pooled"] = pooled; t["pooled_se"] = pse; t["pooled_p1"] = float(norm.sf(pooled / pse)); t["same_sign"] = bool(all(x["est"] > 0 for x in e))
    if t["disc_pass"]:
        carried.append(k)
order = sorted(carried, key=lambda k: disc[k]["pooled_p1"]); flag = True
for i, k in enumerate(order):
    flag = flag and disc[k]["pooled_p1"] < 0.05 / (len(order) - i)
    disc[k]["confirmed"] = bool(flag and disc[k]["same_sign"])
for k in TESTS:
    disc[k].setdefault("confirmed", False)

# robustness (run for all, reported)
R = res["robust"]
for p in POPS:
    d = D[D["pop"] == p]
    R[p] = dict(H1_crowd=reg(d, "keep5", extra=" + C(crowd)"), H2_crowd=reg(d, "net", extra=" + C(crowd)"),
                H2_net45=reg(d, "net45"), H1_r10=reg(d, "keep5", x="bal10"), H2_r10=reg(d, "net", x="bal10"),
                H3_net45=strat(d, "net45"), H3_crowd=strat(d.assign(c2=d.ez + d.crowd), "net", cellcol="c2"),
                H1_ours_theirs=reg(d, "keep5", x="ours", extra=" + theirs"), H1_theirs=reg(d, "keep5", x="theirs", extra=" + ours"))
json.dump(res, open(os.path.join(ROOT, "results", "agent_secondball.json"), "w"), indent=1, default=float)
for k in TESTS:
    t = disc[k]
    print(k, {p: (round(t[p]["est"], 4), round(t[p]["se"], 4), t[p]["n"]) for p in POPS}, "q", round(t["disc_q"], 4), "pooled", round(t["pooled"], 4), round(t["pooled_se"], 4), "conf", t["confirmed"])
print(json.dumps(res["usage"], indent=0)); print(json.dumps(res["descriptive"]["H3_high_keep5"]))
for p in POPS:
    print(p, {k: (round(v["est"], 4), round(v["se"], 4)) for k, v in R[p].items()})

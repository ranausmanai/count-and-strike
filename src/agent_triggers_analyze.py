"""Agent 'triggers': press-vs-hold under pressing triggers, within cells; interaction tests; confirmation; robustness.
-> results/agent_triggers.json"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import ROOT

D0 = pd.read_parquet(os.path.join(ROOT, "data", "proc", "agent_triggers_receipts.parquet"))
TRIG = ["T_backward", "T_touchline", "T_poor_touch", "T_long_air", "T_cm_crowded"]
POPS = ["discovery", "confirm_club", "confirm_women"]
MINN = 5; B = 300
rng = np.random.default_rng(7)


def arrays(D, trig, cellcol, y):
    """match x cell x press x trig sums/counts as dense arrays [M, K, 2, 2]."""
    mi, mlev = pd.factorize(D.match_id); ci, clev = pd.factorize(D[cellcol])
    M, K = len(mlev), len(clev)
    n = np.zeros((M, K, 2, 2)); s = np.zeros((M, K, 2, 2))
    np.add.at(n, (mi, ci, D.press.values.astype(int), D[trig].values.astype(int)), 1)
    np.add.at(s, (mi, ci, D.press.values.astype(int), D[trig].values.astype(int)), D[y].values)
    return n, s


def est(n, s):
    """n,s: [K,2(press),2(trig)] -> (conditional effect under trigger, interaction)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = s / n
        okT = (n[:, 1, 1] >= MINN) & (n[:, 0, 1] >= MINN)
        cT = mu[:, 1, 1] - mu[:, 0, 1]; wT = 1 / (1 / n[:, 1, 1] + 1 / n[:, 0, 1])
        C = np.sum(np.where(okT, wT * cT, 0)) / np.sum(np.where(okT, wT, 0))
        okI = okT & (n[:, 1, 0] >= MINN) & (n[:, 0, 0] >= MINN)
        cN = mu[:, 1, 0] - mu[:, 0, 0]; wI = 1 / (1 / n[:, 1, 1] + 1 / n[:, 0, 1] + 1 / n[:, 1, 0] + 1 / n[:, 0, 0])
        I = np.sum(np.where(okI, wI * (cT - cN), 0)) / np.sum(np.where(okI, wI, 0))
    return C, I, int(okT.sum()), int(okI.sum())


def analyse(D, trig, cellcol="cell", y="net"):
    n, s = arrays(D, trig, cellcol, y)
    C, I, kC, kI = est(n.sum(0), s.sum(0))
    M = n.shape[0]; bs = []
    for _ in range(B):
        w = rng.multinomial(M, np.ones(M) / M).astype(float)
        bs.append(est(np.tensordot(w, n, 1), np.tensordot(w, s, 1))[:2])
    bs = np.array(bs)
    return dict(C=float(C), C_se=float(np.nanstd(bs[:, 0])), I=float(I), I_se=float(np.nanstd(bs[:, 1])),
                cells_C=kC, cells_I=kI, n=int(len(D)), n_trig=int(D[trig].sum()),
                n_trig_press=int((D[trig] & D.press).sum()))


def run(D, cellcol="cell", y="net", label="main"):
    R = {}
    for t in TRIG:
        R[t] = {p: analyse(D[D["pop"] == p], t, cellcol, y) for p in POPS}
        print(label, t, {p: (round(v["C"], 4), round(v["C_se"], 4), round(v["I"], 4), round(v["I_se"], 4)) for p, v in R[t].items()})
    return R


def decide(R):
    tests = []
    for t in TRIG:
        for k in ("I", "C"):
            d = R[t]["discovery"]; z = d[k] / d[k + "_se"]
            tests.append(dict(trigger=t, stat=k, disc=d[k], disc_se=d[k + "_se"], p_two=float(2 * norm.sf(abs(z)))))
    q = multipletests([x["p_two"] for x in tests], method="fdr_bh")[1]
    for x, qq in zip(tests, q):
        x["q"] = float(qq); x["disc_hit"] = bool(qq < 0.05 and x["disc"] > 0)
        cs = [(R[x["trigger"]][p][x["stat"]], R[x["trigger"]][p][x["stat"] + "_se"]) for p in POPS[1:]]
        w = np.array([1 / c[1] ** 2 for c in cs]); pooled = float((w * np.array([c[0] for c in cs])).sum() / w.sum()); pse = float(1 / np.sqrt(w.sum()))
        x.update(club=cs[0][0], club_se=cs[0][1], women=cs[1][0], women_se=cs[1][1], pooled=pooled, pooled_se=pse,
                 p1_pooled=float(norm.sf(pooled / pse)), same_sign=bool(all(c[0] > 0 for c in cs)))
    hits = sorted([x for x in tests if x["disc_hit"]], key=lambda x: x["p1_pooled"]); flag = True
    for i, x in enumerate(hits):
        flag = flag and x["p1_pooled"] < 0.05 / (len(hits) - i); x["confirmed"] = bool(flag and x["same_sign"])
    for x in tests:
        x.setdefault("confirmed", False)
    return tests


def main():
    D = D0.copy()
    R = run(D)
    tests = decide(R)
    for x in tests:
        print(f"{x['trigger']:14s} {x['stat']} disc {x['disc']:+.4f}±{x['disc_se']:.4f} q={x['q']:.3f} | club {x['club']:+.4f}±{x['club_se']:.4f} "
              f"women {x['women']:+.4f}±{x['women_se']:.4f} pooled p1={x['p1_pooled']:.3f} hit={x['disc_hit']} conf={x['confirmed']}")
    out = dict(prereg="PREREGISTRATION_AGENT_triggers.md", n=D.groupby("pop").size().to_dict(),
               press_rate=D.groupby("pop").press.mean().round(4).to_dict(),
               trigger_rate={t: D.groupby("pop")[t].mean().round(4).to_dict() for t in TRIG},
               main=R, tests=tests)
    # Robustness (run for all triggers, reported for interpretation of anything confirmed or near-confirmed)
    D["cell_fine"] = D.cell + np.select([D.our_goalside <= 5, D.our_goalside <= 8], ["0", "1"], "2") + D.already.astype(str)
    out["robust_h45"] = run(D, y="net45", label="h45")
    out["robust_finecells"] = run(D, cellcol="cell_fine", label="fine")
    out["robust_not_already"] = run(D[D.already == 0], label="notalready")
    # poor-touch variant without miscontrol (incomplete receipts only) to gauge leakage
    D["T_incomplete"] = D.incomplete
    out["poor_touch_incomplete_only"] = {p: analyse(D[D["pop"] == p], "T_incomplete") for p in POPS}
    print("incomplete-only", {p: (round(v["C"], 4), round(v["I"], 4)) for p, v in out["poor_touch_incomplete_only"].items()})
    json.dump(out, open(os.path.join(ROOT, "results", "agent_triggers.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

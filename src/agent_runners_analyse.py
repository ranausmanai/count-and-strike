"""Agent 'runners' analysis per PREREGISTRATION_AGENT_runners.md."""
import json, os, sys
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
ROOT = os.path.join(os.path.dirname(__file__), "..")
R = pd.read_parquet(os.path.join(ROOT, "data", "proc", "agent_runners_regains.parquet"))
R["rx"] = R.x / 120; R["cent"] = 1 - (R.y - 40).abs() / 40; R["clk"] = (R.t / 60 + np.where(R.period == 2, 45, 0)) / 90
R["gpos"] = (R.gd > 0).astype(float); R["gneg"] = (R.gd < 0).astype(float)
R["band"] = np.select([R.opp_goalside <= 4, R.opp_goalside <= 6], ["le4", "5_6"], "ge7")
R["mbin"] = np.select([R.mates_ahead <= 2, R.mates_ahead <= 4, R.mates_ahead <= 6], ["a0_2", "b3_4", "c5_6"], "d7p")
R["net20"] = R.own20 - R.opp20; R["net20_30"] = R.own20 - R.opp30; R["net45"] = R.own45 - R.opp45
CTRL = "rx + I(rx**2) + I(rx**3) + cent + gpos + gneg + clk + opp_goalside + n_visible_opp"
POPS = ("discovery", "confirm_club", "confirm_women")
out = {}


def fit(d, f, fam="pois"):
    g = pd.factorize(d.match_id)[0]
    if fam == "pois":
        return smf.glm(f, data=d, family=sm.families.Poisson()).fit(scale="X2", cov_type="cluster", cov_kwds={"groups": g})
    return smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": g})


def coef_test(name, y, extra="", term="mates_ahead", fam="pois"):
    res = {}
    for p in POPS:
        d = R[R["pop"] == p]
        m = fit(d, f"{y} ~ {CTRL}{extra} + mates_ahead", fam)
        b, se = float(m.params[term]), float(m.bse[term])
        res[p] = dict(n=len(d), b=b, se=se, mult=float(np.exp(b)) if fam == "pois" else None, p_one=float(norm.sf(b / se)))
    return res


def pooled(res):
    e = [(res[p]["b"], res[p]["se"]) for p in POPS[1:]]
    w = np.array([1 / s ** 2 for _, s in e]); b = float(sum(w[i] * e[i][0] for i in range(2)) / w.sum()); se = float(1 / np.sqrt(w.sum()))
    return dict(b=b, se=se, p_one=float(norm.sf(b / se)))


# H1
out["H1"] = coef_test("H1", "own20")
# H3
out["H3"] = coef_test("H3", "opp30")
# H2: slope difference le4 - ge7
h2 = {}
for p in POPS:
    d = R[R["pop"] == p]
    m = fit(d, f"own20 ~ {CTRL} + C(band, Treatment('le4')) + mates_ahead + mates_ahead:C(band, Treatment('le4'))")
    t = [k for k in m.params.index if k.startswith("mates_ahead:") and "ge7" in k][0]
    b, se = -float(m.params[t]), float(m.bse[t])
    h2[p] = dict(n=len(d), b=b, se=se, p_one=float(norm.sf(b / se)), slope_le4=float(m.params["mates_ahead"]),
                 slope_ge7=float(m.params["mates_ahead"] + m.params[t]))
out["H2"] = h2
for h in ("H1", "H2", "H3"):
    out[h]["pooled_confirm"] = pooled(out[h])
    out[h]["all_three_one_sided_05"] = bool(all(out[h][p]["p_one"] < 0.05 for p in POPS))
# brief rule
pd2 = [2 * norm.sf(abs(out[h]["discovery"]["b"] / out[h]["discovery"]["se"])) for h in ("H1", "H2", "H3")]
q = multipletests(pd2, method="fdr_bh")[1]
for i, h in enumerate(("H1", "H2", "H3")):
    out[h]["disc_bh_q"] = float(q[i])
    out[h]["brief_rule"] = bool(q[i] < 0.05 and out[h]["discovery"]["b"] > 0 and all(out[h][p]["b"] > 0 for p in POPS[1:])
                                and out[h]["pooled_confirm"]["p_one"] < 0.05)

# H4: sweet spot per band
def adj_means(d, y):
    m = fit(d, f"{y} ~ C(mbin) + {CTRL}", "ols")
    return m


h4 = {}
for yv in ("net20", "net20_30", "net45"):
    h4[yv] = {}
    for band in ("le4", "5_6", "ge7"):
        cnt = R[R.band == band].groupby(["pop", "mbin"]).size().unstack(0)
        bins = [b for b in cnt.index if (cnt.loc[b, list(POPS)] >= 30).all()]
        fits = {p: adj_means(R[(R["pop"] == p) & (R.band == band) & R.mbin.isin(bins)], yv) for p in POPS}
        def eff(m, b):
            return 0.0 if b == bins[0] else float(m.params[f"C(mbin)[T.{b}]"])
        def contrast(m, b1, b2):
            L = np.zeros(len(m.params));
            for b, s in ((b1, 1), (b2, -1)):
                if b != bins[0]:
                    L[list(m.params.index).index(f"C(mbin)[T.{b}]")] += s
            est = float(L @ m.params.values); se = float(np.sqrt(L @ m.cov_params().values @ L))
            return est, se
        effs = {p: {b: eff(fits[p], b) for b in bins} for p in POPS}
        sweet = max(bins, key=lambda b: effs["discovery"][b])
        i = bins.index(sweet); neigh = [bins[j] for j in (i - 1, i + 1) if 0 <= j < len(bins)]
        cons = {}
        for nb in neigh:
            c = {p: dict(zip(("b", "se"), contrast(fits[p], sweet, nb))) for p in POPS}
            for p in POPS:
                c[p]["p_one"] = float(norm.sf(c[p]["b"] / c[p]["se"]))
            c["pooled_confirm"] = pooled(c)
            c["sign_all_three"] = bool(all(c[p]["b"] > 0 for p in POPS))
            cons[f"{sweet}_vs_{nb}"] = c
        nob = {p: {b: int(cnt.loc[b, p]) for b in bins} for p in POPS}
        h4[yv][band] = dict(bins=bins, n=nob, adj_effect_vs_first_bin=effs, sweet=sweet, contrasts=cons)
# Holm on primary net20 contrasts
pc = [(band, k, v["pooled_confirm"]["p_one"]) for band, bb in h4["net20"].items() for k, v in bb["contrasts"].items()]
order = sorted(range(len(pc)), key=lambda i: pc[i][2]); ok = True
for r, i in enumerate(order):
    band, k, pv = pc[i]; ok = ok and pv < 0.05 / (len(pc) - r)
    c = h4["net20"][band]["contrasts"][k]; c["holm_pass"] = bool(ok); c["counts"] = bool(ok and c["sign_all_three"])
out["H4"] = h4

# robustness: 45 s horizon + finer controls for H1/H3
rob = {}
rob["H1_own45"] = coef_test("", "own45"); rob["H1_own45"]["pooled_confirm"] = pooled(rob["H1_own45"])
rob["H1_fine"] = coef_test("", "own20", extra=" + n_vis_mates + nearest_opp"); rob["H1_fine"]["pooled_confirm"] = pooled(rob["H1_fine"])
rob["H3_opp45"] = coef_test("", "opp45"); rob["H3_opp45"]["pooled_confirm"] = pooled(rob["H3_opp45"])
rob["H3_fine"] = coef_test("", "opp30", extra=" + n_vis_mates + nearest_opp"); rob["H3_fine"]["pooled_confirm"] = pooled(rob["H3_fine"])
rob["net20_ols"] = coef_test("", "net20", fam="ols"); rob["net20_ols"]["pooled_confirm"] = pooled(rob["net20_ols"])
rob["net45_ols"] = coef_test("", "net45", fam="ols"); rob["net45_ols"]["pooled_confirm"] = pooled(rob["net45_ols"])
out["robustness"] = rob
out["raw_means"] = {p: R[R["pop"] == p].groupby(["band", "mbin"])[["own20", "opp20", "net20"]].mean().round(4).reset_index().to_dict("records") for p in POPS}
json.dump(out, open(os.path.join(ROOT, "results", "agent_runners.json"), "w"), indent=1)

def show(k, d):
    print(k, {p: (round(d[p]["b"], 4), round(d[p]["se"], 4), round(d[p]["p_one"], 4)) for p in POPS}, "pooledC", {a: round(v, 4) for a, v in d["pooled_confirm"].items()})
for h in ("H1", "H2", "H3"):
    show(h, out[h]); print("   all3:", out[h]["all_three_one_sided_05"], "brief:", out[h]["brief_rule"], "q", round(out[h]["disc_bh_q"], 4))
print("H2 slopes", {p: (round(h2[p]["slope_le4"], 4), round(h2[p]["slope_ge7"], 4)) for p in POPS})
for yv in h4:
    for band, bb in h4[yv].items():
        print(yv, band, "sweet", bb["sweet"], {p: {b: round(v, 4) for b, v in bb["adj_effect_vs_first_bin"][p].items()} for p in POPS})
        for k, c in bb["contrasts"].items():
            print("    ", k, {p: (round(c[p]["b"], 4), round(c[p]["se"], 4)) for p in POPS}, "pooledC p", round(c["pooled_confirm"]["p_one"], 4), "sign3", c["sign_all_three"], c.get("counts"))
for k, d in rob.items():
    show(k, d)

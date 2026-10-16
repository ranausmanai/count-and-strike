"""Agent chain: H1a-H3 tests + H4 tiers (PREREGISTRATION_AGENT_chain.md) -> results/agent_chain.json"""
import json, os, sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import PROC, ROOT
from r22b_underdog import tiers

POPS = ["discovery", "confirm_club", "confirm_women"]
CTRL = "C(ez) + C(height) + gk + pressure"


def load():
    L = pd.read_parquet(os.path.join(PROC, "agent_chain_L.parquet"))
    L["zx"] = pd.cut(L.ex, [-1, 40, 60, 80, 102, 121], labels=False).astype(int)
    L["ez"] = L.zx.astype(str) + np.select([(L.ey - 40).abs() <= 13, L.ey < 40], ["1", "0"], "2")
    L["sup"] = L.ours10 - L.theirs10; L["supc"] = L.sup.clip(-2, 2); L["ours10c"] = L.ours10.clip(0, 4); L["theirs10c"] = L.theirs10.clip(0, 4)
    f = L.fx.notna()
    L.loc[f, "fzx"] = pd.cut(L.loc[f, "fx"], [39, 60, 80, 96, 108, 121], labels=False)
    L["fzy"] = np.select([(L.fy - 40).abs() <= 13, L.fy < 40], ["1", "0"], "2")
    L["cell"] = (L.fzx.astype("Int64").astype(str) + L.fzy + np.select([L.opp_goalside <= 3, L.opp_goalside <= 6], ["0", "1"], "2")
                 + np.select([L.nearest_opp < 3, L.nearest_opp <= 7], ["0", "1"], "2"))
    L["fz"] = L.fzx.astype("Int64").astype(str) + L.fzy
    L["O"] = np.where(L.target_ov >= 1, 1, np.where(L.target_ov <= -1, 0, np.nan))
    L["S"] = (L.sup > 0).astype(int)
    G = pd.read_parquet(os.path.join(PROC, "agent_chain_G.parquet"))
    G["sb"] = np.sign(G.score_diff).astype(int); G["mb"] = pd.cut(G.minute, [-1, 29, 59, 74, 200], labels=False)
    G["pb"] = np.select([G.press <= 1, G.press <= 3], ["0", "1"], "2")
    G["gcell"] = G.sb.astype(str) + G.mb.astype(str) + G.pb
    G["T"] = np.where((G.length >= 35) & (G.vis == 1) & (G.ours10 > G.theirs10), 1, np.where(G.length < 30, 0, np.nan))
    G["Tany"] = np.where(G.length >= 35, 1, np.where(G.length < 30, 0, np.nan))
    G["Tns"] = np.where((G.length >= 35) & ~((G.vis == 1) & (G.ours10 > G.theirs10)), 1, np.where(G.length < 30, 0, np.nan))
    return L, G


def within_cell(d, tcol, y, cell, mn):
    d = d.dropna(subset=[tcol, y])
    a = d[d[tcol] == 1].groupby(cell)[y].agg(["mean", "var", "size"]); b = d[d[tcol] == 0].groupby(cell)[y].agg(["mean", "var", "size"])
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner"); j = j[(j.size_a >= mn) & (j.size_b >= mn)]
    if j.empty:
        return dict(est=np.nan, se=np.nan, n=0, cells=0)
    w = np.minimum(j.size_a, j.size_b)
    diff = ((j.mean_a - j.mean_b) * w).sum() / w.sum()
    var = ((w / w.sum()) ** 2 * (j.var_a.fillna(0) / j.size_a + j.var_b.fillna(0) / j.size_b)).sum()
    return dict(est=float(diff), se=float(np.sqrt(var)), n=int(j.size_a.sum() + j.size_b.sum()), cells=int(len(j)), n_T=int(j.size_a.sum()))


def ols(d, f, terms):
    d = d.dropna(subset=[f.split("~")[0].strip()])
    m = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.match_id)[0]})
    c = np.zeros(len(m.params))
    for t, s in terms.items():
        c[list(m.params.index).index(t)] = s
    est = float(c @ m.params.values); se = float(np.sqrt(c @ m.cov_params().values @ c))
    return dict(est=est, se=se, n=int(m.nobs)), m


def s_h1(L):
    return L[(L.f_has_frame == 1) & (L.fx >= 40) & L.O.notna()]


def s_h1c(L):
    return s_h1(L)[(s_h1(L).vis == 1) & (s_h1(L).sup != 0)]


def s_h2(L):
    return L[L.vis == 1]


TESTS = {
    "H1a_overload_net20F": lambda L, G: within_cell(s_h1(L), "O", "net20F", "cell", 10),
    "H1b_overload_net45F": lambda L, G: within_cell(s_h1(L), "O", "net45F", "cell", 10),
    "H1c_chain_SxO_net45L": lambda L, G: ols(s_h1c(L), f"net45L ~ S*O + {CTRL} + C(fz)", {"S:O": 1})[0],
    "H2a_cost_opp45_per_ours10": lambda L, G: ols(s_h2(L), f"opp45L ~ ours10c + C(theirs10c) + {CTRL}", {"ours10c": 1})[0],
    "H2b_sup1_minus_sup2plus_net45": lambda L, G: ols(s_h2(L), f"net45L ~ C(supc, Treatment(0)) + {CTRL}",
                                                      {"C(supc, Treatment(0))[T.1]": 1, "C(supc, Treatment(0))[T.2]": -1})[0],
    "H2c_net45_per_ours10": lambda L, G: ols(s_h2(L), f"net45L ~ ours10c + C(theirs10c) + {CTRL}", {"ours10c": 1})[0],
    "H3_gk_long_sup_vs_short_net45": lambda L, G: within_cell(G, "T", "net45G", "gcell", 5),
}
TWO_SIDED = {"H2b_sup1_minus_sup2plus_net45"}


def pool(e):
    e = [x for x in e if x["se"] and np.isfinite(x["se"]) and x["se"] > 0]
    if not e:
        return np.nan, np.nan
    w = np.array([1 / x["se"] ** 2 for x in e]); return float(sum(wi * x["est"] for wi, x in zip(w, e)) / w.sum()), float(1 / np.sqrt(w.sum()))


def main():
    L, G = load(); res = {"n": {}, "usage": {}, "tests": {}, "descriptive": {}, "robust": {}, "H4_tiers": {}}
    for p in POPS:
        l = L[L["pop"] == p]
        res["n"][p] = dict(long_balls=int(len(l)), visible=int(l.vis.sum()), retained=int(l.retained.sum()), F_frame=int(l.f_has_frame.sum()),
                           H1=int(len(s_h1(l))), H1c=int(len(s_h1c(l))), goal_kicks=int((G["pop"] == p).sum()),
                           gk_T=int((G[G["pop"] == p]["T"] == 1).sum()), gk_C=int((G[G["pop"] == p]["T"] == 0).sum()))
    T = res["tests"]
    for k, f in TESTS.items():
        T[k] = {p: f(L[L["pop"] == p], G[G["pop"] == p]) for p in POPS}
    ps = []
    for k in TESTS:
        d = T[k]["discovery"]; ps.append(2 * norm.sf(abs(d["est"] / d["se"])) if d["se"] > 0 else 1.0)
    q = multipletests(ps, method="fdr_bh")[1]; carried = []
    for k, pv, qv in zip(TESTS, ps, q):
        t = T[k]; d = t["discovery"]["est"]; sign = np.sign(d) if k in TWO_SIDED else 1.0
        t["disc_p2"] = float(pv); t["disc_q"] = float(qv); t["sign"] = float(sign); t["disc_pass"] = bool(qv < .05 and d * sign > 0)
        c = [t["confirm_club"], t["confirm_women"]]; pe, pse = pool(c)
        t["pooled"] = pe; t["pooled_se"] = pse; t["pooled_p1"] = float(norm.sf(sign * pe / pse)) if pse == pse else 1.0
        t["same_sign"] = bool(all(x["est"] * sign > 0 for x in c))
        if t["disc_pass"]:
            carried.append(k)
    flag = True
    for i, k in enumerate(sorted(carried, key=lambda k: T[k]["pooled_p1"])):
        flag = flag and T[k]["pooled_p1"] < .05 / (len(carried) - i)
        T[k]["confirmed"] = bool(flag and T[k]["same_sign"])
    for k in TESTS:
        T[k].setdefault("confirmed", False)
    # descriptive
    D = res["descriptive"]
    for p in POPS:
        l = L[L["pop"] == p]; c = s_h1c(l); g = G[G["pop"] == p]; v = s_h2(l)
        D[p] = dict(
            chain_cell_means_net45L=c.groupby(["S", "O"]).net45L.agg(["mean", "size"]).round(4).reset_index().to_dict("records"),
            H1c_main=ols(c, f"net45L ~ S*O + {CTRL} + C(fz)", {"S": 1})[0] | {"term": "S (O=0)"},
            H1c_O_main=ols(c, f"net45L ~ S*O + {CTRL} + C(fz)", {"O": 1})[0] | {"term": "O (S=0)"},
            H1c_chain_vs_neither=ols(c, f"net45L ~ S*O + {CTRL} + C(fz)", {"S": 1, "O": 1, "S:O": 1})[0],
            H1_overload_net45L=within_cell(s_h1(l), "O", "net45L", "cell", 10),
            retained_by_sup=v.groupby("supc").retained.agg(["mean", "size"]).round(3).reset_index().to_dict("records"),
            H2_by_supc_net45=v.groupby("supc")[["own45L", "opp45L", "net45L"]].mean().round(4).reset_index().to_dict("records"),
            H2_by_ours10_opp45=v.groupby("ours10c")[["own45L", "opp45L", "net45L"]].agg("mean").round(4).reset_index().to_dict("records"),
            H2_own45_per_ours10=ols(v, f"own45L ~ ours10c + C(theirs10c) + {CTRL}", {"ours10c": 1})[0],
            H2_supc_coefs={str(s): ols(v, f"net45L ~ C(supc, Treatment(0)) + {CTRL}", {f"C(supc, Treatment(0))[T.{s}]": 1})[0] for s in (-2, -1, 1, 2)},
            H3_any_long_vs_short=within_cell(g, "Tany", "net45G", "gcell", 5),
            H3_long_notsup_vs_short=within_cell(g, "Tns", "net45G", "gcell", 5),
            H3_opp45G_longsup_vs_short=within_cell(g, "T", "opp45G", "gcell", 5),
            H3_raw=g.groupby("T")[["net45G", "opp45G"]].agg(["mean", "size"]).round(4).pipe(lambda x: x.set_axis(["_".join(c) for c in x.columns], axis=1)).reset_index().to_dict("records"))
    # robustness (reported for all; required for confirmed)
    R = res["robust"]
    for p in POPS:
        l = L[L["pop"] == p].copy(); h = s_h1(l).assign(cell2=lambda d: d.cell + d.height.str[0] + d.gk.astype(str))
        R[p] = dict(H1a_finer=within_cell(h, "O", "net20F", "cell2", 10), H1b_finer=within_cell(h, "O", "net45F", "cell2", 10),
                    H1_fwd_only_net20F=within_cell(h[h.f_fwd == 1], "O", "net20F", "cell", 10),
                    H1c_net20L=ols(s_h1c(l), f"net20L ~ S*O + {CTRL} + C(fz)", {"S:O": 1})[0],
                    H2a_opp20=ols(s_h2(l).assign(opp20L=lambda d: d.own20L - d.net20L), f"opp20L ~ ours10c + C(theirs10c) + {CTRL}", {"ours10c": 1})[0],
                    H2c_net20=ols(s_h2(l), f"net20L ~ ours10c + C(theirs10c) + {CTRL}", {"ours10c": 1})[0])
    # H4 tiers
    TR = tiers(); L4 = L.merge(TR[["match_id", "team", "tier"]], on=["match_id", "team"], how="left")
    G4 = G.merge(TR[["match_id", "team", "tier"]], on=["match_id", "team"], how="left")
    H4 = res["H4_tiers"]
    for k in ["H1a_overload_net20F", "H1b_overload_net45F", "H1c_chain_SxO_net45L", "H2a_cost_opp45_per_ours10", "H2c_net45_per_ours10", "H3_gk_long_sup_vs_short_net45"]:
        H4[k] = {}
        for p in POPS:
            H4[k][p] = {}
            for tr in ("under", "fav"):
                try:
                    H4[k][p][tr] = TESTS[k](L4[(L4["pop"] == p) & (L4.tier == tr)], G4[(G4["pop"] == p) & (G4.tier == tr)])
                except Exception as ex:
                    H4[k][p][tr] = dict(est=np.nan, se=np.nan, n=0, err=str(ex)[:80])
        dif = []
        for p in ("confirm_club", "confirm_women"):
            u, f = H4[k][p]["under"], H4[k][p]["fav"]
            if u["se"] == u["se"] and f["se"] == f["se"]:
                dif.append(dict(est=u["est"] - f["est"], se=float(np.hypot(u["se"], f["se"]))))
        pe, pse = pool(dif); H4[k]["pooled_confirm_under_minus_fav"] = dict(est=pe, se=pse)
    us = {}
    for p in POPS:
        v = s_h2(L4[L4["pop"] == p])
        us[p] = {tr: dict(sup_pos=float((g.sup > 0).mean()), sup_neg=float((g.sup < 0).mean()), retained=float(g.retained.mean()),
                          overload_follow=float(s_h1(g).O.mean()) if len(s_h1(g)) else np.nan, n=int(len(g)))
                 for tr, g in v.groupby("tier")}
    res["usage"] = us
    json.dump(res, open(os.path.join(ROOT, "results", "agent_chain.json"), "w"), indent=1, default=float)
    print(json.dumps(res["n"], indent=0))
    for k in TESTS:
        t = T[k]
        print(f"{k:34s}", " ".join(f"{p[:6]} {t[p]['est']:+.4f}({t[p]['se']:.4f}) n={t[p]['n']}" for p in POPS),
              f"| q={t['disc_q']:.3f} pooled {t['pooled']:+.4f}({t['pooled_se']:.4f}) p1={t['pooled_p1']:.3f} CONF={t['confirmed']}")


if __name__ == "__main__":
    main()

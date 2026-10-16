"""Agent 'cross': pre-registered tests H1-H4 (PREREGISTRATION_AGENT_cross.md) -> results/agent_cross.json"""
import json, os, sys, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(__file__), "..")
IN = os.environ.get("AGENT_CROSS_OUT", os.path.join(ROOT, "data", "proc", "agent_cross_actions.parquet"))
POPS = ("discovery", "confirm_club", "confirm_women")
ALTS = ("cutback", "inside", "carry", "recycle", "other_pass")
OURS_LAB = ["0-1", "2", "3", "4+"]


def prep(D):
    D = D.copy()
    D["ours_ord"] = np.clip(D.ours - 1, 0, 3).astype(int)
    D["theirs_c"] = np.select([D.theirs <= 4, D.theirs <= 6], [0, 1], 2)
    D["press"] = (D.nearest < 3).astype(int)
    D["cell"] = D.ours_ord.astype(str) + D.theirs_c.astype(str) + D.press.astype(str)
    D["cross"] = D.choice.isin(["cross_high", "cross_low"]).astype(int)
    D["low"] = D.choice.isin(["cross_low", "cutback"]).astype(int)
    D["fam"] = np.where(D.cross == 1, "cross", D.choice)
    return D


def fit(d, term, y="net20", formula=None):
    if len(d) < 20 or (term in d and d[term].nunique() < 2):
        return dict(est=np.nan, se=np.nan, n=int(len(d)))
    f = formula or f"{y} ~ C(cell) + {term}"
    r = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.match_id)[0]})
    name = term if term in r.params else [k for k in r.params.index if k.endswith(term)][0]
    return dict(est=float(r.params[name]), se=float(r.bse[name]), n=int(len(d)))


def tests(D, alt_h2, alt_h3, y="net20"):
    out = {}
    out["H1"] = fit(D, "cross:ours_ord", y, f"{y} ~ C(cell) + cross + cross:ours_ord")
    for h, mask, alt in (("H2", D.ours_ord >= 2, alt_h2), ("H3", D.ours_ord == 0, alt_h3)):
        d = D[mask & ((D.cross == 1) | (D.choice == alt))]
        r = fit(d, "cross", y); r.update(n_cross=int(d.cross.sum()), n_alt=int((d.cross == 0).sum()), alt=alt); out[h] = r
    d = D[(D.theirs_c == 2) & D.choice.isin(["cross_high", "cross_low", "cutback"])]
    r = fit(d, "low", y); r.update(n_low=int(d.low.sum()), n_high=int((d.low == 0).sum())); out["H4"] = r
    return out


DIRN = dict(H1=1, H2=1, H3=-1, H4=1)


def main():
    A = prep(pd.read_parquet(IN))
    P = A[A.vis]
    disc = P[P["pop"] == "discovery"]
    # usual alternative per OURS subset, fixed in discovery (counts only)
    alt_h2 = disc[(disc.ours_ord >= 2) & (disc.cross == 0)].choice.value_counts().idxmax()
    alt_h3 = disc[(disc.ours_ord == 0) & (disc.cross == 0)].choice.value_counts().idxmax()
    counts = {p: P[P["pop"] == p].pivot_table(index="ours_ord", columns="fam", values="net20", aggfunc="size", fill_value=0)
              .rename(index=dict(enumerate(OURS_LAB))).to_dict() for p in POPS}
    res = {p: tests(P[P["pop"] == p], alt_h2, alt_h3) for p in POPS}
    # ---- success rule ----
    H = ["H1", "H2", "H3", "H4"]
    dp = np.array([2 * norm.sf(abs(res["discovery"][h]["est"] / res["discovery"][h]["se"])) for h in H])
    q = multipletests(dp, method="fdr_bh")[1]
    verdict = {}
    passed = []
    for i, h in enumerate(H):
        d = res["discovery"][h]
        ok = bool(q[i] < 0.05 and np.sign(d["est"]) == DIRN[h])
        ests = [(res[p][h]["est"], res[p][h]["se"]) for p in POPS[1:]]
        w = np.array([1 / s ** 2 for _, s in ests]); pooled = float((w * np.array([e for e, _ in ests])).sum() / w.sum()); pse = float(1 / np.sqrt(w.sum()))
        p1 = float(norm.sf(DIRN[h] * pooled / pse))
        same = all(np.sign(e) == DIRN[h] for e, _ in ests)
        verdict[h] = dict(disc_p=float(dp[i]), disc_q=float(q[i]), disc_pass=ok, pooled=pooled, pooled_se=pse, pooled_p1=p1, same_sign_both=bool(same))
        if ok:
            passed.append(h)
    k = len(passed); flag = True
    for j, h in enumerate(sorted(passed, key=lambda h: verdict[h]["pooled_p1"])):
        flag = flag and verdict[h]["pooled_p1"] < 0.05 / (k - j)
        verdict[h]["confirmed"] = bool(flag and verdict[h]["same_sign_both"])
    for h in H:
        verdict[h].setdefault("confirmed", False)
    # ---- secondary: dose-response descriptive (cross minus all alternatives per OURS level) and best-alt-from-discovery ----
    dose = {p: {OURS_LAB[k]: fit(P[(P["pop"] == p) & (P.ours_ord == k)], "cross") for k in range(4)} for p in POPS}
    best = {}
    for k in range(4):
        d = disc[disc.ours_ord == k]
        r = smf.ols("net20 ~ C(cell) + C(fam, Treatment('cross'))", d).fit()
        co = {a: r.params.get(f"C(fam, Treatment('cross'))[T.{a}]", np.nan) for a in ALTS if (d.fam == a).sum() >= 30}
        ba = max(co, key=co.get)
        best[OURS_LAB[k]] = dict(alt=ba, **{p: fit(P[(P["pop"] == p) & (P.ours_ord == k) & ((P.cross == 1) | (P.choice == ba))], "cross") for p in POPS})
    # ---- robustness (reported for all; required for confirmed) ----
    rob = {"net45": {p: tests(P[P["pop"] == p], alt_h2, alt_h3, "net45") for p in POPS},
           "all_frames": {p: tests(A[A["pop"] == p], alt_h2, alt_h3) for p in POPS}}
    A2 = A.copy(); A2["cell"] = A2.cell + np.select([A2.x >= 108, A2.x >= 102], ["2", "1"], "0") + (A2.y > 40).astype(int).astype(str)
    rob["finer_cells_xzone_side"] = {p: tests(A2[A2.vis & (A2["pop"] == p)], alt_h2, alt_h3) for p in POPS}
    out = dict(n_actions={p: int((P["pop"] == p).sum()) for p in POPS}, n_matches={p: int(P[P["pop"] == p].match_id.nunique()) for p in POPS},
               usual_alt=dict(H2=alt_h2, H3=alt_h3), counts=counts, primary=res, verdict=verdict, dose=dose, best_alt=best, robustness=rob)
    json.dump(out, open(os.path.join(ROOT, "results", "agent_cross.json"), "w"), indent=1, default=float)
    print("usual alt", alt_h2, alt_h3, out["n_actions"], out["n_matches"])
    for h in H:
        print(h, " | ".join(f"{p[:9]} {res[p][h]['est']:+.4f}±{res[p][h]['se']:.4f} n={res[p][h]['n']}" for p in POPS), "|", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in verdict[h].items()})
    print("dose (cross - all alts):")
    for p in POPS:
        print(" ", p, {k: f"{v['est']:+.4f}±{v['se']:.4f}" for k, v in dose[p].items()})
    print("best alt from discovery:")
    for k, v in best.items():
        print(" ", k, v["alt"], {p: f"{v[p]['est']:+.4f}±{v[p]['se']:.4f}" for p in POPS})
    for nm, R in rob.items():
        print(nm)
        for h in H:
            print("  ", h, " | ".join(f"{p[:9]} {R[p][h]['est']:+.4f}±{R[p][h]['se']:.4f}" for p in POPS))


if __name__ == "__main__":
    main()

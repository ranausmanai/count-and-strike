"""Agent 'combos': NON-pre-registered sensitivity checks, appended to results/agent_combos.json under 'sensitivity'.
1) type-level effects with n-weighting (instead of inverse-variance) and match-cluster bootstrap SE;
2) completion-matched cell rules where the reference itself became third_man: third_man vs 'other' among completed first passes."""
import json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import PROC, ROOT
from agent_combos_test import TYPES, POPS


def nweighted(D, y="net20"):
    D = D[D.cat != "shot"]
    cnt = D.groupby(["cell", "cat"]).size().unstack(fill_value=0)
    ref = cnt.idxmax(axis=1)
    mu = D.groupby(["cell", "cat"])[y].mean().unstack()
    out = {}
    for t in TYPES:
        refn = np.array([cnt.loc[c, r] for c, r in ref.items()]); refm = np.array([mu.loc[c, r] for c, r in ref.items()])
        ok = (ref.values != t) & (cnt[t].values >= 5) & (refn >= 5)
        d = mu[t].values[ok] - refm[ok]; w = cnt[t].values[ok]
        out[t] = float((d * w).sum() / w.sum()) if w.sum() else np.nan
    return out


def boot(D, B=200, seed=0):
    rng = np.random.default_rng(seed); est = nweighted(D)
    ms = D.match_id.unique(); g = {m: i for m, i in D.groupby("match_id").indices.items()}
    bs = []
    for _ in range(B):
        idx = np.concatenate([g[m] for m in rng.choice(ms, len(ms))])
        bs.append(nweighted(D.iloc[idx]))
    bs = pd.DataFrame(bs)
    return {t: dict(est=est[t], se=float(bs[t].std())) for t in TYPES}


def main():
    D = pd.read_parquet(os.path.join(PROC, "agent_combos_P.parquet"))
    sens = {"nweighted_cluster_boot": {p: boot(D[D["pop"] == p]) for p in POPS}}
    R = json.load(open(os.path.join(ROOT, "results", "agent_combos.json")))
    cm = {}
    Dc = D[D.a1_complete & ~D.cat.isin(["solo_carry", "shot"])]
    for r in R["cell_level"]["rules"]:
        if not r["confirmed"]:
            continue
        cm[r["cell"]] = {}
        for p in POPS:
            X = Dc[(Dc["pop"] == p) & (Dc.cell == r["cell"])]
            a = X[X.cat == "third_man"].net20; b = X[X.cat == "other"].net20
            cm[r["cell"]][p] = dict(diff=float(a.mean() - b.mean()), se=float(np.sqrt(a.var() / len(a) + b.var() / len(b))), n=len(a), n_ref=len(b))
    sens["completion_matched_third_man_vs_other"] = cm
    R["sensitivity"] = sens
    json.dump(R, open(os.path.join(ROOT, "results", "agent_combos.json"), "w"), indent=1, default=float)
    for p in POPS:
        print(p, {t: (round(v["est"], 4), round(v["se"], 4)) for t, v in sens["nweighted_cluster_boot"][p].items()})
    for c, v in cm.items():
        print(c, {p: (round(x["diff"], 4), round(x["se"], 4), x["n"], x["n_ref"]) for p, x in v.items()})


if __name__ == "__main__":
    main()

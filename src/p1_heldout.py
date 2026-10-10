"""P1: does (lagged) style improve out-of-sample prediction of segment xG on held-out EPL 2015/16?"""
import json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from frontier import *

s = load_segments()
need = ["lam_own"] + [c + "_lag" for c in STYLE] + [c + "_lag_opp" for c in STYLE] + STYLE + [c + "_opp" for c in STYLE]
s = s.dropna(subset=need)
train = s[s.league != "EPL"]; test = s[s.league == "EPL"]
print("train segs", len(train), "test segs", len(test), "test matches", test.match_id.nunique())
out = {}
for lag in (True, False):
    sc = StyleScaler().fit(train, STYLE)
    Xb, ob = design(train, sc, "base"); Xf, of = design(train, sc, "full", lag=lag)
    rb = fit_glm(train, Xb, ob); rf = fit_glm(train, Xf, of)
    Tb, tob = design(test, sc, "base"); Tf, tof = design(test, sc, "full", lag=lag)
    mub = np.exp(Tb.values @ rb.params + tob.values); muf = np.exp(Tf.values @ rf.params + tof.values)
    db = poisson_dev(test.xg_for, mub); df_ = poisson_dev(test.xg_for, muf)
    # also: goals as target, with xG-fitted rates (do the rates predict actual goals?)
    gb = poisson_dev(test.goals_in_seg_for, mub); gf = poisson_dev(test.goals_in_seg_for, muf)
    # match-level bootstrap of deviance improvement
    t = test.assign(d=db - df_, g=gb - gf)
    per = t.groupby("match_id")[["d", "g"]].sum()
    rng = np.random.default_rng(20261011)
    boots = np.array([per.values[rng.integers(0, len(per), len(per))].sum(0) for _ in range(2000)])
    key = "lagged" if lag else "contemporaneous"
    out[key] = dict(dev_base=float(db.sum()), dev_full=float(df_.sum()), improvement=float((db - df_).sum()),
                    rel_improvement=float((db - df_).sum() / db.sum()),
                    ci95=[float(np.percentile(boots[:, 0], 2.5)), float(np.percentile(boots[:, 0], 97.5))],
                    goals_improvement=float((gb - gf).sum()),
                    goals_ci95=[float(np.percentile(boots[:, 1], 2.5)), float(np.percentile(boots[:, 1], 97.5))],
                    coefs={n: [float(p), float(se)] for n, p, se in zip(Xf.columns, rf.params, rf.bse)})
    print(key, {k: v for k, v in out[key].items() if k != "coefs"})
    for n, p, se in zip(Xf.columns, rf.params, rf.bse):
        print(f"   {n:28s} {p:+.4f} ({se:.4f})")
json.dump(out, open(os.path.join(ROOT, "results", "p1_heldout.json"), "w"), indent=1)

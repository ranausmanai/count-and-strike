"""Sweet-spot curves: how underdog points vary with each tactic knob (stage-1 tactics), plus a GBM partial dependence."""
import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import partial_dependence
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
S = json.load(open(os.path.join(ROOT, "results", "sim_search.json")))
names = S["names"]
X = np.array([r["tactic"] for r in S["stage1"]]); y = np.array([r["pts"] for r in S["stage1"]])
gbm = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0).fit(X, y)
from sklearn.model_selection import cross_val_score
r2 = cross_val_score(GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0), X, y, cv=5, scoring="r2").mean()
imp = dict(zip(names, gbm.feature_importances_.round(3)))
print("cross-validated R2 of points from tactics:", round(r2, 3)); print("importance:", dict(sorted(imp.items(), key=lambda kv: -kv[1])))
fig, axs = plt.subplots(2, 5, figsize=(14, 5.4), sharey=True)
curves = {}
for k, ax in enumerate(axs.ravel()):
    pdp = partial_dependence(gbm, X, [k], grid_resolution=25)
    gx, gy = pdp["grid_values"][0], pdp["average"][0]
    curves[names[k]] = dict(x=gx.tolist(), pts=gy.tolist(), best_x=float(gx[np.argmax(gy)]))
    ax.plot(gx, gy, color="#2a78d6", lw=2); ax.axhline(S["baseline"]["pts"], color="#9a9993", ls="--", lw=1)
    ax.set_title(names[k], fontsize=10); ax.grid(True, color="#ecebe7")
axs[0, 0].set_ylabel("underdog points / match"); axs[1, 0].set_ylabel("underdog points / match")
fig.suptitle("Simulated underdog: points per match vs each tactic knob (partial dependence; dashed = standard tactic)", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "sim_sweetspots.png"), dpi=170, facecolor="white")
json.dump(dict(r2=r2, importance=imp, curves=curves), open(os.path.join(ROOT, "results", "sim_sweetspots.json"), "w"), indent=1, default=float)
for n, c in curves.items():
    print(f"{n:10s} best at {c['best_x']:.2f}  range of effect {min(c['pts']):.3f}–{max(c['pts']):.3f}")

"""Figures for the Count & Strike paper."""
import json, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(__file__), ".."); FIG = os.path.join(ROOT, "figures")
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.dpi": 200})
# Fig A: counter trigger (pooled 360 populations)
A = pd.read_parquet(os.path.join(ROOT, "data", "proc", "r13_actions.parquet"))
R = A[A.regain_start & (A.x < 80)].copy()
R["zone"] = pd.cut(R.x, [-1, 40, 60, 80], labels=["Own third", "Own half", "Middle third"])
R["back"] = pd.cut(R.opp_goalside, [-1, 3, 5, 7, 11], labels=["≤3", "4–5", "6–7", "8+"])
T = R.groupby(["zone", "back"]).own20.mean().unstack() * 1000
f, ax = plt.subplots(figsize=(6.2, 2.6))
cols = [BLUE, AQUA, ORANGE]
for i, z in enumerate(T.index):
    ax.plot(range(4), T.loc[z].values, "o-", color=cols[i], lw=2, ms=7, mec="white", mew=1.2, label=z)
    ax.text(3.08, T.loc[z].values[-1], z, va="center", fontsize=8.5, color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels(list(T.columns)); ax.set_xlabel("Opponents behind the ball when you win it"); ax.set_xlim(-0.2, 3.9)
ax.set_ylabel("Counter value\n(xG ×1000, next 20 s)"); ax.grid(True, axis="y", color="#ecebe7")
ax.set_title("COUNT: at the moment you win the ball, count the opponents behind it")
f.savefig(os.path.join(FIG, "cs_fig_count.png"), bbox_inches="tight", facecolor="white"); plt.close(f)
# Fig B: strike-through rules with 3-population estimates
pb = json.load(open(os.path.join(ROOT, "results", "r15_patience.json")))
labs = ["In box, pressed → shoot\n(vs lateral pass)", "In box, pressed → shoot\n(vs turning away)", "Edge, 4–6 ahead, pressed → shoot\n(vs lateral)",
        "Edge, 4–6 ahead, space → shoot\n(vs turning)", "25–40 yds, packed, pressed → shoot\n(vs lateral)", "Edge, 4–6 ahead, pressed → shoot\n(vs back pass)",
        "Edge, packed (7+), pressed → shoot\n(vs lateral)", "Edge, packed (7+), pressed → shoot\n(vs turning)"]
f, ax = plt.subplots(figsize=(6.4, 3.8))
y = np.arange(len(pb))[::-1]
for yi, r, l in zip(y, pb, labs):
    for W, off, c in (("W20", 0.15, BLUE), ("W60", -0.15, ORANGE)):
        v = r[W]; ax.plot([v["adv"] - 1.96 * v["se"], v["adv"] + 1.96 * v["se"]], [yi + off] * 2, color=c, lw=2)
        ax.plot(v["adv"], yi + off, "o", color=c, ms=6, mec="white")
ax.axvline(0, color=INK, lw=0.8); ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=7.5)
ax.plot([], [], "o-", color=BLUE, label="next 20 s"); ax.plot([], [], "o-", color=ORANGE, label="next 60 s (patience test)")
ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.set_xlabel("Advantage of shooting (net xG per decision), same space and shooting lane")
ax.set_title("STRIKE THROUGH: near the box, shooting beats recycling — and patience doesn't pay later")
f.savefig(os.path.join(FIG, "cs_fig_strike.png"), bbox_inches="tight", facecolor="white"); plt.close(f)
print(T.round(1))

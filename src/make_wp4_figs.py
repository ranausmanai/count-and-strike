"""Figures for white paper v4 ("What actually works"): corrected count diagram, bait diagram, second-ball retention by superiority."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_whitepaper_figs import full_pitch, half_pitch, player, arrow, save, AMBER, LINE, INK, RED, BLUE  # noqa: E402  (re-runs v2 figs; harmless)

ROOT = os.path.join(os.path.dirname(__file__), "..")
GREY = "#9aa69f"

# 1. COUNT (corrected): always 10 outfield opponents; n behind the ball, the rest caught upfield. Keepers not shown.
behind = {3: [(96, 34), (104, 48), (90, 60)],
          8: [(95, 30), (104, 44), (98, 58), (84, 22), (100, 52), (90, 40), (104, 64), (80, 50)]}
ahead = [(40, 20), (36, 58), (30, 40), (46, 64), (44, 12), (24, 28), (28, 54)]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.7))
for ax, n in ((axs[0], 3), (axs[1], 8)):
    full_pitch(ax)
    player(ax, 52, 42, "us", ball=True)
    for x, y in behind[n] + ahead[:10 - n]:
        player(ax, x, y, "them")
    if n == 3:
        player(ax, 62, 22, "us"); player(ax, 64, 62, "us")
        arrow(ax, (62, 22), (92, 16), color=AMBER, ls="--"); arrow(ax, (64, 62), (96, 70), color=AMBER, ls="--")
        arrow(ax, (54, 42), (88, 22), color=AMBER)
        ax.set_title("3 BEHIND THE BALL, 7 CAUGHT UPFIELD", color=INK, fontsize=11.5, weight="bold", pad=6)
    else:
        player(ax, 44, 30, "us"); arrow(ax, (54, 42), (45, 32), color=LINE)
        ax.set_title("8 BEHIND THE BALL, BLOCK SET", color=INK, fontsize=11.5, weight="bold", pad=6)
fig.text(0.5, 0.06, "We win the ball (red, with ball), attacking right. Blue = their 10 outfield players (keepers not shown).",
         ha="center", fontsize=9.5, color="#4a554f")
save(fig, "count_v4.png")

# 2. BAIT: work it down one side, defenders shift, far side opens
fig, ax = plt.subplots(figsize=(5.2, 3.6))
half_pitch(ax, x0=50)
for x, y in ((70, 66), (78, 72), (84, 60)):
    player(ax, x, y, "us")
player(ax, 74, 64, "us", ball=True)
for x, y in ((80, 66), (86, 70), (90, 58), (78, 56), (92, 46), (96, 64)):
    player(ax, x, y, "them")
for x, y in ((98, 30), (94, 22)):
    player(ax, x, y, "them")
player(ax, 88, 14, "us"); player(ax, 96, 10, "us")
arrow(ax, (72, 70), (82, 64), color=LINE, ls="--"); arrow(ax, (80, 62), (86, 68), color=LINE, ls="--")
for (a, b) in (((92, 40), (92, 46)), ((78, 50), (78, 56))):
    arrow(ax, a, b, color="#9ec3ff", lw=1.4)
arrow(ax, (76, 62), (87, 16), color=AMBER, rad=0.15)
ax.text(70, 77, "1  work it down one side", color="white", fontsize=8.5, weight="bold")
ax.text(54, 30, "2  their block slides\n    across", color="#cfe0ff", fontsize=8.5, weight="bold")
ax.text(92, 3.5, "3  far side opens", color=AMBER, fontsize=8.5, weight="bold", ha="center")
ax.set_ylim(-1, 81)
save(fig, "bait_v4.png")

# 3. Second-ball retention by local superiority (pooled 3 populations)
J = json.load(open(os.path.join(ROOT, "results", "agent_chain.json")))["descriptive"]
agg = {}
for p, d in J.items():
    for r in d.get("retained_by_sup", []):
        k = max(-1, min(2, r["supc"]))
        a = agg.setdefault(k, [0.0, 0])
        a[0] += r["mean"] * r["size"]; a[1] += r["size"]
labs = {-1: "Outnumbered", 0: "Level", 1: "+1 of us", 2: "+2 or more"}
ks = sorted(agg)
vals = [agg[k][0] / agg[k][1] * 100 for k in ks]
fig, ax = plt.subplots(figsize=(5.4, 2.8))
cols = [GREY, GREY, RED, GREY]
ax.bar(range(len(ks)), vals, color=cols[:len(ks)], width=0.62)
for i, v in enumerate(vals):
    ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=10, weight="bold", color=INK)
ax.set_xticks(range(len(ks))); ax.set_xticklabels([labs[k] for k in ks], fontsize=9.5)
ax.set_ylim(0, 105); ax.set_yticks([])
ax.set_title("Still have the ball 5 s after a long ball, by players around the landing spot", fontsize=9, loc="left")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
save(fig, "secondball_v4.png")
print({labs[k]: (round(v, 1), agg[k][1]) for k, v in zip(ks, vals)})

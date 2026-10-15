"""Pitch diagrams and charts for the Count & Strike white paper."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc, FancyArrowPatch, Polygon, Circle

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "figures", "wp"); os.makedirs(OUT, exist_ok=True)
FELT, LINE, RED, BLUE, AMBER, INK = "#1e4a38", "#d7e6db", "#d9553a", "#3b6fc0", "#f0b13c", "#17201b"
plt.rcParams.update({"font.family": "DejaVu Sans"})


def half_pitch(ax, x0=60):
    """Attacking half (x 60..120 in StatsBomb yards), goal on the right."""
    ax.add_patch(Rectangle((x0, 0), 120 - x0, 80, fc=FELT, ec="none", zorder=0))
    for i in range(int((120 - x0) / 6)):
        if i % 2 == 0:
            ax.add_patch(Rectangle((x0 + i * 6, 0), 6, 80, fc="#21503d", ec="none", zorder=0))
    kw = dict(color=LINE, lw=1.6, zorder=1)
    ax.plot([x0, 120, 120, x0, x0], [0, 0, 80, 80, 0], **kw)
    ax.plot([102, 102, 120], [18, 62, 62], **kw); ax.plot([102, 120], [18, 18], **kw)
    ax.plot([114, 114, 120], [30, 50, 50], **kw); ax.plot([114, 120], [30, 30], **kw)
    ax.plot([120, 121.5, 121.5, 120], [36, 36, 44, 44], **kw)
    ax.add_patch(Arc((108, 40), 20, 20, theta1=127, theta2=233, **kw))
    ax.plot([108], [40], "o", ms=2.5, color=LINE, zorder=1)
    if x0 <= 60:
        ax.add_patch(Arc((60, 40), 20, 20, theta1=-90, theta2=90, **kw))
    ax.set_xlim(x0 - 1, 123); ax.set_ylim(-1, 81); ax.set_aspect("equal"); ax.axis("off")


def full_pitch(ax):
    ax.add_patch(Rectangle((0, 0), 120, 80, fc=FELT, ec="none", zorder=0))
    for i in range(20):
        if i % 2 == 0:
            ax.add_patch(Rectangle((i * 6, 0), 6, 80, fc="#21503d", ec="none", zorder=0))
    kw = dict(color=LINE, lw=1.4, zorder=1)
    ax.plot([0, 120, 120, 0, 0], [0, 0, 80, 80, 0], **kw); ax.plot([60, 60], [0, 80], **kw)
    ax.add_patch(Circle((60, 40), 10, fill=False, **kw))
    for s in (0, 1):
        x = 0 if s == 0 else 120; d = 1 if s == 0 else -1
        ax.plot([x, x + 18 * d, x + 18 * d, x], [18, 18, 62, 62], **kw)
        ax.plot([x, x + 6 * d, x + 6 * d, x], [30, 30, 50, 50], **kw)
    ax.set_xlim(-1, 121); ax.set_ylim(-1, 81); ax.set_aspect("equal"); ax.axis("off")


def player(ax, x, y, team, label=None, ball=False):
    c = RED if team == "us" else BLUE
    ax.add_patch(Circle((x, y), 1.9, fc=c, ec="white", lw=1.2, zorder=4))
    if ball:
        ax.add_patch(Circle((x + 2.2, y - 1.6), 0.9, fc="white", ec=INK, lw=0.8, zorder=5))
    if label:
        ax.text(x, y + 3.4, label, color="white", ha="center", fontsize=8.5, weight="bold", zorder=6)


def arrow(ax, a, b, color="white", style="-|>", ls="-", lw=2.2, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=16, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=3))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=220, bbox_inches="tight", pad_inches=0.05, facecolor="white"); plt.close(fig)


# 1. COUNT — two panels: few back (go) vs many back (keep)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.7))
for ax, n_back, verdict, col in ((axs[0], 3, "3 BEHIND THE BALL  →  GO NOW", AMBER), (axs[1], 8, "8 BEHIND THE BALL  →  KEEP IT", LINE)):
    full_pitch(ax)
    player(ax, 52, 42, "us", ball=True)
    ys = np.linspace(14, 66, n_back)
    xs = 60 + 40 * (np.arange(n_back) % 2 == 0) * 0 + np.linspace(70, 98, n_back) * 0 + (np.array([72, 88, 96, 80, 92, 100, 76, 86][:n_back]))
    xs = np.array([95, 104, 98, 84, 100, 90, 104, 80][:n_back]) if n_back == 8 else np.array([96, 104, 90])
    ys = np.array([30, 44, 58, 22, 52, 40, 62, 46][:n_back]) if n_back == 8 else np.array([34, 48, 60])
    for x, y in zip(xs, ys):
        player(ax, x, y, "them")
    for x, y in ((40, 20), (36, 58), (30, 40), (46, 64), (44, 12)):
        player(ax, x, y, "them")
    if n_back == 3:
        player(ax, 62, 22, "us"); player(ax, 64, 62, "us")
        arrow(ax, (62, 22), (92, 16), color=AMBER, ls="--"); arrow(ax, (64, 62), (96, 70), color=AMBER, ls="--")
        arrow(ax, (54, 42), (88, 22), color=AMBER)
    else:
        player(ax, 44, 30, "us"); arrow(ax, (54, 42), (45, 32), color=LINE)
    ax.set_title(verdict, color=INK, fontsize=12, weight="bold", pad=6)
fig.text(0.5, 0.06, "We win the ball (red, with ball). Count the opponents (blue) between the ball and their goal.", ha="center", fontsize=9.5, color="#4a554f")
save(fig, "rule_count.png")

# 2. STRIKE THROUGH — edge of box vs set block
fig, ax = plt.subplots(figsize=(6.2, 4.6)); half_pitch(ax, x0=72)
block = [(103, 26), (104, 34), (104, 46), (103, 54), (110, 32), (111, 44), (109, 38), (99, 45), (97, 31), (97, 52)]
for x, y in block:
    player(ax, x, y, "them")
ax.add_patch(Circle((118.5, 40), 1.9, fc=AMBER, ec="white", lw=1.2, zorder=4))
player(ax, 92, 40, "us", ball=True); player(ax, 90, 18, "us"); player(ax, 90, 62, "us")
ax.add_patch(Polygon([(94, 39), (120, 36), (120, 44)], closed=True, fc=AMBER, alpha=0.18, ec="none", zorder=2))
arrow(ax, (94.5, 39), (120, 41.5), color=AMBER, lw=2.6)
arrow(ax, (92, 38), (90.5, 21), color=LINE, ls=":", lw=1.6)
ax.text(96, 4, "SHOOT  +0.04 to +0.05 xG", color=AMBER, fontsize=10, weight="bold", ha="center")
ax.text(73.5, 12, "sideways pass\n(what teams usually do)", color=LINE, fontsize=8, ha="left")
save(fig, "rule_strike.png")

# 3. RUN INTO SPACE
fig, ax = plt.subplots(figsize=(6.2, 4.6)); half_pitch(ax, x0=72)
for x, y in ((104, 30), (106, 44), (99, 58), (112, 38)):
    player(ax, x, y, "them")
ax.add_patch(Circle((118.5, 40), 1.9, fc=AMBER, ec="white", lw=1.2, zorder=4))
player(ax, 84, 14, "us", ball=True); player(ax, 92, 52, "us")
ax.add_patch(Rectangle((86, 6), 16, 18, fc=AMBER, alpha=0.14, ec="none", zorder=2))
arrow(ax, (86, 13), (101, 17), color=AMBER, lw=2.6)
arrow(ax, (86, 14), (95, 27), color=LINE, ls=":", lw=1.6, rad=-0.4)
ax.text(96, 3, "CARRY INTO SPACE  +0.04 xG", color=AMBER, fontsize=8.6, weight="bold", ha="center")
ax.text(84, 31, "dribble at a man", color=LINE, fontsize=8, ha="center")
save(fig, "rule_run.png")

# 4. PRESS THE EXPOSED (our box on the left, mirror: draw own half)
fig, ax = plt.subplots(figsize=(6.2, 4.6)); half_pitch(ax, x0=72); ax.invert_xaxis()
ax.text(117, 4, "our goal ←", color=LINE, fontsize=8, ha="left")
player(ax, 98, 66, "them", label="receiver", ball=True); player(ax, 86, 56, "them"); player(ax, 84, 72, "them")
for x, y in ((106, 40), (108, 30), (104, 50), (112, 44), (96, 34)):
    player(ax, x, y, "us")
arrow(ax, (96.5, 36), (98, 62), color=AMBER, lw=2.6)
ax.text(96, 16, "≤5 of us goal-side,\nreceiver unmarked:\nPRESS WITHIN 2 s", color=AMBER, fontsize=9, weight="bold", ha="center")
save(fig, "rule_press.png")

# 5. DEAD BALLS — elasticity bars across datasets
fig, ax = plt.subplots(figsize=(6.2, 3.2))
labs = ["StatsBomb\n2015/16", "Wyscout\n2017/18", "Unseen women's\n& ISL leagues"]
sp, op = [0.41, 0.28, 0.72], [1.09, 0.89, 1.02]
x = np.arange(3); w = 0.36
ax.bar(x - w / 2, op, w, color="#9aa69f", label="Open play")
ax.bar(x + w / 2, sp, w, color=RED, label="Set pieces")
for i in range(3):
    ax.text(x[i] - w / 2, op[i] + 0.03, f"{op[i]:.2f}", ha="center", fontsize=9)
    ax.text(x[i] + w / 2, sp[i] + 0.03, f"{sp[i]:.2f}", ha="center", fontsize=9, color=RED, weight="bold")
ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=9); ax.set_ylim(0, 1.3)
ax.set_ylabel("How much chances depend\non team strength (elasticity)", fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
save(fig, "rule_deadballs.png")

# 6. VALUE — club outcomes before/after
J = json.load(open(os.path.join(ROOT, "results", "clubs_playbook.json")))
rows = [("Man City", "p_title", "Title"), ("Chelsea", "p_top4", "Top 4"), ("Newcastle", "p_top4", "Top 4"), ("Brighton", "p_top4", "Top 4"),
        ("Leeds", "p_releg", "Relegation"), ("Sunderland", "p_releg", "Relegation"), ("Wolves", "p_releg", "Relegation")]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for i, (c, k, lab) in enumerate(rows[::-1]):
    b, p = J[c]["base"][k] * 100, J[c]["playbook"][k] * 100
    good = p > b if k != "p_releg" else p < b
    ax.plot([b, p], [i, i], color="#c7cfca", lw=4, solid_capstyle="round", zorder=1)
    ax.plot(b, i, "o", color="#9aa69f", ms=9, zorder=2); ax.plot(p, i, "o", color=RED if k != "p_releg" else BLUE, ms=11, zorder=3)
    ax.text(-3, i, f"{c} · {lab}", ha="right", va="center", fontsize=9.5)
    ax.text(max(b, p) + 2.5, i, f"{b:.0f}% → {p:.0f}%", va="center", fontsize=9, weight="bold")
ax.set_xlim(0, 112); ax.set_yticks([]); ax.set_xlabel("Probability over a simulated 2025/26 season (%)", fontsize=9)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
save(fig, "value_clubs.png")
print("figures in", OUT)

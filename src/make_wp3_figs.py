"""Figures for white paper v3 (round 22): strike stress ladder, overload gap, club value v3."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "figures", "wp")
RED, BLUE, AMBER, INK, GREY, FELT = "#d9553a", "#3b6fc0", "#f0b13c", "#17201b", "#9aa69f", "#1e4a38"
plt.rcParams.update({"font.family": "DejaVu Sans"})
POPS = ("discovery", "confirm_club", "confirm_women")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=220, bbox_inches="tight", pad_inches=0.05, facecolor="white"); plt.close(fig)


def clean(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


# 1. Strike stress ladder
S = json.load(open(os.path.join(ROOT, "results", "r22a_strike.json")))
avg = lambda d: np.mean([d[p]["est"] for p in POPS])
rows = [("Same zone, space and lane", np.mean([S["raw_and_cell"][p]["cell"][0] for p in POPS]), GREY),
        ("+ keeper position", avg(S["S1_keeper_cells"]["per_pop"]), GREY),
        ("+ 19 details (doubly robust)", avg(S["S2_aipw_ate"]["per_pop"]), AMBER),
        ("Extra shots a coach would add", avg(S["S2_aipw_atu"]["per_pop"]), RED),
        ("…measured 60 s later", avg(S["robust_60s_atu"]["per_pop"]), RED),
        ("Placebo: the 20 s BEFORE the shot", avg(S["S4_placebo_pre20"]["per_pop"]), BLUE)]
fig, ax = plt.subplots(figsize=(7.2, 3.3))
for i, (lab, v, c) in enumerate(rows[::-1]):
    ax.barh(i, v, color=c, height=0.62)
    ax.text(v + 0.0015, i, f"+{v:.3f}", va="center", fontsize=9, weight="bold", color=INK)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=9.5)
ax.set_xlabel("Shoot minus recycle (xG per decision, average of 3 populations)", fontsize=9); ax.set_xlim(0, 0.095); clean(ax)
ax.axhline(0.5, color="#c7cfca", lw=1, ls="--")
save(fig, "strike_stress.png")

# 2. Overload gap: what players choose when both lanes exist, by tier
L = json.load(open(os.path.join(ROOT, "results", "r22d_ledger.json")))
fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.25))
ax = axs[0]; tiers = [("under", "Underdogs"), ("mid", "Even games"), ("fav", "Favourites")]
for i, (k, lab) in enumerate(tiers[::-1]):
    o = L["overload"]["by_tier"][k]["overload"] * 100; c = L["overload"]["by_tier"][k]["crowded"] * 100
    ax.barh(i, c, color=GREY, height=0.6); ax.barh(i, o, left=c, color=RED, height=0.6)
    ax.text(c / 2, i, f"crowded {c:.0f}%", ha="center", va="center", color="white", fontsize=8.5, weight="bold")
    ax.text(c + o / 2, i, f"{o:.0f}%", ha="center", va="center", color="white", fontsize=8.5, weight="bold")
ax.set_yticks(range(3)); ax.set_yticklabels([t[1] for t in tiers[::-1]], fontsize=9); ax.set_xlim(0, 100); ax.set_xticks([])
ax.set_title("Pass into: crowded (grey) or overload (red)", fontsize=9.5, loc="left"); clean(ax); ax.spines["bottom"].set_visible(False)
ax = axs[1]
for i, (k, lab) in enumerate(tiers[::-1]):
    v = L["second_ball"]["by_tier"][k]["outnumbered"] * 100
    ax.barh(i, v, color=BLUE if k == "under" else GREY, height=0.6); ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9, weight="bold")
ax.set_yticks(range(3)); ax.set_yticklabels([]); ax.set_xlim(0, 40); ax.set_xticks([])
ax.set_title("Long balls into outnumbered zones", fontsize=9.5, loc="left"); clean(ax); ax.spines["bottom"].set_visible(False)
save(fig, "overload_gap.png")

# 3. Club value v3
J = json.load(open(os.path.join(ROOT, "results", "clubs_playbook_v3.json")))
rows = [("Man City", "p_title", "Title"), ("Chelsea", "p_top4", "Top 4"), ("Newcastle", "p_top4", "Top 4"), ("Brighton", "p_top4", "Top 4"),
        ("Leeds", "p_releg", "Relegation"), ("Sunderland", "p_releg", "Relegation"), ("Wolves", "p_releg", "Relegation")]
fig, ax = plt.subplots(figsize=(7.2, 3.4))
for i, (c, k, lab) in enumerate(rows[::-1]):
    b, p = J[c]["base"][k] * 100, J[c]["playbook"][k] * 100
    ax.plot([b, p], [i, i], color="#c7cfca", lw=4, solid_capstyle="round", zorder=1)
    ax.plot(b, i, "o", color=GREY, ms=9, zorder=2); ax.plot(p, i, "o", color=RED if k != "p_releg" else BLUE, ms=11, zorder=3)
    ax.text(-3, i, f"{c} · {lab}", ha="right", va="center", fontsize=9.5)
    ax.text(max(b, p) + 2.5, i, f"{b:.0f}% → {p:.0f}%", va="center", fontsize=9, weight="bold")
ax.set_xlim(0, 112); ax.set_yticks([]); ax.set_xlabel("Probability over a simulated 2025/26 season (%)", fontsize=9); clean(ax)
save(fig, "value_clubs_v3.png")
print("ok")

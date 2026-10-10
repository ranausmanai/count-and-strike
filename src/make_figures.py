"""All publication figures (figures/fig*.png) from results/*.json."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import dp

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = lambda f: json.load(open(os.path.join(ROOT, "results", f)))
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c9c8c3", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.dpi": 200,
                     "grid.color": "#ecebe7", "grid.linewidth": 0.6})


def save(f, name):
    f.savefig(os.path.join(FIG, name), bbox_inches="tight", facecolor="white"); plt.close(f)


# ---- Fig 1: the underdog's exchange rate (theory) ----
def ep(lf, la, K=95):
    rf = np.full((K, 2 * dp.GMAX + 1, 1), lf / K); ra = np.full_like(rf, la / K)
    V, _ = dp.solve(rf, ra); return V[0, dp.GMAX]
S = np.linspace(0.22, 0.55, 34); T = np.linspace(1.4, 4.0, 27)
Z = np.array([[ep(s * t, (1 - s) * t) for s in S] for t in T])
f, ax = plt.subplots(figsize=(6.2, 3.9))
cs = ax.contour(S, T, Z, levels=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3], colors=BLUE, linewidths=1.2)
ax.clabel(cs, fmt="%.1f pts", fontsize=8, colors=INK)
ax.axvline(0.5, color=GRAY, lw=0.8, ls="--"); ax.text(0.505, 3.85, "even match", color=MUTED, fontsize=8)
ax.annotate("", xy=(0.33, 1.9), xytext=(0.33, 3.1), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6))
ax.text(0.338, 1.75, "fewer goals, same share:\nthe underdog gains", color=ORANGE, fontsize=8.5, bbox=dict(fc="white", ec="none", pad=1.5))
ax.set_xlabel("Underdog's share of chances  S = λ_for / (λ_for + λ_against)")
ax.set_ylabel("Match intensity  T (total expected goals)")
ax.set_title("Fig 1. Expected points for the side with share S\n(below S≈0.45, lower intensity raises the weaker side's points)")
ax.grid(True); save(f, "fig1_exchange_rate.png")

# ---- Fig 2: pre-registered scorecard ----
p1 = R("p1_heldout.json"); p2 = R("p2_policy.json"); p3 = R("p3_style_market.json"); p4 = R("p4_tempo_market.json")
rows = [
    ("P1  lagged style → held-out EPL xG\n(Δ deviance, + = style helps)", p1["lagged"]["improvement"], p1["lagged"]["ci95"], "fail"),
    ("P2  DP policy gain, cross-fitted\n(pts/match; threshold 0.05)", p2["A_crossfit"]["mean_gain"], p2["A_crossfit"]["ci95"], "fail"),
    ("P3  habitual style vs market, EPL\n(pts residual per unit closeness)", p3["EPL_registered"]["coef"], p3["EPL_registered"]["ci"], "null"),
    ("P4  habitual tempo vs market\n(pts per extra shot/match ×100)", 100 * p4["all_underdogs"]["coef_per_shot"], [100 * x for x in p4["all_underdogs"]["ci"]], "pass"),
]
f, axs = plt.subplots(len(rows), 1, figsize=(6.2, 4.6))
for ax, (lab, est, ci, verdict) in zip(axs, rows):
    col = {"pass": AQUA, "fail": ORANGE, "null": GRAY}[verdict]
    ax.plot(ci, [0, 0], color=col, lw=2.2, solid_capstyle="round"); ax.plot([est], [0], "o", ms=8, color=col, mec="white", mew=1.5)
    ax.axvline(0, color=INK, lw=0.8); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    span = max(abs(ci[0]), abs(ci[1])) * 1.35; ax.set_xlim(-span, span)
    ax.text(-span, 0.55, lab, fontsize=8.3, color=INK, va="bottom")
    ax.text(span, 0.55, {"pass": "SUPPORTED", "fail": "NOT SUPPORTED", "null": "NULL"}[verdict], fontsize=8, color=INK, ha="right", va="bottom", weight="bold")
    ax.set_ylim(-0.6, 1.6)
axs[0].set_title("Fig 2. Pre-registered tests (point estimate, 95% CI). P5 shown in Table 3")
f.tight_layout(); save(f, "fig2_scorecard.png")

# ---- Fig 3: channel elasticities, two datasets ----
sb = R("channels.json"); wy = R("wyscout_replication.json")
chs = ["build_up", "counter", "high_regain", "penalty", "corner", "throw_in", "free_kick"]
lab = {"build_up": "Build-up play", "counter": "Counter-attack", "high_regain": "High regain", "penalty": "Penalty",
       "corner": "Corner", "throw_in": "Attacking throw-in", "free_kick": "Attacking free kick"}
f, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True, gridspec_kw=dict(wspace=0.08))
y = np.arange(len(chs))[::-1]
for d, off, col, name in ((sb, 0.15, BLUE, "StatsBomb 2015/16"), (wy, -0.15, ORANGE, "Wyscout 2017/18")):
    e = [d["elasticity_all"][c]["e_own"] for c in chs]; se = [d["elasticity_all"][c]["se_own"] for c in chs]
    a1.errorbar(e, y + off, xerr=1.96 * np.array(se), fmt="o", ms=5, color=col, ecolor=col, elinewidth=1.2, capsize=0, label=name)
    sh = [d["underdog_share_mismatch"][c] * 100 for c in chs]
    a2.plot(sh, y + off, "o", ms=6, color=col, mec="white", mew=1.0)
for ax in (a1, a2):
    ax.grid(True, axis="x")
a1.axvline(1, color=GRAY, lw=0.8, ls="--"); a1.axvline(0, color=INK, lw=0.8)
a1.set_yticks(y); a1.set_yticklabels([lab[c] for c in chs])
a1.set_xlabel("Elasticity to team attack strength"); a1.legend(frameon=False, fontsize=8, loc="lower right")
a2.axvline(sb["underdog_share_mismatch"]["xg_total"] * 100, color=GRAY, lw=0.8, ls="--")
a2.set_xlabel("Underdog share in mismatches (%)")
a1.set_title("Fig 3. Where quality matters less: set pieces are flat, build-up is steep", loc="left", x=-0.45)
save(f, "fig3_channels.png")

# ---- Fig 4: cheap-skill map ----
f, ax = plt.subplots(figsize=(6.2, 4.0))
nice = {"corner_xg_per_corner": "xG per corner", "throw_xg_per_att_throw": "xG per att. throw", "fk_xg_per_att_fk": "xG per att. free kick",
        "counter_xg_per_match": "Counter xG", "buildup_xg_per_match": "Build-up xG", "corners_won_per_match": "Corners won",
        "corner_xg_conceded_per_match": "Corner xG conceded", "counter_xg_conceded_per_match": "Counter xG conceded",
        "buildup_xg_conceded_per_match": "Build-up xG conceded"}
ax.axvspan(-0.25, 0.25, color="#eef6f1", zorder=0); ax.text(0, 0.98, "weakly tied to wealth", ha="center", color=MUTED, fontsize=8, transform=ax.get_xaxis_transform())
for k, n in nice.items():
    a, b = sb["cheap_skill_map"][k], wy["cheap_skill_map"][k]
    ax.plot([a["wealth_corr"], b["wealth_corr"]], [a["reliability"], b["reliability"]], color="#d6d5d0", lw=1, zorder=1)
    ax.plot(a["wealth_corr"], a["reliability"], "o", color=BLUE, ms=6, mec="white", zorder=2)
    ax.plot(b["wealth_corr"], b["reliability"], "s", color=ORANGE, ms=5.5, mec="white", zorder=2)
    dy = {"corner_xg_per_corner": -0.035, "throw_xg_per_att_throw": 0.035, "corners_won_per_match": 0.03, "counter_xg_per_match": -0.03}.get(k, 0)
    ax.text(a["wealth_corr"] + 0.03, a["reliability"] + dy, n, fontsize=7.5, color=INK, va="center")
ax.plot([], [], "o", color=BLUE, label="StatsBomb 2015/16"); ax.plot([], [], "s", color=ORANGE, label="Wyscout 2017/18")
ax.legend(frameon=False, fontsize=8, loc="lower left"); ax.grid(True)
ax.set_xlabel("Correlation with team market rating"); ax.set_ylabel("Split-half reliability (persistence)")
ax.set_xlim(-0.9, 1.15); ax.set_ylim(-0.05, 1.0)
ax.set_title("Fig 4. Cheap skills (persistent, not bought): only set-piece efficiency\nqualifies, and its persistence is modest")
save(f, "fig4_cheap_skills.png")

# ---- Fig 5: when a level underdog should open up ----
mins = np.arange(0, 95); shares = np.linspace(0.25, 0.5, 26)
M = np.zeros((len(shares), len(mins)))
for i, s_ in enumerate(shares):
    t = 2.7; lf, la = s_ * t, (1 - s_) * t; m = np.array([0.91, 1.0, 1.09])
    RF = np.broadcast_to(lf / 95 * m, (95, 2 * dp.GMAX + 1, 3)).copy(); RA = np.broadcast_to(la / 95 * m, (95, 2 * dp.GMAX + 1, 3)).copy()
    _, pol = dp.solve(RF, RA); M[i] = pol[:, dp.GMAX]
f, ax = plt.subplots(figsize=(6.2, 3.3))
from matplotlib.colors import ListedColormap
ax.imshow(M, aspect="auto", origin="lower", cmap=ListedColormap([BLUE, "#f0efec", ORANGE]), vmin=0, vmax=2,
          extent=[0, 95, shares[0] * 100, shares[-1] * 100])
ax.set_xlabel("Minute"); ax.set_ylabel("Underdog share of chances (%)")
ax.text(30, 30, "LOW tempo: protect the draw", color="white", fontsize=9, weight="bold")
ax.text(70, 46.5, "HIGH tempo:\ngo for the win", color="white", fontsize=8.5, weight="bold", ha="center")
ax.set_title("Fig 5. Optimal tempo at a level score (T = 2.7 goals, tempo menu ±9%)")
save(f, "fig5_switch_map.png")

# ---- Fig 6: Brentford season scenarios ----
ss = R("season_sim.json"); sens = R("season_sensitivity.json")
sc = [("Baseline (2025/26 strength)", ss["baseline"]), ("Set pieces to 90th pct (low persistence)", sens["setpiece_p90_lowrel"]),
      ("Set pieces to 90th pct", ss["setpiece_p90"]), ("Tempo policy (upper bound)", ss["tempo_dp_c09"]),
      ("Set pieces + tempo", ss["setpiece+tempo"])]
f, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=True, gridspec_kw=dict(wspace=0.08, width_ratios=[1.2, 1]))
y = np.arange(len(sc))[::-1]
a1.barh(y, [v["E_pts"] for _, v in sc], color=[GRAY] + [BLUE] * 4, height=0.55)
for yi, (_, v) in zip(y, sc):
    a1.text(v["E_pts"] + 0.3, yi, f"{v['E_pts']:.1f}", va="center", fontsize=8, color=INK)
a1.set_xlim(45, 57); a1.set_yticks(y); a1.set_yticklabels([n for n, _ in sc]); a1.set_xlabel("Expected points")
a2.barh(y, [v["p_top4"] * 100 for _, v in sc], color=[GRAY] + [AQUA] * 4, height=0.55)
for yi, (_, v) in zip(y, sc):
    a2.text(v["p_top4"] * 100 + 0.2, yi, f"{v['p_top4']*100:.1f}%  (title {v['p_title']*100:.2f}%)", va="center", fontsize=7.5, color=INK)
a2.set_xlim(0, 16); a2.set_xlabel("P(top 4), %")
a1.set_title("Fig 6. Brentford: what the evidence-backed playbook is worth over 38 matches")
save(f, "fig6_brentford.png")

# ---- Fig 7: Leicester decomposition ----
L = R("leicester.json")["leicester"]
f, ax = plt.subplots(figsize=(6.2, 2.3))
steps = [("Market expectation", L["xpts_mkt"]), ("Chance quality (xG)", L["xpts_xg"]), ("Actual points", L["pts"])]
ax.barh([2], [L["xpts_mkt"]], color=GRAY, height=0.5)
ax.barh([1], [L["xpts_xg"] - L["xpts_mkt"]], left=L["xpts_mkt"], color=BLUE, height=0.5)
ax.barh([0], [L["pts"] - L["xpts_xg"]], left=L["xpts_xg"], color=ORANGE, height=0.5)
ax.set_yticks([2, 1, 0]); ax.set_yticklabels(["Market expected points", "+ better chances than priced", "+ finishing / keeping / luck"])
for yv, x, t in ((2, L["xpts_mkt"], f"{L['xpts_mkt']:.0f}"), (1, L["xpts_xg"], f"+{L['xpts_xg']-L['xpts_mkt']:.0f} → {L['xpts_xg']:.0f}"), (0, L["pts"], f"+{L['pts']-L['xpts_xg']:.0f} → {L['pts']:.0f}")):
    ax.text(x + 0.6, yv, t, va="center", fontsize=8.5, color=INK)
ax.set_xlim(0, 92); ax.set_xlabel("Points, Premier League 2015/16")
ax.set_title("Fig 7. Leicester 2015/16: two thirds of the miracle sat above their chances")
save(f, "fig7_leicester.png")
print("figures written")

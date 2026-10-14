"""Shareable Count & Strike rule card (1080x1350)."""
import os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT = os.path.join(os.path.dirname(__file__), "..")
BG, INK, MUTED, ACC, ACC2 = "#0f1720", "#ffffff", "#a9b4c0", "#2fd08a", "#ffb547"
f = plt.figure(figsize=(10.8, 13.5), dpi=100); f.patch.set_facecolor(BG)
ax = f.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1080); ax.set_ylim(0, 1350); ax.axis("off")
ax.text(70, 1260, "COUNT & STRIKE", color=INK, fontsize=46, weight="bold", family="DejaVu Sans")
ax.text(72, 1205, "5 football rules that survived testing on 3.6 million decisions\nmen's & women's football · 4 independent datasets", color=MUTED, fontsize=15, va="top", linespacing=1.4)
rules = [("1", "COUNT", "Win the ball → count opponents behind it.", "≤3 back: attack NOW.  8+ back: keep it.", "Counter worth 2×"),
         ("2", "STRIKE THROUGH", "Near the box vs a set defence:", "shoot — don't pass sideways or back.", "+0.04 to +0.11 xG per decision"),
         ("3", "RUN INTO SPACE", "Free in the final third, few defenders:", "carry it forward — don't dribble.", "+0.03 to +0.04 xG"),
         ("4", "PRESS THE EXPOSED", "They receive near your box, unmarked,", "few of you behind the ball: press in 2 s.", "+0.008 xG, ~once a match"),
         ("5", "DEAD BALLS", "Set pieces depend least on quality.", "Train them like your best weapon.", "Weaker teams lose least here")]
y = 1080
for n, title, l1, l2, val in rules:
    ax.add_patch(FancyBboxPatch((60, y - 150), 960, 165, boxstyle="round,pad=0,rounding_size=22", fc="#18232f", ec="none"))
    ax.text(100, y - 10, n, color=ACC, fontsize=44, weight="bold", va="top")
    ax.text(180, y - 18, title, color=INK, fontsize=22, weight="bold", va="top")
    ax.text(180, y - 62, l1, color=MUTED, fontsize=16, va="top")
    ax.text(180, y - 92, l2, color=INK, fontsize=16, va="top", weight="bold")
    ax.text(990, y - 22, val, color=ACC2, fontsize=13, va="top", ha="right")
    y -= 185
ax.text(70, 95, "Worth ≈ +4 points a season for any club. Thousands of other ideas failed.", color=INK, fontsize=15, weight="bold")
ax.text(70, 55, "Predictions locked publicly before the next data release · Project Underdog · data: StatsBomb Open Data", color=MUTED, fontsize=12.5)
f.savefig(os.path.join(ROOT, "figures", "count_and_strike_card.png"), facecolor=BG); print("ok")

"""Shareable Count & Strike rule card (1080x1350)."""
import os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT = os.path.join(os.path.dirname(__file__), "..")
BG, INK, MUTED, ACC, ACC2 = "#0f1720", "#ffffff", "#a9b4c0", "#2fd08a", "#ffb547"
f = plt.figure(figsize=(10.8, 13.5), dpi=100); f.patch.set_facecolor(BG)
ax = f.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1080); ax.set_ylim(0, 1350); ax.axis("off")
ax.text(70, 1260, "COUNT & STRIKE", color=INK, fontsize=46, weight="bold", family="DejaVu Sans")
ax.text(72, 1205, "8 football rules that survived testing on 3.6 million decisions\nmen's & women's football \u00b7 independent datasets \u00b7 pre-registered", color=MUTED, fontsize=15, va="top", linespacing=1.4)
rules = [("1", "COUNT", "Win the ball \u2192 count opponents behind it.", "\u22643 back: attack NOW.  8+ back: keep it.", "Counter worth 2\u00d7"),
         ("2", "STRIKE THROUGH", "Near the box vs a set defence, any position:", "shoot \u2014 don't pass sideways or back.", "+0.04 to +0.11 xG"),
         ("3", "PLAY INTO THE OVERLOAD", "Count ours vs theirs in each lane ahead:", "pass into the lane where we outnumber them.", "Players do it only 23%"),
         ("4", "WIN THE SECOND BALL", "Going long?", "aim where we outnumber them at the landing.", "+5 pts ball retention/player"),
         ("5", "RUN INTO SPACE", "Free in the final third, few defenders:", "carry it forward \u2014 don't dribble.", "+0.04 xG"),
         ("6", "THIRD-MAN RELEASE", "Around the box:", "release first time to the runner beyond.", "+0.10 xG wide in box"),
         ("7", "PRESS THE EXPOSED", "Unmarked receiver near your box, few of you back:", "press within 2 seconds.", "once a match"),
         ("8", "DEAD BALLS", "Set pieces depend least on quality.", "Train them like your best weapon.", "Weaker teams lose least")]
y = 1125
for n, title, l1, l2, val in rules:
    ax.add_patch(FancyBboxPatch((60, y - 106), 960, 114, boxstyle="round,pad=0,rounding_size=18", fc="#18232f", ec="none"))
    ax.text(98, y - 8, n, color=ACC, fontsize=34, weight="bold", va="top")
    ax.text(165, y - 12, title, color=INK, fontsize=18, weight="bold", va="top")
    ax.text(165, y - 48, l1, color=MUTED, fontsize=13.5, va="top")
    ax.text(165, y - 74, l2, color=INK, fontsize=13.5, va="top", weight="bold")
    ax.text(992, y - 16, val, color=ACC2, fontsize=11.5, va="top", ha="right")
    y -= 124
ax.text(70, 95, "Worth \u2248 +8 points a season (simulated). Thousands of other ideas failed.", color=INK, fontsize=15, weight="bold")
ax.text(70, 55, "Predictions locked publicly before the next data release · Project Underdog · data: StatsBomb Open Data", color=MUTED, fontsize=12.5)
f.savefig(os.path.join(ROOT, "figures", "count_and_strike_card.png"), facecolor=BG); print("ok")

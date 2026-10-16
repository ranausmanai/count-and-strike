# Agent bait: bait and switch

**Question.** Does a bait make the far lane overloaded, and does switching to it then pay off? A bait here means at least 2 of our actions on one side in the last 8 s, with more opponents on the ball's side than at the start.

**Data.** StatsBomb 360: side passes from x ≥ 40 in possessions at least 8 s old. There are 67,368 decisions in discovery, 60,180 in men's club and 36,699 in women's, and about 28% are baited. Switches after a bait: 1,525, 1,191 and 660. Switches without one: 1,594, 907 and 738.

| Test (cell-matched) | Disc | Club | Women | Pooled |
|---|---|---|---|---|
| H1 far-lane overload, bait − none | +.105 (.010) | +.079 (.011) | +.052 (.015) | +.069 (.009) |
| H2 switch, bait vs none (net xG 20 s) | −.0007 (.0019) | +.0023 (.0035) | +.0019 (.0039) | +.0022 (.0026) |
| H3 after bait, switch − stay | +.0003 (.0014) | +.0043 (.0018) | +.0053 (.0027) | +.0046 (.0015) |
| H4 possessions, bait→switch − bait only | −.0027 (.0015) | −.0022 (.0016) | −.0018 (.0022) | −.0021 (.0013) |

**Verdict: MIXED.**
- **Mechanism (H1): CONFIRMED.** Inside a bait, the far-lane overload rises from the bait's start by +0.11, +0.16 and +0.04. Without a bait it barely changes.
- **Payoff (H2–H4): NOT CONFIRMED.** All three are null in discovery. AIPW in discovery gives +.0003, −.0019 and −.0042. Placebos are clean.
- **Usage.** Teams switch more after a bait: 7.9% vs 4.8%, 7.0% vs 3.0%, and 6.3% vs 4.2%.

**Key caveat.** The camera follows the ball, so the far lane is often off-screen (median visibility about 0.3). Part of the overload may be a visibility artefact, and teams choose when to bait.

**Coaching.** Holding the ball on one side does open the far lane, but switching into it showed no measurable xG gain.

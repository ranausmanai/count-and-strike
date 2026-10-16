# Project Underdog — Round 19 pre-registration: the sweet-spot search

*Written 2026-10-15 before running. Same 360 split: discovery = men's international; confirmation = men's club AND women's (both required).*

## Families and grids (~850 candidate rules)

* **T, tempo (first-time vs controlled).** For passes in each of the 486 situation cells: first-time (a pass within 1.0 s of the same player's ball receipt) vs controlled (≥ 1.0 s).
* **P, pressing traps.** Opponent receipts. Cells: zone (6×3) × touchline band (receiver ≤ 10 yds from a touchline vs not) × our players within 10 yds of the receiver (0 / 1 / 2+) × receiver support (<8 / 8–15 / >15 yds). Choice: press within 2 s vs hold.
* **C, count threshold sweep.** At regains outside the final third, the first action is *go* (pass or carry gaining ≥ 10 yds within 3 s) or *keep* (otherwise). For each opponents-goal-side value k = 2…9 and each regain zone (own third / own half / middle third), compute the go − keep advantage. The **sweet spot** is the largest k at which go beats keep in all three populations.
* **O, overload sweep.** Zone (5 bands from x ≥ 40) × threshold t ∈ {1, 2, 3}: passes into a lane with overload ≥ t vs ≤ −t.

## Rules

* **Families T and P:** BH 5% in discovery. Confirmed = same sign in both confirmation populations and pooled one-sided p < 0.05/k (Holm, within family).
* **Family C:** reported as a curve with all populations; a threshold is "confirmed" if the advantage is positive with one-sided p < 0.05 in each of the three populations.
* **Family O:** confirmed if positive with one-sided p < 0.05 in each of the three populations.

## Season simulation sweep

All 2⁶ combinations of the six confirmed Count & Strike rules (plus any new confirmed rules), × adoption levels {25, 50, 75, 100%}, × 9 clubs, 30,000 seasons each. This gives points, top-4, title and relegation for each combination, and the best package per club. The value of each rule comes from its confirmed estimates. This step is simulation, not evidence.

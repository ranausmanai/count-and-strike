# Round 26 pre-registration: effect-calibrated simulator, then grid search

*Written 2026-10-17, before calibration and search.*

## Why this round exists
Round 25's top-10 tactics were all exploits: no pressing, no far-side marking, lopsided attacks, with gains of +1.8 to +2.1 points per match. Escape alone gained +0.51, which is too large.

## The simulator (`engine5`)
`engine5` adds tracking-data realism:
- Block shift slope 0.38 with a 1.5 s lag. Tracking data gives 0.38 and 1.4–1.7 s.
- Pressers only from ≤ 15 m, arriving goal-side, with a sprint factor `cal[8]`.
- The far-side marker stays within `cal[9]` m of the back line.

## Calibration
Ten parameters are fitted to two kinds of target.

**Average targets (equal teams):**

| Measure | Target |
|---|---|
| Goals per team | 1.35 |
| Shots | 12.5 |
| Pass completion | 0.83 |
| Tackles | 45 |
| Counter ratio (≤3 back vs 8+) | 2.0 |

**Effect anchors (equal teams, points difference on common seeds).** Each target is 0, because real-data tests found no edge:
- **N1:** pressing with 2 vs 0 pressers. Real data: 1 of 102 pressing situations helped and the 5 classic triggers were null.
- **N2:** bait-and-switch (asym 0.8 + release 1) vs none. Real data: agent bait H2–H4 null.
- **N3:** marking the far side vs not. There's no real-data test; the anchor encodes "sensible marking doesn't wreck your defence".

**Loss:** the sum of squared relative errors on the averages, plus (anchor / 0.12)² for each anchor.

## Grid search
The same procedure as `PREREGISTRATION_25.md` (3,000-point grid, then stages 1–3, the same guards and support rule), run on engine5 with the new calibration.
- **Underdog strength:** re-tuned to 0.9 points per match.

## Additional supported-in-simulation rule
A tactic's gain must also exceed the size of the biggest residual anchor error.

## Interpretation
- A tactic that survives is a hypothesis the effect-calibrated simulator can't explain away. It is still not proof.

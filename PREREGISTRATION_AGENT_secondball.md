# Pre-registration — agent "secondball": win the second ball (local balance at the landing zone)

*Written 2026-10-16 before any outcome (next-possession team, net xG) is computed. Data: StatsBomb 360 (477 matches).
Discovery = men's international (`DISC`); confirmation = men's club AND women's (`WOMEN`), both required.*

## Decision set (long balls)
- `Pass` events, periods 1–2, `pass.length` ≥ 35 yds, with a 360 freeze frame for the pass.
- Pass type: open play (no `pass.type`) or `Goal Kick`. (Open-play keeper kicks are included automatically.) Corners, free kicks,
  throw-ins and kick-offs are excluded.
- Clearances: StatsBomb `Clearance` events carry no end location, so they cannot be placed; they are NOT included (deviation from the
  question wording, declared here up front). Passes flagged as clearances are not a separate type in the data.
- Visibility: keep only frames whose `visible_area` polygon contains the pass `end_location` (ray-casting point-in-polygon).

## Exposure
From the freeze frame AT the pass: `ours` = visible outfield teammates (not actor, not keeper) within 15 yds of the end location;
`theirs` = visible outfield opponents (not keeper) within 15 yds. `bal = ours − theirs`.
Usage reported: share of long balls with bal < 0 (outnumbered), = 0, > 0, per population.

## Outcomes
- `keep5`: 1 if the `possession_team` of the first event with timestamp ≥ t0 + 5 s (same period) is the passing team, else 0.
  Drop if no such event in the period.
- `net`: own − opponent non-penalty xG in the 20 s after the pass (`r15_360.windows`).

## Controls
End zone = x band of end location (<40, 40–60, 60–80, 80–102, ≥102) × lane (|y−40| ≤ 13 central, else left/right) → up to 15 cells;
pass height (Ground / Low / High); `under_pressure` on the passer (bool); plus goal-kick indicator.

## Hypotheses (directional, all "+")
- H1: in an OLS linear probability model `keep5 ~ bal + C(endzone) + C(height) + pressure + goalkick`, coef(bal) > 0.
- H2: `net ~ bal + same controls`, coef(bal) > 0.
- H3 (choice rule): High passes only. Superior = bal ≥ 1, Outnumbered = bal ≤ −1 (bal = 0 dropped). Within end-zone cell,
  diff = mean(net | superior) − mean(net | outnumbered); cells need ≥ 5 in each arm; pooled with weights n_sup·n_out/(n_sup+n_out),
  pooled SE = sqrt(Σw²SE²)/Σw. Secondary (descriptive, not in the test family): same estimator on keep5.
- SEs for H1/H2: cluster-robust by match.

## Success rule
- Discovery: H1, H2, H3 two-sided p (z = est/SE), Benjamini–Hochberg 5%, with hypothesised sign.
- Confirmation: each passing test must have the same sign in men's club AND women's, inverse-variance pooled one-sided p < 0.05,
  Holm-corrected across tests carried to confirmation.
- Robustness for any confirmed result: (a) 45 s horizon net; (b) finer matching — add counts of all visible players within 15 yds
  (crowding = ours + theirs, as a categorical 0–2/3–5/6+) as a control for H1/H2 and as an extra cell dimension for H3; (c) radius 10 yds.

## Anticipated caveats
Freeze frame is at the pass moment, not at landing; players move during a 2–3 s flight. Visibility filter selects passes whose landing
zone is on camera (likely shorter/more central). Passers aim at teammates, so bal partly encodes the intended recipient being free
(i.e. selection on the pass being "on"). Not causal.

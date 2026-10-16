# Pre-registration — agent "cross": when should a team cross?

Written before any outcome (net xG) was computed. Code: `src/agent_cross_*.py`.

## Question
Coaching lore: "only cross when you have enough bodies in the box". Does the value of crossing
(relative to what players otherwise do in the same wide attacking-third situation) depend on
how many of our players are in the box?

## Sample (StatsBomb 360, periods 1–2)
On-ball decisions by the team in possession: Pass, Carry (moved ≥1 yd), Dribble, starting at
x ≥ 96 and |y − 40| > 18, with a 360 freeze frame. Open play only: passes of a restart type
(corner, free kick, throw-in, goal kick, kick-off) excluded, and possessions whose play pattern
is From Corner / From Free Kick / From Throw In / From Kick Off / From Goal Kick excluded
(i.e. keep Regular Play, From Counter, From Keeper, Other). Shots are excluded.

Populations: discovery = men's international (`DISC`); confirmation = men's club and women's (`WOMEN`), separately.

## Choices
Classified in this order:
- **cutback**: pass with `cut_back`.
- **cross**: pass with `cross`, or any pass ending in the penalty area (x ≥ 102, 18 ≤ y ≤ 62).
  Sub-type **cross_high** (height = High Pass) / **cross_low** (Ground or Low Pass).
- **recycle**: other pass with dx ≤ −5.
- **inside**: other pass that moves ≥ 5 yds towards the centre line (|y−40| reduced by ≥ 5).
- **carry**: Carry or Dribble.
- **other_pass**: any remaining pass (down the line, lateral, long, etc.).

CROSS = cross_high + cross_low. Alternatives = cutback, inside, carry, recycle, other_pass.

## Cells (matching)
- OURS: visible outfield teammates (excluding actor and keeper) inside the penalty area: 0–1 / 2 / 3 / 4+.
- THEIRS: visible outfield opponents (excluding keeper) inside the penalty area: ≤4 / 5–6 / 7+.
- PRESS: nearest visible opponent to the ball carrier < 3 yds (yes/no).
24 cells. All comparisons are within cell (cell fixed effects).

Primary sample requires that the freeze frame's visible area contains the penalty spot (108, 40)
so the box counts are meaningful (otherwise counts are trivially low). Robustness: all frames.

## Outcome
net = own xG − opponent xG (non-penalty shots, StatsBomb xG) in the 20 s after the action
(window conventions of `r15_360.windows`). Robustness horizon 45 s.

## Estimation
OLS of net on choice indicators with cell fixed effects, standard errors clustered by match.

## Hypotheses (directional)
- **H1 (dose-response)**: the cross-vs-alternative gap increases with OURS. Model on cross + all
  alternatives: net ~ C(cell) + cross + cross × ours_ord (ours_ord = 0,1,2,3). Test: slope > 0.
  "Best alternative" interpretation (secondary, not in the success family): for each OURS level,
  the alternative with the highest cell-adjusted mean is chosen in DISCOVERY only, then the
  cross-minus-that-alternative gap is re-estimated in confirmation (no re-selection).
- **H2**: with OURS ≥ 3 (levels 3 and 4+), CROSS − usual alternative > 0.
- **H3**: with OURS 0–1, CROSS − usual alternative < 0.
  "Usual alternative" = the most frequent non-cross choice in that OURS subset in DISCOVERY
  (fixed then for confirmation). Sample = cross + that alternative, cell FE.
- **H4**: with THEIRS 7+, (cross_low + cutback) − cross_high > 0. Sample = those three choices, cell FE.

## Success rule
Discovery: two-sided p of the four primary tests, Benjamini–Hochberg at 5%, AND the sign in the
predicted direction. A hypothesis that passes discovery is CONFIRMED if the estimate has the same
sign in BOTH confirmation populations and the inverse-variance pooled one-sided p < 0.05,
Holm-corrected across the hypotheses taken to confirmation. Confirmation estimates are reported
for every hypothesis regardless. Robustness for any confirmed result: 45 s horizon and the
all-frames sample. Minimum: a contrast is "not testable" if either arm has < 30 actions in discovery.

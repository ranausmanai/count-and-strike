# Pre-registration — agent "shooter": WHO should shoot?

Written before any outcome (net xG) for the analysis subset was inspected.

## Question
The confirmed rule says that near the box against a set defence, SHOT beats recycling. Does the shot advantage hold for every
player role, or only for forwards? Does prior finishing skill matter?

## Data
- `data/proc/r15_attack.parquet` (StatsBomb 360 on-ball open-play actions with `net` = own − opponent xG in the next 20 s,
  shot's own xG included, as in Round 15). Joined on (match_id, idx) to the raw event to obtain player id and position name.
- Populations: discovery = men's international (`DISC`); confirmation = men's club and women's (both required).
- Prior finishing skill: non-penalty shots from all StatsBomb data available locally (sb 2015/16 big-five, sb2 women's leagues,
  sb360) dated strictly BEFORE the decision's match date.

## Decision set
- Region: x ≥ 96 and |y − 40| ≤ 20 (edge of box and in the box). Period 1–2, open play (r15 filters).
- SHOT = `action == "shot"`. RECYCLE (the usual alternative) = `lateral`, `back_pass`, `short_fwd_ground`, `short_fwd_air`.
  All other actions are dropped.
- Role groups from the event `position.name`:
  - FW: Center Forward, Left/Right Center Forward, Secondary Striker
  - AM/W: Center/Left/Right Attacking Midfield, Left/Right Wing, Left/Right Midfield
  - CM/DM: Center/Left/Right Center Midfield, Center/Left/Right Defensive Midfield
  - DEF: Center/Left/Right Center Back, Left/Right Back, Left/Right Wing Back. Goalkeepers dropped.
- Situation cell = zone (x 96–108 / ≥108) × opponents goal-side (≤3 / 4–6 / 7+) × nearest opponent (<3 / 3–7 / >7 yd)
  × shooting-lane blockers in the shooter→posts triangle (0 / 1 / 2+). (For H3 the zone is replaced by the band x 90–102.)

## Estimator
Within each cell (separately per population and role group): diff = mean(net | SHOT) − mean(net | RECYCLE),
SE = sqrt(var_s/n_s + var_r/n_r). Cells used only if n_s ≥ 5 and n_r ≥ 5. Cells pooled with weights proportional to n_s
(the advantage on the decisions where the role actually shot); pooled SE = sqrt(Σ w² SE²)/Σ w.

## Hypotheses (all directional)
- H1 (four tests, one per role): advantage > 0 for FW, AM/W, CM/DM, DEF.
- H2 (two tests): advantage(FW) − advantage(DEF) > 0; advantage(FW) − advantage(CM/DM) > 0 (independent groups, SEs add in quadrature).
- H3 (one test): for CM/DM in x 90–102, |y−40| ≤ 20, advantage > 0.
- H4 (one test): among decisions by players with ≥ 1 prior shot, advantage(above-median finisher) − advantage(below-median) > 0,
  pooled over all role groups, cells = situation cell × role group. Skill = (G − xG)/(n_shots + 50) (shrinkage k = 50 shots).
  Median = median skill over distinct players with prior shots in that population.

## Success rule
- Discovery: the 8 tests above, two-sided p from z = est/SE, Benjamini–Hochberg 5%; must also have the hypothesised sign.
- Confirmation: for each test passing discovery, same sign in men's club AND women's, inverse-variance pooled one-sided p < 0.05,
  Holm-corrected across the number of tests carried to confirmation.
- Robustness for anything confirmed: 60 s horizon (net60), same estimator, pooled confirmation populations.
- Reported also (descriptive, not tested): per-population estimates for every test regardless of discovery outcome.

## Caveats anticipated
Choice is not randomised; a shot is taken when the player sees a chance not captured by the cell. Role differences may reflect
where/what kind of shots each role takes within a cell. Prior-skill estimates are noisy (few shots per player).

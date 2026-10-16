# Pre-registration — agent "restarts": why do quick restarts help? (360 mechanism test)

Written before any outcome (net xG) is computed on the restart sample built here. Only data-structure checks (event/frame fields) were made.

## Background
Round 21: free kicks and throw-ins taken quickly (<6 s) beat slow ones (>15 s) on net xG(20 s) in all 4 populations, but small and not confirmed (SB_360comps +0.0004 ± 0.0012).

## Sample
- StatsBomb 360 matches (`data/raw/sb360`), periods 1–2. Discovery = men's international (`r15_360.DISC`); confirmation = women's (`WOMEN`) and men's club (rest). Both confirmations required.
- Restarts = Pass events with pass type Free Kick or Throw-in, ball x ≥ 30 (own-half 30–60 zone added to the attacking half), WITH a 360 freeze frame at the restart event. Free kicks that are shots are not passes and are excluded by construction; penalties/corners not included.
- Delay = restart timestamp − (previous event timestamp + previous event duration), previous event = preceding event in file order within the same period. Exclude delay < 0 or > 90 s (implausible / stoppage). Quick = delay < 6 s, slow = delay > 15 s (same as R21); 6–15 s used only in the distribution description.
- Freeze-frame measures (actor attacks toward x = 120; keepers excluded from all counts):
  - `opp_gs` = visible opponents with x > ball x (goal-side of the ball);
  - `opp10` = visible opponents within 10 yds of the ball;
  - `mates_ahead` = visible restarting-team outfielders (non-actor) with x > ball x;
  - `n_vis` = visible players (recorded to check visibility bias).
- Outcome: net xG over next 20 s (own − opponent, non-penalty StatsBomb xG), window [t, t+20].

## Cells
- Base cell = kind (FK / TI) × zone (30–60, 60–80, 80–96, 96+) × channel (central |y−40| ≤ 20 / wide).
- Defence band for H2 = `opp_gs` in {0–4, 5–6, 7–8, 9–10}.
- Estimator: within-cell quick − slow difference, weighted by min(n_quick, n_slow); cells need ≥ 5 quick and ≥ 5 slow. SE by match-cluster bootstrap (500 reps). One-sided z tests in the stated direction.

## Hypotheses (directions fixed)
- **H1a (mechanism)**: quick restarts face FEWER opponents goal-side (quick − slow `opp_gs` < 0), base cells.
- **H1b**: quick restarts face fewer opponents within 10 yds (< 0).
- **H1c (descriptive, no test)**: quick restarts have fewer own players ahead of the ball.
- **H0 (replication)**: quick − slow net xG > 0 in base cells (x ≥ 30, all).
- **H2 (speed beyond defence count)**: quick − slow net xG > 0 within base cell × `opp_gs` band.
  Interpretation rule: if H0 is supported and H2 is confirmed → speed matters beyond defenders-goal-side. If H0 is supported but the H2 estimate is < 50% of H0 (same weighting) and H2 not confirmed → the advantage works via catching the defence unset. If H0 itself fails, neither conclusion is drawn.
- **H3a**: quick advantage larger in attacking half (x ≥ 60) than own half (30–60): difference > 0.
- **H3b**: within attacking half, quick advantage larger for throw-ins than free kicks: TI − FK > 0.
- **H4 (own-half counter launch)**: in the own-half zone (30–60), quick − slow net xG > 0.

## Success rule
Seven tests (H1a, H1b, H0, H2, H3a, H3b, H4): Benjamini–Hochberg 5% in discovery; then same sign in BOTH confirmation populations and pooled (inverse-variance) one-sided p < 0.05, Holm-corrected over the hypotheses that pass discovery. Robustness for anything confirmed: (i) 45 s horizon, (ii) finer matching adding `opp10` band (0, 1, 2+) to H2 cells.

## Caveats acknowledged in advance
360 frames show only visible players; slow restarts may have more players in view. `n_vis` will be compared and goal-side counts reported also as share of visible opponents. Delay is a choice confounded with game state and with why play stopped (foul vs ball out); within-cell matching only partly addresses this.

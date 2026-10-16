# Pre-registration — agent "runners": how many of OUR players should commit forward at a regain?

Written before any outcome (own20 / opp20 / net / opp30) for the analysis subset was inspected. Only covariate
distributions (counts of regains by population, mates_ahead, opp_goalside) were looked at.

## Context
Confirmed "Count" rule (Round 14A, `src/r13_360.py:trigger()`): counter value (own xG in 20 s) falls 11–19 % per extra
opponent goal-side at the regain. Question: holding the opponent count fixed, does it pay to have more of our own players
already ahead of the ball, and what does it cost at the other end?

## Data
- `data/proc/r13_actions.parquet`, rows with `regain_start` (first on-ball action of a Regular-Play possession won from the
  opponent), restricted to x < 80 (outside the final third), periods 1–2. Expected n: discovery 6,664 (166 matches),
  men's club 5,623 (126), women's 5,871 (125).
- `mates_ahead` = visible outfield team-mates with x greater than the ball (360 freeze frame; keeper and actor excluded).
- New extraction (`src/agent_runners_build.py`): from raw sb360 events, non-penalty shots (period, t, team, xG) to compute
  `opp30` = opponent xG in (t, t+30 s], `own45`, `opp45` (robustness horizon), and `n_vis_mates` (visible outfield team-mates)
  from the freeze frame of the same event. Joined on (match_id, period, t, team, x, y).
- Populations: discovery = men's international (`DISC`); confirmation = men's club and women's (`WOMEN`).

## Controls (all models)
rx = x/120 with rx, rx², rx³; cent = 1 − |y−40|/40; opp_goalside (linear); n_visible_opp; gpos/gneg (leading/trailing);
clk = match minute/90. Exactly the `trigger()` specification plus `mates_ahead`.

## Hypotheses
- **H1** (primary, directional +): Poisson GLM (quasi, scale = X², match-clustered SE)
  `own20 ~ controls + mates_ahead`; β(mates_ahead) > 0.
- **H2** (directional +): same GLM with `mates_ahead × band`, band = opp_goalside ≤4 / 5–6 / ≥7 (band main effects added;
  linear opp_goalside retained). Test β_slope(≤4) − β_slope(≥7) > 0 (Wald from the clustered covariance).
- **H3** (cost side, directional +, reported): Poisson GLM `opp30 ~ controls + mates_ahead`; β > 0 (more committed ⇒ more exposed).
- **H4** (net sweet spot): within each band (≤4, 5–6, ≥7) and population, OLS `net ~ C(mbin) + controls` (match-clustered SE),
  mbin = mates_ahead 0–2 / 3–4 / 5–6 / 7+. A bin–band cell is used only if n ≥ 30 in every population.
  Sweet spot per band = bin with the largest adjusted net in DISCOVERY. It "counts" only if, in ALL three populations, the
  sweet-spot bin beats every usable adjacent bin (contrast > 0), and the inverse-variance pooled confirmation contrast is
  one-sided p < 0.05 (Holm across the contrasts carried). An end bin (0–2 or 7+) as sweet spot = monotone rule ("fewer/more is better").
  Primary net = own20 − opp20 (as in the table); secondary (reported) own20 − opp30.

## Success rule
- H1, H2, H3: hypothesised sign AND one-sided p < 0.05 in each of the three populations (user's rule; stricter than the brief's).
  Also reported per the brief: discovery BH 5 % over {H1, H2, H3} two-sided, same sign in both confirmations, pooled
  (inverse-variance) confirmation one-sided p < 0.05, Holm-corrected.
- H4 as above. Verdict: CONFIRMED if H1 confirmed; MIXED if H1 not but H2/H3/H4 partly; else NOT CONFIRMED.

## Robustness (for anything confirmed)
1. 45 s horizon (own45 / opp45 / net45). 2. Finer controls: + n_vis_mates + nearest_opp (mates_ahead may partly proxy
frame visibility). 3. Sweet-spot: same contrasts with net45.

## Caveats anticipated
mates_ahead and opp_goalside are strongly linked (both reflect how far forward the game was when the ball was won); a regain
with many team-mates ahead is often a high regain in a broken game. Not randomised; "players ahead" is partly a description of
the moment, not a choice made after the regain. Visibility of the 360 frame limits counts.

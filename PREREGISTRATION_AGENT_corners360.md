# Pre-registration — agent `corners360`: how to set up a corner (360 frames)

Written before any outcome was computed. Only coverage of the visible area was inspected (median 92% of the penalty area visible at the corner frame; ~35% of corners in a 1-in-12 match sample have no frame).

## Unit and measurement
- Unit: every StatsBomb `Pass` with `pass.type = Corner`, periods 1–2, with a 360 frame.
- Frame counts (attacking team = taker's team; keepers excluded from outfield counts; taker excluded):
  - `our_six`: our players in the six-yard box (x ≥ 114, 30 ≤ y ≤ 50).
  - `our_box`: our players in the penalty area (x ≥ 102, 18 ≤ y ≤ 62), includes `our_six`.
  - `our_edge`: our visible players outside the box with x ≥ 90 (edge / recycling).
  - `our_back` = 9 − our_box − our_edge (players not near the box: rest defence + short options; assumes 11 men).
  - `opp_box`: opponent outfielders in the penalty area; `opp_six` likewise.
  - `side`: corner from y < 40 vs y > 40.
- Coverage filter (primary): ≥ 85% of the penalty area (grid of points) inside `visible_area` AND 100% of the six-yard box visible. Sensitivity: ≥ 95% of the penalty area. Coverage is reported.
- Outcomes (non-penalty shots, StatsBomb xG):
  - `own20`: our xG in [t, t+20 s] of the corner.
  - `opp30`: opponent xG in (t, t+30 s] (counter cost).
  - `net` = own20 − opp30.
- Selection control: team×competition-season fixed effects (within-demeaning of outcome and regressors), SE clustered by match. Controls: `opp_box` (linear) and `side`.

## Hypotheses (directions fixed in advance)
- **H1 (dose–response)**: slope of `own20` on `our_box` (linear) > 0, controlling for `opp_box`, `side`, team FE.
- **H2 (overload the six)**: among corners with `our_six` ≥ 3 or ≤ 1, indicator(`our_six` ≥ 3) has a positive effect on `own20`, with `our_box` stratum fixed effects (one dummy per count), `opp_box`, `side`, team FE.
- **H3 (rest-defence cost)**: slope of `opp30` on `our_box` > 0, same controls. Net-value sweet spot: FE-adjusted mean of `net` by `our_box` bin (≤3, 4, 5, 6, ≥7) reported descriptively only (no success claim).

## Success rule
- Discovery (men's international: WC22, Euro 20/24, AFCON 23): one-sided p in the pre-registered direction, Benjamini–Hochberg 5% over the 3 hypotheses.
- Confirmation: for those passing, same sign in BOTH men's club and women's, and pooled (inverse-variance) one-sided p < 0.05, Holm-corrected over the hypotheses carried forward.
- Robustness for any confirmed result: (a) 95% coverage filter, (b) horizon 45 s for own xG / 60 s for counter xG, (c) excluding short corners (pass end outside the penalty area).
- Also reported regardless of verdict: estimates in all three populations, and a pooled-all estimate.

## Caveats anticipated
Box count is chosen in reaction to the opponent and scoreline; FE removes stable team quality, not in-match context. Frames show positions at the kick, not the run pattern. Small samples (corners ~10/match) — power to detect a 0.005 xG/player effect is limited; a null is a null of that resolution.

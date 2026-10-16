# Pre-registration — agent "bait": bait and switch (engineering the far-side overload)

*Written 2026-10-16 BEFORE any outcome (net xG, and far-lane overload as a function of bait) is computed. Only non-outcome
distributions were inspected (treatment prevalence, switch counts, visibility of the far lane, opponent-count changes).
Data: StatsBomb 360 (477 matches). Discovery = men's international (`r15_360.DISC`); confirmation = men's club AND women's
(`WOMEN`), both required.*

## Decision table (`src/agent_bait_build.py` → `data/proc/agent_bait_passes.parquet`)
- Decision points: `Pass` events by the possession team, periods 1–2, not a restart (`RESTART`), start x ≥ 40, with a 360 frame,
  with an end location, and ball on a SIDE: |y − 40| > 5 (central 10-yd strip excluded; side sd = left if y < 40 else right).
- Settled possessions only: time since first event of the StatsBomb possession ≥ 8 s (so the look-back window lies inside it).
- On-ball actions = `Pass`, `Carry`, `Dribble` by the possession team (any location, restarts included as actions).
- Frames: actor attacks toward x = 120; for the same team consecutive frames are comparable.
- Lane overload (as r18a): lanes y < 26.7 / 26.7–53.3 / > 53.3; ov[lane] = visible outfield teammates (not actor) − visible outfield
  opponents with x > ball_x − 5. `far_ov` = ov of the side lane OPPOSITE the ball; `ball_ov` = ov of the ball's lane.
- `opp_bh` = visible outfield opponents on the ball's half of the width (any x).
- `far_vis` = share of an 8×5 grid over the far side lane (x from ball_x − 5 to 120) inside the frame's `visible_area`.

## Treatment definitions (fixed now)
- Look-back window W = [t − 8 s, t), same possession.
- **Bait (loose)**: ≥ 2 on-ball actions in W and ALL of them on the same side (|y−40| > 5, same half) as the current ball.
- **Bait (strict, primary, "B+")**: bait loose AND `draw` = opp_bh(now) − opp_bh(frame of the earliest framed action in W) ≥ 1.
- **No bait ("B0")**: NOT bait loose. Loose-but-not-strict decisions are excluded from B+ vs B0 comparisons.
- **Switch**: end lane differs from start lane by 2, OR lateral distance |ey − y| ≥ 30 yds with end on the opposite half.
- **Stay ball-side**: end location on the ball's half ((ey−40)(y−40) > 0).

## Matching cell (as r18a)
zone x band (40–60, 60–80, 80–96, 96–108, ≥108) × y band (left/central/right, |y−40| ≤ 13 central) × opp_goalside (≤3, 4–6, ≥7)
× nearest_opp (<3, 3–7, >7). H1 adds far_vis bin (0.5–0.8, ≥0.8).

## Estimator
Within-cell difference of means (A − B), cells with ≥ 5 in each arm, weights min(nA, nB), SE from cell variances (r18a style).

## Outcomes
- `net20` (primary), `net45`: own − opponent non-penalty xG in the 20 / 45 s after t, same period (own shots in (t−0.01, t+h],
  opponent shots in (t+0.01, t+h]), from `data/proc/r22_shots.parquet`.
- H1 outcome: `far_ov` (mechanism; not xG).

## Hypotheses (all direction "+")
- **H1 (mechanism)**: among settled side-of-pitch decisions with far_vis ≥ 0.5, mean far_ov is higher after B+ than after B0
  (cell × far_vis bin). Secondary (not in family): all frames regardless of visibility; and change far_ov(now) − far_ov(bait start).
- **H2**: among switches, net20 after B+ > after B0 (same cell).
- **H3**: among B+ decisions, switch > stay ball-side on net20 (same cell). Report switch rates after B+ and after B0.
- **H4 (possession level)**: possessions (match, period, possession) with ≥ 1 B+ decision; anchor = first B+ decision. T = some
  pass at/after the anchor in that possession is a switch made from a B+ state; control = no switch of any kind at/after the anchor
  (possessions switching only from non-B+ states are dropped). Outcome net20 from the anchor time; cell at the anchor.
  Known bias: possessions lost immediately cannot switch (survivorship favours T); the placebo and AIPW are reported but do not
  remove this.

## Success rule
- Discovery: H1–H4 two-sided p (z = est/SE), Benjamini–Hochberg 5% with hypothesised sign.
- Confirmation: same sign in men's club AND women's, inverse-variance pooled one-sided p < 0.05, Holm across tests carried forward.
- **Stress battery** (run for H2, H3, H4 regardless of the above, reported): cross-fitted AIPW ATE (5-fold GroupKFold by match,
  `r22c_stress_all.crossfit` + `r22a_strike.aipw/verdict`), and 20-s pre-decision placebo (`add_pre`, same features; placebo fails
  if pooled confirmation > 0 with one-sided p < .05). Features: H2: x, ady, opp_goalside, nearest_opp, far_vis, poss_age, minute;
  H3/H4: those + far_ov, ball_ov, mid_ov, opp_bh, mate_bh. A rule SURVIVES only if confirmed by the cell test AND AIPW verdict
  passes AND placebo does not fail.
- Robustness for any confirmed test: net45 horizon; finer cell (+ far_ov bin ≤−1/0/≥1); far_vis ≥ 0.5 subset.
- Descriptive: switch rate after B+ vs B0; share of B+ decisions with far_ov ≥ 1; loose-bait versions of H2/H3 (not in family).

## Interpretation
Even a pass would show association, not causation: bait states are chosen, and camera framing (far lane often off-screen,
median far_vis ≈ 0.3) shifts visible counts. Nulls will be reported plainly.

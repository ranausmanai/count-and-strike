# Pre-registration — agent "avail": is the overload pass actually ON?

*Written 2026-10-16 before any outcome (net xG, completion, target-lane share) is computed for the new build.
Data: StatsBomb 360 (477 matches). Discovery = men's international (`r15_360.DISC`); confirmation = men's club AND women's (`WOMEN`), both required.*

## Critic's objection
The "play into the overload" finding (r18a) may be an artefact: the overload lane pass may often not be available (blocked, too far,
receiver marked) or the overload may be created by players outside the camera's visible area. We test the finding only where the
overload pass is genuinely ON, and separately in full-width frames.

## Decision set
Rebuilt from raw 360 + events with the exact r18a logic (`src/r18a_overload.py:one`): `Pass` events in periods 1–2 by the possession team,
not a restart type (`RESTART`), location x ≥ 40, with a freeze frame. Lanes y<26.7 / 26.7–53.3 / >53.3; ov[lane] = visible outfield
teammates (not actor, not keeper) minus visible outfield opponents (not keeper) with x > ball_x − 5.
**Choice set** = passes where max_ov ≥ 1 AND min_ov ≤ −1 (both an overload and a crowded lane exist).
**T** = 1 if the pass ends in a lane with ov ≥ 1 ("overload pass"), T = 0 if it ends in a lane with ov ≤ −1 ("crowded pass"); target ov = 0 dropped.

## Availability (new)
A teammate is an **available** receiver if: (a) inside the `visible_area` polygon (ray casting; points within 1 yd of the boundary count
as inside); (b) 5–35 yds from the ball; (c) nearest opponent (all opponents incl. keeper) > 3 yds; (d) no opponent (incl. keeper) within
2 yds of the segment ball→teammate (excluding the teammate's own end point region is NOT done — the 3-yd rule in (c) already clears it);
and, to be one of the players creating the overload, x > ball_x − 5.
- `overload_on` = at least one available teammate whose lane has ov ≥ 1. `n_on_ov` = number of such teammates.
- `crowded_on` = same, for a lane with ov ≤ −1 (used for H2b).
- `full_width` = visible_area y-range covers ≤ 5 and ≥ 75 (`r20_bridge.full_width`).
Cell = r18a cell (x-zone × y-zone × opp_goalside band × nearest_opp band).
Outcomes: `net` (own − opp non-penalty xG, 20 s, `r15_360.windows`), `net45` (same, 45 s), `complete` (no pass outcome).

## Hypotheses
- **H1 (descriptive)**: among overload_on choice situations, share of passes into the crowded lane (of T∈{0,1}) is > 50%.
  Reported per population with match-clustered bootstrap 95% CI. "Holds" if the CI lower bound > 0.5 in all three populations. Not in test family.
- **H2 (primary)**: within overload_on, cell-matched mean net(T=1) − net(T=0) > 0 (r18a `within_cell_diff`: cells need ≥ 10 per arm,
  weights min(n_a, n_b), SE from within-cell variances).
- **H3 (primary)**: H2 restricted to overload_on AND full_width frames.
- **H4 (primary)**: within overload_on, cross-fitted AIPW ATE of T on net (`r22a_strike.aipw`, nuisance via `r22c_stress_all.crossfit`,
  5-fold GroupKFold by match), features x, y, opp_goalside, nearest_opp, start_ov, max_ov, min_ov, n_on_ov, crowded_on, full_width,
  n_mates_vis, n_opps_vis. **Placebo**: same AIPW on `pre_net` (net xG in the 20 s BEFORE the pass, `r22c_stress_all.add_pre`); H4
  "survives" only if the effect passes AND the placebo does not show pooled positive one-sided p < .05.
- Secondary (same estimator, reported, not in the family): completion difference within overload_on (H2c); H2b = H2 restricted to
  situations where both overload_on and crowded_on (both options genuinely on); contrast group: H2 in NOT overload_on situations.

## Success rule
- Discovery: H2, H3, H4 effect; two-sided p from z, Benjamini–Hochberg 5%, with hypothesised sign (+).
- Confirmation: each passing test needs the same sign in men's club AND women's, inverse-variance pooled one-sided p < 0.05, Holm across
  tests carried to confirmation. (`r22a_strike.verdict` logic, plus Holm.)
- Robustness for any confirmed test: net45 horizon with the same estimator; finer matching by adding n_on_ov band (1 / 2+) and
  crowded_on to the cell.
- Also reported: overload_on situations per team per match (mean, median), per population, all and full-width only.

## Verdict mapping
CONFIRMED = H2 and H4 both confirmed (and H4 placebo clean). MIXED = exactly one of H2/H3/H4 confirmed, or confirmation sign split.
NOT CONFIRMED otherwise.

## Anticipated caveats
Freeze frames show only visible players, so (a) is near-trivial; full_width is the real artefact check. Availability is geometric, not
a model of pass difficulty or of receiver body shape. Selection: players choose among options with information we lack. Not causal.
